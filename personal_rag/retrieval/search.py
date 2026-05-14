"""Primitive scorers — BM25, dense (numpy + pgvector), RRF, reranker.

Each scorer returns the same shape so the orchestrator can pipe them
together freely:

    list of (chunk_idx_flat, score, meta_tuple, chunk_text)

``meta_tuple`` is ``(doc_id, doc_name, chunk_idx_within_doc)`` for local
documents. RRF fuses any number of such ranked lists; rerank takes the
fused list back into a reordered list of the same shape.
"""

import re

import numpy as np
import streamlit as st

from ..auth.supabase_io import _supabase_client
from ..config import RERANKER_MODEL
from ..data.storage import _EMBEDDER_TABLE_MAP
from ..llm.clients import load_embedder, load_reranker


def _tokenize_for_bm25(text: str):
    # Mixed-script tokenization: words/CJK/numbers. Lowercase ASCII.
    text = (text or '').lower()
    return re.findall(r"[\w가-힯一-鿿]+", text)


def build_bm25_over_docs(docs: list):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None, None
    if not docs:
        return None, None
    all_chunks, meta = [], []
    for doc in docs:
        for i, ch in enumerate(doc['chunks']):
            all_chunks.append(ch)
            meta.append((doc['id'], doc['name'], i))
    tokenized = [_tokenize_for_bm25(c) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, meta


def _flatten_chunks(docs: list):
    all_chunks, all_embs, meta = [], [], []
    for doc in docs:
        embs = doc.get('embeddings')
        if embs is None or len(embs) != len(doc['chunks']):
            continue
        for i, ch in enumerate(doc['chunks']):
            all_chunks.append(ch)
            meta.append((doc['id'], doc['name'], i))
        all_embs.append(embs)
    if not all_embs:
        return [], np.zeros((0, 1), dtype=np.float32), []
    return all_chunks, np.vstack(all_embs), meta


def _dense_search_pgvector(query: str, top_n: int):
    """Top-k cosine search via Supabase pgvector RPC. Returns the same shape
    as the in-memory variant — (chunk_idx_within_doc, score, meta, text) —
    where meta is (doc_id, doc_name, chunk_idx). Falls back to None on any
    issue so the caller can revert to the in-memory path."""
    client = _supabase_client()
    if client is None:
        return None
    embedder_id = st.session_state['embedder_model']
    mapping = _EMBEDDER_TABLE_MAP.get(embedder_id)
    if mapping is None:
        return None
    _short_name, _vec_col, _dim = mapping
    rpc_name = ('match_chunks_minilm' if embedder_id.endswith('MiniLM-L12-v2')
                else 'match_chunks_bgem3')

    embedder = load_embedder(embedder_id)
    q = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Honor the chat-side document filter so pgvector doesn't return chunks
    # the user has hidden.
    doc_filter = st.session_state.get('chat_doc_filter') or []
    docs = st.session_state.get('docs') or []
    if doc_filter and len(doc_filter) < len(docs):
        p_doc_ids = list(doc_filter)
    else:
        p_doc_ids = None

    try:
        params = {
            'p_user_id': st.session_state.get('user_id', '_local'),
            'p_query_embedding': q.tolist(),
            'p_match_count': int(top_n),
            'p_doc_ids': p_doc_ids,
        }
        resp = client.rpc(rpc_name, params).execute()
        rows = resp.data or []
    except Exception as e:
        st.session_state['_pgv_search_last_err'] = (
            f'{rpc_name}: {type(e).__name__}: {str(e)[:600]}'
        )
        return None

    if not rows:
        # Empty result is a legitimate answer; surface it as such instead
        # of silently falling back, so users notice if their data isn't in
        # pgvector yet (e.g. ingested before the schema was applied).
        st.session_state['_pgv_search_last_n'] = 0
        return []

    out = []
    for r in rows:
        meta = (r.get('doc_id'), r.get('doc_name'), int(r.get('chunk_idx', 0)))
        out.append((
            int(r.get('id', 0)),
            float(r.get('score', 0.0)),
            meta,
            r.get('text') or '',
        ))
    st.session_state['_pgv_search_last_n'] = len(out)
    return out


def dense_search(query: str, top_n: int):
    # Route to pgvector when the user has opted in (Phase 2b). On any
    # failure or when the result is unusable, transparently fall back to
    # the in-memory numpy path.
    if st.session_state.get('use_pgvector_search'):
        pg = _dense_search_pgvector(query, top_n)
        if pg is not None:
            return pg

    docs = st.session_state['docs']
    chunks, embs, meta = _flatten_chunks(docs)
    if not chunks:
        return []
    embedder = load_embedder(st.session_state['embedder_model'])
    q = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    sims = embs @ q
    idx = np.argsort(-sims)[:top_n]
    return [(int(i), float(sims[i]), meta[i], chunks[i]) for i in idx]


def bm25_search(query: str, top_n: int):
    docs = st.session_state['docs']
    bm25, meta = build_bm25_over_docs(docs)
    if bm25 is None or not meta:
        return []
    chunks, _, _ = _flatten_chunks(docs)
    scores = bm25.get_scores(_tokenize_for_bm25(query))
    idx = np.argsort(-scores)[:top_n]
    return [(int(i), float(scores[i]), meta[i], chunks[i]) for i in idx]


def rrf_fuse(rankings: list, k: int = 60):
    """rankings: list of [(idx, score, meta, chunk), ...]; fuse via Reciprocal Rank Fusion."""
    rrf_scores = {}
    payload = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            idx = item[0]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
            payload[idx] = item
    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])
    out = []
    for idx, s in fused:
        i, _, m, c = payload[idx]
        out.append((i, s, m, c))
    return out


def rerank(query: str, candidates: list, top_k: int):
    if not candidates:
        return []
    reranker = load_reranker(RERANKER_MODEL)
    pairs = [[query, c[3]] for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: -float(x[1]))
    out = []
    for (i, _, m, c), s in scored[:top_k]:
        out.append((i, float(s), m, c))
    return out
