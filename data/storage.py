"""Per-user disk vector store + optional Supabase pgvector dual-write.

Each ingested document lands at
``<DATA_DIR>/<user_id>/<safe(embedder_id)>/<doc_id>/`` with three artifacts:

- ``meta.json`` — name, raw_text, chunks, chunk_pages, page_count, flags.
- ``embeddings.npy`` — the (n_chunks, dim) numpy matrix.
- ``pages/<page_no>.png`` — optional rendered pages for multimodal turns.

When Supabase is configured AND ``db_schema_pgvector.sql`` has been applied,
every save also pushes chunk rows into ``public.doc_chunks`` so the
embeddings survive Cloud container restarts. On a fresh container with no
disk content, ``load_all_for_current_embedder`` falls back to rebuilding
``st.session_state['docs']`` from pgvector via ``_restore_docs_from_pgvector``.
"""

import base64
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import streamlit as st

from auth.supabase_io import (
    _scrub_for_postgres,
    _supabase_client,
)
from auth.users import _log_event, _user_data_dir


def _safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)


def _embedder_dir(embedder_id: str) -> Path:
    return _user_data_dir() / _safe_name(embedder_id)


def compute_doc_id(name: str, raw_text: str, chunk_size: int, chunk_overlap: int) -> str:
    h = hashlib.sha256()
    h.update(name.encode('utf-8'))
    h.update(b'\0')
    h.update(raw_text.encode('utf-8', errors='ignore'))
    h.update(f'\0{chunk_size}\0{chunk_overlap}'.encode())
    return h.hexdigest()[:16]


def save_doc(embedder_id: str, doc: dict) -> None:
    d = _embedder_dir(embedder_id) / doc['id']
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        'id': doc['id'],
        'name': doc['name'],
        'chunks': doc['chunks'],
        'chunk_pages': doc.get('chunk_pages', []),
        'page_count': doc.get('page_count', 0),
        'has_page_images': bool(doc.get('has_page_images', False)),
        'is_pdf': bool(doc.get('is_pdf', False)),
        'raw_text': doc['raw_text'],
        'chunk_size': doc['chunk_size'],
        'chunk_overlap': doc['chunk_overlap'],
    }
    (d / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False))
    np.save(d / 'embeddings.npy', doc['embeddings'])


# -----------------------------------------------------------------------------
# pgvector dual-write (optional)
# -----------------------------------------------------------------------------
# Maps the embedder model id we use locally to the short name we store in
# ``doc_chunks.embedder`` and the vector column that holds the actual values.
_EMBEDDER_TABLE_MAP = {
    'BAAI/bge-m3': ('bge-m3', 'embedding_bgem3', 1024),
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2':
        ('paraphrase-multilingual-MiniLM-L12-v2', 'embedding_minilm', 384),
}


def _pgvector_upsert_doc(embedder_id: str, doc: dict) -> None:
    """Best-effort: send every chunk of `doc` into public.doc_chunks so the
    embeddings survive container restarts and can be queried at the DB layer
    when pgvector retrieval is enabled. No-op when Supabase isn't wired up
    or when the embedder isn't one we have a column for.

    Tracks success/failure counters in session_state (separate from the
    generic logging counters) so the cache tab can surface state. Never
    raises — local on-disk numpy arrays remain the source of truth."""
    client = _supabase_client()
    if client is None:
        return
    mapping = _EMBEDDER_TABLE_MAP.get(embedder_id)
    if mapping is None:
        return
    short_name, vec_col, _dim = mapping

    user_id = st.session_state.get('user_id', '_local')
    chunks = doc.get('chunks') or []
    embs = doc.get('embeddings')
    pages = doc.get('chunk_pages') or [[] for _ in chunks]
    if embs is None or len(embs) != len(chunks):
        return

    rows = []
    for i, ch in enumerate(chunks):
        v = embs[i]
        # supabase-py serializes lists as pgvector literals automatically;
        # numpy.ndarray needs .tolist() first.
        if hasattr(v, 'tolist'):
            v = v.tolist()
        rows.append({
            'user_id': user_id,
            'doc_id': doc['id'],
            'doc_name': doc.get('name', ''),
            'chunk_idx': i,
            'text': ch,
            'pages': pages[i] if i < len(pages) else [],
            'embedder': short_name,
            vec_col: v,
        })
    if not rows:
        return

    # Batch the upsert to keep individual requests under a few hundred KB
    # — large PDFs can have hundreds of chunks.
    BATCH = 100
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        st.session_state['_pgv_attempts'] = (
            st.session_state.get('_pgv_attempts', 0) + len(batch)
        )
        try:
            (client.table('doc_chunks')
                .upsert(
                    _scrub_for_postgres(batch),
                    on_conflict='user_id,doc_id,chunk_idx,embedder',
                )
                .execute())
            st.session_state['_pgv_successes'] = (
                st.session_state.get('_pgv_successes', 0) + len(batch)
            )
        except Exception as e:
            st.session_state['_pgv_failures'] = (
                st.session_state.get('_pgv_failures', 0) + len(batch)
            )
            st.session_state['_pgv_last_err'] = (
                f'doc_chunks batch: {type(e).__name__}: {str(e)[:600]}'
            )
            # If the very first batch fails (e.g. table missing), don't
            # keep hammering — let the rest abort silently.
            return


def _pgvector_delete_doc(doc_id: str) -> None:
    """Best-effort cleanup: when a user deletes a document locally, also
    remove its chunks from pgvector so the two stores stay aligned."""
    client = _supabase_client()
    if client is None:
        return
    user_id = st.session_state.get('user_id', '_local')
    try:
        (client.table('doc_chunks')
            .delete()
            .eq('user_id', user_id)
            .eq('doc_id', doc_id)
            .execute())
    except Exception:
        pass


def load_doc(embedder_id: str, doc_id: str):
    d = _embedder_dir(embedder_id) / doc_id
    mp = d / 'meta.json'
    ep = d / 'embeddings.npy'
    if not (mp.exists() and ep.exists()):
        return None
    try:
        meta = json.loads(mp.read_text())
        embs = np.load(ep)
    except Exception:
        return None
    return {
        'id': doc_id,
        'name': meta['name'],
        'raw_text': meta.get('raw_text', ''),
        'chunks': meta['chunks'],
        'chunk_pages': meta.get('chunk_pages', []),
        'page_count': meta.get('page_count', 0),
        'has_page_images': meta.get('has_page_images', False),
        'is_pdf': meta.get('is_pdf', False),
        'chunk_size': meta.get('chunk_size'),
        'chunk_overlap': meta.get('chunk_overlap'),
        'embeddings': embs,
    }


def _pages_dir(embedder_id: str, doc_id: str) -> Path:
    return _embedder_dir(embedder_id) / doc_id / 'pages'


@st.cache_data(show_spinner=False)
def load_page_image_b64(embedder_id: str, doc_id: str, page_no: int):
    """Read and base64-encode a rendered PDF page. Cached so multimodal turns
    don't re-read the same PNG every time."""
    p = _pages_dir(embedder_id, doc_id) / f'{page_no}.png'
    if not p.exists():
        return None
    try:
        return base64.b64encode(p.read_bytes()).decode('ascii')
    except Exception:
        return None


def delete_saved_doc(embedder_id: str, doc_id: str) -> None:
    d = _embedder_dir(embedder_id) / doc_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def list_saved_doc_ids(embedder_id: str):
    d = _embedder_dir(embedder_id)
    if not d.exists():
        return []
    return [p.name for p in d.iterdir() if p.is_dir() and (p / 'meta.json').exists()]


def _restore_docs_from_pgvector(embedder_id: str) -> list:
    """Pull every chunk this user has stored in Supabase doc_chunks and
    rebuild the in-memory docs[] list. Used on first login from a fresh
    Cloud container — the local .data/ tree is ephemeral, but the
    embeddings live in pgvector and survive restarts.

    Returns the rebuilt list (also persists each doc back to local disk so
    subsequent operations don't have to re-query). Empty list on any failure
    or when the user has no chunks yet."""
    client = _supabase_client()
    if client is None:
        return []
    mapping = _EMBEDDER_TABLE_MAP.get(embedder_id)
    if mapping is None:
        return []
    short_name, vec_col, _dim = mapping

    user_id = st.session_state.get('user_id', '_local')
    try:
        # Pull only what we need to reconstruct. Embeddings come back as
        # the vector type — supabase-py decodes pgvector to a Python list.
        # `.range(0, 9999)` is required because PostgREST default cap is 1000.
        resp = (client.table('doc_chunks')
                .select(f'doc_id, doc_name, chunk_idx, "text", pages, {vec_col}')
                .eq('user_id', user_id)
                .eq('embedder', short_name)
                .order('doc_id')
                .order('chunk_idx')
                .range(0, 9999)
                .execute())
        rows = resp.data or []
    except Exception:
        return []
    if not rows:
        return []

    # Group rows by doc_id, preserving chunk_idx order.
    docs_by_id = {}
    for r in rows:
        did = r.get('doc_id')
        if not did:
            continue
        d = docs_by_id.setdefault(did, {
            'id': did,
            'name': r.get('doc_name') or '',
            'chunks': [],
            'chunk_pages': [],
            'embeddings_list': [],
        })
        d['chunks'].append(r.get('text') or '')
        d['chunk_pages'].append(r.get('pages') or [])
        emb = r.get(vec_col)
        # PostgREST serializes pgvector values as text — e.g.
        # '[0.012, -0.005, ...]'. Decode to a Python list before we try to
        # stack them into a numpy matrix.
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except Exception:
                emb = None
        if not isinstance(emb, list) or len(emb) != mapping[2]:
            # Bad row (dim mismatch / null / decode failed) — fill with
            # zeros so the chunk index still aligns, retrieval will just
            # never match it.
            emb = [0.0] * mapping[2]
        d['embeddings_list'].append(emb)

    rebuilt = []
    for did, d in docs_by_id.items():
        if not d['chunks']:
            continue
        try:
            embs = np.asarray(d['embeddings_list'], dtype=np.float32)
        except Exception:
            # Defensive: if rows somehow have inconsistent dims after the
            # per-row check above, skip this doc entirely rather than
            # crashing the whole restore.
            continue
        # Re-normalize defensively — cosine math elsewhere assumes unit
        # length. supabase usually returns the original normalized values
        # but converting through json + back can drift floating-point.
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        embs = embs / norms

        doc = {
            'id': did,
            'name': d['name'] or did,
            'raw_text': '\n\n'.join(d['chunks']),
            'chunks': d['chunks'],
            'chunk_pages': d['chunk_pages'],
            'page_count': max(
                (max(p) for p in d['chunk_pages'] if p), default=0,
            ),
            'has_page_images': False,
            'is_pdf': False,  # cannot tell from pgvector alone
            'embeddings': embs,
            'chunk_size': 0,
            'chunk_overlap': 0,
        }
        try:
            save_doc(embedder_id, doc)
        except Exception:
            # Disk save is best-effort; the in-memory doc is still usable.
            pass
        rebuilt.append(doc)

    return rebuilt


def load_all_for_current_embedder():
    """Populate st.session_state['docs'] for the current embedder. Tries
    local disk first (fast); on a fresh Cloud container the disk is empty,
    so falls back to rebuilding from Supabase pgvector if configured.
    Skips the work if we already loaded the same embedder this session."""
    eid = st.session_state['embedder_model']
    if st.session_state.get('_loaded_for_embedder') == eid:
        return
    docs = []
    for did in list_saved_doc_ids(eid):
        d = load_doc(eid, did)
        if d is not None:
            docs.append(d)
    # Nothing on disk → try pgvector restore for logged-in users on Cloud.
    if not docs and _supabase_client() is not None:
        docs = _restore_docs_from_pgvector(eid)
        if docs:
            _log_event('docs_restored_from_pgvector', {
                'embedder': eid,
                'n_docs': len(docs),
                'n_chunks_total': sum(len(d['chunks']) for d in docs),
            })
    st.session_state['docs'] = docs
    st.session_state['_loaded_for_embedder'] = eid
