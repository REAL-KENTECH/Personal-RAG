"""End-to-end retrieval orchestration: variants → search → fuse → rerank → balance.

The top-level entry is ``retrieve(query)``. It calls ``retrieve_local`` for
the user's documents (with optional query expansion, hybrid search, RRF
fusion, reranker, and per-doc balancing) and, if web search is enabled,
appends results from ``web_search`` for a combined context the chat
builder hands to the LLM.

Per-doc balancing (``_select_with_per_doc_min``) is on by default when
the user has multiple documents loaded, and automatically bumps the
per-doc reserve when the query looks like a comparison ("compare A and
B", "차이", "공통점", etc.) so a question about two sources doesn't end
up with all chunks coming from one of them.
"""

import streamlit as st

from .expansion import expand_queries
from .search import bm25_search, dense_search, rerank, rrf_fuse
from .web import web_search


_COMPARISON_KEYWORDS = (
    '비교', '대조', '차이', '공통점', '공통', '유사점', '다른점', '대비', '구분',
    'compare', 'comparison', 'differ', 'difference', 'similar', 'versus', ' vs ',
)


def _is_comparison_query(query: str) -> bool:
    """Heuristic — does this query ask to compare/contrast across sources?"""
    q = (query or '').lower()
    return any(kw.lower() in q for kw in _COMPARISON_KEYWORDS)


def _select_with_per_doc_min(ranked: list, top_k: int, per_doc_min: int) -> list:
    """Pick top_k from a ranked list while reserving at least `per_doc_min` slots
    per document. Web results are treated as a single 'web' bucket.

    Phase 1: walk ranked list, assign each item to its doc's quota until quota fills.
    Phase 2: fill remaining slots with the highest-scoring not-yet-selected items."""
    if per_doc_min <= 0 or top_k <= 0:
        return ranked[:top_k]
    selected = []
    used = set()
    by_doc = {}

    def _bucket(item):
        # ranked item = (idx, score, meta, chunk). meta = (doc_id, doc_name, chunk_idx) for doc.
        # For web items we pass through a different shape; treat as 'web' bucket.
        try:
            meta = item[2]
            doc_id = meta[0] if isinstance(meta, tuple) else None
            return doc_id or 'web'
        except Exception:
            return 'web'

    for item in ranked:
        if len(selected) >= top_k:
            break
        b = _bucket(item)
        if by_doc.get(b, 0) < per_doc_min:
            selected.append(item)
            used.add(id(item))
            by_doc[b] = by_doc.get(b, 0) + 1

    for item in ranked:
        if len(selected) >= top_k:
            break
        if id(item) not in used:
            selected.append(item)
            used.add(id(item))
    return selected


def _effective_per_doc_min(query: str) -> int:
    """Apply user toggles + comparison autodetect. Returns the per-doc reserve to
    use for this query, considering how many documents the user has."""
    n_docs = len(st.session_state.get('docs', []))
    if n_docs < 2:
        return 0
    reserve = 0
    if st.session_state.get('per_doc_balance'):
        reserve = max(reserve, int(st.session_state.get('per_doc_reserve', 1)))
    if (st.session_state.get('comparison_autodetect')
            and _is_comparison_query(query)):
        reserve = max(reserve, 2)
    # Don't reserve more than would fit
    final_top_k = int(st.session_state.get('final_top_k', 5))
    if reserve * n_docs > final_top_k:
        # Allow at most floor(top_k / n_docs); minimum 1 if any reserve requested
        reserve = max(1, final_top_k // n_docs)
    return reserve


def _single_query_local_search(q: str, mode: str, top_n: int):
    """Run dense+/or BM25 for one query, return RRF-fused ranking (or single)."""
    rankings = []
    if mode in ('dense', 'hybrid'):
        rankings.append(dense_search(q, top_n))
    if mode in ('bm25', 'hybrid'):
        rankings.append(bm25_search(q, top_n))
    rankings = [r for r in rankings if r]
    if not rankings:
        return []
    if len(rankings) == 1:
        return rankings[0]
    return rrf_fuse(rankings)


def retrieve_local(query: str) -> list:
    """Hybrid retrieval with optional HyDE/multi-query expansion + reranker."""
    if not st.session_state['docs']:
        st.session_state['_last_variants'] = [query]
        return []
    mode = st.session_state['retrieval_mode']
    top_n = int(st.session_state['retrieve_top_n'])
    top_k = int(st.session_state['final_top_k'])

    variants = expand_queries(query)
    st.session_state['_last_variants'] = variants
    # variants[0] is the contextually-rewritten query (or the original if no
    # rewriting happened); used as the rerank target since it best captures intent.
    base_query = variants[0] if variants else query

    per_query_rankings = []
    for q in variants:
        ranking = _single_query_local_search(q, mode, top_n)
        if ranking:
            per_query_rankings.append(ranking)

    if not per_query_rankings:
        return []
    if len(per_query_rankings) == 1:
        fused = per_query_rankings[0][:top_n]
    else:
        fused = rrf_fuse(per_query_rankings)[:top_n]

    # Rerank operates on the full top_n candidates so we have a high-quality
    # ranked list to apply per-doc balancing on top of.
    if st.session_state['use_reranker']:
        try:
            fused = rerank(base_query, fused, top_k=len(fused))
        except Exception as e:
            st.warning(f'Reranker 실패 ({e}). RRF 결과를 그대로 사용합니다.')

    per_doc_min = _effective_per_doc_min(query)
    if per_doc_min > 0:
        fused = _select_with_per_doc_min(fused, top_k, per_doc_min)
    else:
        fused = fused[:top_k]

    docs_by_id = {d['id']: d for d in st.session_state['docs']}
    out = []
    for idx, score, meta, chunk in fused:
        doc_id, doc_name, chunk_idx = meta
        doc = docs_by_id.get(doc_id, {})
        pages_list = doc.get('chunk_pages') or []
        pages = pages_list[chunk_idx] if chunk_idx < len(pages_list) else []
        out.append({
            'source': 'doc',
            'doc_id': doc_id,
            'doc': doc_name,
            'chunk_idx': chunk_idx,
            'pages': pages,
            'text': chunk,
            'score': score,
        })
    return out


def retrieve(query: str) -> list:
    """Combine local document retrieval and web search results."""
    results = retrieve_local(query)
    if st.session_state['web_enabled']:
        results = results + web_search(query)
    return results
