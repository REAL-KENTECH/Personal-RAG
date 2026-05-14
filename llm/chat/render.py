"""Streamlit-side assistant turn rendering — answer, citations, reasoning.

``render_assistant`` is the one entry point view code (or chat history
replay) uses to paint a single assistant turn. Citation summaries get
their own expanders, with cited vs. uncited results visually separated.
"""

import streamlit as st

from .citations import format_answer_with_citations, parse_citations
from .messages import _format_pages


def render_assistant(answer: str, reasoning: str, retrieved: list, turn_idx: int,
                     variants: list = None):
    if variants and len(variants) > 1:
        with st.expander(
            f'검색에 사용된 쿼리 변형 {len(variants)}개', expanded=False
        ):
            for i, v in enumerate(variants):
                label = '원본 질문' if i == 0 else f'변형 {i}'
                preview = v if len(v) <= 300 else v[:300] + '...'
                st.markdown(f'**{label}**: {preview}')
    if reasoning:
        with st.expander(f'추론 과정 (turn {turn_idx + 1})', expanded=False):
            st.markdown(reasoning)

    n = len(retrieved)
    pretty = format_answer_with_citations(answer, n) if n else (answer or '')
    st.markdown(pretty or '*(empty response)*')

    if not retrieved:
        return

    cited = parse_citations(answer, n)
    cited_items = [(i + 1, r) for i, r in enumerate(retrieved) if (i + 1) in cited]
    uncited_items = [(i + 1, r) for i, r in enumerate(retrieved) if (i + 1) not in cited]

    if cited_items:
        st.markdown(f"**출처** · 인용 {len(cited_items)} / 검색 {n}")
        for j, r in cited_items:
            with st.expander(_citation_summary(j, r), expanded=False):
                _citation_body(r)
        if uncited_items:
            with st.expander(
                f'인용되지 않은 검색 결과 {len(uncited_items)}개 보기', expanded=False
            ):
                for j, r in uncited_items:
                    st.markdown(f"**{_citation_summary(j, r)}**")
                    _citation_body(r)
                    st.divider()
    else:
        with st.expander(
            f'검색된 자료 {n}개 (모델이 [N] 인용 표기를 사용하지 않음)', expanded=False
        ):
            for i, r in enumerate(retrieved):
                st.markdown(f"**{_citation_summary(i + 1, r)}**")
                _citation_body(r)
                st.divider()


def _citation_summary(j: int, r: dict) -> str:
    from urllib.parse import urlparse
    if r.get('source') == 'web':
        host = urlparse(r.get('url', '')).netloc or '웹'
        title = (r.get('doc') or '').strip()
        return f"[{j}] 웹 · {host} — {title[:60]}"
    score = r.get('score', 0.0)
    pages = r.get('pages') or []
    page_part = _format_pages(pages)
    return (
        f"[{j}] {r.get('doc', '')}{page_part} · "
        f"chunk {r.get('chunk_idx', 0)} · score {score:.3f}"
    )


def _citation_body(r: dict):
    if r.get('source') == 'web':
        url = r.get('url', '')
        if url:
            st.markdown(f"[{url}]({url})")
    st.text(r.get('text', ''))
