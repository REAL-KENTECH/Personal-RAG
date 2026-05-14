"""LLM-driven query expansion: contextual rewrite, multi-query, HyDE.

The chat history rewriter (``rewrite_with_context``) resolves pronouns
and implied subjects so the embedder sees a self-contained question.
``expand_queries`` then optionally adds paraphrases and/or a HyDE
hypothetical answer paragraph — each retrieved against the index
independently, then RRF-fused upstream.
"""

import re

import streamlit as st

from ..llm.clients import get_openai_client
from ..llm.params import _build_completion_params, _thinking_off_extra_body


def rewrite_with_context(query: str) -> str:
    """If conversation history exists, rewrite the current query into a
    standalone form (resolving pronouns / implied subjects). Cheap fallback
    to original on any failure."""
    history_u = st.session_state['user_inputs']
    history_a = st.session_state['generated_responses']
    if not history_u:
        return query
    try:
        client = get_openai_client()
    except Exception:
        return query
    # Use only the last 3 turns to keep this cheap.
    hist_pairs = list(zip(history_u[-3:], history_a[-3:]))
    history_str = '\n'.join(f"User: {u}\nAssistant: {a}" for u, a in hist_pairs)
    prompt = (
        "Given the conversation history and the user's latest message, rewrite "
        "the latest message as a fully self-contained question. Resolve pronouns "
        "(그것/그게/it/this), fill in implied subjects, and make the question "
        "stand on its own. If it is already self-contained, return it unchanged. "
        "Output only the rewritten sentence — no preface, no quotes.\n\n"
        f"History:\n{history_str}\n\n"
        f"Latest: {query}\n\nRewritten:"
    )
    try:
        params = _build_completion_params(
            model=st.session_state['model'],
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=200, temperature=0.0,
            extra_body=_thinking_off_extra_body(),
        )
        resp = client.chat.completions.create(**params)
        rewritten = (resp.choices[0].message.content or '').strip()
        rewritten = rewritten.strip('"').strip("'").strip()
        return rewritten or query
    except Exception as e:
        st.warning(f'쿼리 재작성 실패 ({e}). 원본 질문으로 검색합니다.')
        return query


def expand_queries(query: str) -> list:
    """Return [effective_query, paraphrase1, ..., hyde]. The first variant is
    the contextually-rewritten query if rewriting is enabled and history exists;
    otherwise it is the original query."""
    base = query
    if st.session_state.get('use_contextual_rewrite') and st.session_state['user_inputs']:
        rewritten = rewrite_with_context(query)
        if rewritten and rewritten != query:
            base = rewritten
    variants = [base]
    # Always also keep the literal user query so retrieval is robust if the
    # rewrite drifted semantically.
    if base != query:
        variants.append(query)

    if not (st.session_state.get('use_multi_query') or st.session_state.get('use_hyde')):
        return variants
    try:
        client = get_openai_client()
    except Exception:
        return variants
    model = st.session_state['model']
    eb = _thinking_off_extra_body()

    if st.session_state.get('use_multi_query'):
        n = int(st.session_state.get('n_paraphrases', 3))
        prompt = (
            f"Rewrite the following question in {n} different ways while preserving "
            f"the meaning. Output one paraphrase per line. No numbering, no quotes, "
            f"no explanation.\n\nQuestion: {base}"
        )
        try:
            mq_params = _build_completion_params(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=300, temperature=0.7,
                extra_body=eb,
            )
            resp = client.chat.completions.create(**mq_params)
            text = (resp.choices[0].message.content or '').strip()
            added = 0
            for line in text.split('\n'):
                clean = line.strip().lstrip('-•*').strip()
                # Strip leading numbering like "1." or "1)"
                m = re.match(r'^\d+\s*[.)]\s*(.*)$', clean)
                if m:
                    clean = m.group(1)
                if clean and clean not in variants:
                    variants.append(clean)
                    added += 1
                    if added >= n:
                        break
        except Exception as e:
            st.warning(f'Multi-query 생성 실패: {e}')

    if st.session_state.get('use_hyde'):
        prompt = (
            "Write a concise factual paragraph that would hypothetically answer "
            "the following question, as if extracted from an authoritative document. "
            "No preface, no commentary — just the paragraph.\n\n"
            f"Question: {base}"
        )
        try:
            hy_params = _build_completion_params(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=250, temperature=0.3,
                extra_body=eb,
            )
            resp = client.chat.completions.create(**hy_params)
            hyde = (resp.choices[0].message.content or '').strip()
            if hyde:
                variants.append(hyde)
        except Exception as e:
            st.warning(f'HyDE 생성 실패: {e}')

    return variants
