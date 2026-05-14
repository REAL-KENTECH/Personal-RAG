"""Agent task execution + audit logging.

``run_agent_task`` is the single entry point the view layer calls. It:

1. Builds the retrieval query via the template's formula and runs
   ``retrieve_local`` (optionally with a per-doc filter and the
   per-doc balance boost for the comparison task) plus ``web_search``.
2. Hands the template's messages to ``_agent_run_llm``, which respects
   the user's stream / thinking / sampling preferences.
3. Persists the run to ``./logs/{user_id}/agents.jsonl`` and to
   Supabase ``agent_runs`` for cross-device audit.
"""

import datetime
import json
import time

import streamlit as st

from ..auth.supabase_io import _supabase_insert
from ..auth.users import _agent_log_path
from ..llm.chat import non_stream_chat, split_thinking, stream_chat
from ..llm.clients import get_openai_client
from ..llm.params import (
    _build_completion_params,
    _is_dashscope_endpoint,
    _is_hf_router_endpoint,
    _is_openai_endpoint,
    _provider_supports_top_k,
)
from ..retrieval.pipeline import retrieve_local
from ..retrieval.web import web_search
from .templates import AGENT_TASKS


def _agent_run_llm(messages: list, model: str):
    """Common runner: builds params, streams or non-streams, post-processes."""
    extra_body = {}
    if _provider_supports_top_k():
        extra_body['top_k'] = int(st.session_state['sampling_top_k'])
    if not st.session_state['enable_thinking']:
        if _is_dashscope_endpoint():
            extra_body['enable_thinking'] = False
        elif not _is_openai_endpoint() and not _is_hf_router_endpoint():
            extra_body['chat_template_kwargs'] = {'enable_thinking': False}

    params = _build_completion_params(
        model=model, messages=messages,
        max_tokens=st.session_state['max_tokens'],
        temperature=st.session_state['temperature'],
        top_p=st.session_state['top_p'],
        presence_penalty=st.session_state['presence_penalty'],
        extra_body=extra_body,
    )
    client = get_openai_client()
    t0 = time.time()
    if st.session_state['stream']:
        full_text, reasoning = stream_chat(client, params)
    else:
        with st.spinner('생성 중...'):
            full_text, reasoning = non_stream_chat(client, params)
    elapsed = time.time() - t0

    if full_text and not reasoning:
        rt, ct = split_thinking(full_text)
        if rt:
            reasoning, full_text = rt, ct
    return full_text, reasoning, elapsed


def _agent_log(task_key: str, inputs: dict, output: str, retrieved: list,
               model: str, elapsed: float):
    """Append one agent run to ./logs/agents.jsonl. Same schema family as chat
    JSONL — analysis-friendly."""
    # Slim: 200-char preview + reference, matches log_turn_structured.
    retrieved_records = []
    for j, r in enumerate(retrieved or [], start=1):
        rec = {'rank': j, 'source': r.get('source'),
               'score': r.get('score'),
               'text_preview': (r.get('text') or '')[:200]}
        if r.get('source') == 'web':
            rec['title'] = r.get('doc'); rec['url'] = r.get('url')
        else:
            rec['doc_id'] = r.get('doc_id')
            rec['doc_name'] = r.get('doc')
            rec['chunk_index'] = r.get('chunk_idx')
            rec['pages'] = r.get('pages') or []
        retrieved_records.append(rec)

    record = {
        'kind': 'agent',
        'task': task_key,
        'timestamp': datetime.datetime.now().isoformat(timespec='microseconds'),
        'user_id': st.session_state.get('user_id', '_local'),
        'inputs': inputs,
        'output': output or '',
        'model': model,
        'provider': st.session_state.get('provider', ''),
        'elapsed_seconds': float(elapsed) if elapsed is not None else None,
        'retrieved': retrieved_records,
        'n_retrieved': len(retrieved_records),
    }
    try:
        with _agent_log_path().open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception as e:
        st.warning(f'에이전트 로그 저장 실패: {e}')
    _supabase_insert('agent_runs', record)


def run_agent_task(task_key: str, inputs: dict, doc_ids_filter: list = None):
    """Run one agent task: retrieve → build messages → call LLM → log → return.

    If doc_ids_filter is given and shorter than the full doc list, retrieval is
    scoped to just those documents (state is swapped during retrieval and
    restored in finally).
    """
    task = AGENT_TASKS[task_key]

    query = task['search_query'](inputs)
    retrieved = []
    if query.strip():
        if st.session_state['docs']:
            all_docs = st.session_state['docs']
            # None        → use every loaded doc (selector wasn't shown)
            # empty list  → user explicitly unchecked all → retrieve nothing
            # subset      → filter to those doc_ids
            # full set    → equivalent to None, skip the swap
            need_filter = (
                doc_ids_filter is not None
                and len(doc_ids_filter) < len(all_docs)
            )
            if need_filter:
                filt_set = set(doc_ids_filter)
                st.session_state['docs'] = [d for d in all_docs if d['id'] in filt_set]
            try:
                if task_key == 'comparison':
                    _saved_bal = st.session_state.get('per_doc_balance')
                    st.session_state['per_doc_balance'] = True
                    try:
                        retrieved = retrieve_local(query.strip())
                    finally:
                        st.session_state['per_doc_balance'] = _saved_bal
                else:
                    retrieved = retrieve_local(query.strip())
            finally:
                if need_filter:
                    st.session_state['docs'] = all_docs
        if st.session_state['web_enabled']:
            retrieved = retrieved + web_search(query.strip())

    messages = task['build_messages'](inputs, retrieved)
    full_text, reasoning, elapsed = _agent_run_llm(messages, st.session_state['model'])

    # Augment inputs with doc selection for the audit log
    log_inputs = dict(inputs)
    if doc_ids_filter is not None:
        id_to_name = {d['id']: d['name'] for d in st.session_state['docs']}
        log_inputs['_selected_documents'] = [
            id_to_name.get(did, did) for did in doc_ids_filter
        ]

    _agent_log(task_key, log_inputs, full_text, retrieved,
               st.session_state['model'], elapsed)
    return full_text, reasoning, retrieved, elapsed
