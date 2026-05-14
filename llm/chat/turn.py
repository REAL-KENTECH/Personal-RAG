"""Top-level chat turn orchestrator.

``handle_chat_turn`` wires retrieval → message build → LLM call →
render → persistence into one function. Every other module in
``llm.chat`` provides one of these pieces.
"""

import datetime
import time

import streamlit as st

from data.sessions import _new_session_id, save_current_session
from llm.clients import _active_api_key, get_openai_client
from llm.params import (
    _build_completion_params,
    _is_dashscope_endpoint,
    _is_hf_router_endpoint,
    _is_openai_endpoint,
    _provider_supports_top_k,
)
from retrieval.pipeline import retrieve

from .agentic import agentic_chat_pass
from .citations import split_thinking
from .errors import _show_llm_error
from .messages import build_messages
from .render import render_assistant
from .session_io import auto_title_session, log_turn_structured
from .streaming import non_stream_chat, stream_chat


def handle_chat_turn(user_input: str):
    """Process one user turn: retrieve, call LLM, render, persist to history."""
    if (not st.session_state['model']
            or not _active_api_key()
            or not st.session_state['base_url']):
        provider = st.session_state.get('provider', 'Hugging Face Router')
        st.error(
            f'설정 탭에서 모델 · 엔드포인트 · {provider} 용 API 키를 모두 입력해 주세요.'
        )
        return

    general_only = bool(st.session_state.get('general_chat_mode'))
    local_active = (not general_only) and bool(st.session_state['docs'])
    web_active = (not general_only) and bool(st.session_state['web_enabled'])
    rag_active = local_active or web_active

    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        retrieved = []
        if rag_active:
            if local_active and web_active:
                msg = '문서 및 웹 검색 중...'
            elif web_active:
                msg = '웹 검색 중...'
            else:
                msg = '문서 검색 중...'
            # Apply chat-time document filter (None = use all)
            all_docs_full = st.session_state['docs']
            chat_filter = st.session_state.get('chat_doc_filter')
            need_filter = (
                chat_filter is not None
                and 0 < len(chat_filter) < len(all_docs_full)
            )
            if need_filter:
                filt_set = set(chat_filter)
                st.session_state['docs'] = [
                    d for d in all_docs_full if d['id'] in filt_set
                ]
            try:
                with st.spinner(msg):
                    retrieved = retrieve(user_input)
            finally:
                if need_filter:
                    st.session_state['docs'] = all_docs_full

        messages = build_messages(user_input, retrieved)

        is_openai = _is_openai_endpoint()
        is_dashscope = _is_dashscope_endpoint()
        extra_body = {}
        if _provider_supports_top_k():
            extra_body['top_k'] = int(st.session_state['sampling_top_k'])
        if not st.session_state['enable_thinking']:
            if is_dashscope:
                extra_body['enable_thinking'] = False
            elif not is_openai and not _is_hf_router_endpoint():
                # vLLM / SGLang / self-hosted only. HF Router providers vary
                # and several (e.g. Cerebras-served Llama / gpt-oss) reject
                # chat_template_kwargs with HTTP 400.
                extra_body['chat_template_kwargs'] = {'enable_thinking': False}

        params = _build_completion_params(
            model=st.session_state['model'],
            messages=messages,
            max_tokens=st.session_state['max_tokens'],
            temperature=st.session_state['temperature'],
            top_p=st.session_state['top_p'],
            presence_penalty=st.session_state['presence_penalty'],
            extra_body=extra_body,
        )

        full_text, reasoning_text = '', ''
        elapsed_sec = None
        try:
            client = get_openai_client()
            _t0 = time.time()
            # Agentic mode: only meaningful when we have local docs to search
            # against. Web-only retrieval would need a different tool.
            if st.session_state.get('use_agentic_search') and local_active:
                max_iters = int(st.session_state.get('agentic_max_iters', 3))
                full_text, reasoning_text, retrieved = agentic_chat_pass(
                    client, params, retrieved, max_iters=max_iters,
                )
            elif st.session_state['stream']:
                full_text, reasoning_text = stream_chat(client, params)
            else:
                with st.spinner('생각 중…'):
                    full_text, reasoning_text = non_stream_chat(client, params)
            elapsed_sec = time.time() - _t0
        except Exception as e:
            _show_llm_error(e)
            full_text, reasoning_text = '', ''

        if full_text and not reasoning_text:
            rt, ct = split_thinking(full_text)
            if rt:
                reasoning_text = rt
                full_text = ct

        if full_text or reasoning_text:
            turn_variants = st.session_state.get('_last_variants') or [user_input]
            st.session_state['user_inputs'].append(user_input)
            st.session_state['generated_responses'].append(full_text)
            st.session_state['thinking_traces'].append(reasoning_text)
            st.session_state['retrieved_per_turn'].append(retrieved)
            st.session_state['query_variants_per_turn'].append(turn_variants)
            render_assistant(
                full_text, reasoning_text, retrieved,
                len(st.session_state['generated_responses']) - 1,
                variants=turn_variants,
            )
            # Persist this conversation. Create a session ID on first turn.
            if not st.session_state.get('current_session_id'):
                st.session_state['current_session_id'] = _new_session_id()
                st.session_state['current_session_created_at'] = (
                    datetime.datetime.now().isoformat()
                )
            save_current_session()
            # Auto-title once, after the very first turn.
            if (len(st.session_state['user_inputs']) == 1
                    and not st.session_state.get('current_session_title')):
                title = auto_title_session()
                if title:
                    st.session_state['current_session_title'] = title
                    save_current_session()
            # Append this turn to the per-session structured JSONL log.
            log_turn_structured(
                user_input=user_input,
                response_text=full_text,
                reasoning=reasoning_text,
                retrieved=retrieved,
                model=st.session_state['model'],
                elapsed_sec=elapsed_sec,
                query_variants=turn_variants,
            )
