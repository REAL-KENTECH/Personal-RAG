"""
Personal RAG Chatbot — 2026 stack.

Features
--------
- OpenAI-compatible chat completions (HF Router / DashScope / vLLM / local)
- Streaming responses with reasoning_content split (Gemma-4 / R1 / Qwen3)
- BGE-M3 (default) or MiniLM multilingual embeddings — sidebar selectable
- Hybrid retrieval: BM25 + dense + Reciprocal Rank Fusion
- Cross-encoder reranker (BAAI/bge-reranker-v2-m3) — optional
- Persistent vector store on disk (`./.data/{embedder}/{doc_hash}/`)
- Inline [N] citation parsing + per-chunk popover annotations
- HF Hub cache list/delete
- Configurable chunking (size, overlap)
"""

import base64
import datetime
import hashlib
import json
import os
import re
import shutil
import uuid
import warnings
from pathlib import Path

# Silence transformers deprecation logs before any heavy import.
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message=r'.*__path__.*')

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from personal_rag.auth.prefs import _load_user_prefs, _save_user_prefs
from personal_rag.auth.supabase_io import (
    _scrub_for_postgres,
    _supabase_client,
    _supabase_insert,
    _supabase_login,
    _supabase_save_prefs,
    _supabase_signup,
    _supabase_users_enabled,
)
from personal_rag.auth.users import (
    USERS_FROM_SECRETS,
    _agent_log_path,
    _auth_gate,
    _events_log_path,
    _is_streamlit_cloud,
    _log_event,
    _safe_uid,
    _user_data_dir,
    _user_logs_dir,
    _user_sessions_dir,
)
from personal_rag.branding import FAVICON, LOGO_URI
from personal_rag.config import (
    APP_CSS,
    DATA_DIR,
    EMBEDDER_CHOICES,
    LOGS_DIR,
    PROVIDER_MODELS,
    PROVIDER_NAMES,
    PROVIDERS,
    RERANKER_MODEL,
    _CUSTOM,
)
from personal_rag.data.sessions import (
    _new_session_id,
    _session_jsonl_path,
    _session_path,
    _supabase_list_sessions,
    _supabase_load_session,
    delete_session,
    list_sessions,
    load_session,
    rename_session,
    save_current_session,
    start_new_session,
)
from personal_rag.data.storage import (
    _EMBEDDER_TABLE_MAP,
    _embedder_dir,
    _pages_dir,
    _pgvector_delete_doc,
    _pgvector_upsert_doc,
    _restore_docs_from_pgvector,
    _safe_name,
    compute_doc_id,
    delete_saved_doc,
    list_saved_doc_ids,
    load_all_for_current_embedder,
    load_doc,
    load_page_image_b64,
    save_doc,
)
from personal_rag.llm.chat import (
    RAG_SYSTEM_PROMPT,
    _CITE_PATTERN,
    _SEARCH_TOOL_DEF,
    _citation_body,
    _citation_summary,
    _collect_page_image_parts,
    _context_label,
    _format_pages,
    _format_tool_search_result,
    _record_response_model,
    _show_llm_error,
    agentic_chat_pass,
    auto_title_session,
    build_messages,
    format_answer_with_citations,
    handle_chat_turn,
    log_turn_structured,
    non_stream_chat,
    parse_citations,
    render_assistant,
    split_thinking,
    stream_chat,
)
from personal_rag.llm.clients import (
    _active_api_key,
    _make_openai_client,
    get_openai_client,
    load_embedder,
    load_reranker,
)
from personal_rag.processing.chunking import (
    _SENT_SPLIT_RE,
    _expand_into_paragraphs,
    _split_long_text_to_sentences,
    chunk_elements,
    chunk_text,
)
from personal_rag.processing.ingestion import (
    _embed_with_progress,
    ingest_files,
    reindex_all,
    remove_doc,
)
from personal_rag.retrieval.expansion import expand_queries, rewrite_with_context
from personal_rag.retrieval.pipeline import (
    _COMPARISON_KEYWORDS,
    _effective_per_doc_min,
    _is_comparison_query,
    _select_with_per_doc_min,
    _single_query_local_search,
    retrieve,
    retrieve_local,
)
from personal_rag.retrieval.search import (
    _dense_search_pgvector,
    _flatten_chunks,
    _tokenize_for_bm25,
    bm25_search,
    build_bm25_over_docs,
    dense_search,
    rerank,
    rrf_fuse,
)
from personal_rag.retrieval.web import web_search
from personal_rag.agents.runner import _agent_log, _agent_run_llm, run_agent_task
from personal_rag.agents.templates import (
    AGENT_TASKS,
    _agent_format_context,
    _analysis_build,
    _comparison_build,
    _email_build,
    _report_build,
    _summary_build,
)
from personal_rag.ui.helpers import _chip, _empty, _section
from personal_rag.ui.sidebar import NAV, NAV_KEYS, render_sidebar
from personal_rag.ui.widgets import model_picker
from personal_rag.processing.parsing import (
    _read_bytes,
    parse_csv_bytes,
    parse_docx_bytes,
    parse_file,
    parse_hwpx_bytes,
    parse_pdf_docling,
    parse_pdf_ocr,
    parse_pdf_pypdf,
    render_pdf_pages_to_dir,
)
from personal_rag.llm.params import (
    _DEPRECATED_MODEL_SWAPS,
    _build_completion_params,
    _is_anthropic_endpoint,
    _is_dashscope_endpoint,
    _is_fireworks_endpoint,
    _is_hf_router_endpoint,
    _is_openai_endpoint,
    _is_openai_gpt5_family,
    _is_openai_reasoning_model,
    _provider_supports_top_k,
    _resolve_deprecated_model,
    _thinking_off_extra_body,
    _uses_max_completion_tokens,
)
from personal_rag.state import _init_state

load_dotenv(Path(__file__).parent / '.env')
load_dotenv(Path(__file__).parent.parent / '.env')

# Bridge: when running on Streamlit Cloud, secrets live in st.secrets, not in
# a .env file. Promote them to os.environ so the existing os.getenv code path
# works unchanged. Local .env values (loaded above) take priority on the dev
# machine because we only set keys that aren't already in os.environ.
try:
    if hasattr(st, 'secrets'):
        for _k, _v in dict(st.secrets).items():
            if isinstance(_v, (str, int, float, bool)) and _k not in os.environ:
                os.environ[_k] = str(_v)
except Exception:
    pass

st.set_page_config(
    page_title='Personal RAG',
    page_icon=FAVICON,
    layout='wide',
    initial_sidebar_state='expanded',
)

# CSS — minimal custom styling, defined in personal_rag/config.py.
# We deliberately do NOT force display/width on Streamlit's own layout
# elements (stSidebar, stHeader, etc.) so the framework can manage its own
# responsive behavior and the sidebar can be collapsed/reopened normally.
st.markdown(APP_CSS, unsafe_allow_html=True)

# Ensure the on-disk roots exist before any view tries to write to them.
# DATA_DIR / LOGS_DIR are declared as Path constants in personal_rag/config.py.
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Per-user filesystem layout, Supabase RPCs, event logging, and the login
# gate live in personal_rag/auth/. Provider/model/embedder catalogs live in
# personal_rag/config.py. The imports above re-expose those names locally so
# existing call sites further down this file still resolve.


# =============================================================================
# Session state
# =============================================================================
# Default values for ``st.session_state`` live in personal_rag/state.py.
# We call _init_state() here so every later code path can read settings
# off st.session_state without needing to seed defaults itself.
_init_state()


# ---------- Boot sequence ----------
# The login gate must resolve a user_id BEFORE prefs load, because prefs are
# read from per-user paths under DATA_DIR/<user_id>/. Both live in
# personal_rag/auth/ and only the call sites remain here.
_auth_gate()
_load_user_prefs()


# =============================================================================
# Persistent vector store + Conversation sessions
# =============================================================================
# Disk + pgvector storage lives in personal_rag/data/storage.py; conversation
# session CRUD lives in personal_rag/data/sessions.py. Names are imported at
# the top of this file. log_turn_structured + auto_title_session remain here
# because they straddle the chat-helpers boundary and will move alongside
# handle_chat_turn in a later pass.








# =============================================================================
# Boot: load persisted docs for current embedder
# =============================================================================

load_all_for_current_embedder()


# =============================================================================
# Sidebar (brand, nav, sessions, status, user)
# =============================================================================
# Implementation lives in personal_rag/ui/sidebar.py; this call runs on every
# rerun so the rail reflects whichever button just triggered the re-execution.
render_sidebar()


# =============================================================================
# View renderers
# =============================================================================

def view_chat():
    # Auto-restore the last active conversation if session_state was wiped
    # by an idle reconnect. preferences.json carries current_session_id; the
    # actual messages live in .data/{user}/sessions/{id}.json on disk and
    # are pulled in here by load_session().
    _restore_sid = st.session_state.get('current_session_id')
    if _restore_sid and not st.session_state.get('user_inputs'):
        try:
            load_session(_restore_sid)
        except Exception:
            pass

    local_active = bool(st.session_state['docs'])
    web_active = bool(st.session_state['web_enabled'])

    # Top bar — title + model picker + (optional) doc filter.
    has_multi_docs = len(st.session_state['docs']) >= 2
    if has_multi_docs:
        top_left, top_mid, top_right = st.columns([3, 1, 2])
    else:
        top_left, top_right = st.columns([3, 2])
        top_mid = None

    with top_left:
        title = st.session_state.get('current_session_title') or '새 대화'
        current_sid_for_rename = st.session_state.get('current_session_id')
        # Title behaves as a popover trigger when there's an actual saved
        # session — gives the user an inline rename UI. Pre-session it's
        # just static text since there's nothing to persist yet.
        if current_sid_for_rename and hasattr(st, 'popover'):
            with st.popover(title, use_container_width=False):
                with st.form('_rename_form', clear_on_submit=False):
                    new_title = st.text_input(
                        '대화 이름',
                        value=(st.session_state.get('current_session_title') or ''),
                        max_chars=60,
                        placeholder='예: 회의록 요약 / 신입사원 매뉴얼 Q&A',
                    )
                    submit = st.form_submit_button(
                        '저장', use_container_width=True, type='primary',
                    )
                if submit:
                    if rename_session(current_sid_for_rename, new_title):
                        st.success('이름을 변경했습니다.')
                        st.rerun()
                    else:
                        st.error('이름 변경에 실패했습니다.')
        else:
            st.markdown(
                f"<div style='font-size:15px; font-weight:600; padding-top:4px;'>{title}</div>",
                unsafe_allow_html=True,
            )

    # Doc filter popover (only when multiple docs are loaded).
    if has_multi_docs and top_mid is not None and hasattr(st, 'popover'):
        all_docs = st.session_state['docs']
        doc_id_to_name = {d['id']: d['name'] for d in all_docs}
        all_ids = list(doc_id_to_name.keys())
        all_id_set = set(all_ids)
        chat_filter = st.session_state.get('chat_doc_filter')
        n_total = len(all_ids)
        n_active = (
            len(chat_filter) if chat_filter is not None else n_total
        )
        with top_mid:
            with st.popover(f"문서 {n_active}/{n_total}",
                            use_container_width=True):
                # Default = filter if set (and still valid), else all
                if chat_filter is not None:
                    default_value = [d for d in chat_filter if d in all_id_set]
                else:
                    default_value = all_ids
                sel = st.multiselect(
                    '답변에 사용할 문서',
                    options=all_ids,
                    default=default_value,
                    format_func=lambda did: doc_id_to_name[did],
                    help='기본은 전체 문서. 일부만 사용하려면 체크를 조정하세요. '
                    '대화 중에도 언제든 변경할 수 있고, 다음 메시지부터 적용됩니다.',
                )
                new_filter = None if set(sel) == all_id_set else sel
                if new_filter != chat_filter:
                    st.session_state['chat_doc_filter'] = new_filter
                    st.rerun()

    with top_right:
        model_label = st.session_state['model'] or '모델 미설정'
        if len(model_label) > 36:
            model_label = model_label[:33] + '...'
        if hasattr(st, 'popover'):
            with st.popover(f"모델: {model_label}", use_container_width=True):
                provider = st.session_state.get('provider', '?')
                st.caption(f"공급자: {provider}")
                model_picker('모델 선택', key_prefix='inline', instant=True)
                if st.button('상세 설정 열기', key='inline_open_settings',
                             use_container_width=True):
                    st.session_state['active_view'] = 'settings'
                    st.rerun()
        else:
            st.caption(f"모델: {model_label}")
    st.divider()

    # Render existing history first.
    for i in range(len(st.session_state['generated_responses'])):
        with st.chat_message('user'):
            st.markdown(st.session_state['user_inputs'][i])
        with st.chat_message('assistant'):
            reasoning = (
                st.session_state['thinking_traces'][i]
                if i < len(st.session_state['thinking_traces']) else ''
            )
            retrieved = (
                st.session_state['retrieved_per_turn'][i]
                if i < len(st.session_state['retrieved_per_turn']) else []
            )
            variants = (
                st.session_state['query_variants_per_turn'][i]
                if i < len(st.session_state['query_variants_per_turn']) else []
            )
            render_assistant(
                st.session_state['generated_responses'][i],
                reasoning, retrieved, i, variants=variants,
            )

    # Process pending input from a suggestion click (rerun after click sets it).
    pending = st.session_state.pop('_pending_input', None)
    if pending:
        handle_chat_turn(pending)
    elif not st.session_state['generated_responses']:
        # Hero with full brand lockup + suggestion chips.
        logo_html = (
            f'<img src="{LOGO_URI}" '
            f'style="width:100%; max-width:340px; height:auto; '
            f'margin:0 auto 18px auto; display:block; opacity:0.95;" />'
            if LOGO_URI else ''
        )
        st.markdown(
            f'<div class="empty-hero">'
            f'{logo_html}'
            f'<h2>오늘은 어떤 걸 도와드릴까요?</h2>'
            f'<p>문서를 올려 그 내용으로 답을 받거나, 일반 챗으로 바로 시작할 수 있습니다.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.session_state['docs']:
            suggestions = [
                '업로드한 문서들을 한 문단으로 요약해 줘',
                '문서에서 가장 중요한 개념 세 가지는?',
                '문서들 사이의 주요 차이점을 비교해 줘',
                '내가 어떤 질문을 해볼 수 있을까',
            ]
        else:
            suggestions = [
                'Personal RAG가 어떻게 동작하는지 설명해 줘',
                '문서를 업로드하려면 어떻게 하는가',
                '임베딩 모델은 무엇을 골라야 좋을까',
                '검색 품질을 높이는 팁은',
            ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f'suggest_{i}', use_container_width=True):
                    st.session_state['_pending_input'] = s
                    st.rerun()

    if st.session_state['generated_responses']:
        cols = st.columns([1, 5])
        with cols[0]:
            if st.button('대화 초기화', use_container_width=True):
                st.session_state['user_inputs'] = []
                st.session_state['generated_responses'] = []
                st.session_state['thinking_traces'] = []
                st.session_state['retrieved_per_turn'] = []
                st.session_state['query_variants_per_turn'] = []
                st.rerun()

    if local_active and web_active:
        _placeholder = '문서와 웹에서 검색하여 답합니다'
    elif local_active:
        _placeholder = '문서에 대해 질문하세요'
    elif web_active:
        _placeholder = '웹 검색 후 답합니다'
    else:
        _placeholder = '메시지를 입력하세요 (.txt / .md / .pdf 파일 첨부 가능)'

    # Streamlit 1.42+: chat_input supports inline file attachments.
    # Fall back to plain chat_input on older versions.
    try:
        submitted = st.chat_input(
            _placeholder,
            accept_file='multiple',
            file_type=['txt', 'md', 'pdf', 'docx', 'csv', 'hwpx'],
        )
    except TypeError:
        submitted = st.chat_input(_placeholder)

    if submitted:
        # Normalize: newer Streamlit returns an object with .text and .files;
        # older returns a bare string.
        text_part = ''
        file_part = []
        if hasattr(submitted, 'text') or hasattr(submitted, 'files'):
            text_part = (getattr(submitted, 'text', '') or '').strip()
            file_part = list(getattr(submitted, 'files', []) or [])
        else:
            text_part = (submitted or '').strip()

        # Ingest any attached files first so they're searchable on the same turn.
        # ingest_files() appends to st.session_state['docs'] and saves to disk,
        # so the files immediately show up in the Documents tab too.
        added = 0
        if file_part:
            # ingest_files() shows its own st.status with per-batch progress.
            added = ingest_files(file_part)
            if added > 0:
                try:
                    st.toast(
                        f'{added}개 문서 인덱싱 완료 — 문서 탭에 추가되었고 '
                        '이번 질문부터 검색 대상입니다.'
                    )
                except Exception:
                    pass

        if text_part:
            handle_chat_turn(text_part)

        # Rerun after any file ingest so the sidebar status counter and
        # the Documents tab list refresh immediately (otherwise they stay
        # stale until the next user interaction).
        if added > 0:
            st.rerun()


# =============================================================================
# Documents view
# =============================================================================

def view_docs():
    _section(
        '문서 업로드',
        '.txt / .md / .pdf 파일을 업로드하면 자동으로 청킹·임베딩되어 디스크에 영속 저장됩니다. '
        '같은 파일을 다시 올려도 내용 해시가 일치하면 즉시 캐시에서 복원됩니다.',
    )
    uploaded = st.file_uploader(
        ' ',
        type=['txt', 'md', 'pdf', 'docx', 'csv', 'hwpx'],
        accept_multiple_files=True,
        label_visibility='collapsed',
    )
    if uploaded:
        # ingest_files() shows its own st.status with per-batch progress.
        added = ingest_files(uploaded)
        if added > 0:
            st.success(f'{added}개 새 문서 인덱싱 완료')
            st.rerun()

    st.write('')
    _section('인덱싱된 문서')

    if not st.session_state['docs']:
        _empty('아직 업로드된 문서가 없습니다. 위 영역에 파일을 끌어다 놓으세요.')
    else:
        total_chunks = sum(len(d['chunks']) for d in st.session_state['docs'])
        total_chars = sum(len(d.get('raw_text', '')) for d in st.session_state['docs'])
        st.caption(
            f"문서 {len(st.session_state['docs'])}개 · "
            f"청크 {total_chunks}개 · {total_chars:,}자"
        )
        for d in list(st.session_state['docs']):
            with st.container(border=True):
                row = st.columns([4, 1, 1, 1, 1])
                row[0].markdown(f"**{d['name']}**")
                row[0].caption(
                    f"id: `{d['id']}` · "
                    f"chunk_size={d.get('chunk_size', '?')} · "
                    f"overlap={d.get('chunk_overlap', '?')}"
                )
                row[1].metric('chunks', len(d['chunks']))
                row[2].metric('pages', d.get('page_count', 0) or '—')
                row[3].metric('images', 'on' if d.get('has_page_images') else 'off')
                with row[4]:
                    st.write('')
                    if st.button('삭제', key=f"del_doc_{d['id']}", use_container_width=True):
                        remove_doc(d['id'])
                        st.rerun()
        if st.button('모든 문서 삭제 (디스크 포함)', type='secondary'):
            for d in list(st.session_state['docs']):
                remove_doc(d['id'])
            st.rerun()

    st.write('')
    with st.expander('고급 청킹 설정'):
        st.caption(
            '값을 바꾸면 현재 보유한 모든 문서가 자동으로 재청킹·재임베딩됩니다. '
            '이전 설정의 캐시는 보존되어 되돌릴 수 있습니다.'
        )
        cs_cols = st.columns(2)
        with cs_cols[0]:
            new_size = st.number_input(
                '청크 크기 (문자)', 100, 4000,
                int(st.session_state['chunk_size']),
                help='하나의 청크에 들어가는 최대 글자 수.',
            )
        with cs_cols[1]:
            new_overlap = st.number_input(
                '청크 겹침 (문자)', 0, 1000,
                int(st.session_state['chunk_overlap']),
                help='이웃한 청크 사이에 중복으로 포함시킬 글자 수. 문맥 유실 방지.',
            )
        if (new_size, new_overlap) != (
            st.session_state['chunk_size'], st.session_state['chunk_overlap']
        ):
            st.session_state['chunk_size'] = new_size
            st.session_state['chunk_overlap'] = new_overlap
            if st.session_state['docs']:
                with st.spinner('재청킹 및 재임베딩 중...'):
                    reindex_all()
                st.rerun()

    st.write('')
    _section(
        '검색 미리보기',
        'LLM 호출 없이 검색 결과만 확인합니다. 어떤 청크가 어떤 점수로 잡히는지 점검하는 용도.',
    )
    preview_cols = st.columns([5, 1])
    with preview_cols[0]:
        preview_query = st.text_input(
            '검색어', key='preview_query', label_visibility='collapsed',
            placeholder='예: 매출 성장률 정의',
        )
    with preview_cols[1]:
        run_preview = st.button(
            '검색 실행',
            disabled=(not preview_query.strip() or not st.session_state['docs']),
            use_container_width=True,
        )
    if run_preview and preview_query.strip():
        with st.spinner('검색 중...'):
            results = retrieve_local(preview_query.strip())
        if not results:
            st.warning('결과 없음.')
        else:
            st.caption(f"{len(results)}개 청크 반환")
            for j, r in enumerate(results, start=1):
                with st.expander(_citation_summary(j, r), expanded=False):
                    _citation_body(r)


# =============================================================================
# Settings view
# =============================================================================

def view_settings():
    _section(
        '설정',
        '모델·검색·웹·응답을 카테고리별로 정리. 변경은 즉시 저장됩니다.',
    )

    # ----- Top status snapshot -----
    active_p = st.session_state.get('provider', 'Hugging Face Router')
    model_short = (st.session_state.get('model') or '—')
    if len(model_short) > 30:
        model_short = model_short[:27] + '...'
    embedder_label = {
        'BAAI/bge-m3': 'BGE-M3',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': 'MiniLM',
    }.get(st.session_state.get('embedder_model', ''), '—')
    has_key = bool(_active_api_key())
    with st.container(border=True):
        m = st.columns(4)
        m[0].metric('공급자', active_p)
        m[1].metric('모델', model_short)
        m[2].metric('임베더', embedder_label)
        m[3].metric('API 키', '설정됨' if has_key else '없음')

    # ----- Tabbed sections -----
    tab_llm, tab_search, tab_web, tab_response, tab_advanced = st.tabs(
        ['LLM 공급자', 'RAG 검색', '웹 검색', '응답 / 멀티모달', '고급']
    )

    # ============== LLM ==============
    with tab_llm:
        with st.container(border=True):
            st.markdown('##### 공급자 선택')
            current_provider = st.session_state.get('provider', PROVIDER_NAMES[0])
            if current_provider not in PROVIDER_NAMES:
                current_provider = PROVIDER_NAMES[0]
            new_provider = st.selectbox(
                '공급자', PROVIDER_NAMES,
                index=PROVIDER_NAMES.index(current_provider),
                help='프리셋을 선택하면 엔드포인트 주소·기본 모델·환경변수에서 API 키를 자동으로 채웁니다.',
                label_visibility='collapsed',
            )
            if new_provider != current_provider:
                cfg = PROVIDERS[new_provider]
                if cfg['base_url']:
                    st.session_state['base_url'] = cfg['base_url']
                if cfg['default_model']:
                    st.session_state['model'] = cfg['default_model']
                st.session_state['provider'] = new_provider
                st.rerun()
            else:
                st.session_state['provider'] = new_provider
            model_picker('모델', key_prefix='settings')

        # API keys — staged via pending state. Typing into a field doesn't
        # touch session_state[<active_key>] until the user presses 저장.
        # That keeps every keystroke from triggering _save_user_prefs and
        # the disk/Supabase round-trips it carries, and gives the user a
        # clear "I'm done editing" moment.
        with st.container(border=True):
            st.markdown('##### API 키')
            st.caption(
                '공급자별로 따로 저장됩니다. "(사용 중)" 표시가 현재 공급자 키. '
                '입력 후 아래 **저장** 버튼을 눌러야 반영됩니다.'
            )

            def _key_label(name, owner):
                return f'{name} (사용 중)' if owner == active_p else name

            _key_specs = [
                ('hf_api_key',         'Hugging Face',         'Hugging Face Router',
                 'hf_...',
                 'HF Inference Router (Gemma / DeepSeek / Qwen 등). '
                 'fine-grained 토큰 권장: https://huggingface.co/settings/tokens'),
                ('openai_api_key',     'OpenAI',               'OpenAI',
                 'sk-...',
                 'gpt-4o / gpt-5 / o3 등. https://platform.openai.com/api-keys'),
                ('anthropic_api_key',  'Anthropic (Claude)',   'Anthropic (Claude)',
                 'sk-ant-...',
                 'Claude 4.x. https://console.anthropic.com/settings/keys'),
                ('fireworks_api_key',  'Fireworks AI',         'Fireworks AI',
                 'fw_...',
                 '오픈모델 빠른 추론. https://fireworks.ai/account/api-keys'),
                ('dashscope_api_key',  'DashScope (Qwen)',     'DashScope (Qwen)',
                 'sk-...',
                 'Qwen 공식 API.'),
                ('custom_api_key',     'Custom / vLLM',        'vLLM / local',
                 '(self-host endpoint key)',
                 'vLLM 등 셀프 호스팅 / Custom OpenAI-호환 endpoint.'),
            ]

            def _render_key_row(active_key, label_text, owner, ph, help_text):
                pending_key = f'_pending_{active_key}'
                # First-time render seeds the widget from the active value.
                # On subsequent reruns Streamlit retains the user's typing
                # via the widget key, so we don't re-seed.
                if pending_key not in st.session_state:
                    st.session_state[pending_key] = (
                        st.session_state.get(active_key, '') or ''
                    )
                st.text_input(
                    _key_label(label_text, owner),
                    type='password',
                    placeholder=ph,
                    help=help_text,
                    key=pending_key,
                )
                pending_val = st.session_state.get(pending_key, '') or ''
                active_val = st.session_state.get(active_key, '') or ''
                if pending_val != active_val:
                    bcols = st.columns([2, 1, 1])
                    bcols[0].caption('변경 대기 — 적용을 눌러야 호출에 반영됩니다.')
                    if bcols[1].button(
                        '취소', key=f'reset_{active_key}',
                        use_container_width=True,
                    ):
                        st.session_state[pending_key] = active_val
                        st.rerun()
                    if bcols[2].button(
                        '적용', key=f'apply_{active_key}',
                        type='primary', use_container_width=True,
                    ):
                        st.session_state[active_key] = pending_val
                        try:
                            _save_user_prefs()
                        except Exception:
                            pass
                        st.rerun()

            kc1, kc2 = st.columns(2)
            for i, spec in enumerate(_key_specs):
                col = kc1 if i % 2 == 0 else kc2
                with col:
                    _render_key_row(*spec)

            # Saved-count summary at the bottom.
            set_count = sum(
                1 for active_key, *_ in _key_specs
                if (st.session_state.get(active_key, '') or '').strip()
            )
            st.caption(f'현재 {set_count}/{len(_key_specs)} 개의 키가 저장돼 있습니다.')

    # ============== RAG 검색 ==============
    with tab_search:
        # Mode + embedder
        with st.container(border=True):
            st.markdown('##### 동작 모드')
            st.session_state['general_chat_mode'] = st.checkbox(
                '일반 대화 모드 (RAG / 웹 검색 끔)',
                value=st.session_state['general_chat_mode'],
                help='업로드 문서·웹 검색을 모두 건너뛰고 LLM 본연 지식으로 답합니다.',
            )
            prev_embedder = st.session_state['embedder_model']
            emb_labels = {
                'BAAI/bge-m3': 'BGE-M3 (한국어 강함, 2.2GB)',
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2':
                    'MiniLM 다국어 (가벼움, 470MB)',
            }
            emb_idx = (
                EMBEDDER_CHOICES.index(st.session_state['embedder_model'])
                if st.session_state['embedder_model'] in EMBEDDER_CHOICES else 0
            )
            st.session_state['embedder_model'] = st.selectbox(
                '임베딩 모델', EMBEDDER_CHOICES,
                index=emb_idx,
                format_func=lambda x: emb_labels.get(x, x),
                help='문서·질문을 벡터로 변환하는 모델.',
            )
            if st.session_state['embedder_model'] != prev_embedder:
                st.session_state['_loaded_for_embedder'] = None
                load_all_for_current_embedder()
                st.rerun()

        # Retrieval pipeline
        with st.container(border=True):
            st.markdown('##### 검색 파이프라인')
            mode_labels = {'hybrid': '하이브리드 (권장)',
                           'dense': '의미 기반만', 'bm25': '키워드만'}
            st.session_state['retrieval_mode'] = st.radio(
                '검색 방식',
                ['hybrid', 'dense', 'bm25'],
                index=['hybrid', 'dense', 'bm25'].index(
                    st.session_state['retrieval_mode']
                ),
                horizontal=True,
                format_func=lambda x: mode_labels[x],
            )
            cc1, cc2 = st.columns(2)
            with cc1:
                st.session_state['use_reranker'] = st.checkbox(
                    '재정렬 모델 사용 (Cross-encoder)',
                    value=st.session_state['use_reranker'],
                    help='정확도 향상, 응답 약간 느려짐.',
                )
                st.session_state['use_contextual_rewrite'] = st.checkbox(
                    '이어지는 질문 자동 보완',
                    value=st.session_state['use_contextual_rewrite'],
                    help='이전 대화 맥락으로 self-contained 질문 재작성.',
                )
                st.session_state['per_doc_balance'] = st.checkbox(
                    '여러 문서 균형 검색',
                    value=st.session_state['per_doc_balance'],
                    help='상위 결과가 한 문서에 쏠리지 않게 강제.',
                )
                st.session_state['comparison_autodetect'] = st.checkbox(
                    '비교 질문 자동 감지',
                    value=st.session_state['comparison_autodetect'],
                )
            with cc2:
                _pgv_available = _supabase_client() is not None
                st.session_state['use_pgvector_search'] = st.checkbox(
                    'pgvector 의미 검색 (Supabase)',
                    value=st.session_state['use_pgvector_search'],
                    help='Supabase pgvector 로 dense 검색. 미연결 시 자동 폴백.',
                    disabled=not _pgv_available,
                )
                if not _pgv_available:
                    st.caption('Supabase 미연동 — 비활성')
                st.session_state['use_agentic_search'] = st.checkbox(
                    '에이전트 검색 (LLM 추가 검색)',
                    value=st.session_state['use_agentic_search'],
                    help='LLM 이 도구 호출로 추가 검색 발행. function calling 지원 모델 필요.',
                )
                if st.session_state['use_agentic_search']:
                    st.session_state['agentic_max_iters'] = st.slider(
                        '최대 라운드', 1, 5,
                        int(st.session_state.get('agentic_max_iters', 3)),
                    )

    # ============== 웹 ==============
    with tab_web:
        with st.container(border=True):
            st.markdown('##### 실시간 웹 검색')
            st.caption(
                '질문 시 웹 검색 결과를 컨텍스트에 함께 포함. DuckDuckGo 는 API 키 불필요.'
            )
            st.session_state['web_enabled'] = st.checkbox(
                '웹 검색 사용',
                value=st.session_state['web_enabled'],
            )
            if st.session_state['web_enabled']:
                wp_labels = {
                    'duckduckgo': 'DuckDuckGo (키 불필요)',
                    'tavily': 'Tavily (LLM 최적화, 키 필요)',
                    'brave': 'Brave (키 필요)',
                }
                wc1, wc2 = st.columns(2)
                with wc1:
                    st.session_state['web_provider'] = st.selectbox(
                        '검색 제공자', ['duckduckgo', 'tavily', 'brave'],
                        index=['duckduckgo', 'tavily', 'brave'].index(
                            st.session_state['web_provider']
                        ),
                        format_func=lambda x: wp_labels[x],
                    )
                with wc2:
                    st.session_state['web_top_n'] = st.number_input(
                        '결과 수', 1, 20, int(st.session_state['web_top_n']),
                    )
                if st.session_state['web_provider'] == 'tavily':
                    st.session_state['tavily_key'] = st.text_input(
                        'Tavily API 키', st.session_state['tavily_key'],
                        type='password', placeholder='tvly-...',
                    )
                elif st.session_state['web_provider'] == 'brave':
                    st.session_state['brave_key'] = st.text_input(
                        'Brave API 키', st.session_state['brave_key'],
                        type='password',
                    )
            else:
                st.caption('웹 검색이 꺼져 있어 추가 설정이 보이지 않습니다.')

    # ============== 응답 / 멀티모달 ==============
    with tab_response:
        with st.container(border=True):
            st.markdown('##### 응답 동작')
            rc1, rc2 = st.columns(2)
            with rc1:
                st.session_state['stream'] = st.checkbox(
                    '스트리밍 응답', value=st.session_state['stream'],
                    help='응답을 토큰 단위로 실시간 표시.',
                )
            with rc2:
                st.session_state['enable_thinking'] = st.checkbox(
                    '추론 모드 사용', value=st.session_state['enable_thinking'],
                    help='지원 모델에서 reasoning 토큰을 분리 표시.',
                )

        with st.container(border=True):
            st.markdown('##### 멀티모달 (이미지)')
            st.caption(
                'PDF 페이지 이미지를 LLM 에 함께 전달해 표·차트·도식을 이해하게 합니다. '
                '비전 입력 지원 모델 필요.'
            )
            st.session_state['include_page_images'] = st.checkbox(
                'PDF 페이지 이미지 첨부',
                value=st.session_state['include_page_images'],
            )
            if st.session_state['include_page_images']:
                st.session_state['max_page_images'] = st.number_input(
                    '한 턴에 보낼 이미지 수', 1, 10,
                    int(st.session_state['max_page_images']),
                )

    # ============== 고급 ==============
    with tab_advanced:
        with st.container(border=True):
            st.markdown('##### 엔드포인트')
            st.session_state['base_url'] = st.text_input(
                '엔드포인트 주소', st.session_state['base_url'],
                help='OpenAI 호환 endpoint. {base_url}/chat/completions 가 호출됩니다.',
            )

        with st.container(border=True):
            st.markdown('##### 샘플링 / 응답 길이')
            ac1, ac2 = st.columns(2)
            with ac1:
                st.session_state['max_tokens'] = st.number_input(
                    '최대 응답 토큰', 16, 131072,
                    int(st.session_state['max_tokens']),
                )
                st.session_state['temperature'] = st.slider(
                    'temperature', 0.0, 2.0,
                    float(st.session_state['temperature']), 0.05,
                )
                st.session_state['top_p'] = st.slider(
                    'top_p', 0.0, 1.0, float(st.session_state['top_p']), 0.01,
                )
            with ac2:
                st.session_state['sampling_top_k'] = st.number_input(
                    'top_k', 1, 200, int(st.session_state['sampling_top_k']),
                )
                st.session_state['presence_penalty'] = st.slider(
                    'presence_penalty', 0.0, 2.0,
                    float(st.session_state['presence_penalty']), 0.1,
                )

        with st.container(border=True):
            st.markdown('##### 검색 정밀도')
            ad1, ad2 = st.columns(2)
            with ad1:
                st.session_state['retrieve_top_n'] = st.number_input(
                    '1차 후보 수', 1, 200,
                    int(st.session_state['retrieve_top_n']),
                    help='재정렬 전에 가져올 후보 청크 수.',
                )
                st.session_state['final_top_k'] = st.number_input(
                    '최종 청크 수', 1, 50,
                    int(st.session_state['final_top_k']),
                    help='LLM 에 컨텍스트로 전달할 최종 청크 수.',
                )
            with ad2:
                st.session_state['per_doc_reserve'] = st.number_input(
                    '문서당 최소 청크', 1, 5,
                    int(st.session_state['per_doc_reserve']),
                )
            st.session_state['use_multi_query'] = st.checkbox(
                '다중 쿼리 (paraphrase)',
                value=st.session_state['use_multi_query'],
                help='질문을 여러 표현으로 변형 후 합집합 검색.',
            )
            if st.session_state['use_multi_query']:
                st.session_state['n_paraphrases'] = st.number_input(
                    '변형 개수', 1, 8,
                    int(st.session_state['n_paraphrases']),
                )
            st.session_state['use_hyde'] = st.checkbox(
                'HyDE (가상 답안 검색)',
                value=st.session_state['use_hyde'],
            )


# =============================================================================
# Cache view
# =============================================================================

def view_cache():
    _section(
        '데이터 & 저장소',
        '클라우드 영속 저장, 로컬 디스크, 모델 캐시를 한 곳에서 관리.',
    )

    user_logs = _user_logs_dir()
    user_dd = _user_data_dir()

    # ----- Compute compact stats for the overview row -----
    sb_connected = _supabase_client() is not None
    sb_attempts = st.session_state.get('_sb_attempts', 0)
    sb_failures = st.session_state.get('_sb_failures', 0)
    pgv_successes = st.session_state.get('_pgv_successes', 0)
    pgv_failures = st.session_state.get('_pgv_failures', 0)

    # Local disk usage (sum of all files under user_dd).
    local_bytes = 0
    if user_dd.exists():
        for p in user_dd.rglob('*'):
            if p.is_file():
                try:
                    local_bytes += p.stat().st_size
                except Exception:
                    pass

    # HF model cache size.
    hf_cached = []
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        hf_cached = [(r.repo_id, r.size_on_disk_str, str(r.repo_path), r.size_on_disk)
                     for r in info.repos]
    except Exception:
        hf_cached = []
    hf_total_bytes = sum(x[3] for x in hf_cached)

    def _fmt_size(b):
        if b < 1024:
            return f'{b} B'
        if b < 1024 * 1024:
            return f'{b / 1024:.1f} KB'
        if b < 1024 ** 3:
            return f'{b / 1024 / 1024:.1f} MB'
        return f'{b / 1024 ** 3:.2f} GB'

    # ----- Top overview: 4 metric cards -----
    with st.container(border=True):
        m = st.columns(4)
        if sb_connected:
            sb_label = '정상' if sb_failures == 0 else f'{sb_failures}건 실패'
            m[0].metric('영속 로깅', sb_label, delta=f'{sb_attempts} 시도' if sb_attempts else None)
        else:
            m[0].metric('영속 로깅', '미설정')
        if sb_connected and pgv_successes:
            pgv_label = f'{pgv_successes}건' if pgv_failures == 0 else f'{pgv_failures}건 실패'
            m[1].metric('pgvector', pgv_label)
        elif sb_connected:
            m[1].metric('pgvector', '대기')
        else:
            m[1].metric('pgvector', '미연결')
        m[2].metric('로컬 디스크', _fmt_size(local_bytes))
        m[3].metric('HF 모델 캐시', _fmt_size(hf_total_bytes))

    # ----- Tabbed body -----
    tab_supabase, tab_local, tab_hf = st.tabs(
        ['클라우드 영속 (Supabase)', '로컬 디스크', 'HF 모델 캐시']
    )

    # ============== Supabase ==============
    with tab_supabase:
        if not sb_connected:
            with st.container(border=True):
                if _is_streamlit_cloud():
                    st.warning(
                        '영속 로깅 미설정 — 아래 로컬 JSONL 은 컨테이너 재시작 시 사라집니다.'
                    )
                    st.markdown(
                        '**활성화 방법:** Manage app → Settings → Secrets 에 다음 추가:\n'
                        '```toml\n'
                        'SUPABASE_URL = "https://xxxx.supabase.co"\n'
                        'SUPABASE_KEY = "eyJ..."\n'
                        '```\n'
                        '그 후 SQL Editor 에서 `db_schema.sql` / `db_schema_pgvector.sql` / `db_schema_users.sql` 실행.'
                    )
                else:
                    st.info(
                        '로컬 개발 중에는 JSONL 만으로 충분합니다. '
                        'Cloud 배포 시 `SUPABASE_URL` / `SUPABASE_KEY` 설정 권장.'
                    )
        else:
            # Status cards
            with st.container(border=True):
                st.markdown('##### 로깅 (chat_turns / agent_runs / events)')
                successes = st.session_state.get('_sb_successes', 0)
                last_err = st.session_state.get('_sb_last_err')
                if sb_attempts == 0:
                    st.info('이 세션에서 INSERT 시도 없음. 아래 진단 버튼으로 즉시 테스트 가능.')
                elif sb_failures == 0:
                    st.success(f'INSERT {successes}/{sb_attempts} 성공. 컨테이너 재시작에도 보존.')
                else:
                    st.error(
                        f'INSERT {sb_failures}/{sb_attempts} 실패. '
                        'RLS / 스키마 누락 등 점검 필요.'
                    )
                    if last_err:
                        with st.expander('마지막 실패 메시지', expanded=True):
                            st.code(last_err)

                if st.button(
                    '연결 진단 (events 1행 INSERT + DELETE)',
                    key='sb_diagnose_btn',
                ):
                    import time as _t
                    client = _supabase_client()
                    probe = {
                        'event_type': 'diagnostic_probe',
                        'user_id': st.session_state.get('user_id', '_local'),
                        'payload': {'ts': _t.time(), 'note': '캐시 탭 진단 버튼'},
                    }
                    try:
                        ins = client.table('events').insert(probe).execute()
                        ins_id = (ins.data[0]['id']
                                  if getattr(ins, 'data', None) and ins.data else None)
                        st.success(f'INSERT 성공 (id={ins_id}). 정리 중...')
                        if ins_id is not None:
                            try:
                                client.table('events').delete().eq('id', ins_id).execute()
                                st.info(f'정리 완료 (id={ins_id} 삭제). DB 정상.')
                            except Exception as de:
                                st.warning(
                                    f'INSERT 됐는데 DELETE 실패 ({de}).'
                                )
                    except Exception as e:
                        st.error(f'INSERT 실패: {type(e).__name__}')
                        st.code(str(e)[:1500])
                        msg = str(e).lower()
                        if 'row-level security' in msg or 'rls' in msg or 'policy' in msg:
                            st.markdown(
                                '**원인: RLS 차단.** SQL Editor 에서 한 번 실행:\n\n'
                                '```sql\n'
                                'alter table public.events     disable row level security;\n'
                                'alter table public.chat_turns disable row level security;\n'
                                'alter table public.agent_runs disable row level security;\n'
                                '```'
                            )
                        elif 'does not exist' in msg or '42p01' in msg:
                            st.markdown('**원인: 테이블 없음.** `db_schema.sql` 실행 필요.')
                        elif '401' in msg or '403' in msg or 'unauthorized' in msg:
                            st.markdown('**원인: 키 권한.** anon public 키인지 확인.')

            # pgvector status
            with st.container(border=True):
                st.markdown('##### pgvector (청크 임베딩 영속화)')
                pgv_attempts = st.session_state.get('_pgv_attempts', 0)
                pgv_last_err = st.session_state.get('_pgv_last_err')
                if pgv_attempts == 0:
                    st.info(
                        '이 세션 인덱싱 없음. 문서 업로드 시 청크 임베딩이 '
                        '`doc_chunks` 테이블에 자동 저장됩니다. '
                        '(스키마: `db_schema_pgvector.sql`)'
                    )
                elif pgv_failures == 0:
                    st.success(
                        f'청크 임베딩 {pgv_successes}/{pgv_attempts} 영속화 성공.'
                    )
                else:
                    st.error(
                        f'영속화 {pgv_failures}/{pgv_attempts} 실패. '
                        '`db_schema_pgvector.sql` 적용 확인.'
                    )
                    if pgv_last_err:
                        with st.expander('마지막 pgvector 에러', expanded=False):
                            st.code(pgv_last_err)

                # Active retrieval source.
                if st.session_state.get('use_pgvector_search'):
                    n = st.session_state.get('_pgv_search_last_n')
                    err = st.session_state.get('_pgv_search_last_err')
                    if err:
                        st.warning('의미 검색 경로: pgvector (직전 호출 실패 → 로컬 폴백)')
                        with st.expander('검색 에러', expanded=False):
                            st.code(err)
                    elif n is None:
                        st.caption('의미 검색 경로: pgvector (아직 호출 없음)')
                    else:
                        st.caption(f'의미 검색 경로: pgvector — 직전 호출 {n}건')
                else:
                    st.caption('의미 검색 경로: 로컬 numpy (in-memory)')

    # ============== 로컬 디스크 ==============
    with tab_local:
        # Aggregate logs first (agents + events)
        with st.container(border=True):
            st.markdown('##### 통합 로그 파일')
            found_any = False
            for fname, label_text in (
                ('events.jsonl', '로그인 / 문서 / 세션 / LLM 에러 이벤트'),
                ('agents.jsonl', '에이전트 실행 기록'),
            ):
                fpath = user_logs / fname
                if not fpath.exists():
                    continue
                found_any = True
                try:
                    n_lines = sum(1 for _ in fpath.open('r', encoding='utf-8'))
                except Exception:
                    n_lines = '?'
                size_kb = fpath.stat().st_size / 1024
                unit = 'events' if 'events' in fname else 'runs'
                cols = st.columns([5, 1, 1])
                cols[0].markdown(f"**`{fname}`** · {label_text}")
                cols[1].caption(f'{n_lines} {unit} · {size_kb:.1f} KB')
                with cols[2]:
                    try:
                        st.download_button(
                            '다운로드', data=fpath.read_bytes(),
                            file_name=fname, mime='application/x-jsonlines',
                            key=f'dl_{fname}', use_container_width=True,
                        )
                    except Exception:
                        pass
            if not found_any:
                st.caption('아직 통합 로그 파일이 없습니다.')

        # Per-session chat logs
        _AGGREGATE_LOGS = {'agents.jsonl', 'events.jsonl'}
        jsonl_files = sorted(
            [p for p in user_logs.glob('*.jsonl') if p.name not in _AGGREGATE_LOGS],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        with st.container(border=True):
            st.markdown('##### 세션별 대화 로그')
            st.caption(f'경로: `{user_logs}`')
            if not jsonl_files:
                st.caption('대화 로그는 아직 없습니다.')
            else:
                for p in jsonl_files[:30]:
                    try:
                        n_lines = sum(1 for _ in p.open('r', encoding='utf-8'))
                    except Exception:
                        n_lines = '?'
                    size_kb = p.stat().st_size / 1024
                    cols = st.columns([5, 1, 1])
                    cols[0].markdown(f"`{p.name}`")
                    cols[1].caption(f'{n_lines} turns · {size_kb:.1f} KB')
                    with cols[2]:
                        try:
                            st.download_button(
                                '다운로드', data=p.read_bytes(),
                                file_name=p.name,
                                mime='application/x-jsonlines',
                                key=f'dl_jsonl_{p.name}',
                                use_container_width=True,
                            )
                        except Exception:
                            pass
                if len(jsonl_files) > 30:
                    st.caption(f'표시 30개 / 전체 {len(jsonl_files)}개')

        # Local vector store
        with st.container(border=True):
            st.markdown('##### 로컬 벡터 스토어')
            st.caption(f'경로: `{user_dd}` — 임베더 모델마다 하위 폴더.')
            if not user_dd.exists():
                st.caption('아직 저장된 벡터 인덱스가 없습니다.')
            else:
                rows = []
                for sub in sorted(user_dd.iterdir()):
                    if not sub.is_dir() or sub.name == 'sessions':
                        continue
                    doc_dirs = [p for p in sub.iterdir() if p.is_dir()]
                    total = 0
                    for p in sub.rglob('*'):
                        if p.is_file():
                            try:
                                total += p.stat().st_size
                            except Exception:
                                pass
                    rows.append((sub.name, len(doc_dirs), total))
                if not rows:
                    st.caption('아직 저장된 벡터 인덱스가 없습니다.')
                else:
                    for name, n, size in rows:
                        cols = st.columns([4, 1, 1])
                        cols[0].markdown(f"`{name}`")
                        cols[1].caption(f'{n} docs')
                        cols[2].caption(_fmt_size(size))

    # ============== HF 모델 캐시 ==============
    with tab_hf:
        with st.container(border=True):
            st.markdown('##### Hugging Face 다운로드 모델')
            st.caption(
                '임베더 / reranker / 자가호스팅 모델 등 로컬 다운로드된 가중치. '
                '사용하지 않는 항목은 삭제해 디스크 회수.'
            )
            if not hf_cached:
                st.caption('캐시된 모델이 없습니다.')
            else:
                for repo_id, size_str, path, _bytes in hf_cached:
                    cols = st.columns([4, 1, 1])
                    cols[0].markdown(f"`{repo_id}`")
                    cols[1].caption(size_str)
                    if cols[2].button(
                        '삭제', key=f'cache_{repo_id}',
                        use_container_width=True,
                    ):
                        try:
                            shutil.rmtree(path)
                            st.success(f'{repo_id} 삭제 완료')
                            st.rerun()
                        except Exception as e:
                            st.error(f'삭제 실패: {e}')


def view_agents():
    _section(
        '에이전트',
        '내 문서를 토대로 이메일/보고서/요약/데이터 분석/비교 분석을 자동 생성합니다. '
        '각 작업은 retrieval로 근거를 가져온 뒤 작업별 전용 프롬프트로 LLM에 요청합니다.',
    )

    # Task picker
    task_keys = list(AGENT_TASKS.keys())
    if 'agent_task_key' not in st.session_state:
        st.session_state['agent_task_key'] = task_keys[0]
    selected = st.radio(
        '작업 선택',
        task_keys,
        index=task_keys.index(st.session_state.get('agent_task_key', task_keys[0])),
        format_func=lambda k: AGENT_TASKS[k]['label'],
        horizontal=True,
    )
    st.session_state['agent_task_key'] = selected
    task = AGENT_TASKS[selected]
    st.caption(task['description'])

    if task.get('requires_docs') and not st.session_state['docs']:
        st.warning(
            '이 작업은 문서가 필요합니다. **문서** 탭에서 먼저 파일을 업로드해 주세요.'
        )

    # Input form
    inputs = {}
    selected_doc_ids = None
    with st.form(f'agent_form_{selected}'):
        # Per-task document subset selector. Shown whenever the user has
        # 2+ documents loaded — every agent task uses retrieved chunks, so
        # the choice "which docs do you want this draft to lean on" is
        # always meaningful even when requires_docs=False (e.g. email).
        if len(st.session_state['docs']) >= 2:
            doc_id_to_name = {d['id']: d['name'] for d in st.session_state['docs']}
            all_ids = list(doc_id_to_name.keys())
            selected_doc_ids = st.multiselect(
                '참고할 문서',
                options=all_ids,
                default=all_ids,
                format_func=lambda did: doc_id_to_name[did],
                help='이 작업이 검색·인용할 문서를 선택합니다. 기본은 전체 문서. '
                '특정 문서만 골라 좁히면 응답이 더 정확해집니다. '
                '문서가 1개라면 선택지가 없으니 자동 사용됩니다.',
            )

        for f in task['fields']:
            label = f['label']
            placeholder = f.get('placeholder', '')
            if f['type'] == 'text':
                inputs[f['key']] = st.text_input(label, placeholder=placeholder)
            elif f['type'] == 'textarea':
                inputs[f['key']] = st.text_area(label, height=120, placeholder=placeholder)
            elif f['type'] == 'select':
                inputs[f['key']] = st.selectbox(label, f['options'])
        submitted = st.form_submit_button('실행', use_container_width=True, type='primary')

    if submitted:
        if (not st.session_state['model']
                or not _active_api_key()
                or not st.session_state['base_url']):
            provider = st.session_state.get('provider', 'Hugging Face Router')
            st.error(
                f'설정 탭에서 공급자/모델/{provider} 용 API 키를 먼저 확인해 주세요.'
            )
            st.stop()
        if task.get('requires_docs') and not st.session_state['docs']:
            st.stop()
        if (task.get('requires_docs') and selected_doc_ids is not None
                and len(selected_doc_ids) == 0):
            st.error('최소 한 개 이상의 문서를 선택해 주세요.')
            st.stop()

        try:
            full_text, reasoning, retrieved, elapsed = run_agent_task(
                selected, inputs, doc_ids_filter=selected_doc_ids,
            )
        except Exception as e:
            _show_llm_error(e)
            st.stop()

        st.write('')
        _section('결과', f'생성 시간: {elapsed:.1f}s')
        if reasoning:
            with st.expander('추론 과정', expanded=False):
                st.markdown(reasoning)
        st.markdown(full_text or '*(빈 응답)*')

        if full_text:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            ext = task.get('output_ext', 'md')
            mime = 'text/markdown' if ext == 'md' else 'text/plain'
            st.download_button(
                '다운로드',
                data=full_text,
                file_name=f"{selected}_{stamp}.{ext}",
                mime=mime,
                use_container_width=True,
                key=f'dl_agent_{selected}_{stamp}',
            )

        if retrieved:
            with st.expander(f'사용된 근거 ({len(retrieved)}개)', expanded=False):
                for j, r in enumerate(retrieved, start=1):
                    if r.get('source') == 'web':
                        from urllib.parse import urlparse
                        host = urlparse(r.get('url', '')).netloc or '웹'
                        st.markdown(f"**[{j}] 웹 · {host}** — {r.get('doc', '')[:80]}")
                        if r.get('url'):
                            st.markdown(f"<{r.get('url')}>")
                    else:
                        pages = r.get('pages') or []
                        page_str = ''
                        if pages:
                            page_str = (f' p.{pages[0]}' if len(pages) == 1
                                        else f' pp.{pages[0]}-{pages[-1]}')
                        st.markdown(
                            f"**[{j}] {r.get('doc', '')}{page_str}** · "
                            f"chunk {r.get('chunk_idx', 0)} · "
                            f"score {r.get('score', 0):.3f}"
                        )
                    st.text((r.get('text', '') or '')[:600])
                    st.divider()


# =============================================================================
# About view
# =============================================================================

def view_about():
    if LOGO_URI:
        st.markdown(
            f'<div style="text-align:center; margin-bottom:18px;">'
            f'<img src="{LOGO_URI}" '
            f'style="width:100%; max-width:280px; height:auto; display:block; margin:0 auto 8px auto;" />'
            f'<div style="font-size:22px; font-weight:700; letter-spacing:-0.01em;">Personal RAG</div>'
            f'<div style="font-size:13px; color:rgba(128,128,128,0.95);">내 문서 기반 질의응답 시스템</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        내 문서를 기반으로 답하는 개인용 RAG (Retrieval-Augmented Generation) 시스템.

        #### 검색 파이프라인
        1. 문서 파싱 — PDF는 **Docling**으로 표·헤딩·리스트 구조를 보존하며 마크다운으로 변환, 페이지 메타를 청크별로 기록.
        2. 청킹 — 문단 단위로 묶고 size 한도를 넘으면 슬라이딩 윈도우로 분할.
        3. 임베딩 — **BGE-M3** (다국어 SOTA) 또는 MiniLM multilingual.
        4. 하이브리드 검색 — Dense 코사인 + BM25 의 **Reciprocal Rank Fusion**.
        5. (옵션) **HyDE** / **Multi-query** / **Contextual rewrite** 로 쿼리 보강.
        6. (옵션) Cross-encoder **bge-reranker-v2-m3** 로 top_n → top_k 재정렬.
        7. (옵션) PDF 페이지 이미지를 **멀티모달 LLM** 에 함께 전달.
        8. LLM 응답을 토큰 단위로 스트리밍, reasoning_content 와 본문을 분리.
        9. 답변 안의 `[N]` 인용 마커를 파싱하여 청크별로 출처 표시.

        #### 저장
        - `./.data/{embedder}/{doc_hash}/` — 임베더별 격리.
        - `meta.json` (청크, 페이지 메타, 원본 텍스트) + `embeddings.npy` + `pages/{page}.png`.
        - 같은 파일·청크 설정으로 다시 업로드 시 캐시에서 즉시 복원.
        - 사용자별 환경설정 (API 키, 모델, 검색 설정 등) 영속 저장: `./.data/{user}/preferences.json`. Streamlit Cloud의 idle 재연결 시 메모리가 초기화되더라도 다음 접속에서 자동 복원됩니다. (Cloud 컨테이너가 재시작되면 사라지므로 영구 보존이 필요하면 Streamlit Secrets 사용 권장.)
        - 대화 세션 메타: `./.data/{user}/sessions/{id}.json` (사이드바 대화 목록의 원천).
        - **대화 로그**: `./logs/{user}/{session_id}.jsonl` — 세션별로 한 파일, 한 줄당 한 턴. 분석·DB 친화적 구조.
          필드: session_id, turn_index, timestamp, user_message, assistant_message, reasoning, model/provider, elapsed_seconds, retrieved (rank·source·score·page·url), citation_numbers_used, query_variants, settings_snapshot (rerank/HyDE/multi-query/per-doc 등 모든 설정 스냅샷).
        - **에이전트 실행 로그**: `./logs/{user}/agents.jsonl` — 에이전트 워크플로(이메일/보고서/요약/분석/비교)의 실행 기록. task, inputs, output, retrieved, model, elapsed_seconds.
        - **이벤트 로그**: `./logs/{user}/events.jsonl` — 로그인/로그아웃, 문서 업로드·삭제, 세션 삭제, LLM 호출 실패 등 비-턴 이벤트. 각 줄에 event_type, timestamp, user_id, payload.
          pandas: `pd.concat([pd.read_json(f, lines=True) for f in glob.glob('logs/*/*.jsonl')])`.
          Postgres: `\\copy turns FROM '...' WITH (FORMAT json)` 또는 batched insert.

        #### 지원 LLM 엔드포인트
        Hugging Face Inference Router · OpenAI · Alibaba DashScope · vLLM / 로컬 OpenAI-호환 서버.

        #### 사용 라이브러리
        Streamlit, OpenAI SDK, sentence-transformers, rank_bm25, Docling, PyMuPDF, pypdf, ddgs.
        """
    )


# =============================================================================
# View router — dispatch to selected view
# =============================================================================

_VIEWS = {
    'chat':     view_chat,
    'docs':     view_docs,
    'agents':   view_agents,
    'settings': view_settings,
    'cache':    view_cache,
    'about':    view_about,
}
_VIEWS.get(st.session_state.get('active_view', 'chat'), view_chat)()

# Persist user preferences after every rerun (cheap; only writes on change).
_save_user_prefs()
