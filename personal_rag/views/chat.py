"""Main chat view — hero, history, suggestion chips, input box with file drop."""

import streamlit as st

from ..branding import LOGO_URI
from ..data.sessions import load_session, rename_session
from ..llm.chat import handle_chat_turn, render_assistant
from ..processing.ingestion import ingest_files
from ..ui.widgets import model_picker


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
