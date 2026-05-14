"""Persistent left sidebar — brand, new chat, navigation, sessions, status, user.

The whole rail is wrapped in ``render_sidebar()`` so app.py can run it
once per rerun in a controlled spot. Streamlit re-executes the script
top-down each interaction, so calling ``render_sidebar()`` once near
the end of app.py (after the boot sequence but before the view router)
keeps the rail in sync with whichever button just triggered the rerun.
"""

import streamlit as st

from auth.prefs import _PERSIST_KEYS
from auth.users import _log_event
from branding import LOGO_URI
from data.sessions import (
    delete_session,
    list_sessions,
    load_session,
    rename_session,
    start_new_session,
)


NAV = [
    ('chat',     '대화'),
    ('docs',     '문서'),
    ('agents',   '업무 도구(에이전트)'),
    ('settings', '설정'),
    ('cache',    '캐시'),
    ('about',    '소개'),
]
NAV_KEYS = [k for k, _ in NAV]


def render_sidebar():
    with st.sidebar:
        if LOGO_URI:
            st.markdown(
                f'<div style="text-align:center; padding:4px 0 4px 0;">'
                f'<img src="{LOGO_URI}" '
                f'style="width:100%; max-width:220px; height:auto; display:block; margin:0 auto;" />'
                f'</div>'
                f'<div style="text-align:center; margin-bottom:8px;">'
                f'<div class="sb-brand" style="font-size:14px;">Personal RAG</div>'
                f'<div class="sb-tagline">내 문서 기반 질의응답</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="sb-brand">Personal RAG</div>', unsafe_allow_html=True)
            st.markdown('<div class="sb-tagline">내 문서 기반 질의응답</div>', unsafe_allow_html=True)
        st.write('')

        if st.button('새 대화', use_container_width=True, type='primary'):
            start_new_session()
            st.session_state['active_view'] = 'chat'
            st.rerun()

        # ----- Menu (pinned, always visible above the fold) -----
        st.markdown('<div class="sb-section">메뉴</div>', unsafe_allow_html=True)
        current_view = st.session_state.get('active_view', 'chat')
        if current_view not in NAV_KEYS:
            current_view = 'chat'
        for key, label in NAV:
            is_active = (key == current_view)
            if st.button(
                label,
                key=f'nav_{key}',
                use_container_width=True,
                type='primary' if is_active else 'secondary',
            ):
                if not is_active:
                    st.session_state['active_view'] = key
                    st.rerun()

        # ----- Saved conversations (scrolls internally; does not push menu off) -----
        sessions = list_sessions()
        current_sid = st.session_state.get('current_session_id')
        st.markdown('<div class="sb-section">대화</div>', unsafe_allow_html=True)
        if not sessions:
            st.caption('저장된 대화가 없습니다.')
        else:
            # Cap visible height; the list scrolls inside its own container so the
            # menu above and status below stay anchored.
            with st.container(height=300, border=False):
                for s in sessions:
                    is_active = (s['id'] == current_sid)
                    label = s['title'] or '(제목 없음)'
                    if len(label) > 22:
                        label = label[:20] + '...'
                    row = st.columns([5, 1, 1])
                    with row[0]:
                        if st.button(
                            label,
                            key=f"sess_{s['id']}",
                            use_container_width=True,
                            type='primary' if is_active else 'secondary',
                        ):
                            if not is_active:
                                load_session(s['id'])
                                st.session_state['active_view'] = 'chat'
                                st.rerun()
                    with row[1]:
                        if hasattr(st, 'popover'):
                            with st.popover(
                                '✎', use_container_width=True,
                                help='대화 이름 변경',
                            ):
                                with st.form(
                                    f'_rename_form_{s["id"]}',
                                    clear_on_submit=False,
                                ):
                                    new_t = st.text_input(
                                        '대화 이름',
                                        value=s['title'] or '',
                                        max_chars=60,
                                        key=f'rename_input_{s["id"]}',
                                    )
                                    if st.form_submit_button(
                                        '저장',
                                        use_container_width=True,
                                        type='primary',
                                    ):
                                        if rename_session(s['id'], new_t):
                                            st.rerun()
                    with row[2]:
                        if st.button('×', key=f"del_sess_{s['id']}",
                                     use_container_width=True,
                                     help='대화 삭제'):
                            delete_session(s['id'])
                            if current_sid == s['id']:
                                start_new_session()
                            st.rerun()

        # ----- Status -----
        st.markdown('<div class="sb-section">현재 상태</div>', unsafe_allow_html=True)
        n_docs = len(st.session_state['docs'])
        n_chunks = sum(len(d['chunks']) for d in st.session_state['docs'])
        model_short = st.session_state['model'] or '없음'
        if len(model_short) > 32:
            model_short = model_short[:29] + '...'
        st.caption(f"모델: `{model_short}`")
        prov_short = st.session_state.get('provider', '')
        if prov_short:
            st.caption(f"공급자: `{prov_short}`")
        # Provider-stamped model id from the last API response — bulletproof
        # confirmation of which model actually served the previous turn. Shown
        # always (not only on mismatch) because LLMs misidentify themselves
        # ("I'm Claude 3" when running Claude 4.6) and this header settles it.
        actual_model = st.session_state.get('_last_response_model')
        if actual_model:
            am_short = actual_model if len(actual_model) <= 36 else actual_model[:33] + '...'
            configured = st.session_state.get('model', '')
            if actual_model == configured:
                st.caption(f"직전 응답 (실제): `{am_short}` ✓ 일치")
            else:
                st.caption(f"직전 응답 (실제): `{am_short}`")
        if st.session_state.get('general_chat_mode'):
            st.caption(
                '모드: 일반 대화 (RAG 끔)'
                + (f' · 문서 {n_docs}개 보유' if n_docs else '')
            )
        elif n_docs:
            st.caption(f"문서 {n_docs}개 · 청크 {n_chunks}개")
        else:
            st.caption('문서 없음 (일반 챗)')
        if st.session_state['web_enabled'] and not st.session_state.get('general_chat_mode'):
            st.caption(f"웹 검색: {st.session_state['web_provider']}")

        # ----- User / logout -----
        uid = st.session_state.get('user_id', '_local')
        # Logged-in state = anything other than the anonymous/local fallbacks.
        is_logged_in = not (uid == '_local' or uid.startswith('_anon_'))
        st.markdown('<div class="sb-section">사용자</div>', unsafe_allow_html=True)
        if is_logged_in:
            st.caption(f"로그인: `{uid}`")
            if st.button('로그아웃', use_container_width=True, key='logout_btn'):
                _log_event('logout', {'username': uid})
                base_clear = (
                    'user_id', 'user_inputs', 'generated_responses',
                    'thinking_traces', 'retrieved_per_turn',
                    'query_variants_per_turn', 'current_session_id',
                    'current_session_title', 'current_session_created_at',
                    'docs', 'doc_embs', 'doc_meta',
                    '_loaded_for_embedder', '_prefs_snapshot', '_prefs_loaded',
                )
                # Also clear all persistable preference keys so the next user does
                # not inherit the previous one's API keys / model / settings.
                for k in base_clear + _PERSIST_KEYS:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()
        elif uid.startswith('_anon_'):
            st.caption(f"익명 세션: `{uid}`")
            st.caption('이 브라우저 탭에서만 유효. 새 탭/창은 별도 데이터.')
        else:
            st.caption(f"로컬 단일 사용자: `{uid}`")
