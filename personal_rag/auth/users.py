"""Per-user filesystem layout + event logging + the Streamlit login gate.

Every authenticated user gets their own subtree under DATA_DIR and LOGS_DIR
so uploaded documents, indexes, conversation history, and agent runs do not
leak between users sharing the same deployment. If no auth backend is
configured the app falls back to a single ``_local`` user (local dev) or a
per-browser anonymous UUID (Streamlit Cloud).

The login backends, in priority order:

1. **Supabase users** (signup + login, bcrypt) — when Supabase is wired up.
2. **Legacy secrets ``[users]`` block** — admin-managed, no signup.
3. **Streamlit Cloud, no auth backend** → anonymous per-browser UUID.
4. **Local dev, no auth backend** → single-tenant ``_local``.
"""

import datetime
import json
import re
import uuid
from pathlib import Path

import streamlit as st

from ..branding import LOGO_URI
from ..config import DATA_DIR, LOGS_DIR
from .supabase_io import (
    _supabase_insert,
    _supabase_login,
    _supabase_signup,
    _supabase_users_enabled,
)


USERS_FROM_SECRETS = {}
try:
    USERS_FROM_SECRETS = dict(st.secrets.get('users', {}) or {})
except Exception:
    USERS_FROM_SECRETS = {}


def _safe_uid(uid: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', uid or '_local') or '_local'


def _user_data_dir() -> Path:
    d = DATA_DIR / _safe_uid(st.session_state.get('user_id', '_local'))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _user_sessions_dir() -> Path:
    d = _user_data_dir() / 'sessions'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _user_logs_dir() -> Path:
    d = LOGS_DIR / _safe_uid(st.session_state.get('user_id', '_local'))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _agent_log_path() -> Path:
    return _user_logs_dir() / 'agents.jsonl'


def _events_log_path() -> Path:
    return _user_logs_dir() / 'events.jsonl'


def _log_event(event_type: str, payload: dict = None) -> None:
    """Append one event to ./logs/{user_id}/events.jsonl.

    Captures everything outside the chat/agent JSONL: login, logout, document
    ingest, document delete, session delete, LLM call failures. One line per
    event so it merges cleanly with the other JSONL files for analytics.
    Best-effort — failures swallowed so logging never breaks the user flow.
    """
    record = {
        'kind': 'event',
        'event_type': event_type,
        'timestamp': datetime.datetime.now().isoformat(timespec='microseconds'),
        'user_id': st.session_state.get('user_id', '_local'),
        'payload': payload or {},
    }
    try:
        with _events_log_path().open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception:
        pass
    _supabase_insert('events', record)


def _render_login_screen():
    """Render brand + login form. Sets user_id on success and reruns; otherwise
    st.stop()s so the rest of the app is gated off.

    Also renders a minimal sidebar (brand only) so the page does not look like
    the sidebar disappeared — it just has no nav until login completes.
    """
    with st.sidebar:
        if LOGO_URI:
            st.markdown(
                f'<div style="text-align:center; padding:4px 0 4px 0;">'
                f'<img src="{LOGO_URI}" '
                f'style="width:100%; max-width:220px; height:auto; display:block; margin:0 auto;" />'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="sb-brand" style="text-align:center; font-size:14px;">Personal RAG</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sb-tagline" style="text-align:center;">로그인 후 이용</div>',
            unsafe_allow_html=True,
        )

    if LOGO_URI:
        st.markdown(
            f'<div style="text-align:center; margin-top:80px; margin-bottom:8px;">'
            f'<img src="{LOGO_URI}" '
            f'style="max-width:240px; height:auto; opacity:0.95;" /></div>',
            unsafe_allow_html=True,
        )

    # Two distinct login backends:
    #  1) Supabase users table (signup + login, bcrypt). Active when Supabase
    #     is configured. New users can register themselves.
    #  2) Legacy secrets [users] block. Admin-managed; no signup.
    use_supabase_auth = _supabase_users_enabled()

    if use_supabase_auth:
        st.markdown(
            '<div style="text-align:center; font-size:14px; color:rgba(128,128,128,0.9); '
            'margin-bottom:16px;">계정으로 로그인하거나 새로 가입하세요.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:center; font-size:14px; color:rgba(128,128,128,0.9); '
            'margin-bottom:24px;">로그인이 필요합니다.</div>',
            unsafe_allow_html=True,
        )

    _, form_col, _ = st.columns([1, 2, 1])
    with form_col:
        if use_supabase_auth:
            tab_login, tab_signup = st.tabs(['로그인', '회원가입'])
            with tab_login:
                with st.form('_login_form', clear_on_submit=False):
                    username = st.text_input(
                        '아이디', key='login_username',
                        placeholder='가입한 아이디',
                    )
                    password = st.text_input(
                        '비밀번호', type='password', key='login_password',
                        placeholder='비밀번호 입력',
                    )
                    submitted = st.form_submit_button(
                        '로그인', use_container_width=True, type='primary',
                    )
                if submitted:
                    ok, msg, _user_id = _supabase_login(username, password)
                    if ok:
                        st.session_state['user_id'] = (username or '').strip()
                        st.session_state['_loaded_for_embedder'] = None
                        _log_event('login', {
                            'method': 'supabase',
                            'username': st.session_state['user_id'],
                        })
                        st.rerun()
                    else:
                        _log_event('login_failed', {
                            'method': 'supabase',
                            'attempted_username': (username or '')[:64],
                            'reason': msg[:200],
                        })
                        st.error(msg)
            with tab_signup:
                with st.form('_signup_form', clear_on_submit=False):
                    new_username = st.text_input(
                        '새 아이디',
                        key='signup_username',
                        placeholder='2자 이상 64자 이하',
                    )
                    new_password = st.text_input(
                        '새 비밀번호', type='password',
                        key='signup_password',
                        placeholder='6자 이상 입력',
                    )
                    new_password2 = st.text_input(
                        '비밀번호 확인', type='password',
                        key='signup_password2',
                        placeholder='위와 동일하게 한 번 더',
                    )
                    signup_submitted = st.form_submit_button(
                        '회원가입', use_container_width=True, type='primary',
                    )
                if signup_submitted:
                    if new_password != new_password2:
                        st.error('비밀번호 확인이 일치하지 않습니다.')
                    else:
                        ok, msg = _supabase_signup(new_username, new_password)
                        if ok:
                            # Auto-login after signup so user doesn't have to
                            # retype credentials.
                            ok2, _msg2, _uid = _supabase_login(
                                new_username, new_password,
                            )
                            if ok2:
                                st.session_state['user_id'] = (
                                    (new_username or '').strip()
                                )
                                st.session_state['_loaded_for_embedder'] = None
                                _log_event('signup', {
                                    'username': st.session_state['user_id'],
                                })
                                _log_event('login', {
                                    'method': 'supabase',
                                    'username': st.session_state['user_id'],
                                    'first_login': True,
                                })
                                st.success('회원가입 완료 — 자동 로그인했습니다.')
                                st.rerun()
                            else:
                                st.success('회원가입 성공. 로그인 탭에서 다시 로그인해 주세요.')
                        else:
                            st.error(msg)
        else:
            with st.form('_login_form', clear_on_submit=False):
                username = st.text_input('사용자 ID')
                password = st.text_input('비밀번호', type='password')
                submitted = st.form_submit_button(
                    '로그인', use_container_width=True, type='primary',
                )
            if submitted:
                if (username in USERS_FROM_SECRETS
                        and str(USERS_FROM_SECRETS[username]) == str(password)):
                    st.session_state['user_id'] = username
                    st.session_state['_loaded_for_embedder'] = None
                    _log_event('login', {'method': 'password', 'username': username})
                    st.rerun()
                else:
                    _log_event('login_failed', {
                        'method': 'password',
                        'attempted_username': (username or '')[:64],
                    })
                    st.error('사용자 ID 또는 비밀번호가 올바르지 않습니다.')
    st.stop()


def _is_streamlit_cloud() -> bool:
    """Detect Streamlit Community Cloud — apps are mounted under /mount/src/."""
    return Path('/mount/src').exists()


def _auth_gate():
    """Resolve the active user_id. See module docstring for the priority order.
    Called once near the top of app.py; afterwards every other module can
    rely on ``st.session_state['user_id']`` being set.
    """
    current_uid = st.session_state.get('user_id', '')

    # Migration: if an auth backend was just enabled but the browser still has
    # a stale anonymous user_id from before the backend existed, drop it so
    # the login screen actually appears. Otherwise the gate keeps short-
    # circuiting on the old anon id and the user never sees a login form.
    if current_uid.startswith('_anon_') and (
            _supabase_users_enabled() or USERS_FROM_SECRETS):
        st.session_state['user_id'] = ''
        current_uid = ''

    if current_uid:
        return  # already logged in (real account or anon/local fallback)

    if _supabase_users_enabled():
        _render_login_screen()
        return

    if USERS_FROM_SECRETS:
        _render_login_screen()
        return

    if _is_streamlit_cloud():
        # New visitor on a shared deployment without auth → give them their
        # own isolated workspace for this browser session. Closing the tab
        # ends the session; previous anonymous data remains on disk under
        # its UUID until an admin cleans it up.
        st.session_state['user_id'] = '_anon_' + uuid.uuid4().hex[:10]
        _log_event('login', {'method': 'anonymous'})
    else:
        st.session_state['user_id'] = '_local'
        if not st.session_state.get('_local_login_logged'):
            _log_event('login', {'method': 'local'})
            st.session_state['_local_login_logged'] = True
