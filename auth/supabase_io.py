"""Thin wrappers around Supabase Postgres + Auth RPCs.

Streamlit Community Cloud's filesystem is ephemeral — every redeploy or
container restart wipes /mount/src/.../logs and .data. To preserve user
activity, this module ALSO pushes inserts to Supabase Postgres when
``SUPABASE_URL`` + ``SUPABASE_KEY`` are configured (via .env or Streamlit
secrets). If neither is set, every helper here is a quiet no-op and the
local JSONL files remain the single source of truth.

Expected schema (apply once via Supabase SQL editor):
- ``db_schema.sql``           — chat_turns, agent_runs, events
- ``db_schema_users.sql``     — signup_user / login_user SECURITY DEFINER RPCs
- ``db_schema_preferences.sql`` — get_prefs / set_prefs RPCs + user_preferences
- ``db_schema_pgvector.sql``  — pgvector chunk store (optional)
- ``db_schema_sessions.sql``  — conversation sessions (optional)
"""

import json
import os

import streamlit as st


@st.cache_resource(show_spinner=False)
def _supabase_client():
    """Return a Supabase client if configured, else None. Cached so we don't
    re-import / re-connect on every turn."""
    url = os.getenv('SUPABASE_URL', '').strip()
    key = os.getenv('SUPABASE_KEY', '').strip() or os.getenv('SUPABASE_SERVICE_KEY', '').strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None


def _scrub_for_postgres(obj):
    """Recursively strip control characters Postgres can't store.

    PDF text extraction routinely embeds NULL bytes (\\x00) into chunk text.
    Postgres rejects them in TEXT and JSONB columns with code 22P05. We
    also drop other ASCII control chars (except tab/newline/cr) since they
    contribute nothing for log analysis and trip JSON validators in some
    Postgres setups."""
    if isinstance(obj, str):
        if '\x00' not in obj and not any(
            ord(c) < 32 and c not in '\t\n\r' for c in obj[:1024]
        ):
            return obj  # fast path — clean string
        return ''.join(
            c for c in obj if ord(c) >= 32 or c in '\t\n\r'
        ).replace('\x00', '')
    if isinstance(obj, dict):
        return {k: _scrub_for_postgres(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_for_postgres(x) for x in obj]
    return obj


# -----------------------------------------------------------------------------
# Supabase-backed user auth (signup / login)
# -----------------------------------------------------------------------------
# Passwords never reach Python — we call two SECURITY DEFINER RPCs in Postgres
# that handle bcrypt hashing/verification, and only see (success, message,
# user_id) back. Configure via db_schema_users.sql.

def _supabase_signup(username: str, password: str) -> tuple:
    """Returns (success: bool, message: str). Requires Supabase configured
    and db_schema_users.sql applied."""
    client = _supabase_client()
    if client is None:
        return False, 'Supabase 가 설정되지 않아 회원가입을 사용할 수 없습니다.'
    try:
        resp = client.rpc(
            'signup_user',
            {'p_username': username, 'p_password': password},
        ).execute()
        rows = resp.data or []
        if not rows:
            return False, '회원가입 처리 중 알 수 없는 오류가 발생했습니다.'
        row = rows[0]
        return bool(row.get('success')), str(row.get('message') or '')
    except Exception as e:
        msg = str(e)
        if 'function' in msg.lower() and 'does not exist' in msg.lower():
            return False, (
                '회원가입 시스템이 아직 설정되지 않았습니다. '
                'Supabase SQL Editor 에서 db_schema_users.sql 을 실행하세요.'
            )
        return False, f'회원가입 실패: {msg[:200]}'


def _supabase_login(username: str, password: str) -> tuple:
    """Returns (success: bool, message: str, user_id_in_db: int|None)."""
    client = _supabase_client()
    if client is None:
        return False, 'Supabase 미설정.', None
    try:
        resp = client.rpc(
            'login_user',
            {'p_username': username, 'p_password': password},
        ).execute()
        rows = resp.data or []
        if not rows:
            return False, '로그인 처리 중 알 수 없는 오류.', None
        row = rows[0]
        return (
            bool(row.get('success')),
            str(row.get('message') or ''),
            row.get('user_id'),
        )
    except Exception as e:
        msg = str(e)
        if 'function' in msg.lower() and 'does not exist' in msg.lower():
            return False, (
                '로그인 시스템이 아직 설정되지 않았습니다. '
                'Supabase SQL Editor 에서 db_schema_users.sql 을 실행하세요.'
            ), None
        return False, f'로그인 실패: {msg[:200]}', None


def _supabase_users_enabled() -> bool:
    """True if Supabase is configured. We assume db_schema_users.sql has
    been applied; the RPC helpers handle the "function missing" case with
    a friendly message instead of crashing."""
    return _supabase_client() is not None


def _should_sync_prefs_to_supabase() -> bool:
    """Sync prefs to Supabase only when a user has a stable identity. Anon
    UUIDs change each visit so syncing them wastes DB writes for no
    benefit; logged-in usernames and the local-dev `_local` slot are fine."""
    client = _supabase_client()
    if client is None:
        return False
    uid = st.session_state.get('user_id', '')
    if not uid or uid.startswith('_anon_'):
        return False
    return True


def _supabase_load_prefs() -> dict:
    """Return this user's prefs blob from Supabase, or {} if none / error.
    Designed to extend (not replace) whatever the disk file already gave us:
    callers should merge with disk dict, Supabase wins on conflicts."""
    if not _should_sync_prefs_to_supabase():
        return {}
    client = _supabase_client()
    uid = st.session_state['user_id']
    try:
        resp = client.rpc('get_prefs', {'p_user_id': uid}).execute()
        # rpc on a scalar-returning function gives us the value directly
        data = resp.data
        if isinstance(data, dict):
            return data
        if isinstance(data, str) and data:
            try:
                return json.loads(data)
            except Exception:
                return {}
        return {}
    except Exception:
        return {}


def _supabase_save_prefs(prefs: dict) -> None:
    """Upsert prefs blob. Best-effort; failures don't break the local save."""
    if not _should_sync_prefs_to_supabase():
        return
    client = _supabase_client()
    uid = st.session_state['user_id']
    try:
        client.rpc(
            'set_prefs',
            {'p_user_id': uid, 'p_prefs': _scrub_for_postgres(prefs)},
        ).execute()
    except Exception as e:
        # Don't loop noise — record once per session.
        if not st.session_state.get('_prefs_sync_warned'):
            st.session_state['_prefs_sync_warned'] = True
            st.session_state['_prefs_sync_last_err'] = (
                f'{type(e).__name__}: {str(e)[:300]}'
            )


def _supabase_insert(table: str, record: dict) -> None:
    """Best-effort INSERT. Never raises — local JSONL remains the source of
    truth for the live container; Supabase is the durable copy. Failures are
    tracked in session_state so the cache tab can surface them instead of
    leaving the user wondering why rows aren't appearing."""
    client = _supabase_client()
    if client is None:
        return
    st.session_state['_sb_attempts'] = st.session_state.get('_sb_attempts', 0) + 1
    try:
        client.table(table).insert(_scrub_for_postgres(record)).execute()
        st.session_state['_sb_successes'] = st.session_state.get('_sb_successes', 0) + 1
    except Exception as e:
        st.session_state['_sb_failures'] = st.session_state.get('_sb_failures', 0) + 1
        st.session_state['_sb_last_err'] = f'{table}: {type(e).__name__}: {str(e)[:600]}'
