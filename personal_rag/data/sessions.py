"""Conversation session CRUD — disk JSON blobs mirrored to Supabase.

Each session is a single ``.json`` file under
``<DATA_DIR>/<user_id>/sessions/<session_id>.json`` containing the user
turns, assistant responses, and reasoning traces from the conversation.
Retrieved chunks and query variants are intentionally NOT persisted — they
are heavy and chunk indices can drift between embedder runs.

When Supabase is configured, the same turns are also appended to
``public.chat_turns`` (one row per turn) by ``log_turn_structured`` (which
still lives in app.py for now — it sits between chat helpers and session
storage). This module's helpers handle list/load/rename/delete on the
sessions themselves, surfacing entries from both stores.

The structured per-turn writer (``log_turn_structured``) and the auto-titler
(``auto_title_session``) need LLM clients and citation parsing that have
not been extracted yet — they will join this module after the chat
helpers are split out.
"""

import datetime
import json
import uuid
from pathlib import Path

import streamlit as st

from ..auth.supabase_io import _supabase_client
from ..auth.users import _log_event, _user_logs_dir, _user_sessions_dir


def _new_session_id() -> str:
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]


def _session_path(sid: str) -> Path:
    return _user_sessions_dir() / f'{sid}.json'


def _session_jsonl_path(sid: str) -> Path:
    return _user_logs_dir() / f'{sid}.jsonl'


def save_current_session():
    """Persist current chat (if there is one). No-op if no user turns yet."""
    sid = st.session_state.get('current_session_id')
    if not sid:
        return
    if not st.session_state['user_inputs']:
        return
    data = {
        'id': sid,
        'title': st.session_state.get('current_session_title', '') or '',
        'created_at': st.session_state.get('current_session_created_at')
            or datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat(),
        'model': st.session_state['model'],
        'provider': st.session_state.get('provider', ''),
        'user_inputs': st.session_state['user_inputs'],
        'generated_responses': st.session_state['generated_responses'],
        'thinking_traces': st.session_state['thinking_traces'],
        # retrieved chunks / variants intentionally not persisted (heavy + chunk ids may shift).
    }
    try:
        _session_path(sid).write_text(json.dumps(data, ensure_ascii=False))
    except Exception as e:
        st.warning(f'대화 저장 실패: {e}')


def _supabase_list_sessions(limit: int = 50) -> list:
    """Aggregate the current user's sessions from Supabase chat_turns.
    Returns [] when Supabase isn't configured, user is anonymous, or the
    RPC isn't available yet."""
    client = _supabase_client()
    if client is None:
        return []
    uid = st.session_state.get('user_id', '')
    if not uid or uid.startswith('_anon_'):
        return []
    try:
        resp = client.rpc(
            'list_user_sessions',
            {'p_user_id': uid, 'p_limit': int(limit)},
        ).execute()
        rows = resp.data or []
    except Exception:
        return []
    out = []
    for r in rows:
        sid = r.get('session_id')
        if not sid:
            continue
        out.append({
            'id': sid,
            'title': r.get('title') or '(제목 없음)',
            'updated_at': r.get('updated_at', '') or '',
            'model': r.get('model', '') or '',
            'n_turns': int(r.get('n_turns') or 0),
            '_source': 'supabase',
        })
    return out


def _supabase_load_session(sid: str) -> bool:
    """Pull every chat_turn row for (user, session) and populate session_state
    so the conversation re-renders. Returns False when no rows / errors."""
    client = _supabase_client()
    if client is None:
        return False
    uid = st.session_state.get('user_id', '')
    if not uid or uid.startswith('_anon_'):
        return False
    try:
        resp = (client.table('chat_turns')
                .select('turn_index, "timestamp", session_title, model, '
                        'user_message, assistant_message, reasoning, '
                        'query_variants')
                .eq('user_id', uid)
                .eq('session_id', sid)
                .order('turn_index')
                .execute())
        rows = resp.data or []
    except Exception:
        return False
    if not rows:
        return False

    title = rows[0].get('session_title') or '(제목 없음)'
    created = rows[0].get('timestamp', '')
    st.session_state['current_session_id'] = sid
    st.session_state['current_session_title'] = title
    st.session_state['current_session_created_at'] = created
    st.session_state['user_inputs'] = [r.get('user_message', '') or '' for r in rows]
    st.session_state['generated_responses'] = [
        r.get('assistant_message', '') or '' for r in rows
    ]
    st.session_state['thinking_traces'] = [
        r.get('reasoning', '') or '' for r in rows
    ]
    n = len(rows)
    st.session_state['retrieved_per_turn'] = [[] for _ in range(n)]
    st.session_state['query_variants_per_turn'] = [
        (r.get('query_variants') or []) for r in rows
    ]
    # Save back to disk as a warm cache for next access.
    try:
        save_current_session()
    except Exception:
        pass
    return True


def load_session(sid: str) -> bool:
    """Try local disk first (full retrieved cache); fall back to rebuilding
    the conversation from Supabase chat_turns when the disk file isn't on
    this container (fresh Cloud start, different device, etc.)."""
    p = _session_path(sid)
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = None
        if data:
            st.session_state['current_session_id'] = data.get('id', sid)
            st.session_state['current_session_title'] = data.get('title', '')
            st.session_state['current_session_created_at'] = data.get('created_at', '')
            st.session_state['user_inputs'] = data.get('user_inputs', [])
            st.session_state['generated_responses'] = data.get('generated_responses', [])
            st.session_state['thinking_traces'] = data.get('thinking_traces', [])
            n = len(st.session_state['user_inputs'])
            st.session_state['retrieved_per_turn'] = [[] for _ in range(n)]
            st.session_state['query_variants_per_turn'] = [[] for _ in range(n)]
            return True
    # Disk miss → Supabase
    return _supabase_load_session(sid)


def list_sessions(limit: int = 30):
    """Sessions sorted by updated_at desc. Merges local disk with Supabase
    (logged-in users only); Supabase contributes sessions that aren't on the
    current container's disk. Disk entries win on duplicate id since they
    carry the original session metadata (created_at, etc.)."""
    sd = _user_sessions_dir()
    out = []
    seen_ids = set()
    if sd.exists():
        for p in sd.glob('*.json'):
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            sid = data.get('id', p.stem)
            seen_ids.add(sid)
            out.append({
                'id': sid,
                'title': data.get('title') or '(제목 없음)',
                'updated_at': data.get('updated_at', ''),
                'model': data.get('model', ''),
            })
    for s in _supabase_list_sessions(limit=max(limit * 2, 100)):
        if s['id'] in seen_ids:
            continue
        out.append({
            'id': s['id'],
            'title': s.get('title') or '(제목 없음)',
            'updated_at': s.get('updated_at', ''),
            'model': s.get('model', ''),
        })
    out.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return out[:limit]


def rename_session(sid: str, new_title: str) -> bool:
    """Rename a saved conversation in both disk metadata and Supabase.

    Updates:
      - .data/{user}/sessions/{sid}.json — title field
      - public.chat_turns.session_title — every row of this session, so the
        sidebar list rebuilt from Supabase also reflects the new name
      - st.session_state['current_session_title'] if this is the active one
    Returns True when at least one storage layer updated successfully.
    """
    new_title = (new_title or '').strip()[:60]
    if not new_title:
        return False
    ok_any = False

    # Disk metadata
    p = _session_path(sid)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            data['title'] = new_title
            data['updated_at'] = datetime.datetime.now().isoformat()
            p.write_text(json.dumps(data, ensure_ascii=False))
            ok_any = True
        except Exception:
            pass

    # Supabase chat_turns — update every row of this session for the user
    client = _supabase_client()
    uid = st.session_state.get('user_id', '')
    if client is not None and uid and not uid.startswith('_anon_'):
        try:
            (client.table('chat_turns')
                .update({'session_title': new_title})
                .eq('user_id', uid)
                .eq('session_id', sid)
                .execute())
            ok_any = True
        except Exception:
            pass

    # Active session — mirror the change in session_state too.
    if st.session_state.get('current_session_id') == sid:
        st.session_state['current_session_title'] = new_title

    if ok_any:
        _log_event('session_rename', {'session_id': sid, 'new_title': new_title})
    return ok_any


def delete_session(sid: str):
    """Remove a conversation entirely — disk file + every chat_turn row in
    Supabase for that (user, session). Audit logged."""
    p = _session_path(sid)
    existed = p.exists()
    if existed:
        try:
            p.unlink()
        except Exception:
            pass
    # Also wipe the corresponding chat_turns rows so the session does not
    # reappear after the next Cloud restart / device switch.
    client = _supabase_client()
    uid = st.session_state.get('user_id', '')
    if client is not None and uid and not uid.startswith('_anon_'):
        try:
            (client.table('chat_turns')
                .delete()
                .eq('user_id', uid)
                .eq('session_id', sid)
                .execute())
        except Exception:
            pass
    _log_event('session_delete', {'session_id': sid, 'existed': existed})


def start_new_session():
    """Reset state for a new conversation. ID is assigned lazily on first message."""
    st.session_state['current_session_id'] = None
    st.session_state['current_session_title'] = ''
    st.session_state['current_session_created_at'] = ''
    st.session_state['user_inputs'] = []
    st.session_state['generated_responses'] = []
    st.session_state['thinking_traces'] = []
    st.session_state['retrieved_per_turn'] = []
    st.session_state['query_variants_per_turn'] = []
