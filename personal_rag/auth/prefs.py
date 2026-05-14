"""Persist per-user UI/config selections across Streamlit reruns and reboots.

Streamlit Cloud disconnects idle WebSocket sessions, and reconnection resets
``st.session_state``. We persist the slowly-changing config (API keys,
provider, model, retrieval / sampling settings, active view, current
session id) to disk so the user does not lose them when they step away.
Conversation history itself is already persisted via the session JSONL files.

Writes go to two places:

1. **Local disk** — fast, works offline, recovers the warm container case.
2. **Supabase** — survives container restarts and follows the user across
   devices. Authoritative on conflict for logged-in users.
"""

import json

import streamlit as st

from .supabase_io import _supabase_load_prefs, _supabase_save_prefs
from .users import _user_data_dir


_PERSIST_KEYS = (
    'provider', 'model', 'base_url',
    'hf_api_key', 'openai_api_key', 'anthropic_api_key', 'fireworks_api_key',
    'dashscope_api_key', 'custom_api_key',
    'tavily_key', 'brave_key',
    'embedder_model', 'chunk_size', 'chunk_overlap',
    'retrieval_mode', 'retrieve_top_n', 'final_top_k', 'use_reranker',
    'use_pgvector_search', 'use_agentic_search', 'agentic_max_iters',
    'general_chat_mode',
    'use_contextual_rewrite', 'use_multi_query', 'use_hyde', 'n_paraphrases',
    'per_doc_balance', 'per_doc_reserve', 'comparison_autodetect',
    'include_page_images', 'max_page_images',
    'web_enabled', 'web_provider', 'web_top_n',
    'max_tokens', 'temperature', 'top_p', 'sampling_top_k',
    'presence_penalty', 'stream', 'enable_thinking',
    'chat_doc_filter', 'active_view',
    # Active conversation pointer — combined with auto-restore in view_chat
    # this lets the user pick up exactly where they were after a Cloud
    # idle reset, not just "new chat on the chat tab".
    'current_session_id',
)


def _user_prefs_path():
    return _user_data_dir() / 'preferences.json'


def _load_user_prefs():
    """Override session_state defaults with whatever was last saved for this
    user. Tries disk first (fast, available locally), then merges in
    Supabase prefs (survives Cloud container restarts and follows the user
    across devices). Supabase values win on conflict — they are the
    authoritative source for logged-in users.

    Crucial: this must run only ONCE per Streamlit session. Otherwise every
    rerun (e.g. when a sidebar button sets active_view='settings') would
    immediately overwrite the just-clicked value with the previously-saved
    disk value, making the UI feel unresponsive.

    After an idle reconnect, session_state is wiped → the _prefs_loaded flag
    is gone → this function runs again and restores the user's last state.
    """
    if st.session_state.get('_prefs_loaded'):
        return

    data = {}

    # Disk first — present on warm Cloud container or local dev.
    p = _user_prefs_path()
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}

    # Supabase next — wins on conflict so cross-device sync works.
    sb_prefs = _supabase_load_prefs()
    if isinstance(sb_prefs, dict) and sb_prefs:
        data.update(sb_prefs)

    for k in _PERSIST_KEYS:
        if k not in data:
            continue
        v = data[k]
        if v is None:
            continue
        if isinstance(v, str) and not v:
            # Don't clobber an env-loaded key with an empty saved string.
            continue
        st.session_state[k] = v
    # Fallback: deprecated models (Responses-only / Korean-native not on
    # HF Inference Providers) should be swapped before any API call. We
    # only have the swap table after PROVIDERS is defined further down the
    # file, so just check by name here — the table itself is referenced at
    # call time by _resolve_deprecated_model(). This pass handles "user's
    # picker still shows the bad name" on session start.
    _DEPRECATED_NOW = {
        'gpt-5-pro', 'o1-pro', 'gpt-5.5-pro', 'gpt-5.4-pro',
        'LGAI-EXAONE/EXAONE-4.5-33B',
        'naver-hyperclovax/HyperCLOVAX-SEED-Think-32B',
    }
    if st.session_state.get('model') in _DEPRECATED_NOW:
        # If we know the provider, swap to its default; otherwise leave a
        # neutral safe choice. The call-time guard handles the rest.
        prov = st.session_state.get('provider', '')
        if prov == 'OpenAI':
            st.session_state['model'] = 'gpt-5-mini'
        elif prov == 'Hugging Face Router':
            st.session_state['model'] = 'Qwen/Qwen3-Next-80B-A3B-Instruct'
        else:
            st.session_state['model'] = 'Qwen/Qwen3-Next-80B-A3B-Instruct'
    st.session_state['_prefs_loaded'] = True


def _save_user_prefs():
    """Persist current configurable session_state. Writes to local disk
    (fast, works offline) AND to Supabase user_preferences (survives
    container restarts, follows the user across devices). Each side is
    best-effort; if one fails the other still happened."""
    data = {k: st.session_state.get(k) for k in _PERSIST_KEYS}
    try:
        encoded = json.dumps(data, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        return
    if st.session_state.get('_prefs_snapshot') == encoded:
        return
    try:
        _user_prefs_path().write_text(encoded)
    except Exception:
        pass
    _supabase_save_prefs(data)
    st.session_state['_prefs_snapshot'] = encoded
