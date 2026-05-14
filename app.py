"""Personal RAG Chatbot — entry point.

Run with ``streamlit run app.py``.

This file is intentionally short. Every feature — retrieval, parsing,
chat orchestration, agents, the sidebar, and each view tab — lives in
``personal_rag/`` and is composed here in the order Streamlit needs to
execute on every rerun:

1. Silence transformers warnings before any heavy import.
2. Load env from local .env files; bridge ``st.secrets`` onto
   ``os.environ`` so secret-based deployments still work.
3. ``st.set_page_config`` + global CSS — required to be early.
4. Make sure on-disk roots exist.
5. Boot sequence: seed session state, gate on auth, hydrate prefs,
   load this user's persisted docs into memory.
6. Render the persistent sidebar.
7. Dispatch to whichever view the user selected.
8. Persist any pref changes the rerun produced.

For newcomers: open ``personal_rag/`` and read the ``__init__.py`` docstring
in each subpackage — they describe what each module owns.
"""

import os
import warnings
from pathlib import Path

# Silence transformers deprecation logs before any heavy import.
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message=r'.*__path__.*')

import streamlit as st
from dotenv import load_dotenv

from personal_rag.auth.prefs import _load_user_prefs, _save_user_prefs
from personal_rag.auth.users import _auth_gate
from personal_rag.branding import FAVICON
from personal_rag.config import APP_CSS, DATA_DIR, LOGS_DIR
from personal_rag.data.storage import load_all_for_current_embedder
from personal_rag.state import _init_state
from personal_rag.ui.sidebar import render_sidebar
from personal_rag.views.about import view_about
from personal_rag.views.agents import view_agents
from personal_rag.views.cache import view_cache
from personal_rag.views.chat import view_chat
from personal_rag.views.docs import view_docs
from personal_rag.views.settings import view_settings


# -----------------------------------------------------------------------------
# Environment + Streamlit page boilerplate
# -----------------------------------------------------------------------------

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

# CSS — minimal custom styling, defined in personal_rag/config.py. We
# deliberately do NOT force display/width on Streamlit's own layout elements
# (stSidebar, stHeader, etc.) so the framework can manage its own responsive
# behavior and the sidebar can be collapsed/reopened normally.
st.markdown(APP_CSS, unsafe_allow_html=True)

# Ensure the on-disk roots exist before any view tries to write to them.
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Boot sequence
# -----------------------------------------------------------------------------
# Order matters: state defaults first → auth gate resolves user_id → prefs
# load reads from per-user paths → vector store hydrates the embedder's docs.

_init_state()
_auth_gate()
_load_user_prefs()
load_all_for_current_embedder()


# -----------------------------------------------------------------------------
# Sidebar + view dispatch
# -----------------------------------------------------------------------------

render_sidebar()

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
