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

LOGO_PATH = Path(__file__).parent / 'logo' / 'real_logo.png'           # full lockup
FAVICON_PATH = Path(__file__).parent / 'logo' / 'real_1_엠블럼.png'    # square emblem for browser tab


def _b64(path: Path) -> str:
    if not path.exists():
        return ''
    try:
        return base64.b64encode(path.read_bytes()).decode('ascii')
    except Exception:
        return ''


_LOGO_B64 = _b64(LOGO_PATH)
_LOGO_URI = f'data:image/png;base64,{_LOGO_B64}' if _LOGO_B64 else ''


# Favicon needs a square-ish source so the tab icon is recognizable.
_FAVICON = str(FAVICON_PATH) if FAVICON_PATH.exists() else (
    str(LOGO_PATH) if LOGO_PATH.exists() else None
)


st.set_page_config(
    page_title='Personal RAG',
    page_icon=_FAVICON,
    layout='wide',
    initial_sidebar_state='expanded',
)

# CSS — minimal custom styling. We deliberately do NOT force display/width on
# Streamlit's own layout elements (stSidebar, stHeader, etc.) so the framework
# can manage its own responsive behavior, the sidebar can be collapsed and
# reopened normally, and future Streamlit DOM changes don't break our layout.
st.markdown(
    """
    <style>
      /* App brand inside sidebar. */
      .sb-brand { font-size: 18px; font-weight: 700; letter-spacing: -0.01em; margin-bottom: 2px; }
      .sb-tagline { color: rgba(128,128,128,0.95); font-size: 12px; margin-bottom: 4px; }
      .sb-section { color: rgba(128,128,128,0.95); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 16px; margin-bottom: 6px; }

      /* Section labels inside main views. */
      .section-title { font-size: 18px; font-weight: 600; margin: 6px 0 6px 0; }
      .section-sub   { color: rgba(128,128,128,0.95); font-size: 13px; margin-bottom: 10px; line-height: 1.5; }

      /* Empty-state cards and chat hero. */
      .empty-state { padding: 28px; border: 1px dashed rgba(128,128,128,0.28); border-radius: 12px; text-align: center; color: rgba(128,128,128,0.95); line-height: 1.6; }
      .empty-hero { text-align: center; padding: 64px 16px 24px 16px; }
      .empty-hero h2 { font-size: 26px; font-weight: 600; margin: 0 0 8px 0; letter-spacing: -0.01em; }
      .empty-hero p { color: rgba(128,128,128,0.95); font-size: 14px; margin: 0 0 24px 0; }

      /* Status chips used inside sidebar bottom status. */
      .chip { display: inline-block; padding: 3px 9px; background: rgba(120,120,120,0.12); border: 1px solid rgba(120,120,120,0.22); border-radius: 999px; font-size: 11px; line-height: 1.4; margin-right: 4px; margin-bottom: 4px; }
      .chip.active { background: rgba(46,160,67,0.12); border-color: rgba(46,160,67,0.30); }
      .chip.muted  { color: rgba(128,128,128,0.95); }

      /* Make hr/divider lines softer. */
      hr { border-color: rgba(128,128,128,0.16) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).parent / '.data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path(__file__).parent / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Multi-user isolation ----------
# Each user gets their own subtree under DATA_DIR and LOGS_DIR so uploaded
# documents, indexes, conversation history, and agent runs do not leak between
# users sharing this deployment. If no users are configured in secrets, the
# app falls back to a single '_local' user (suitable for local dev).

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

EMBEDDER_CHOICES = [
    'BAAI/bge-m3',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
]
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'

# Provider presets for OpenAI-compatible endpoints.
PROVIDERS = {
    'Hugging Face Router': {
        'base_url': 'https://router.huggingface.co/v1',
        'env_key': 'HF_TOKEN',
        'default_model': 'google/gemma-4-31B-it',
    },
    'OpenAI': {
        'base_url': 'https://api.openai.com/v1',
        'env_key': 'OPENAI_API_KEY',
        'default_model': 'gpt-5-mini',
    },
    'DashScope (Qwen)': {
        'base_url': 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
        'env_key': 'DASHSCOPE_API_KEY',
        'default_model': 'qwen3.6-27b',
    },
    'vLLM / local': {
        'base_url': 'http://localhost:8000/v1',
        'env_key': None,
        'default_model': '',
    },
    'Custom': {
        'base_url': '',
        'env_key': None,
        'default_model': '',
    },
}
PROVIDER_NAMES = list(PROVIDERS.keys())

# Popular model IDs per provider. "직접 입력" sentinel lets the user type any
# other model ID; the actual model string sent to the API is always the raw ID.
_CUSTOM = '__custom__'
PROVIDER_MODELS = {
    'Hugging Face Router': [
        # DeepSeek — flagship general purpose
        'deepseek-ai/DeepSeek-V4-Pro',
        'deepseek-ai/DeepSeek-V4-Flash',
        'deepseek-ai/DeepSeek-V3.2',
        'deepseek-ai/DeepSeek-R1',
        # Qwen3 — strong multilingual incl. Korean
        'Qwen/Qwen3.6-35B-A3B',
        'Qwen/Qwen3-Next-80B-A3B-Instruct',
        'Qwen/Qwen3-235B-A22B-Instruct-2507',
        'Qwen/Qwen3-235B-A22B-Thinking-2507',
        'Qwen/Qwen3-VL-30B-A3B-Instruct',
        'Qwen/Qwen2.5-7B-Instruct',
        # Google Gemma 4 (multimodal)
        'google/gemma-4-31B-it',
        'google/gemma-4-26B-A4B-it',
        # Meta Llama
        'meta-llama/Llama-3.3-70B-Instruct',
        'meta-llama/Llama-3.1-8B-Instruct',
        # Moonshot Kimi K2 (long-context, agentic)
        'moonshotai/Kimi-K2.6',
        'moonshotai/Kimi-K2-Thinking',
        # GLM (Z.ai)
        'zai-org/GLM-5.1',
        'zai-org/GLM-4.7',
        # OpenAI open-source
        'openai/gpt-oss-120b',
        'openai/gpt-oss-20b',
        # MiniMax
        'MiniMaxAI/MiniMax-M2.7',
        # Cohere
        'CohereLabs/c4ai-command-a-03-2025',
    ],
    'OpenAI': [
        # GPT-5 latest tier
        'gpt-5.5',
        'gpt-5.5-pro',
        'gpt-5.4',
        'gpt-5.4-pro',
        'gpt-5.4-mini',
        'gpt-5.4-nano',
        'gpt-5.3-chat-latest',
        'gpt-5.2',
        'gpt-5.1',
        'gpt-5',
        'gpt-5-pro',
        'gpt-5-mini',
        'gpt-5-nano',
        'gpt-5-chat-latest',
        # GPT-4.1 series
        'gpt-4.1',
        'gpt-4.1-mini',
        'gpt-4.1-nano',
        # GPT-4o family
        'gpt-4o',
        'gpt-4o-mini',
        # Reasoning models
        'o4-mini',
        'o3',
        'o3-mini',
        'o1',
        'o1-pro',
    ],
    'DashScope (Qwen)': [
        'qwen3.6-27b',
        'qwen3-72b',
        'qwen-max',
        'qwen-plus',
        'qwen-turbo',
        'qwen-vl-max',
    ],
    'vLLM / local': [],
    'Custom': [],
}


# =============================================================================
# Session state
# =============================================================================

def _init_state():
    defaults = {
        # chat
        'user_inputs': [],
        'generated_responses': [],
        'thinking_traces': [],
        'retrieved_per_turn': [],

        # model / endpoint
        'provider': 'Hugging Face Router',
        'model': 'google/gemma-4-31B-it',
        'base_url': os.getenv('OPENAI_BASE_URL', 'https://router.huggingface.co/v1'),

        # Per-provider API keys. Stored separately so switching providers
        # does not overwrite a manually entered key.
        'hf_api_key': os.getenv('HF_TOKEN', ''),
        'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
        'dashscope_api_key': os.getenv('DASHSCOPE_API_KEY', ''),
        'custom_api_key': '',

        # generation
        'max_tokens': 8192,
        'temperature': 1.0,
        'top_p': 0.95,
        'sampling_top_k': 64,
        'presence_penalty': 0.0,
        'enable_thinking': True,
        'stream': True,

        # rag config
        'embedder_model': EMBEDDER_CHOICES[0],
        'chunk_size': 600,
        'chunk_overlap': 80,
        'retrieval_mode': 'hybrid',   # 'dense' | 'bm25' | 'hybrid'
        'retrieve_top_n': 20,
        'final_top_k': 5,
        'use_reranker': True,

        # query expansion config
        'use_hyde': False,
        'use_multi_query': False,
        'n_paraphrases': 3,
        'use_contextual_rewrite': True,

        # cross-document retrieval
        'per_doc_balance': True,
        'per_doc_reserve': 1,
        'comparison_autodetect': True,

        # multimodal config
        'include_page_images': False,
        'max_page_images': 3,

        # web search config
        'web_enabled': False,
        'web_provider': 'duckduckgo',  # 'duckduckgo' | 'tavily' | 'brave'
        'web_top_n': 5,
        'tavily_key': os.getenv('TAVILY_API_KEY', ''),
        'brave_key': os.getenv('BRAVE_API_KEY', ''),

        # per-turn parallel arrays
        'query_variants_per_turn': [],

        # navigation
        'active_view': 'chat',
        '_pending_input': None,

        # chat-time document filter (None = use all loaded docs)
        'chat_doc_filter': None,

        # session (conversation) tracking
        'current_session_id': None,
        'current_session_title': '',
        'current_session_created_at': '',

        # in-memory doc store: list of dicts {id, name, raw_text, chunks, embeddings, chunk_size, chunk_overlap}
        'docs': [],
        '_loaded_for_embedder': None,   # tracks which embedder's persisted docs are loaded
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


# ---------- Login gate ----------

def _render_login_screen():
    """Render brand + login form. Sets user_id on success and reruns; otherwise
    st.stop()s so the rest of the app is gated off.

    Also renders a minimal sidebar (brand only) so the page does not look like
    the sidebar disappeared — it just has no nav until login completes.
    """
    with st.sidebar:
        if _LOGO_URI:
            st.markdown(
                f'<div style="text-align:center; padding:4px 0 4px 0;">'
                f'<img src="{_LOGO_URI}" '
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

    if _LOGO_URI:
        st.markdown(
            f'<div style="text-align:center; margin-top:80px; margin-bottom:8px;">'
            f'<img src="{_LOGO_URI}" '
            f'style="max-width:240px; height:auto; opacity:0.95;" /></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="text-align:center; font-size:14px; color:rgba(128,128,128,0.9); '
        'margin-bottom:24px;">로그인이 필요합니다.</div>',
        unsafe_allow_html=True,
    )
    _, form_col, _ = st.columns([1, 2, 1])
    with form_col:
        with st.form('_login_form', clear_on_submit=False):
            username = st.text_input('사용자 ID')
            password = st.text_input('비밀번호', type='password')
            submitted = st.form_submit_button(
                '로그인', use_container_width=True, type='primary'
            )
        if submitted:
            if (username in USERS_FROM_SECRETS
                    and str(USERS_FROM_SECRETS[username]) == str(password)):
                st.session_state['user_id'] = username
                st.session_state['_loaded_for_embedder'] = None
                st.rerun()
            else:
                st.error('사용자 ID 또는 비밀번호가 올바르지 않습니다.')
    st.stop()


def _is_streamlit_cloud() -> bool:
    """Detect Streamlit Community Cloud — apps are mounted under /mount/src/."""
    return Path('/mount/src').exists()


def _auth_gate():
    """Resolve the active user_id:
      - users configured in st.secrets['users'] → render login form,
        set user_id to the verified username (persistent identity).
      - no users configured AND deployed on Streamlit Cloud → each browser
        session gets a fresh anonymous user_id (per-visitor isolation).
      - no users configured AND running locally → single-tenant '_local'
        (so a developer keeps their persisted data between reruns).
    """
    if 'user_id' in st.session_state and st.session_state['user_id']:
        return

    if USERS_FROM_SECRETS:
        _render_login_screen()
        return

    if _is_streamlit_cloud():
        # New visitor on a shared deployment without login → give them their
        # own isolated workspace for this browser session. Closing the tab
        # ends the session; previous anonymous data remains on disk under
        # its UUID until an admin cleans it up.
        st.session_state['user_id'] = '_anon_' + uuid.uuid4().hex[:10]
    else:
        st.session_state['user_id'] = '_local'


_auth_gate()


# =============================================================================
# Persistent vector store
# =============================================================================

def _safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', s)


def _embedder_dir(embedder_id: str) -> Path:
    return _user_data_dir() / _safe_name(embedder_id)


def compute_doc_id(name: str, raw_text: str, chunk_size: int, chunk_overlap: int) -> str:
    h = hashlib.sha256()
    h.update(name.encode('utf-8'))
    h.update(b'\0')
    h.update(raw_text.encode('utf-8', errors='ignore'))
    h.update(f'\0{chunk_size}\0{chunk_overlap}'.encode())
    return h.hexdigest()[:16]


def save_doc(embedder_id: str, doc: dict) -> None:
    d = _embedder_dir(embedder_id) / doc['id']
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        'id': doc['id'],
        'name': doc['name'],
        'chunks': doc['chunks'],
        'chunk_pages': doc.get('chunk_pages', []),
        'page_count': doc.get('page_count', 0),
        'has_page_images': bool(doc.get('has_page_images', False)),
        'is_pdf': bool(doc.get('is_pdf', False)),
        'raw_text': doc['raw_text'],
        'chunk_size': doc['chunk_size'],
        'chunk_overlap': doc['chunk_overlap'],
    }
    (d / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False))
    np.save(d / 'embeddings.npy', doc['embeddings'])


def load_doc(embedder_id: str, doc_id: str):
    d = _embedder_dir(embedder_id) / doc_id
    mp = d / 'meta.json'
    ep = d / 'embeddings.npy'
    if not (mp.exists() and ep.exists()):
        return None
    try:
        meta = json.loads(mp.read_text())
        embs = np.load(ep)
    except Exception:
        return None
    return {
        'id': doc_id,
        'name': meta['name'],
        'raw_text': meta.get('raw_text', ''),
        'chunks': meta['chunks'],
        'chunk_pages': meta.get('chunk_pages', []),
        'page_count': meta.get('page_count', 0),
        'has_page_images': meta.get('has_page_images', False),
        'is_pdf': meta.get('is_pdf', False),
        'chunk_size': meta.get('chunk_size'),
        'chunk_overlap': meta.get('chunk_overlap'),
        'embeddings': embs,
    }


def _pages_dir(embedder_id: str, doc_id: str) -> Path:
    return _embedder_dir(embedder_id) / doc_id / 'pages'


# ---------- Conversation sessions ----------

def _new_session_id() -> str:
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]


def _session_path(sid: str) -> Path:
    return _user_sessions_dir() / f'{sid}.json'


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


def load_session(sid: str) -> bool:
    p = _session_path(sid)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
    except Exception:
        return False
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


def list_sessions(limit: int = 30):
    """Return sessions sorted by updated_at desc, with light metadata only."""
    sd = _user_sessions_dir()
    if not sd.exists():
        return []
    out = []
    for p in sd.glob('*.json'):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        out.append({
            'id': data.get('id', p.stem),
            'title': data.get('title') or '(제목 없음)',
            'updated_at': data.get('updated_at', ''),
            'model': data.get('model', ''),
        })
    out.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return out[:limit]


def delete_session(sid: str):
    p = _session_path(sid)
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


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


def _session_jsonl_path(sid: str) -> Path:
    return _user_logs_dir() / f'{sid}.jsonl'


def log_turn_structured(user_input: str, response_text: str, reasoning: str,
                        retrieved: list, model: str, elapsed_sec,
                        query_variants: list = None):
    """Append one JSON record per turn to ./logs/{session_id}.jsonl.
    Postgres-ready schema. One line per turn; analysis-friendly.

    Schema fields: session_id, session_title, turn_index, timestamp,
      user_message, assistant_message, reasoning,
      model, provider, base_url, elapsed_seconds,
      retrieved[] (rank/source/score/text/doc_id/doc_name/chunk_index/pages/url/title),
      n_retrieved, citation_numbers_used[], n_cited,
      query_variants[], settings_snapshot{} (all retrieval/sampling/feature flags).
    """
    sid = st.session_state.get('current_session_id')
    if not sid:
        return
    path = _session_jsonl_path(sid)

    # Normalize retrieved entries
    retrieved_records = []
    for j, r in enumerate(retrieved or [], start=1):
        rec = {
            'rank': j,
            'source': r.get('source'),
            'score': r.get('score'),
            'text': (r.get('text') or '')[:1200],
        }
        if r.get('source') == 'web':
            rec['title'] = r.get('doc')
            rec['url'] = r.get('url')
        else:
            rec['doc_id'] = r.get('doc_id')
            rec['doc_name'] = r.get('doc')
            rec['chunk_index'] = r.get('chunk_idx')
            rec['pages'] = r.get('pages') or []
        retrieved_records.append(rec)

    n_chunks = len(retrieved) if retrieved else 0
    cited_nums = sorted(parse_citations(response_text, n_chunks)) if n_chunks else []

    settings = {
        'embedder_model': st.session_state.get('embedder_model'),
        'retrieval_mode': st.session_state.get('retrieval_mode'),
        'retrieve_top_n': st.session_state.get('retrieve_top_n'),
        'final_top_k': st.session_state.get('final_top_k'),
        'use_reranker': st.session_state.get('use_reranker'),
        'use_hyde': st.session_state.get('use_hyde'),
        'use_multi_query': st.session_state.get('use_multi_query'),
        'n_paraphrases': st.session_state.get('n_paraphrases'),
        'use_contextual_rewrite': st.session_state.get('use_contextual_rewrite'),
        'per_doc_balance': st.session_state.get('per_doc_balance'),
        'per_doc_reserve': st.session_state.get('per_doc_reserve'),
        'comparison_autodetect': st.session_state.get('comparison_autodetect'),
        'chat_doc_filter': st.session_state.get('chat_doc_filter'),
        'web_enabled': st.session_state.get('web_enabled'),
        'web_provider': st.session_state.get('web_provider'),
        'web_top_n': st.session_state.get('web_top_n'),
        'include_page_images': st.session_state.get('include_page_images'),
        'max_page_images': st.session_state.get('max_page_images'),
        'temperature': st.session_state.get('temperature'),
        'top_p': st.session_state.get('top_p'),
        'sampling_top_k': st.session_state.get('sampling_top_k'),
        'presence_penalty': st.session_state.get('presence_penalty'),
        'max_tokens': st.session_state.get('max_tokens'),
        'stream': st.session_state.get('stream'),
        'enable_thinking': st.session_state.get('enable_thinking'),
        'chunk_size': st.session_state.get('chunk_size'),
        'chunk_overlap': st.session_state.get('chunk_overlap'),
    }

    record = {
        'session_id': sid,
        'session_title': st.session_state.get('current_session_title', ''),
        'turn_index': len(st.session_state['user_inputs']),
        'timestamp': datetime.datetime.now().isoformat(timespec='microseconds'),
        'user_message': user_input,
        'assistant_message': response_text or '',
        'reasoning': reasoning or '',
        'model': model,
        'provider': st.session_state.get('provider', ''),
        'base_url': st.session_state.get('base_url', ''),
        'elapsed_seconds': float(elapsed_sec) if elapsed_sec is not None else None,
        'retrieved': retrieved_records,
        'n_retrieved': len(retrieved_records),
        'citation_numbers_used': cited_nums,
        'n_cited': len(cited_nums),
        'query_variants': list(query_variants or []),
        'settings_snapshot': settings,
    }

    try:
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception as e:
        st.warning(f'구조화 로그 저장 실패: {e}')


def auto_title_session() -> str:
    """LLM-generate a short Korean title from the first turn."""
    if not st.session_state['user_inputs'] or not st.session_state['generated_responses']:
        return ''
    user_msg = st.session_state['user_inputs'][0][:300]
    asst_msg = st.session_state['generated_responses'][0][:300]
    prompt = (
        '다음 대화 첫 턴을 보고 8글자 내외의 간결한 한국어 제목을 만드세요. '
        '큰따옴표/마침표/이모지 없이 제목만 출력하세요.\n\n'
        f'사용자: {user_msg}\n어시스턴트: {asst_msg}\n\n제목:'
    )
    try:
        client = get_openai_client()
        tparams = _build_completion_params(
            model=st.session_state['model'],
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=80,
            temperature=0.3,
            extra_body=_thinking_off_extra_body(),
        )
        resp = client.chat.completions.create(**tparams)
        title = (resp.choices[0].message.content or '').strip()
        title = title.strip('"').strip("'").strip('「').strip('」').strip()
        if len(title) > 30:
            title = title[:30]
        return title or '(제목 없음)'
    except Exception:
        return ''


@st.cache_data(show_spinner=False)
def load_page_image_b64(embedder_id: str, doc_id: str, page_no: int):
    """Read and base64-encode a rendered PDF page. Cached so multimodal turns
    don't re-read the same PNG every time."""
    p = _pages_dir(embedder_id, doc_id) / f'{page_no}.png'
    if not p.exists():
        return None
    try:
        return base64.b64encode(p.read_bytes()).decode('ascii')
    except Exception:
        return None


def delete_saved_doc(embedder_id: str, doc_id: str) -> None:
    d = _embedder_dir(embedder_id) / doc_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def list_saved_doc_ids(embedder_id: str):
    d = _embedder_dir(embedder_id)
    if not d.exists():
        return []
    return [p.name for p in d.iterdir() if p.is_dir() and (p / 'meta.json').exists()]


def load_all_for_current_embedder():
    """Populate st.session_state['docs'] from disk for the current embedder."""
    eid = st.session_state['embedder_model']
    if st.session_state.get('_loaded_for_embedder') == eid:
        return
    docs = []
    for did in list_saved_doc_ids(eid):
        d = load_doc(eid, did)
        if d is not None:
            docs.append(d)
    st.session_state['docs'] = docs
    st.session_state['_loaded_for_embedder'] = eid


# =============================================================================
# Models (cached)
# =============================================================================

@st.cache_resource(show_spinner=False)
def _make_openai_client(base_url: str, api_key: str):
    """Build (and cache) an OpenAI client keyed by base_url+api_key.
    Changing either creates a fresh client; identical configs share one."""
    return OpenAI(base_url=base_url, api_key=api_key)


def _active_api_key() -> str:
    """Return the API key for the currently selected provider."""
    p = st.session_state.get('provider', 'Hugging Face Router')
    if p == 'Hugging Face Router':
        return st.session_state.get('hf_api_key', '') or ''
    if p == 'OpenAI':
        return st.session_state.get('openai_api_key', '') or ''
    if p == 'DashScope (Qwen)':
        return st.session_state.get('dashscope_api_key', '') or ''
    # vLLM / local / Custom
    return st.session_state.get('custom_api_key', '') or ''


def get_openai_client():
    """Resolve the active OpenAI-compatible client from session state."""
    return _make_openai_client(
        st.session_state.get('base_url') or '',
        _active_api_key(),
    )


@st.cache_resource(show_spinner=False)
def load_embedder(model_id: str):
    try:
        from transformers.utils import logging as _hf_logging
        _hf_logging.set_verbosity_error()
        _hf_logging.disable_progress_bar()
    except Exception:
        pass
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id)


@st.cache_resource(show_spinner=False)
def load_reranker(model_id: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_id)


# =============================================================================
# Parsing & chunking
# =============================================================================

def _read_bytes(file) -> bytes:
    """Read full bytes from a Streamlit UploadedFile, resetting position."""
    pos = file.tell() if hasattr(file, 'tell') else 0
    data = file.read()
    try:
        file.seek(pos)
    except Exception:
        pass
    return data


def parse_pdf_docling(pdf_bytes: bytes) -> dict:
    """Parse PDF with Docling. Returns {elements, page_count, ok}.

    elements is a flat list of {text, page} ordered as on the page.
    Tables are emitted as markdown table strings; lists/headings preserved.
    """
    import tempfile
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return {'elements': [], 'page_count': 0, 'ok': False}
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        d = result.document
    except Exception as e:
        st.warning(f'Docling 파싱 실패, pypdf로 폴백합니다: {e}')
        return {'elements': [], 'page_count': 0, 'ok': False}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    elements = []
    pages_seen = set()
    for item, _level in d.iterate_items():
        text = ''
        try:
            if hasattr(item, 'export_to_markdown'):
                # Tables, lists, headings: structured markdown.
                text = item.export_to_markdown(d)
        except Exception:
            text = ''
        if not text and hasattr(item, 'text'):
            text = item.text or ''
        text = (text or '').strip()
        if not text:
            continue
        page = None
        prov = getattr(item, 'prov', None)
        if prov:
            try:
                page = prov[0].page_no
            except Exception:
                page = None
        if page is not None:
            pages_seen.add(page)
        elements.append({'text': text, 'page': page})

    page_count = max(pages_seen) if pages_seen else 0
    return {'elements': elements, 'page_count': page_count, 'ok': True}


def parse_pdf_pypdf(pdf_bytes: bytes) -> dict:
    """Fallback PDF parser. Page boundaries known but no structure."""
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    elements = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or '').strip()
        if text:
            elements.append({'text': text, 'page': i + 1})
    return {'elements': elements, 'page_count': len(reader.pages), 'ok': True}


def parse_file(file) -> dict:
    """Returns a dict: {raw_text, elements, page_count, is_pdf, pdf_bytes}.

    elements: list of {text, page}.  page is int (1-indexed) for PDFs,
    None for plain text files.  pdf_bytes is retained so the caller can
    render page images later (only set when is_pdf=True).
    """
    name = file.name.lower()
    if name.endswith('.pdf'):
        data = _read_bytes(file)
        parsed = parse_pdf_docling(data)
        if not parsed['ok'] or not parsed['elements']:
            try:
                parsed = parse_pdf_pypdf(data)
            except Exception as e:
                st.error(f'PDF 파싱 실패 ({file.name}): {e}')
                return {'raw_text': '', 'elements': [], 'page_count': 0,
                        'is_pdf': True, 'pdf_bytes': data}
        raw = '\n\n'.join(e['text'] for e in parsed['elements'])
        return {
            'raw_text': raw,
            'elements': parsed['elements'],
            'page_count': parsed['page_count'],
            'is_pdf': True,
            'pdf_bytes': data,
        }
    try:
        text = file.read().decode('utf-8', errors='ignore')
    except Exception as e:
        st.error(f'파일 읽기 실패 ({file.name}): {e}')
        return {'raw_text': '', 'elements': [], 'page_count': 0,
                'is_pdf': False, 'pdf_bytes': None}
    return {
        'raw_text': text,
        'elements': [{'text': text, 'page': None}] if text.strip() else [],
        'page_count': 0,
        'is_pdf': False,
        'pdf_bytes': None,
    }


def chunk_elements(elements: list, size: int, overlap: int):
    """Group elements (each {text, page}) into chunks of ~size chars.

    Returns (chunks: list[str], chunk_pages: list[list[int]]).
    """
    if not elements:
        return [], []
    size = max(50, int(size))
    overlap = max(0, min(int(overlap), size - 1))

    chunks, chunk_pages = [], []
    buf_text, buf_pages = '', set()

    def flush():
        nonlocal buf_text, buf_pages
        if buf_text:
            chunks.append(buf_text)
            chunk_pages.append(sorted(buf_pages))
            buf_text, buf_pages = '', set()

    for el in elements:
        text = (el.get('text') or '').strip()
        if not text:
            continue
        page = el.get('page')
        candidate = (buf_text + '\n\n' + text).strip() if buf_text else text
        if len(candidate) <= size:
            buf_text = candidate
            if page is not None:
                buf_pages.add(page)
            continue
        flush()
        if len(text) > size:
            step = max(1, size - overlap)
            for i in range(0, len(text), step):
                chunks.append(text[i:i + size])
                chunk_pages.append([page] if page is not None else [])
        else:
            buf_text = text
            buf_pages = {page} if page is not None else set()
    flush()

    # Merge tiny trailing chunk into its predecessor.
    if len(chunks) >= 2 and len(chunks[-1]) < max(50, size // 5):
        chunks[-2] = (chunks[-2] + '\n\n' + chunks[-1])[: size + overlap]
        merged_pages = sorted(set(chunk_pages[-2]) | set(chunk_pages[-1]))
        chunk_pages[-2] = merged_pages
        chunks.pop()
        chunk_pages.pop()

    return chunks, chunk_pages


def chunk_text(text: str, size: int, overlap: int) -> list:
    """Backward-compatible: plain-text chunking returning list of strings only."""
    chunks, _pages = chunk_elements(
        [{'text': text, 'page': None}], size, overlap
    )
    return chunks


def render_pdf_pages_to_dir(pdf_bytes: bytes, out_dir: Path, dpi: int = 144) -> int:
    """Render each PDF page to PNG into out_dir/{page}.png. Returns number of pages."""
    import fitz
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with fitz.open(stream=pdf_bytes, filetype='pdf') as d:
        for i, page in enumerate(d):
            pix = page.get_pixmap(dpi=dpi)
            pix.save(str(out_dir / f'{i + 1}.png'))
            n = i + 1
    return n


# =============================================================================
# Retrieval
# =============================================================================

def _tokenize_for_bm25(text: str):
    # Mixed-script tokenization: words/CJK/numbers. Lowercase ASCII.
    text = (text or '').lower()
    return re.findall(r"[\w가-힯一-鿿]+", text)


def build_bm25_over_docs(docs: list):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return None, None
    if not docs:
        return None, None
    all_chunks, meta = [], []
    for doc in docs:
        for i, ch in enumerate(doc['chunks']):
            all_chunks.append(ch)
            meta.append((doc['id'], doc['name'], i))
    tokenized = [_tokenize_for_bm25(c) for c in all_chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, meta


def _flatten_chunks(docs: list):
    all_chunks, all_embs, meta = [], [], []
    for doc in docs:
        embs = doc.get('embeddings')
        if embs is None or len(embs) != len(doc['chunks']):
            continue
        for i, ch in enumerate(doc['chunks']):
            all_chunks.append(ch)
            meta.append((doc['id'], doc['name'], i))
        all_embs.append(embs)
    if not all_embs:
        return [], np.zeros((0, 1), dtype=np.float32), []
    return all_chunks, np.vstack(all_embs), meta


def dense_search(query: str, top_n: int):
    docs = st.session_state['docs']
    chunks, embs, meta = _flatten_chunks(docs)
    if not chunks:
        return []
    embedder = load_embedder(st.session_state['embedder_model'])
    q = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    sims = embs @ q
    idx = np.argsort(-sims)[:top_n]
    return [(int(i), float(sims[i]), meta[i], chunks[i]) for i in idx]


def bm25_search(query: str, top_n: int):
    docs = st.session_state['docs']
    bm25, meta = build_bm25_over_docs(docs)
    if bm25 is None or not meta:
        return []
    chunks, _, _ = _flatten_chunks(docs)
    scores = bm25.get_scores(_tokenize_for_bm25(query))
    idx = np.argsort(-scores)[:top_n]
    return [(int(i), float(scores[i]), meta[i], chunks[i]) for i in idx]


def rrf_fuse(rankings: list, k: int = 60):
    """rankings: list of [(idx, score, meta, chunk), ...]; fuse via Reciprocal Rank Fusion."""
    rrf_scores = {}
    payload = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            idx = item[0]
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)
            payload[idx] = item
    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])
    out = []
    for idx, s in fused:
        i, _, m, c = payload[idx]
        out.append((i, s, m, c))
    return out


def rerank(query: str, candidates: list, top_k: int):
    if not candidates:
        return []
    reranker = load_reranker(RERANKER_MODEL)
    pairs = [[query, c[3]] for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: -float(x[1]))
    out = []
    for (i, _, m, c), s in scored[:top_k]:
        out.append((i, float(s), m, c))
    return out


def _is_openai_endpoint() -> bool:
    """Detect OpenAI's official API (rejects non-standard params like top_k)."""
    return 'api.openai.com' in (st.session_state.get('base_url') or '').lower()


def _is_dashscope_endpoint() -> bool:
    return 'dashscope' in (st.session_state.get('base_url') or '').lower()


def _is_hf_router_endpoint() -> bool:
    return 'router.huggingface.co' in (st.session_state.get('base_url') or '').lower()


def _provider_supports_top_k() -> bool:
    """top_k via extra_body is provider-specific. Live-tested:
      - OpenAI: rejects.
      - HF Router: many sub-providers reject (e.g. Cerebras for Llama-3.1-8B,
        gpt-oss-120b). Safest to omit unless DashScope/vLLM/Custom.
      - DashScope (Qwen): supports.
      - vLLM/local & Custom: typically supports."""
    if _is_openai_endpoint():
        return False
    if _is_hf_router_endpoint():
        return False
    return True


def _is_openai_reasoning_model(model: str) -> bool:
    """OpenAI o-series — strictest constraints: no temperature/top_p/penalty."""
    m = (model or '').lower()
    return m.startswith('o1') or m.startswith('o3') or m.startswith('o4')


def _is_openai_gpt5_family(model: str) -> bool:
    """GPT-5/5.x chat models — require max_completion_tokens, reject non-default
    temperature/top_p/penalty."""
    return (model or '').lower().startswith('gpt-5')


def _uses_max_completion_tokens(model: str) -> bool:
    """Models that require max_completion_tokens instead of max_tokens (OpenAI new tier)."""
    return _is_openai_reasoning_model(model) or _is_openai_gpt5_family(model)


def _build_completion_params(
    model: str, messages: list,
    *,
    max_tokens=None, temperature=None, top_p=None, presence_penalty=None,
    extra_body: dict = None, stream: bool = False,
) -> dict:
    """Build chat.completions.create kwargs that respect per-model constraints.
    Drops sampling params for GPT-5/o-series; uses max_completion_tokens for those."""
    out = {'model': model, 'messages': messages}
    if stream:
        out['stream'] = True

    is_openai = _is_openai_endpoint()
    if max_tokens is not None:
        budget = int(max_tokens)
        if is_openai and _is_openai_reasoning_model(model):
            # Reasoning eats budget internally; ensure room for actual output.
            budget = max(budget, 1500)
        if is_openai and _uses_max_completion_tokens(model):
            out['max_completion_tokens'] = budget
        else:
            out['max_tokens'] = budget

    # OpenAI new-tier rejects non-default sampling params; safest to omit them.
    restrict_sampling = is_openai and (
        _is_openai_reasoning_model(model) or _is_openai_gpt5_family(model)
    )
    if not restrict_sampling:
        if temperature is not None:
            out['temperature'] = float(temperature)
        if top_p is not None:
            out['top_p'] = float(top_p)
        if presence_penalty is not None:
            out['presence_penalty'] = float(presence_penalty)

    # extra_body — OpenAI rejects unknown fields, our caller is responsible for
    # not setting top_k there, but as a final safety net strip it for OpenAI.
    if extra_body:
        if is_openai:
            extra_body = {k: v for k, v in extra_body.items()
                          if k not in ('top_k', 'chat_template_kwargs', 'enable_thinking')}
        if extra_body:
            out['extra_body'] = extra_body
    return out


def _thinking_off_extra_body() -> dict:
    """Return extra_body that disables thinking, branched by provider.
    OpenAI has no thinking mode and rejects unknown fields, so return {} there."""
    if _is_openai_endpoint():
        return {}
    if _is_dashscope_endpoint():
        return {'enable_thinking': False}
    return {'chat_template_kwargs': {'enable_thinking': False}}


def rewrite_with_context(query: str) -> str:
    """If conversation history exists, rewrite the current query into a
    standalone form (resolving pronouns / implied subjects). Cheap fallback
    to original on any failure."""
    history_u = st.session_state['user_inputs']
    history_a = st.session_state['generated_responses']
    if not history_u:
        return query
    try:
        client = get_openai_client()
    except Exception:
        return query
    # Use only the last 3 turns to keep this cheap.
    hist_pairs = list(zip(history_u[-3:], history_a[-3:]))
    history_str = '\n'.join(f"User: {u}\nAssistant: {a}" for u, a in hist_pairs)
    prompt = (
        "Given the conversation history and the user's latest message, rewrite "
        "the latest message as a fully self-contained question. Resolve pronouns "
        "(그것/그게/it/this), fill in implied subjects, and make the question "
        "stand on its own. If it is already self-contained, return it unchanged. "
        "Output only the rewritten sentence — no preface, no quotes.\n\n"
        f"History:\n{history_str}\n\n"
        f"Latest: {query}\n\nRewritten:"
    )
    try:
        params = _build_completion_params(
            model=st.session_state['model'],
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=200, temperature=0.0,
            extra_body=_thinking_off_extra_body(),
        )
        resp = client.chat.completions.create(**params)
        rewritten = (resp.choices[0].message.content or '').strip()
        rewritten = rewritten.strip('"').strip("'").strip()
        return rewritten or query
    except Exception as e:
        st.warning(f'쿼리 재작성 실패 ({e}). 원본 질문으로 검색합니다.')
        return query


def expand_queries(query: str) -> list:
    """Return [effective_query, paraphrase1, ..., hyde]. The first variant is
    the contextually-rewritten query if rewriting is enabled and history exists;
    otherwise it is the original query."""
    base = query
    if st.session_state.get('use_contextual_rewrite') and st.session_state['user_inputs']:
        rewritten = rewrite_with_context(query)
        if rewritten and rewritten != query:
            base = rewritten
    variants = [base]
    # Always also keep the literal user query so retrieval is robust if the
    # rewrite drifted semantically.
    if base != query:
        variants.append(query)

    if not (st.session_state.get('use_multi_query') or st.session_state.get('use_hyde')):
        return variants
    try:
        client = get_openai_client()
    except Exception:
        return variants
    model = st.session_state['model']
    eb = _thinking_off_extra_body()

    if st.session_state.get('use_multi_query'):
        n = int(st.session_state.get('n_paraphrases', 3))
        prompt = (
            f"Rewrite the following question in {n} different ways while preserving "
            f"the meaning. Output one paraphrase per line. No numbering, no quotes, "
            f"no explanation.\n\nQuestion: {base}"
        )
        try:
            mq_params = _build_completion_params(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=300, temperature=0.7,
                extra_body=eb,
            )
            resp = client.chat.completions.create(**mq_params)
            text = (resp.choices[0].message.content or '').strip()
            added = 0
            for line in text.split('\n'):
                clean = line.strip().lstrip('-•*').strip()
                # Strip leading numbering like "1." or "1)"
                m = re.match(r'^\d+\s*[.)]\s*(.*)$', clean)
                if m:
                    clean = m.group(1)
                if clean and clean not in variants:
                    variants.append(clean)
                    added += 1
                    if added >= n:
                        break
        except Exception as e:
            st.warning(f'Multi-query 생성 실패: {e}')

    if st.session_state.get('use_hyde'):
        prompt = (
            "Write a concise factual paragraph that would hypothetically answer "
            "the following question, as if extracted from an authoritative document. "
            "No preface, no commentary — just the paragraph.\n\n"
            f"Question: {base}"
        )
        try:
            hy_params = _build_completion_params(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=250, temperature=0.3,
                extra_body=eb,
            )
            resp = client.chat.completions.create(**hy_params)
            hyde = (resp.choices[0].message.content or '').strip()
            if hyde:
                variants.append(hyde)
        except Exception as e:
            st.warning(f'HyDE 생성 실패: {e}')

    return variants


_COMPARISON_KEYWORDS = (
    '비교', '대조', '차이', '공통점', '공통', '유사점', '다른점', '대비', '구분',
    'compare', 'comparison', 'differ', 'difference', 'similar', 'versus', ' vs ',
)


def _is_comparison_query(query: str) -> bool:
    """Heuristic — does this query ask to compare/contrast across sources?"""
    q = (query or '').lower()
    return any(kw.lower() in q for kw in _COMPARISON_KEYWORDS)


def _select_with_per_doc_min(ranked: list, top_k: int, per_doc_min: int) -> list:
    """Pick top_k from a ranked list while reserving at least `per_doc_min` slots
    per document. Web results are treated as a single 'web' bucket.

    Phase 1: walk ranked list, assign each item to its doc's quota until quota fills.
    Phase 2: fill remaining slots with the highest-scoring not-yet-selected items."""
    if per_doc_min <= 0 or top_k <= 0:
        return ranked[:top_k]
    selected = []
    used = set()
    by_doc = {}

    def _bucket(item):
        # ranked item = (idx, score, meta, chunk). meta = (doc_id, doc_name, chunk_idx) for doc.
        # For web items we pass through a different shape; treat as 'web' bucket.
        try:
            meta = item[2]
            doc_id = meta[0] if isinstance(meta, tuple) else None
            return doc_id or 'web'
        except Exception:
            return 'web'

    for item in ranked:
        if len(selected) >= top_k:
            break
        b = _bucket(item)
        if by_doc.get(b, 0) < per_doc_min:
            selected.append(item)
            used.add(id(item))
            by_doc[b] = by_doc.get(b, 0) + 1

    for item in ranked:
        if len(selected) >= top_k:
            break
        if id(item) not in used:
            selected.append(item)
            used.add(id(item))
    return selected


def _effective_per_doc_min(query: str) -> int:
    """Apply user toggles + comparison autodetect. Returns the per-doc reserve to
    use for this query, considering how many documents the user has."""
    n_docs = len(st.session_state.get('docs', []))
    if n_docs < 2:
        return 0
    reserve = 0
    if st.session_state.get('per_doc_balance'):
        reserve = max(reserve, int(st.session_state.get('per_doc_reserve', 1)))
    if (st.session_state.get('comparison_autodetect')
            and _is_comparison_query(query)):
        reserve = max(reserve, 2)
    # Don't reserve more than would fit
    final_top_k = int(st.session_state.get('final_top_k', 5))
    if reserve * n_docs > final_top_k:
        # Allow at most floor(top_k / n_docs); minimum 1 if any reserve requested
        reserve = max(1, final_top_k // n_docs)
    return reserve


def _single_query_local_search(q: str, mode: str, top_n: int):
    """Run dense+/or BM25 for one query, return RRF-fused ranking (or single)."""
    rankings = []
    if mode in ('dense', 'hybrid'):
        rankings.append(dense_search(q, top_n))
    if mode in ('bm25', 'hybrid'):
        rankings.append(bm25_search(q, top_n))
    rankings = [r for r in rankings if r]
    if not rankings:
        return []
    if len(rankings) == 1:
        return rankings[0]
    return rrf_fuse(rankings)


def retrieve_local(query: str) -> list:
    """Hybrid retrieval with optional HyDE/multi-query expansion + reranker."""
    if not st.session_state['docs']:
        st.session_state['_last_variants'] = [query]
        return []
    mode = st.session_state['retrieval_mode']
    top_n = int(st.session_state['retrieve_top_n'])
    top_k = int(st.session_state['final_top_k'])

    variants = expand_queries(query)
    st.session_state['_last_variants'] = variants
    # variants[0] is the contextually-rewritten query (or the original if no
    # rewriting happened); used as the rerank target since it best captures intent.
    base_query = variants[0] if variants else query

    per_query_rankings = []
    for q in variants:
        ranking = _single_query_local_search(q, mode, top_n)
        if ranking:
            per_query_rankings.append(ranking)

    if not per_query_rankings:
        return []
    if len(per_query_rankings) == 1:
        fused = per_query_rankings[0][:top_n]
    else:
        fused = rrf_fuse(per_query_rankings)[:top_n]

    # Rerank operates on the full top_n candidates so we have a high-quality
    # ranked list to apply per-doc balancing on top of.
    if st.session_state['use_reranker']:
        try:
            fused = rerank(base_query, fused, top_k=len(fused))
        except Exception as e:
            st.warning(f'Reranker 실패 ({e}). RRF 결과를 그대로 사용합니다.')

    per_doc_min = _effective_per_doc_min(query)
    if per_doc_min > 0:
        fused = _select_with_per_doc_min(fused, top_k, per_doc_min)
    else:
        fused = fused[:top_k]

    docs_by_id = {d['id']: d for d in st.session_state['docs']}
    out = []
    for idx, score, meta, chunk in fused:
        doc_id, doc_name, chunk_idx = meta
        doc = docs_by_id.get(doc_id, {})
        pages_list = doc.get('chunk_pages') or []
        pages = pages_list[chunk_idx] if chunk_idx < len(pages_list) else []
        out.append({
            'source': 'doc',
            'doc_id': doc_id,
            'doc': doc_name,
            'chunk_idx': chunk_idx,
            'pages': pages,
            'text': chunk,
            'score': score,
        })
    return out


def web_search(query: str) -> list:
    """Return list of normalized web search results."""
    provider = st.session_state['web_provider']
    top_n = int(st.session_state['web_top_n'])
    try:
        if provider == 'duckduckgo':
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            results = list(DDGS().text(query, max_results=top_n))
            out = []
            for r in results:
                out.append({
                    'source': 'web',
                    'doc': r.get('title') or r.get('href', ''),
                    'url': r.get('href') or r.get('link', ''),
                    'chunk_idx': 0,
                    'text': r.get('body') or r.get('snippet', ''),
                    'score': 0.0,
                })
            return out

        if provider == 'tavily':
            key = st.session_state['tavily_key']
            if not key:
                st.warning(
                    'Tavily API 키가 비어 있습니다. '
                    '.env의 TAVILY_API_KEY에 추가하거나 사이드바에서 입력하세요. '
                    '키 발급: https://app.tavily.com'
                )
                return []
            try:
                from tavily import TavilyClient
            except ImportError:
                st.error(
                    'tavily-python 패키지가 설치되지 않았습니다. '
                    '터미널에서 `pip install tavily-python`를 실행해 주세요.'
                )
                return []
            client = TavilyClient(api_key=key)
            resp = client.search(query=query, max_results=top_n, search_depth='basic')
            out = []
            for r in resp.get('results', []):
                out.append({
                    'source': 'web',
                    'doc': r.get('title', ''),
                    'url': r.get('url', ''),
                    'chunk_idx': 0,
                    'text': r.get('content', ''),
                    'score': float(r.get('score', 0.0)),
                })
            return out

        if provider == 'brave':
            key = st.session_state['brave_key']
            if not key:
                st.warning('Brave API 키가 비어 있습니다 (.env BRAVE_API_KEY).')
                return []
            import requests
            r = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                params={'q': query, 'count': top_n},
                headers={
                    'X-Subscription-Token': key,
                    'Accept': 'application/json',
                },
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            out = []
            for item in data.get('web', {}).get('results', []):
                out.append({
                    'source': 'web',
                    'doc': item.get('title', ''),
                    'url': item.get('url', ''),
                    'chunk_idx': 0,
                    'text': item.get('description', ''),
                    'score': 0.0,
                })
            return out

    except Exception as e:
        st.warning(f'웹 검색 실패 ({provider}): {e}')
        return []
    return []


def retrieve(query: str) -> list:
    """Combine local document retrieval and web search results."""
    results = retrieve_local(query)
    if st.session_state['web_enabled']:
        results = results + web_search(query)
    return results


# =============================================================================
# Doc ingestion
# =============================================================================

def ingest_files(files):
    """For each new file: parse (Docling for PDF) → chunk with page metadata
    → embed → render page images (PDF) → save → register."""
    eid = st.session_state['embedder_model']
    size = st.session_state['chunk_size']
    overlap = st.session_state['chunk_overlap']
    existing_ids = {d['id'] for d in st.session_state['docs']}
    existing_names = {d['name'] for d in st.session_state['docs']}
    new_count = 0

    for f in files:
        if f.name in existing_names:
            continue
        parsed = parse_file(f)
        raw = parsed['raw_text']
        if not raw.strip():
            st.warning(f'{f.name}: 추출된 텍스트가 없어 인덱스에서 제외합니다.')
            continue
        chunks, chunk_pages = chunk_elements(parsed['elements'], size, overlap)
        if not chunks:
            st.warning(f'{f.name}: 청크가 생성되지 않았습니다.')
            continue
        did = compute_doc_id(f.name, raw, size, overlap)
        if did in existing_ids:
            continue
        cached = load_doc(eid, did)
        if cached is not None:
            doc = cached
        else:
            embedder = load_embedder(eid)
            embs = embedder.encode(
                chunks, convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )
            has_imgs = False
            if parsed['is_pdf'] and parsed['pdf_bytes']:
                try:
                    n_pages = render_pdf_pages_to_dir(
                        parsed['pdf_bytes'], _pages_dir(eid, did)
                    )
                    has_imgs = n_pages > 0
                except Exception as e:
                    st.warning(f'{f.name}: 페이지 이미지 렌더 실패 ({e}). 멀티모달 기능 사용 시 제한됨.')
            doc = {
                'id': did, 'name': f.name, 'raw_text': raw,
                'chunks': chunks, 'chunk_pages': chunk_pages,
                'page_count': parsed['page_count'],
                'has_page_images': has_imgs,
                'is_pdf': parsed['is_pdf'],
                'embeddings': embs,
                'chunk_size': size, 'chunk_overlap': overlap,
            }
            save_doc(eid, doc)
        st.session_state['docs'].append(doc)
        existing_ids.add(did)
        existing_names.add(doc['name'])
        new_count += 1
    return new_count


def remove_doc(doc_id: str):
    eid = st.session_state['embedder_model']
    delete_saved_doc(eid, doc_id)
    st.session_state['docs'] = [d for d in st.session_state['docs'] if d['id'] != doc_id]


def reindex_all():
    """Re-chunk + re-embed all current docs with current settings.

    Note: re-chunking uses the stored raw_text, which loses Docling page
    boundaries. Page metadata is preserved only for original ingestion.
    """
    eid = st.session_state['embedder_model']
    size = st.session_state['chunk_size']
    overlap = st.session_state['chunk_overlap']
    new_docs = []
    embedder = load_embedder(eid)
    for d in st.session_state['docs']:
        raw = d.get('raw_text', '')
        if not raw:
            new_docs.append(d)
            continue
        chunks = chunk_text(raw, size, overlap)
        if not chunks:
            continue
        new_id = compute_doc_id(d['name'], raw, size, overlap)
        cached = load_doc(eid, new_id)
        if cached is not None:
            new_docs.append(cached)
            continue
        embs = embedder.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        nd = {
            'id': new_id, 'name': d['name'], 'raw_text': raw,
            'chunks': chunks,
            'chunk_pages': [[] for _ in chunks],
            'page_count': d.get('page_count', 0),
            'has_page_images': d.get('has_page_images', False),
            'is_pdf': d.get('is_pdf', False),
            'embeddings': embs,
            'chunk_size': size, 'chunk_overlap': overlap,
        }
        save_doc(eid, nd)
        new_docs.append(nd)
    st.session_state['docs'] = new_docs


# =============================================================================
# Chat helpers
# =============================================================================

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Ground every factual claim in the Context "
    "provided in the user's latest message. Also use the prior conversation "
    "to understand the user — resolve pronouns, follow-up references, and "
    "implied subjects from earlier turns. Each Context entry is marked with "
    "[N] and tagged as either a local document or a web result. Cite entry "
    "numbers like [1], [2] for every fact you take from the Context. If a "
    "factual answer is not supported by the Context or recent conversation, "
    "reply exactly: \"제공된 자료에서는 답을 찾을 수 없습니다.\""
)


def _format_pages(pages):
    if not pages:
        return ''
    if len(pages) == 1:
        return f' p.{pages[0]}'
    return f' pp.{pages[0]}-{pages[-1]}'


def _context_label(r: dict) -> str:
    if r.get('source') == 'web':
        url = r.get('url', '')
        return f'웹: {r.get("doc", "")} | {url}'
    pages = r.get('pages') or []
    page_part = _format_pages(pages)
    return f'로컬: {r.get("doc", "")} (chunk {r.get("chunk_idx", 0)}{page_part})'


def _collect_page_image_parts(retrieved: list, max_images: int) -> list:
    """Return a list of {type: image_url, image_url: {url: data:...}} parts
    for unique (doc_id, page) pairs that have rendered images on disk."""
    parts = []
    seen = set()
    embedder_id = st.session_state['embedder_model']
    for r in retrieved:
        if r.get('source') != 'doc':
            continue
        doc_id = r.get('doc_id')
        for p in (r.get('pages') or []):
            key = (doc_id, p)
            if key in seen:
                continue
            seen.add(key)
            b64 = load_page_image_b64(embedder_id, doc_id, p)
            if not b64:
                continue
            parts.append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/png;base64,{b64}'},
            })
            if len(parts) >= max_images:
                return parts
    return parts


def build_messages(user_input: str, retrieved: list) -> list:
    msgs = []
    if retrieved:
        msgs.append({'role': 'system', 'content': RAG_SYSTEM_PROMPT})
    for u, a in zip(
        st.session_state['user_inputs'], st.session_state['generated_responses']
    ):
        msgs.append({'role': 'user', 'content': u})
        msgs.append({'role': 'assistant', 'content': a})

    if retrieved:
        ctx = '\n\n'.join(
            f'[{i + 1}] ({_context_label(r)}) {r["text"]}'
            for i, r in enumerate(retrieved)
        )
        text_content = f'Context:\n{ctx}\n\nQuestion: {user_input}'
    else:
        text_content = user_input

    # Multimodal: attach unique page images for current turn only.
    image_parts = []
    if retrieved and st.session_state.get('include_page_images'):
        max_imgs = int(st.session_state.get('max_page_images', 3))
        image_parts = _collect_page_image_parts(retrieved, max_imgs)

    if image_parts:
        content = [{'type': 'text', 'text': text_content}] + image_parts
        msgs.append({'role': 'user', 'content': content})
    else:
        msgs.append({'role': 'user', 'content': text_content})
    return msgs


def split_thinking(text: str):
    s = text or ''
    m = re.search(r'<think>(.*?)</think>\s*(.*)', s, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(
        r'<\|channel\|?>thought\s*(.*?)<\|?channel\|>\s*(.*)',
        s, flags=re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return '', s.strip()


_CITE_PATTERN = r'\[((?:\d+\s*,\s*)*\d+)\]'


def parse_citations(answer: str, n_chunks: int) -> set:
    nums = set()
    for group in re.findall(_CITE_PATTERN, answer or ''):
        for piece in group.split(','):
            try:
                n = int(piece.strip())
                if 1 <= n <= n_chunks:
                    nums.add(n)
            except ValueError:
                pass
    return nums


def format_answer_with_citations(answer: str, n_chunks: int) -> str:
    def repl(m):
        inner = m.group(1)
        for piece in inner.split(','):
            try:
                n = int(piece.strip())
                if 1 <= n <= n_chunks:
                    return f'**[{inner}]**'
            except ValueError:
                pass
        return m.group(0)
    return re.sub(_CITE_PATTERN, repl, answer or '')


def stream_chat(client, params: dict):
    """Stream response. Returns (full_text, reasoning_text)."""
    placeholder = st.empty()
    full_text = ''
    reasoning_text = ''
    try:
        stream = client.chat.completions.create(stream=True, **params)
        for chunk in stream:
            if not getattr(chunk, 'choices', None):
                continue
            delta = chunk.choices[0].delta
            rc = getattr(delta, 'reasoning_content', None) or ''
            c = getattr(delta, 'content', None) or ''
            if rc:
                reasoning_text += rc
            if c:
                full_text += c
            if full_text:
                placeholder.markdown(full_text)
            elif reasoning_text:
                placeholder.markdown(f'_(생각 중)_\n\n> {reasoning_text}')
        placeholder.empty()
    except Exception as e:
        placeholder.empty()
        raise
    return full_text, reasoning_text


def non_stream_chat(client, params: dict):
    resp = client.chat.completions.create(**params)
    choice = resp.choices[0]
    full_text = choice.message.content or ''
    reasoning = ''
    for attr in ('reasoning_content', 'reasoning'):
        val = getattr(choice.message, attr, None)
        if val:
            reasoning = val
            break
    return full_text, reasoning


def render_assistant(answer: str, reasoning: str, retrieved: list, turn_idx: int,
                     variants: list = None):
    if variants and len(variants) > 1:
        with st.expander(
            f'검색에 사용된 쿼리 변형 {len(variants)}개', expanded=False
        ):
            for i, v in enumerate(variants):
                label = '원본 질문' if i == 0 else f'변형 {i}'
                preview = v if len(v) <= 300 else v[:300] + '...'
                st.markdown(f'**{label}**: {preview}')
    if reasoning:
        with st.expander(f'추론 과정 (turn {turn_idx + 1})', expanded=False):
            st.markdown(reasoning)

    n = len(retrieved)
    pretty = format_answer_with_citations(answer, n) if n else (answer or '')
    st.markdown(pretty or '*(empty response)*')

    if not retrieved:
        return

    cited = parse_citations(answer, n)
    cited_items = [(i + 1, r) for i, r in enumerate(retrieved) if (i + 1) in cited]
    uncited_items = [(i + 1, r) for i, r in enumerate(retrieved) if (i + 1) not in cited]

    if cited_items:
        st.markdown(f"**출처** · 인용 {len(cited_items)} / 검색 {n}")
        for j, r in cited_items:
            with st.expander(_citation_summary(j, r), expanded=False):
                _citation_body(r)
        if uncited_items:
            with st.expander(
                f'인용되지 않은 검색 결과 {len(uncited_items)}개 보기', expanded=False
            ):
                for j, r in uncited_items:
                    st.markdown(f"**{_citation_summary(j, r)}**")
                    _citation_body(r)
                    st.divider()
    else:
        with st.expander(
            f'검색된 자료 {n}개 (모델이 [N] 인용 표기를 사용하지 않음)', expanded=False
        ):
            for i, r in enumerate(retrieved):
                st.markdown(f"**{_citation_summary(i + 1, r)}**")
                _citation_body(r)
                st.divider()


def _citation_summary(j: int, r: dict) -> str:
    from urllib.parse import urlparse
    if r.get('source') == 'web':
        host = urlparse(r.get('url', '')).netloc or '웹'
        title = (r.get('doc') or '').strip()
        return f"[{j}] 웹 · {host} — {title[:60]}"
    score = r.get('score', 0.0)
    pages = r.get('pages') or []
    page_part = _format_pages(pages)
    return (
        f"[{j}] {r.get('doc', '')}{page_part} · "
        f"chunk {r.get('chunk_idx', 0)} · score {score:.3f}"
    )


def _citation_body(r: dict):
    if r.get('source') == 'web':
        url = r.get('url', '')
        if url:
            st.markdown(f"[{url}]({url})")
    st.text(r.get('text', ''))


# =============================================================================
# Boot: load persisted docs for current embedder
# =============================================================================

load_all_for_current_embedder()


# =============================================================================
# UI helpers
# =============================================================================

def _chip(text: str, kind: str = 'default') -> str:
    cls = 'chip'
    if kind == 'active':
        cls += ' active'
    elif kind == 'muted':
        cls += ' muted'
    return f'<span class="{cls}">{text}</span>'


def _section(title: str, sub: str = ''):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def _empty(text: str):
    st.markdown(f'<div class="empty-state">{text}</div>', unsafe_allow_html=True)


def model_picker(label: str, key_prefix: str):
    """Render a per-provider model dropdown with '직접 입력' fallback.
    Reads/writes st.session_state['model']. `key_prefix` namespaces widget keys."""
    provider = st.session_state.get('provider', PROVIDER_NAMES[0])
    known = PROVIDER_MODELS.get(provider, [])
    current = st.session_state.get('model', '')

    if not known:
        # No known list — just a text input.
        st.session_state['model'] = st.text_input(
            label, current, key=f'{key_prefix}_model_text',
        )
        return

    options = known + [_CUSTOM]
    if current in known:
        idx = known.index(current)
    else:
        idx = len(options) - 1   # 직접 입력
    choice = st.selectbox(
        label, options, index=idx,
        format_func=lambda x: '직접 입력...' if x == _CUSTOM else x,
        key=f'{key_prefix}_model_select',
    )
    if choice == _CUSTOM:
        st.session_state['model'] = st.text_input(
            '모델 ID 직접 입력',
            value=current if current not in known else '',
            key=f'{key_prefix}_model_custom',
            placeholder='예: gpt-4o, my-org/my-finetune',
        )
    else:
        st.session_state['model'] = choice


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

    local_active = bool(st.session_state['docs'])
    web_active = bool(st.session_state['web_enabled'])
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
            elif not is_openai:
                # vLLM / SGLang / HF Router (some providers honor this, others ignore)
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
            import time as _time
            client = get_openai_client()
            _t0 = _time.time()
            if st.session_state['stream']:
                full_text, reasoning_text = stream_chat(client, params)
            else:
                with st.spinner('응답 생성 중...'):
                    full_text, reasoning_text = non_stream_chat(client, params)
            elapsed_sec = _time.time() - _t0
        except Exception as e:
            st.error(f'요청 실패: {e}')
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


# =============================================================================
# Sidebar — brand, new chat, navigation, current status
# =============================================================================

NAV = [
    ('chat',     '대화'),
    ('docs',     '문서'),
    ('agents',   '업무 도구(에이전트)'),
    ('settings', '설정'),
    ('cache',    '캐시'),
    ('about',    '소개'),
]
NAV_KEYS = [k for k, _ in NAV]

with st.sidebar:
    if _LOGO_URI:
        st.markdown(
            f'<div style="text-align:center; padding:4px 0 4px 0;">'
            f'<img src="{_LOGO_URI}" '
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
                if len(label) > 24:
                    label = label[:22] + '...'
                row = st.columns([5, 1])
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
                    if st.button('×', key=f"del_sess_{s['id']}",
                                 use_container_width=True):
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
    if n_docs:
        st.caption(f"문서 {n_docs}개 · 청크 {n_chunks}개")
    else:
        st.caption('문서 없음 (일반 챗)')
    if st.session_state['web_enabled']:
        st.caption(f"웹 검색: {st.session_state['web_provider']}")

    # ----- User / logout -----
    uid = st.session_state.get('user_id', '_local')
    st.markdown('<div class="sb-section">사용자</div>', unsafe_allow_html=True)
    if USERS_FROM_SECRETS:
        st.caption(f"로그인: `{uid}`")
        if st.button('로그아웃', use_container_width=True, key='logout_btn'):
            for k in (
                'user_id', 'user_inputs', 'generated_responses',
                'thinking_traces', 'retrieved_per_turn',
                'query_variants_per_turn', 'current_session_id',
                'current_session_title', 'current_session_created_at',
                'docs', 'doc_embs', 'doc_meta',
                '_loaded_for_embedder', 'chat_doc_filter',
            ):
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
    elif uid.startswith('_anon_'):
        st.caption(f"익명 세션: `{uid}`")
        st.caption('이 브라우저 탭에서만 유효. 새 탭/창은 별도 데이터.')
    else:
        st.caption(f"로컬 단일 사용자: `{uid}`")


# =============================================================================
# View renderers
# =============================================================================

def view_chat():
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
                model_picker('모델 선택', key_prefix='inline')
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
            f'<img src="{_LOGO_URI}" '
            f'style="width:100%; max-width:340px; height:auto; '
            f'margin:0 auto 18px auto; display:block; opacity:0.95;" />'
            if _LOGO_URI else ''
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
        _placeholder = '메시지를 입력하세요'

    user_input = st.chat_input(_placeholder)
    if user_input:
        handle_chat_turn(user_input)


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
        type=['txt', 'md', 'pdf'],
        accept_multiple_files=True,
        label_visibility='collapsed',
    )
    if uploaded:
        with st.spinner(f'{len(uploaded)}개 파일 인덱싱 중...'):
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
    left, right = st.columns(2, gap='large')

    # ----- 모델 -----
    with left:
        _section('모델')

        current_provider = st.session_state.get('provider', PROVIDER_NAMES[0])
        if current_provider not in PROVIDER_NAMES:
            current_provider = PROVIDER_NAMES[0]
        new_provider = st.selectbox(
            '공급자', PROVIDER_NAMES,
            index=PROVIDER_NAMES.index(current_provider),
            help='프리셋을 선택하면 엔드포인트 주소·기본 모델·환경변수에서 API 키를 자동으로 채웁니다.',
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

        model_picker('모델 이름', key_prefix='settings')

        # Per-provider API keys — all four shown so user can pre-fill once
        # and switch providers without losing the others. The "(사용 중)"
        # label marks which one is sent to the LLM for the current provider.
        active_p = st.session_state.get('provider', 'Hugging Face Router')

        def _key_label(name, env_name, owner):
            tag = ' (사용 중)' if owner == active_p else ''
            return f'{name} ({env_name}){tag}'

        st.markdown('**API 키 (공급자별로 따로 저장)**')
        st.session_state['hf_api_key'] = st.text_input(
            _key_label('Hugging Face 토큰', 'HF_TOKEN', 'Hugging Face Router'),
            st.session_state.get('hf_api_key', ''), type='password',
            help='Hugging Face Router (gemma / DeepSeek / Qwen 등) 사용 시 필요.',
        )
        st.session_state['openai_api_key'] = st.text_input(
            _key_label('OpenAI API 키', 'OPENAI_API_KEY', 'OpenAI'),
            st.session_state.get('openai_api_key', ''), type='password',
            help='OpenAI (gpt-4o / gpt-5 / o3 등) 사용 시 필요.',
        )
        st.session_state['dashscope_api_key'] = st.text_input(
            _key_label('DashScope API 키', 'DASHSCOPE_API_KEY', 'DashScope (Qwen)'),
            st.session_state.get('dashscope_api_key', ''), type='password',
            help='Alibaba DashScope (Qwen 공식 API) 사용 시 필요.',
        )
        st.session_state['custom_api_key'] = st.text_input(
            _key_label('Custom / vLLM 키', '', 'vLLM / local')
            if active_p in ('vLLM / local', 'Custom')
            else 'Custom / vLLM 키',
            st.session_state.get('custom_api_key', ''), type='password',
            help='vLLM 등 셀프 호스팅 / 직접 입력하는 OpenAI-호환 엔드포인트용.',
        )
        st.session_state['stream'] = st.checkbox(
            '스트리밍 응답', value=st.session_state['stream'],
            help='응답을 토큰 단위로 실시간 표시.',
        )
        st.session_state['enable_thinking'] = st.checkbox(
            '추론 모드 사용', value=st.session_state['enable_thinking'],
            help='지원 모델(예: Qwen3, Gemma 4)에서 추론 과정을 분리해서 보여줍니다.',
        )

        with st.expander('고급 모델 설정'):
            st.session_state['base_url'] = st.text_input(
                '엔드포인트 주소', st.session_state['base_url'],
                help='OpenAI 호환 endpoint. {base_url}/chat/completions 가 호출됩니다.',
            )
            st.session_state['max_tokens'] = st.number_input(
                '최대 응답 토큰', 16, 131072, int(st.session_state['max_tokens'])
            )
            st.session_state['temperature'] = st.slider(
                '응답 다양성 (temperature)', 0.0, 2.0,
                float(st.session_state['temperature']), 0.05,
            )
            st.session_state['top_p'] = st.slider(
                'top_p', 0.0, 1.0, float(st.session_state['top_p']), 0.01,
                help='누적 확률 임계값. 보통 0.95.',
            )
            st.session_state['sampling_top_k'] = st.number_input(
                '샘플링 top_k', 1, 200,
                int(st.session_state['sampling_top_k']),
            )
            st.session_state['presence_penalty'] = st.slider(
                '반복 방지 강도 (presence_penalty)', 0.0, 2.0,
                float(st.session_state['presence_penalty']), 0.1,
            )

        st.write('')
        _section(
            '멀티모달 (이미지)',
            'PDF 페이지 이미지를 LLM에 함께 전달해 표·차트·도식을 이해하게 합니다. '
            '모델과 엔드포인트가 비전 입력을 지원해야 동작합니다.',
        )
        st.session_state['include_page_images'] = st.checkbox(
            'PDF 페이지 이미지를 답변에 활용',
            value=st.session_state['include_page_images'],
        )
        if st.session_state['include_page_images']:
            st.session_state['max_page_images'] = st.number_input(
                '한 턴에 보낼 이미지 수', 1, 10,
                int(st.session_state['max_page_images']),
            )

    # ----- 검색 + 웹 -----
    with right:
        _section('검색')

        prev_embedder = st.session_state['embedder_model']
        emb_labels = {
            'BAAI/bge-m3': 'BGE-M3 (한국어 강함, 2.2GB)',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': 'MiniLM 다국어 (가벼움, 470MB)',
        }
        emb_idx = (
            EMBEDDER_CHOICES.index(st.session_state['embedder_model'])
            if st.session_state['embedder_model'] in EMBEDDER_CHOICES else 0
        )
        st.session_state['embedder_model'] = st.selectbox(
            '임베딩 모델', EMBEDDER_CHOICES,
            index=emb_idx,
            format_func=lambda x: emb_labels.get(x, x),
            help='문서와 질문을 벡터로 변환하는 모델. BGE-M3가 품질이 더 좋지만 무겁습니다.',
        )
        if st.session_state['embedder_model'] != prev_embedder:
            st.session_state['_loaded_for_embedder'] = None
            load_all_for_current_embedder()
            st.rerun()

        mode_labels = {'hybrid': '하이브리드 (권장)', 'dense': '의미 기반만', 'bm25': '키워드만'}
        st.session_state['retrieval_mode'] = st.radio(
            '검색 방식',
            ['hybrid', 'dense', 'bm25'],
            index=['hybrid', 'dense', 'bm25'].index(
                st.session_state['retrieval_mode']
            ),
            horizontal=True,
            format_func=lambda x: mode_labels[x],
            help='하이브리드 = 의미 검색 + 키워드 검색을 함께 사용합니다.',
        )
        st.session_state['use_reranker'] = st.checkbox(
            '정확도 우선 (재정렬 모델 사용)',
            value=st.session_state['use_reranker'],
            help='추가 모델로 검색 결과를 정밀하게 재정렬합니다. 정확도는 올라가고 응답은 약간 느려집니다.',
        )
        st.session_state['use_contextual_rewrite'] = st.checkbox(
            '이어지는 질문 자동 보완',
            value=st.session_state['use_contextual_rewrite'],
            help='이전 대화를 참고해 "그게 뭐야?" 같은 질문을 self-contained 질문으로 재작성합니다.',
        )
        st.session_state['per_doc_balance'] = st.checkbox(
            '여러 문서 균형 검색',
            value=st.session_state['per_doc_balance'],
            help='상위 결과가 한 문서에 쏠리지 않도록, 각 문서에서 최소 N개 청크를 강제로 포함시킵니다. '
            'cross-document 사실 비교에 유리.',
        )
        st.session_state['comparison_autodetect'] = st.checkbox(
            '비교 질문 자동 감지',
            value=st.session_state['comparison_autodetect'],
            help='질문에 "비교", "차이", "공통", "vs" 등이 들어 있으면 문서별 최소 청크 수를 자동으로 올립니다.',
        )

        with st.expander('고급 검색 설정'):
            r_cols = st.columns(2)
            with r_cols[0]:
                st.session_state['retrieve_top_n'] = st.number_input(
                    '1차 검색 결과 수', 1, 200,
                    int(st.session_state['retrieve_top_n']),
                    help='재정렬 전에 가져올 후보 청크 수.',
                )
            with r_cols[1]:
                st.session_state['final_top_k'] = st.number_input(
                    '최종 사용할 결과 수', 1, 50,
                    int(st.session_state['final_top_k']),
                    help='LLM에 컨텍스트로 전달할 최종 청크 수.',
                )
            st.session_state['use_multi_query'] = st.checkbox(
                '다중 쿼리 (LLM이 질문을 여러 표현으로 변형)',
                value=st.session_state['use_multi_query'],
                help='paraphrase한 질문들로 각각 검색해 결과를 합칩니다. 검색 recall 향상.',
            )
            if st.session_state['use_multi_query']:
                st.session_state['n_paraphrases'] = st.number_input(
                    '변형 개수', 1, 8,
                    int(st.session_state['n_paraphrases']),
                )
            st.session_state['use_hyde'] = st.checkbox(
                'HyDE (가상 답안 검색)',
                value=st.session_state['use_hyde'],
                help='LLM이 만든 가상의 답변 문단을 검색 쿼리로 사용합니다. 모호한 질문에 유리.',
            )
            st.session_state['per_doc_reserve'] = st.number_input(
                '문서당 최소 청크 수 (균형 검색)', 1, 5,
                int(st.session_state['per_doc_reserve']),
                help='"여러 문서 균형 검색"이 켜져 있을 때 적용. 비교 질문은 자동으로 2 이상으로 상향됩니다.',
            )

        st.write('')
        _section(
            '웹 검색',
            '질문 시 웹 검색 결과를 함께 컨텍스트에 포함합니다. DuckDuckGo는 API 키가 필요 없습니다.',
        )
        st.session_state['web_enabled'] = st.checkbox(
            '실시간 웹 검색 사용',
            value=st.session_state['web_enabled'],
        )
        if st.session_state['web_enabled']:
            wp_labels = {'duckduckgo': 'DuckDuckGo (키 불필요)',
                         'tavily': 'Tavily (LLM 최적화, 키 필요)',
                         'brave': 'Brave (키 필요)'}
            st.session_state['web_provider'] = st.selectbox(
                '검색 제공자', ['duckduckgo', 'tavily', 'brave'],
                index=['duckduckgo', 'tavily', 'brave'].index(
                    st.session_state['web_provider']
                ),
                format_func=lambda x: wp_labels[x],
            )
            st.session_state['web_top_n'] = st.number_input(
                '가져올 결과 수', 1, 20, int(st.session_state['web_top_n'])
            )
            if st.session_state['web_provider'] == 'tavily':
                st.session_state['tavily_key'] = st.text_input(
                    'Tavily API 키', st.session_state['tavily_key'], type='password',
                )
            elif st.session_state['web_provider'] == 'brave':
                st.session_state['brave_key'] = st.text_input(
                    'Brave API 키', st.session_state['brave_key'], type='password',
                )


# =============================================================================
# Cache view
# =============================================================================

def view_cache():
    _section(
        'Hugging Face Hub cache',
        '로컬에 다운로드된 모델 가중치 목록입니다. 사용하지 않는 모델은 삭제해 디스크를 회수할 수 있습니다.',
    )
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        cached = [(r.repo_id, r.size_on_disk_str, str(r.repo_path)) for r in info.repos]
    except Exception:
        cached = []
    if not cached:
        _empty('캐시된 모델이 없습니다.')
    else:
        for repo_id, size, path in cached:
            with st.container(border=True):
                cols = st.columns([4, 1, 1])
                cols[0].markdown(f"`{repo_id}`")
                cols[1].caption(size)
                if cols[2].button('삭제', key=f'cache_{repo_id}', use_container_width=True):
                    try:
                        shutil.rmtree(path)
                        st.success(f'{repo_id} 삭제 완료')
                        st.rerun()
                    except Exception as e:
                        st.error(f'캐시 삭제 실패: {e}')

    st.write('')
    user_logs = _user_logs_dir()
    _section(
        '대화 로그',
        f'경로: `{user_logs}` — 사용자별로 분리. 각 대화가 `{{session_id}}.jsonl` '
        '파일로 저장됩니다 (한 줄당 한 턴, 분석·DB 적재 친화적).',
    )

    # Agent runs log (separate file)
    agent_log = _agent_log_path()
    if agent_log.exists():
        try:
            n_lines = sum(1 for _ in agent_log.open('r', encoding='utf-8'))
        except Exception:
            n_lines = '?'
        size_kb = agent_log.stat().st_size / 1024
        with st.container(border=True):
            cols = st.columns([5, 1, 1])
            cols[0].markdown(f"**`agents.jsonl`** · 에이전트 실행 기록")
            cols[1].caption(f'{n_lines} runs · {size_kb:.1f} KB')
            with cols[2]:
                try:
                    st.download_button(
                        '다운로드', data=agent_log.read_bytes(),
                        file_name='agents.jsonl', mime='application/x-jsonlines',
                        key='dl_agents_jsonl', use_container_width=True,
                    )
                except Exception:
                    pass

    jsonl_files = sorted(
        [p for p in user_logs.glob('*.jsonl') if p.name != 'agents.jsonl'],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not jsonl_files:
        _empty('대화 로그는 아직 없습니다.')
    else:
        for p in jsonl_files[:30]:
            try:
                n_lines = sum(1 for _ in p.open('r', encoding='utf-8'))
            except Exception:
                n_lines = '?'
            size_kb = p.stat().st_size / 1024
            with st.container(border=True):
                cols = st.columns([5, 1, 1])
                cols[0].markdown(f"`{p.name}`")
                cols[1].caption(f'{n_lines} turns · {size_kb:.1f} KB')
                with cols[2]:
                    try:
                        st.download_button(
                            '다운로드', data=p.read_bytes(),
                            file_name=p.name, mime='application/x-jsonlines',
                            key=f'dl_jsonl_{p.name}', use_container_width=True,
                        )
                    except Exception:
                        pass

    st.write('')
    user_dd = _user_data_dir()
    _section(
        '로컬 벡터 스토어',
        f'경로: `{user_dd}` — 사용자별로 분리. 임베더 모델마다 또 하위 폴더.',
    )
    if user_dd.exists():
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
            _empty('저장된 벡터 인덱스가 없습니다.')
        else:
            for name, n, size in rows:
                with st.container(border=True):
                    cols = st.columns([4, 1, 1])
                    cols[0].markdown(f"`{name}`")
                    cols[1].caption(f'{n} docs')
                    cols[2].caption(f'{size / 1024 / 1024:.1f} MB')


# =============================================================================
# Agentic workflows
# =============================================================================

# NOTE: legacy global. Per-user log path is _agent_log_path(); kept here only
# to avoid accidental import-time errors. Use _agent_log_path() everywhere.


def _agent_format_context(retrieved: list) -> str:
    """Format retrieved chunks as numbered context lines for the LLM prompt."""
    if not retrieved:
        return '(검색된 근거 자료 없음)'
    lines = []
    for i, r in enumerate(retrieved, start=1):
        if r.get('source') == 'web':
            lines.append(f"[{i}] (웹: {r.get('doc', '')} {r.get('url', '')}) {r.get('text', '')}")
        else:
            pages = r.get('pages') or []
            page_str = ''
            if pages:
                page_str = (f' p.{pages[0]}' if len(pages) == 1
                            else f' pp.{pages[0]}-{pages[-1]}')
            lines.append(
                f"[{i}] (로컬: {r.get('doc', '')}{page_str}) {r.get('text', '')}"
            )
    return '\n\n'.join(lines)


def _agent_run_llm(messages: list, model: str):
    """Common runner: builds params, streams or non-streams, post-processes."""
    extra_body = {}
    if _provider_supports_top_k():
        extra_body['top_k'] = int(st.session_state['sampling_top_k'])
    if not st.session_state['enable_thinking']:
        if _is_dashscope_endpoint():
            extra_body['enable_thinking'] = False
        elif not _is_openai_endpoint():
            extra_body['chat_template_kwargs'] = {'enable_thinking': False}

    params = _build_completion_params(
        model=model, messages=messages,
        max_tokens=st.session_state['max_tokens'],
        temperature=st.session_state['temperature'],
        top_p=st.session_state['top_p'],
        presence_penalty=st.session_state['presence_penalty'],
        extra_body=extra_body,
    )
    import time as _time
    client = get_openai_client()
    t0 = _time.time()
    if st.session_state['stream']:
        full_text, reasoning = stream_chat(client, params)
    else:
        with st.spinner('생성 중...'):
            full_text, reasoning = non_stream_chat(client, params)
    elapsed = _time.time() - t0

    if full_text and not reasoning:
        rt, ct = split_thinking(full_text)
        if rt:
            reasoning, full_text = rt, ct
    return full_text, reasoning, elapsed


def _agent_log(task_key: str, inputs: dict, output: str, retrieved: list,
               model: str, elapsed: float):
    """Append one agent run to ./logs/agents.jsonl. Same schema family as chat
    JSONL — analysis-friendly."""
    retrieved_records = []
    for j, r in enumerate(retrieved or [], start=1):
        rec = {'rank': j, 'source': r.get('source'),
               'score': r.get('score'), 'text': (r.get('text') or '')[:1200]}
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


# ---- Task templates ----

def _email_build(inputs, retrieved):
    sys_msg = (
        '당신은 한국어 비즈니스 이메일 작성 도우미입니다. 사용자가 제공한 Context '
        '(사용자 문서에서 발췌)를 사실 근거로 사용해, 요청된 톤·길이·핵심 포인트에 '
        '맞춰 이메일 초안을 작성합니다. 형식은 "제목: ..." 한 줄 + 빈 줄 + 본문. '
        'Context에 사실이 부족하면 추측하지 말고 자연스럽게 일반론으로 처리하세요.'
    )
    user_msg = (
        f"작업: 이메일 초안 작성\n"
        f"수신자: {inputs.get('recipient') or '미지정'}\n"
        f"주제 힌트: {inputs.get('subject_hint') or '미지정'}\n"
        f"톤: {inputs.get('tone') or '공식'}\n"
        f"분량: {inputs.get('length') or '보통'}\n\n"
        f"꼭 포함할 핵심 요점:\n{inputs.get('key_points') or '(자유)'}\n\n"
        f"Context (참고 자료):\n{_agent_format_context(retrieved)}\n\n"
        f"이제 이메일 초안을 작성해 주세요."
    )
    return [{'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': user_msg}]


def _report_build(inputs, retrieved):
    sys_msg = (
        '당신은 보고서 작성 전문가입니다. 제공된 Context를 사실 근거로 사용해 마크다운 '
        '구조의 보고서를 작성합니다. 모든 사실 주장에는 출처 번호를 [1], [2] 형식으로 '
        '인용하세요. 섹션 제목은 ##, 소제목은 ###을 사용하세요. Context에 없는 사실은 '
        '추측하지 말고 "추가 자료 필요"로 표시하세요.'
    )
    user_msg = (
        f"작업: 보고서 작성\n"
        f"주제: {inputs.get('topic') or '미지정'}\n"
        f"대상 독자: {inputs.get('audience') or '일반'}\n"
        f"분량: {inputs.get('length') or '중간 (3~5 섹션)'}\n\n"
        f"포함해야 할 섹션·관점:\n{inputs.get('sections') or '(자동 구성)'}\n\n"
        f"Context (참고 자료):\n{_agent_format_context(retrieved)}\n\n"
        f"위 자료를 근거로 보고서를 작성해 주세요. 마크다운 구조."
    )
    return [{'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': user_msg}]


def _summary_build(inputs, retrieved):
    sys_msg = (
        '당신은 한국어 문서 요약 전문가입니다. 제공된 Context를 종합해 요청된 깊이·초점에 '
        '맞게 요약합니다. 출처는 [1], [2]로 인용하고, 핵심 포인트를 글머리 기호로 정리한 후 '
        '단락 형식의 본문 요약을 덧붙이세요.'
    )
    user_msg = (
        f"작업: 문서 요약\n"
        f"초점: {inputs.get('focus') or '전반'}\n"
        f"깊이: {inputs.get('depth') or '단락 요약'}\n\n"
        f"Context:\n{_agent_format_context(retrieved)}\n\n"
        f"요약해 주세요."
    )
    return [{'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': user_msg}]


def _analysis_build(inputs, retrieved):
    sys_msg = (
        '당신은 데이터/통찰 분석 도우미입니다. 제공된 Context에서 수치·표·핵심 사실을 '
        '추출하고, 사용자의 질문에 대해 마크다운 표 + 해석 + 통찰 순으로 답변합니다. '
        '모든 수치 인용에는 [1], [2] 형식의 출처 번호를 붙이세요. Context에 없는 추정은 '
        '"가정"으로 명시하세요.'
    )
    user_msg = (
        f"작업: 데이터/통찰 분석\n"
        f"분석 질문: {inputs.get('question') or '(미지정)'}\n"
        f"추가 맥락: {inputs.get('context_note') or '(없음)'}\n\n"
        f"Context:\n{_agent_format_context(retrieved)}\n\n"
        f"분석을 진행해 주세요. 구조: (1) 핵심 수치 표, (2) 해석, (3) 통찰/시사점."
    )
    return [{'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': user_msg}]


def _comparison_build(inputs, retrieved):
    sys_msg = (
        '당신은 비교 분석 도우미입니다. 두 대상에 대해 마크다운 비교 표를 만들고, '
        '각 항목 차이의 의미를 해석합니다. 모든 사실에 [N]으로 출처를 표기하세요. '
        'Context에 한쪽 정보가 부족하면 "정보 부족"으로 표시하세요.'
    )
    user_msg = (
        f"작업: 비교 분석\n"
        f"대상 A: {inputs.get('item_a') or '(미지정)'}\n"
        f"대상 B: {inputs.get('item_b') or '(미지정)'}\n"
        f"비교 기준 (필요시): {inputs.get('criteria') or '(자동 선택)'}\n\n"
        f"Context:\n{_agent_format_context(retrieved)}\n\n"
        f"비교 분석을 진행해 주세요. 구조: (1) 비교 표, (2) 주요 차이 해석, (3) 결론."
    )
    return [{'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': user_msg}]


AGENT_TASKS = {
    'email': {
        'label': '이메일 초안',
        'description': '문서 내용을 참고해 비즈니스 이메일 초안을 작성합니다.',
        'fields': [
            {'key': 'recipient', 'label': '수신자', 'type': 'text',
             'placeholder': '예: 박팀장님 / 거래처 ABC사 김부장님'},
            {'key': 'subject_hint', 'label': '주제 / 목적', 'type': 'text',
             'placeholder': '예: Q1 실적 공유 및 다음 분기 전략 미팅 일정 제안'},
            {'key': 'tone', 'label': '톤', 'type': 'select',
             'options': ['공식', '친근', '간결']},
            {'key': 'length', 'label': '분량', 'type': 'select',
             'options': ['짧게', '보통', '상세']},
            {'key': 'key_points', 'label': '꼭 포함할 핵심 요점', 'type': 'textarea',
             'placeholder': '한 줄에 한 가지씩 적어주세요.'},
        ],
        'requires_docs': False,
        'search_query': lambda i: f"{i.get('subject_hint', '')} {i.get('key_points', '')}".strip(),
        'build_messages': _email_build,
        'output_ext': 'md',
    },
    'report': {
        'label': '보고서 작성',
        'description': '문서를 근거로 구조화된 마크다운 보고서를 생성합니다.',
        'fields': [
            {'key': 'topic', 'label': '보고서 주제', 'type': 'text',
             'placeholder': '예: 2026 Q1 매출 분석 및 전략 제안'},
            {'key': 'audience', 'label': '대상 독자', 'type': 'text',
             'placeholder': '예: 임원진 / 실무팀 / 외부 파트너'},
            {'key': 'length', 'label': '분량', 'type': 'select',
             'options': ['짧게 (1~2 섹션)', '중간 (3~5 섹션)', '상세 (6+ 섹션)']},
            {'key': 'sections', 'label': '포함해야 할 섹션·관점', 'type': 'textarea',
             'placeholder': '비워두면 자동 구성됩니다.'},
        ],
        'requires_docs': True,
        'search_query': lambda i: f"{i.get('topic', '')} {i.get('sections', '')}".strip(),
        'build_messages': _report_build,
        'output_ext': 'md',
    },
    'summary': {
        'label': '문서 요약',
        'description': '업로드된 문서를 종합 요약합니다.',
        'fields': [
            {'key': 'focus', 'label': '요약 초점', 'type': 'text',
             'placeholder': '예: 매출과 마진 중심 / 리스크 위주 (비워두면 전반)'},
            {'key': 'depth', 'label': '깊이', 'type': 'select',
             'options': ['핵심만 (3줄)', '단락 요약', '상세 요약']},
        ],
        'requires_docs': True,
        'search_query': lambda i: i.get('focus') or '핵심 주제, 주요 논점, 결론',
        'build_messages': _summary_build,
        'output_ext': 'md',
    },
    'analysis': {
        'label': '데이터 분석',
        'description': '문서에서 수치를 뽑아 표 + 해석 + 통찰을 정리합니다.',
        'fields': [
            {'key': 'question', 'label': '분석 질문', 'type': 'text',
             'placeholder': '예: 분기별 매출 성장률과 마진 추이는?'},
            {'key': 'context_note', 'label': '추가 맥락 (선택)', 'type': 'textarea',
             'placeholder': '분석 시 고려할 추가 정보가 있으면 적어주세요.'},
        ],
        'requires_docs': True,
        'search_query': lambda i: f"{i.get('question', '')} 수치 통계 결과".strip(),
        'build_messages': _analysis_build,
        'output_ext': 'md',
    },
    'comparison': {
        'label': '비교 분석',
        'description': '두 대상을 문서 근거로 비교합니다 (여러 문서 균형 검색 자동 적용).',
        'fields': [
            {'key': 'item_a', 'label': '대상 A', 'type': 'text',
             'placeholder': '예: 자사 제품 A / 2025년 실적'},
            {'key': 'item_b', 'label': '대상 B', 'type': 'text',
             'placeholder': '예: 경쟁사 제품 B / 2026년 실적'},
            {'key': 'criteria', 'label': '비교 기준 (선택)', 'type': 'textarea',
             'placeholder': '비워두면 자동 선택.'},
        ],
        'requires_docs': True,
        'search_query': lambda i: f"{i.get('item_a', '')} {i.get('item_b', '')} 비교 차이",
        'build_messages': _comparison_build,
        'output_ext': 'md',
    },
}


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
            need_filter = (
                doc_ids_filter is not None
                and 0 < len(doc_ids_filter) < len(all_docs)
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
        # Per-task document subset selector (only when ≥2 docs are loaded
        # and the task actually uses documents).
        if task.get('requires_docs') and len(st.session_state['docs']) >= 2:
            doc_id_to_name = {d['id']: d['name'] for d in st.session_state['docs']}
            all_ids = list(doc_id_to_name.keys())
            selected_doc_ids = st.multiselect(
                '사용할 문서',
                options=all_ids,
                default=all_ids,
                format_func=lambda did: doc_id_to_name[did],
                help='기본은 전체 문서. 일부만 사용하려면 체크를 조정하세요. '
                '비교 분석을 두 문서로 좁히고 싶을 때 특히 유용합니다.',
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

        full_text, reasoning, retrieved, elapsed = run_agent_task(
            selected, inputs, doc_ids_filter=selected_doc_ids,
        )

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
    if _LOGO_URI:
        st.markdown(
            f'<div style="text-align:center; margin-bottom:18px;">'
            f'<img src="{_LOGO_URI}" '
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
        - 대화 세션 메타: `./.data/sessions/{id}.json` (사이드바 대화 목록의 원천).
        - **에이전트 실행 로그**: `./logs/agents.jsonl` — 에이전트 워크플로(이메일/보고서/요약/분석/비교)의 실행 기록. task, inputs, output, retrieved, model, elapsed_seconds 포함.
        - **대화 로그**: `./logs/{session_id}.jsonl` — 세션별로 한 파일, 한 줄당 한 턴. 분석·DB 친화적 구조.
          필드: session_id, turn_index, timestamp, user_message, assistant_message, reasoning, model/provider, elapsed_seconds, retrieved (rank·source·score·page·url), citation_numbers_used, query_variants, settings_snapshot (rerank/HyDE/multi-query/per-doc 등 모든 설정 스냅샷).
          pandas: `pd.concat([pd.read_json(f, lines=True) for f in glob.glob('logs/*.jsonl')])`.
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
