"""Cached OpenAI-compatible client + embedder + reranker loaders.

The OpenAI client cache is keyed by (base_url, api_key) so flipping
provider in the sidebar creates a fresh client while keeping the
previous one alive — switching back is then a zero-cost lookup.

``load_embedder`` and ``load_reranker`` use ``@st.cache_resource`` so the
underlying transformer weights are loaded exactly once per process,
regardless of how many turns / views call them.
"""

import streamlit as st
from openai import OpenAI


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
    if p == 'Anthropic (Claude)':
        return st.session_state.get('anthropic_api_key', '') or ''
    if p == 'Fireworks AI':
        return st.session_state.get('fireworks_api_key', '') or ''
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
