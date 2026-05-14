"""Session state initialization.

Streamlit re-runs the script on every interaction, but ``st.session_state``
persists across reruns within the same browser tab. _init_state seeds every
key the rest of the app reads from, using ``setdefault`` so a user's existing
selections (provider, API keys, RAG knobs, current session) survive the rerun.

Defaults that depend on the environment are computed here:

- API keys default to the matching environment variable (HF_TOKEN, OPENAI_API_KEY, ...).
- On Streamlit Cloud (detected via ``/mount/src``) we fall back to the lighter
  MiniLM embedder and disable the reranker — the free tier has ~1 GB of RAM,
  below BGE-M3 (~2.2 GB) + reranker (~580 MB) combined.
"""

import os
from pathlib import Path

import streamlit as st

from .config import EMBEDDER_CHOICES


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
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', ''),
        'fireworks_api_key': os.getenv('FIREWORKS_API_KEY', ''),
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

        # rag config — on Streamlit Cloud free tier RAM is ~1 GB, which is
        # below BGE-M3 (~2.2 GB) + reranker (~580 MB). Default to the lighter
        # MiniLM + reranker off there so the app doesn't OOM on first PDF.
        'embedder_model': (
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            if Path('/mount/src').exists() else EMBEDDER_CHOICES[0]
        ),
        'chunk_size': 600,
        'chunk_overlap': 80,
        'retrieval_mode': 'hybrid',   # 'dense' | 'bm25' | 'hybrid'
        'retrieve_top_n': 20,
        'final_top_k': 5,
        'use_reranker': not Path('/mount/src').exists(),
        # Route dense retrieval through Supabase pgvector when both this flag
        # is ON and a Supabase client is configured. Default OFF — user
        # opts in after running db_schema_pgvector.sql and seeing the cache
        # tab confirm that chunks are being upserted.
        'use_pgvector_search': False,
        # Agentic loop: after the initial retrieval, the LLM may issue
        # follow-up searches via a tool call. Default OFF; toggle in
        # settings tab. Only active when the model/provider supports
        # OpenAI-style tool calling (gpt-5, gpt-4.1, Qwen3, Llama-3.x, ...).
        'use_agentic_search': False,
        'agentic_max_iters': 3,
        # When True, skip retrieval entirely and let the LLM answer from its
        # own knowledge — useful when the user wants a quick general chat
        # without uploading documents or when documents are loaded but
        # the question is off-topic.
        'general_chat_mode': False,

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
