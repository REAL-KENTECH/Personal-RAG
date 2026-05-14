"""Static configuration: provider catalog, model lists, embedder choices, CSS.

Everything in this module is a pure constant — no Streamlit calls, no I/O,
no environment lookups. Anything with a side effect (page config, directory
creation, dotenv loading) stays in app.py so import order remains explicit.
"""

from pathlib import Path


# -----------------------------------------------------------------------------
# Filesystem paths derived from the repository root
# -----------------------------------------------------------------------------
# personal_rag/config.py lives one level under the repo root, so .parent.parent
# resolves to the project directory. These are Path objects only — the actual
# mkdir() happens once at app.py boot so import-time side effects stay
# concentrated in the entry point.

_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / '.data'
LOGS_DIR = _ROOT / 'logs'
LOGO_PATH = _ROOT / 'logo' / 'real_logo.png'           # full lockup
FAVICON_PATH = _ROOT / 'logo' / 'real_1_엠블럼.png'    # square emblem for browser tab


# -----------------------------------------------------------------------------
# Embedding / reranker model IDs
# -----------------------------------------------------------------------------

EMBEDDER_CHOICES = [
    'BAAI/bge-m3',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
]

RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'


# -----------------------------------------------------------------------------
# Provider presets for OpenAI-compatible endpoints
# -----------------------------------------------------------------------------

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
    'Anthropic (Claude)': {
        # Anthropic's OpenAI-SDK compatibility endpoint. Most chat-completions
        # features work; tool calling and some sampling params have caveats.
        'base_url': 'https://api.anthropic.com/v1/',
        'env_key': 'ANTHROPIC_API_KEY',
        'default_model': 'claude-sonnet-4-6',
    },
    'Fireworks AI': {
        # OpenAI-compatible. Model ids follow accounts/fireworks/models/<name>.
        'base_url': 'https://api.fireworks.ai/inference/v1',
        'env_key': 'FIREWORKS_API_KEY',
        'default_model': 'accounts/fireworks/models/llama-v3p3-70b-instruct',
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
        # NOTE: Korean-native models from LG (EXAONE) and Naver (HyperCLOVAX)
        # exist on HF Hub but Inference Providers (Together/Cerebras/Hyperbolic)
        # don't typically deploy them. Calling them through the Router returns
        # "model_not_supported". Self-host via vLLM or Ollama instead — see the
        # 'vLLM / local' provider preset below for suggested model IDs.
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
        # GPT-5 tier — Chat Completions compatible only.
        # NOTE: *-pro variants (gpt-5-pro, gpt-5.5-pro, gpt-5.4-pro, o1-pro)
        # are OpenAI's Responses-API-only models and would 404 here, so they
        # are intentionally omitted. Use gpt-5 or o3 for top-tier reasoning.
        'gpt-5.5',
        'gpt-5.4',
        'gpt-5.4-mini',
        'gpt-5.4-nano',
        'gpt-5.3-chat-latest',
        'gpt-5.2',
        'gpt-5.1',
        'gpt-5',
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
    ],
    'Anthropic (Claude)': [
        # Latest Claude 4.x tier
        'claude-opus-4-7',
        'claude-sonnet-4-6',
        'claude-haiku-4-5',
        # 4.x earlier
        'claude-opus-4-5',
        'claude-sonnet-4-5',
        'claude-haiku-4-1',
        # 3.7 / 3.5 (legacy but stable)
        'claude-3-7-sonnet-latest',
        'claude-3-5-sonnet-latest',
        'claude-3-5-haiku-latest',
    ],
    'Fireworks AI': [
        # NOTE: model IDs change occasionally on Fireworks; some require
        # account tier upgrades (e.g. 405B). If you get 404 NOT_FOUND check
        # the current list at https://fireworks.ai/models.
        # Llama
        'accounts/fireworks/models/llama-v3p3-70b-instruct',
        'accounts/fireworks/models/llama-v3p1-70b-instruct',
        'accounts/fireworks/models/llama-v3p1-8b-instruct',
        # Qwen
        'accounts/fireworks/models/qwen2p5-72b-instruct',
        'accounts/fireworks/models/qwen2p5-coder-32b-instruct',
        # DeepSeek
        'accounts/fireworks/models/deepseek-v3',
        'accounts/fireworks/models/deepseek-r1',
        # Mixtral
        'accounts/fireworks/models/mixtral-8x22b-instruct',
    ],
    'DashScope (Qwen)': [
        'qwen3.6-27b',
        'qwen3-72b',
        'qwen-max',
        'qwen-plus',
        'qwen-turbo',
        'qwen-vl-max',
    ],
    'vLLM / local': [
        # Self-hosted suggestions (you bring the GPU + run `vllm serve <id>`).
        # Strong open multilingual that handle Korean well:
        'Qwen/Qwen3-Next-80B-A3B-Instruct',
        'Qwen/Qwen2.5-7B-Instruct',
        'meta-llama/Llama-3.3-70B-Instruct',
        'meta-llama/Llama-3.1-8B-Instruct',
    ],
    'Custom': [],
}


# -----------------------------------------------------------------------------
# Global CSS injected once at app boot
# -----------------------------------------------------------------------------

APP_CSS = """
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
"""
