"""Provider/model detection + per-provider chat-completion kwargs builder.

Each Chat-Completions-compatible provider has its own list of "this kwarg
will 400 you" surprises. Rather than scatter ``if 'anthropic' in base_url``
checks across the chat code, this module owns:

- Endpoint sniffers (``_is_*_endpoint``) keyed off the configured base_url.
- Model-family sniffers (``_is_openai_reasoning_model``, ``_is_openai_gpt5_family``).
- A deprecation table that auto-swaps model ids we used to expose but
  pulled because the provider no longer serves them.
- ``_build_completion_params`` — the only function chat code should call
  to assemble ``client.chat.completions.create(**params)``.
- ``_thinking_off_extra_body`` — provider-aware way to disable internal
  "thinking" output for models that support it.
"""

import streamlit as st


def _is_openai_endpoint() -> bool:
    """Detect OpenAI's official API (rejects non-standard params like top_k)."""
    return 'api.openai.com' in (st.session_state.get('base_url') or '').lower()


def _is_dashscope_endpoint() -> bool:
    return 'dashscope' in (st.session_state.get('base_url') or '').lower()


def _is_hf_router_endpoint() -> bool:
    return 'router.huggingface.co' in (st.session_state.get('base_url') or '').lower()


def _is_anthropic_endpoint() -> bool:
    """Claude via Anthropic's OpenAI-SDK compatibility layer. Has its own
    parameter restrictions (no top_p, no penalty fields, no top_k)."""
    return 'api.anthropic.com' in (st.session_state.get('base_url') or '').lower()


def _is_fireworks_endpoint() -> bool:
    return 'fireworks.ai' in (st.session_state.get('base_url') or '').lower()


def _provider_supports_top_k() -> bool:
    """top_k via extra_body is provider-specific. Live-tested:
      - OpenAI: rejects.
      - Anthropic OpenAI-compat: rejects.
      - HF Router: many sub-providers reject (e.g. Cerebras for Llama-3.1-8B,
        gpt-oss-120b). Safest to omit unless DashScope/vLLM/Custom.
      - Fireworks: rejects via extra_body.
      - DashScope (Qwen): supports.
      - vLLM/local & Custom: typically supports."""
    if _is_openai_endpoint():
        return False
    if _is_anthropic_endpoint():
        return False
    if _is_hf_router_endpoint():
        return False
    if _is_fireworks_endpoint():
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


# Models we used to expose but pulled because they can't be called through
# their configured endpoint. The runtime guard below swaps them on the fly
# before the API request goes out — protects users whose saved prefs / chat
# state still reference them after we removed them from the dropdown.
_DEPRECATED_MODEL_SWAPS = {
    # Korean-native models: HF Inference Providers don't deploy them.
    'LGAI-EXAONE/EXAONE-4.5-33B': ('Qwen/Qwen3-Next-80B-A3B-Instruct', 'Hugging Face Router'),
    'naver-hyperclovax/HyperCLOVAX-SEED-Think-32B': ('Qwen/Qwen3-Next-80B-A3B-Instruct', 'Hugging Face Router'),
    # OpenAI Responses-API-only models.
    'gpt-5-pro':     ('gpt-5-mini', 'OpenAI'),
    'gpt-5.5-pro':   ('gpt-5-mini', 'OpenAI'),
    'gpt-5.4-pro':   ('gpt-5-mini', 'OpenAI'),
    'o1-pro':        ('o4-mini',     'OpenAI'),
}


def _resolve_deprecated_model(model: str) -> str:
    """Swap a deprecated model id for its replacement at API-call time.
    Also updates st.session_state so the picker UI reflects the swap. Shows
    a one-time-per-session notice so the user knows it happened."""
    if model not in _DEPRECATED_MODEL_SWAPS:
        return model
    replacement, _expected_provider = _DEPRECATED_MODEL_SWAPS[model]
    notified = st.session_state.setdefault('_deprecated_model_notified', set())
    if model not in notified:
        st.info(
            f'`{model}` 은 이 endpoint 에서 서빙되지 않아 '
            f'`{replacement}` 으로 자동 전환했습니다. '
            f'설정 탭에서 다른 모델로 변경할 수도 있습니다.'
        )
        notified.add(model)
    st.session_state['model'] = replacement
    return replacement


def _build_completion_params(
    model: str, messages: list,
    *,
    max_tokens=None, temperature=None, top_p=None, presence_penalty=None,
    extra_body: dict = None, stream: bool = False,
) -> dict:
    """Build chat.completions.create kwargs that respect per-model constraints.
    Drops sampling params for GPT-5/o-series; uses max_completion_tokens for those."""
    model = _resolve_deprecated_model(model)
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

    # Sampling param compatibility per provider.
    # - OpenAI gpt-5/o-series: rejects ANY non-default sampling param.
    # - Anthropic (Claude 4.x via OpenAI compat): rejects temperature,
    #   top_p, and presence_penalty on most current models. Claude moved
    #   to model-managed sampling and returns
    #     'temperature is deprecated for this model.'
    #   when any of these are sent. Safest to omit all three; the
    #   default sampling produces good output.
    is_anthropic = _is_anthropic_endpoint()
    openai_strict = is_openai and (
        _is_openai_reasoning_model(model) or _is_openai_gpt5_family(model)
    )
    if not openai_strict and not is_anthropic:
        if temperature is not None:
            out['temperature'] = float(temperature)
        if top_p is not None:
            out['top_p'] = float(top_p)
        if presence_penalty is not None:
            out['presence_penalty'] = float(presence_penalty)

    # extra_body — strip provider-incompatible fields as a safety net.
    if extra_body:
        if is_openai or is_anthropic:
            extra_body = {k: v for k, v in extra_body.items()
                          if k not in ('top_k', 'chat_template_kwargs', 'enable_thinking')}
        if extra_body:
            out['extra_body'] = extra_body
    return out


def _thinking_off_extra_body() -> dict:
    """Return extra_body that disables thinking, branched by provider.

    - OpenAI: no thinking concept; rejects unknown fields → {}.
    - HF Router: providers vary; Cerebras-served models (Llama-3.1-8B,
      gpt-oss, etc.) reject chat_template_kwargs with HTTP 400. Safer to
      omit; thinking-capable models on the router will still emit <think>
      tags which we parse out in split_thinking().
    - DashScope (Qwen): uses enable_thinking directly in extra_body.
    - vLLM / Custom: chat_template_kwargs is the standard way (SGLang/vLLM).
    """
    if _is_openai_endpoint():
        return {}
    if _is_anthropic_endpoint():
        # Claude exposes extended thinking via a 'thinking' parameter at the
        # native API level, but the OpenAI-compat layer ignores it. Return
        # nothing extra to avoid 400 from unknown fields.
        return {}
    if _is_fireworks_endpoint():
        return {}
    if _is_dashscope_endpoint():
        return {'enable_thinking': False}
    if _is_hf_router_endpoint():
        return {}
    return {'chat_template_kwargs': {'enable_thinking': False}}
