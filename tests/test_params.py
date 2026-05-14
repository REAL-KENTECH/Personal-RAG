"""Model-family detectors are pure (take a model string only).

The endpoint sniffers (_is_openai_endpoint, etc.) read st.session_state
so they're tested through the running app, not here.
"""

from llm.params import (
    _DEPRECATED_MODEL_SWAPS,
    _is_openai_gpt5_family,
    _is_openai_reasoning_model,
    _uses_max_completion_tokens,
)


# -----------------------------------------------------------------------------
# _is_openai_reasoning_model — o-series detection
# -----------------------------------------------------------------------------

def test_reasoning_o1():
    assert _is_openai_reasoning_model('o1')
    assert _is_openai_reasoning_model('o1-pro')


def test_reasoning_o3():
    assert _is_openai_reasoning_model('o3')
    assert _is_openai_reasoning_model('o3-mini')


def test_reasoning_o4():
    assert _is_openai_reasoning_model('o4-mini')


def test_reasoning_case_insensitive():
    assert _is_openai_reasoning_model('O1')
    assert _is_openai_reasoning_model('O3-mini')


def test_reasoning_gpt5_not_reasoning():
    # gpt-5 is a chat model, not an o-series reasoning model
    assert not _is_openai_reasoning_model('gpt-5')


def test_reasoning_other_models():
    assert not _is_openai_reasoning_model('gpt-4o')
    assert not _is_openai_reasoning_model('claude-opus-4-7')
    assert not _is_openai_reasoning_model('Qwen/Qwen3-Next-80B-A3B-Instruct')


def test_reasoning_empty():
    assert not _is_openai_reasoning_model('')
    assert not _is_openai_reasoning_model(None)


# -----------------------------------------------------------------------------
# _is_openai_gpt5_family
# -----------------------------------------------------------------------------

def test_gpt5_family_base():
    assert _is_openai_gpt5_family('gpt-5')
    assert _is_openai_gpt5_family('gpt-5-mini')
    assert _is_openai_gpt5_family('gpt-5-nano')


def test_gpt5_family_dotted():
    assert _is_openai_gpt5_family('gpt-5.4')
    assert _is_openai_gpt5_family('gpt-5.5')
    assert _is_openai_gpt5_family('gpt-5.4-mini')


def test_gpt5_family_excludes_gpt4():
    assert not _is_openai_gpt5_family('gpt-4o')
    assert not _is_openai_gpt5_family('gpt-4.1')


def test_gpt5_family_excludes_o_series():
    assert not _is_openai_gpt5_family('o1')
    assert not _is_openai_gpt5_family('o3-mini')


# -----------------------------------------------------------------------------
# _uses_max_completion_tokens — union of the above
# -----------------------------------------------------------------------------

def test_max_completion_tokens_for_o_series():
    assert _uses_max_completion_tokens('o3')
    assert _uses_max_completion_tokens('o4-mini')


def test_max_completion_tokens_for_gpt5():
    assert _uses_max_completion_tokens('gpt-5')
    assert _uses_max_completion_tokens('gpt-5-mini')


def test_max_completion_tokens_legacy_models_false():
    # Older models that still use max_tokens
    assert not _uses_max_completion_tokens('gpt-4o')
    assert not _uses_max_completion_tokens('gpt-4.1')
    assert not _uses_max_completion_tokens('claude-sonnet-4-6')


# -----------------------------------------------------------------------------
# Deprecation swap table — sanity check
# -----------------------------------------------------------------------------

def test_deprecation_table_has_known_entries():
    # Models that exist on OpenAI's Responses API only — must redirect
    assert 'gpt-5-pro' in _DEPRECATED_MODEL_SWAPS
    assert 'o1-pro' in _DEPRECATED_MODEL_SWAPS


def test_deprecation_swap_targets_are_chat_compatible():
    # Every swap target should NOT itself be in the deprecation table
    for old, (new, _provider) in _DEPRECATED_MODEL_SWAPS.items():
        assert new not in _DEPRECATED_MODEL_SWAPS, (
            f'{old} swaps to {new} which is also deprecated — infinite loop risk'
        )
