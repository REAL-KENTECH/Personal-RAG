"""Streaming + non-streaming chat completion + response-model attribution.

Both paths return ``(full_text, reasoning_text)``. Streaming renders an
idle-tick "생각 중…" placeholder so the user can distinguish "model is
thinking" from "app is hung" — reasoning models can spend 10+ seconds
before they emit anything visible.
"""

import time

import streamlit as st


def _record_response_model(model_id: str) -> None:
    """Capture the model id the provider actually served the response from.
    Useful for verifying that requests went to the model the user picked —
    OpenAI chat completions often respond with a dated variant like
    'gpt-5-2025-08-07', so seeing that next to the configured 'gpt-5'
    confirms the routing without trusting the model's self-introduction."""
    if model_id:
        st.session_state['_last_response_model'] = str(model_id)


def stream_chat(client, params: dict):
    """Stream response. Returns (full_text, reasoning_text).

    Renders an immediate "생각 중…" placeholder so the user can tell the
    difference between "model is thinking" and "app is hung". The
    placeholder updates each second with an elapsed-time counter while
    we wait for the first visible token (reasoning models like
    HyperCLOVAX-Think can take 10s+ before they emit anything)."""
    placeholder = st.empty()
    placeholder.markdown('_생각 중…_')
    t0 = time.time()
    last_idle_tick = t0
    full_text = ''
    reasoning_text = ''
    try:
        stream = client.chat.completions.create(stream=True, **params)
        for chunk in stream:
            # Some providers stamp the model id on every chunk; capture
            # it once we see it.
            cm = getattr(chunk, 'model', None)
            if cm and not st.session_state.get('_stream_model_captured'):
                _record_response_model(cm)
                st.session_state['_stream_model_captured'] = True
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
                elapsed = int(time.time() - t0)
                suffix = f' ({elapsed}초 경과)' if elapsed >= 2 else ''
                placeholder.markdown(
                    f'_생각 중{suffix}_\n\n> {reasoning_text}'
                )
            else:
                # No visible output yet — refresh the elapsed counter so
                # users can tell the connection isn't frozen.
                now = time.time()
                if now - last_idle_tick >= 1.0:
                    placeholder.markdown(
                        f'_생각 중… ({int(now - t0)}초 경과)_'
                    )
                    last_idle_tick = now
        placeholder.empty()
    except Exception:
        placeholder.empty()
        raise
    finally:
        st.session_state.pop('_stream_model_captured', None)
    return full_text, reasoning_text


def non_stream_chat(client, params: dict):
    resp = client.chat.completions.create(**params)
    _record_response_model(getattr(resp, 'model', None))
    choice = resp.choices[0]
    full_text = choice.message.content or ''
    reasoning = ''
    for attr in ('reasoning_content', 'reasoning'):
        val = getattr(choice.message, attr, None)
        if val:
            reasoning = val
            break
    return full_text, reasoning
