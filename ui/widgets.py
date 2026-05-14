"""Reusable Streamlit widgets — currently just the model picker.

``model_picker`` adapts to whichever provider is currently selected
(via ``PROVIDER_MODELS``) and offers an explicit "직접 입력" sentinel for
ids not in the curated list. The ``instant`` toggle lets the same widget
serve both the Settings tab (pending → user must click 적용) and the
chat top-bar quick switch (selecting applies immediately).
"""

import streamlit as st

from auth.prefs import _save_user_prefs
from config import _CUSTOM, PROVIDER_MODELS, PROVIDER_NAMES


def model_picker(label: str, key_prefix: str, instant: bool = False):
    """Per-provider model dropdown with '직접 입력' fallback.

    Two modes:
      - instant=False (default): the choice is *pending* until the user
        clicks the 적용 button. Used in the Settings tab where casual
        clicks on the dropdown shouldn't immediately change the model
        being sent to the API.
      - instant=True: every dropdown change writes straight to
        st.session_state['model']. Used in the chat top-bar quick switch
        where the user clearly wants to apply right away.
    """
    provider = st.session_state.get('provider', PROVIDER_NAMES[0])
    known = PROVIDER_MODELS.get(provider, [])
    current = st.session_state.get('model', '')

    def _commit(value: str):
        if not value or value == current:
            return False
        st.session_state['model'] = value
        try:
            _save_user_prefs()
        except Exception:
            pass
        return True

    if not known:
        if instant:
            new_val = st.text_input(
                label, value=current, key=f'{key_prefix}_model_text',
            )
            if _commit(new_val):
                st.rerun()
            return
        pending = st.text_input(
            label, value=current, key=f'{key_prefix}_model_text',
        )
    else:
        options = known + [_CUSTOM]
        if current in known:
            initial_idx = known.index(current)
        else:
            initial_idx = len(options) - 1   # 직접 입력
        choice = st.selectbox(
            label, options, index=initial_idx,
            format_func=lambda x: '직접 입력...' if x == _CUSTOM else x,
            key=f'{key_prefix}_model_select',
        )
        if instant and choice != _CUSTOM:
            # Quick-switch: any direct selection from the list applies now.
            if _commit(choice):
                st.rerun()
            return
        if choice == _CUSTOM:
            text_val = st.text_input(
                '모델 ID 직접 입력',
                value=current if current not in known else '',
                key=f'{key_prefix}_model_custom',
                placeholder='예: gpt-4o, my-org/my-finetune',
            )
            if instant:
                if _commit(text_val):
                    st.rerun()
                return
            pending = text_val
        else:
            pending = choice

    # ---- pending mode (settings tab) ----
    # Apply / confirm row. Only shown when the user has actually changed
    # the dropdown (or typed a different model id) from the active value.
    if pending and pending != current:
        cols = st.columns([3, 1])
        with cols[0]:
            st.caption(
                f'미적용 변경: `{current or "—"}` → `{pending}` '
                f'· 적용 버튼을 눌러야 채팅에 반영됩니다.'
            )
        with cols[1]:
            if st.button('적용', key=f'{key_prefix}_model_apply',
                         type='primary', use_container_width=True):
                _commit(pending)
                st.rerun()
    elif current:
        st.caption(f'현재 모델: `{current}`')
