"""Small HTML-emitting helpers used across views.

The CSS classes referenced here (`chip`, `section-title`, `section-sub`,
`empty-state`) live in ``personal_rag.config.APP_CSS`` which is injected
once at app boot.
"""

import streamlit as st


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
