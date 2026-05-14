"""실습 streamlit 페이지 공통 스타일 + 작은 UI 헬퍼.

메인 앱 (../ui/helpers.py + ../config.py 의 APP_CSS) 의 톤을 가져와 본
실습에서도 비슷한 룩&필을 쓴다. CSS 는 sb-brand / sb-section /
section-title / section-sub / chip / empty-state 클래스 한 세트.
"""

import streamlit as st


STYLES = """
<style>
  .sb-brand   { font-size: 18px; font-weight: 700; letter-spacing: -0.01em; margin-bottom: 2px; }
  .sb-tagline { color: rgba(128,128,128,0.95); font-size: 12px; margin-bottom: 4px; }
  .sb-section { color: rgba(128,128,128,0.95); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 16px; margin-bottom: 6px; }

  .section-title { font-size: 18px; font-weight: 600; margin: 6px 0 6px 0; }
  .section-sub   { color: rgba(128,128,128,0.95); font-size: 13px; margin-bottom: 10px; line-height: 1.5; }

  .empty-state { padding: 28px; border: 1px dashed rgba(128,128,128,0.28); border-radius: 12px; text-align: center; color: rgba(128,128,128,0.95); line-height: 1.6; }
  .empty-hero  { text-align: center; padding: 48px 16px 16px 16px; }
  .empty-hero h2 { font-size: 24px; font-weight: 600; margin: 0 0 8px 0; }
  .empty-hero p  { color: rgba(128,128,128,0.95); font-size: 14px; margin: 0 0 18px 0; }

  .chip { display: inline-block; padding: 3px 9px; background: rgba(120,120,120,0.12); border: 1px solid rgba(120,120,120,0.22); border-radius: 999px; font-size: 11px; line-height: 1.4; margin-right: 4px; margin-bottom: 4px; }
  .chip.active { background: rgba(46,160,67,0.12); border-color: rgba(46,160,67,0.30); }
  .chip.muted  { color: rgba(128,128,128,0.95); }

  hr { border-color: rgba(128,128,128,0.16) !important; }
</style>
"""


def apply_styles():
    """페이지 최상단에서 한 번 호출."""
    st.markdown(STYLES, unsafe_allow_html=True)


def brand(title: str = "사내 규정 챗봇", tagline: str = "RAG 실습 데모"):
    """사이드바 브랜드 영역. 메인 앱의 사이드바 상단과 같은 톤."""
    st.markdown(f'<div class="sb-brand">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-tagline">{tagline}</div>', unsafe_allow_html=True)
    st.write("")


def sidebar_section(label: str):
    """사이드바 안의 작은 섹션 헤더 (메뉴 / 설정 / 상태 같은 라벨)."""
    st.markdown(
        f'<div class="sb-section">{label}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, sub: str = ""):
    """메인 영역 섹션 제목 + 부제."""
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )
    if sub:
        st.markdown(
            f'<div class="section-sub">{sub}</div>',
            unsafe_allow_html=True,
        )


def empty(text: str):
    """비어있을 때 안내 카드."""
    st.markdown(
        f'<div class="empty-state">{text}</div>',
        unsafe_allow_html=True,
    )


def chip(text: str, kind: str = "default") -> str:
    """상태 표시용 작은 칩. HTML 문자열을 반환하므로 st.markdown 으로 그린다."""
    cls = "chip"
    if kind == "active":
        cls += " active"
    elif kind == "muted":
        cls += " muted"
    return f'<span class="{cls}">{text}</span>'
