"""
Step 8. 파일 업로드 위젯.

step7 까지 PDF 는 코드에 하드코딩돼 있었다. 사용자가 자기 PDF 를
업로드해서 그 문서에 대해 질문하게 만들려면 file_uploader 가 필요하다.

업로드된 객체는 bytes 를 메모리에 들고 있는데, PyPDFLoader 는 경로를
받기 때문에 step9 에서 임시 파일로 한 번 저장해서 넘긴다.

    streamlit run step8_file_upload.py
"""

import streamlit as st

from ui_styles import apply_styles, brand, empty, section, sidebar_section


st.set_page_config(page_title="사내 규정 챗봇", layout="wide", initial_sidebar_state="expanded")
apply_styles()


with st.sidebar:
    brand("사내 규정 챗봇", "step 8 — 파일 업로드")
    sidebar_section("설정")
    uploaded_file = st.file_uploader(
        "PDF 문서를 업로드하세요",
        type=["pdf"],
        help="여러 페이지의 사내 규정/매뉴얼 등",
    )


section("사내 규정 챗봇", "PDF 를 업로드해야 동작한다. step 9 에서 RAG 와 연결.")


if uploaded_file is None:
    empty(
        "좌측 사이드바에서 PDF 파일을 먼저 업로드해 주세요."
        "<br>샘플 PDF 가 같은 폴더에 있다면 끌어다 놓아도 됩니다."
    )
else:
    st.success(f"'{uploaded_file.name}' 업로드 완료")

    cols = st.columns(3)
    size_kb = uploaded_file.size / 1024
    cols[0].metric("파일 크기", f"{size_kb:.1f} KB")
    cols[1].metric("MIME", uploaded_file.type or "—")
    cols[2].metric("문서 수", 1)

    # uploaded_file 은 BytesIO. .getvalue() 로 바이트 전체 추출.
    raw = uploaded_file.getvalue()

    with st.expander("바이트 미리보기 (디버깅용)"):
        st.code(f"첫 20바이트: {raw[:20]}")
        st.caption(f"총 {len(raw):,} 바이트")
