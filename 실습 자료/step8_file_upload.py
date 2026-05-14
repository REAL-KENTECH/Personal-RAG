"""
Step 8 — 파일 업로드 위젯

목표: 사용자가 사이드바에서 PDF 파일을 직접 업로드할 수 있게 한다.
업로드된 파일은 메모리에 BytesIO 형태로 들어옴 → 다음 단계에서 RAG에 연결.

실행:
    streamlit run step8_file_upload.py
"""

import streamlit as st

# 메인 화면 제목
st.title("사내 규정 챗봇 🤖")
st.caption("PDF를 업로드해야 질문에 답할 수 있습니다 (단계 9에서 완성).")

# 사이드바에 파일 업로드 위젯
with st.sidebar:
    st.header("설정")
    uploaded_file = st.file_uploader(
        "PDF 문서를 업로드하세요",
        type=["pdf"],
        help="여러 페이지의 사내 규정/매뉴얼 등 PDF",
    )

# 업로드 상태 확인 + 파일 정보 표시
if uploaded_file is not None:
    st.success(f"✓ '{uploaded_file.name}' 업로드 완료")

    # 파일 메타데이터
    file_size_kb = uploaded_file.size / 1024
    st.write(f"📄 파일 크기: {file_size_kb:.1f} KB")
    st.write(f"📋 MIME 타입: {uploaded_file.type}")

    # uploaded_file 은 BytesIO 객체. .read() 로 바이트 전체 읽기 가능.
    # (실습 9에서 이 바이트를 PyPDFLoader 가 받을 수 있도록 임시 파일로 저장)
    raw_bytes = uploaded_file.getvalue()
    st.write(f"🔢 첫 20 바이트: {raw_bytes[:20]}")
else:
    st.info("좌측 사이드바에서 PDF 파일을 업로드해주세요.")

with st.sidebar:
    st.divider()
    st.caption(
        "단계 9에서:\n"
        "1) 업로드된 PDF → 페이지 로드\n"
        "2) 청크 분할 → 임베딩\n"
        "3) FAISS 인덱스 메모리에 보관\n"
        "4) 사용자 질문 → 검색 + LLM 답변"
    )
