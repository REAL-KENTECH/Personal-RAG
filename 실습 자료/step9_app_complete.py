"""
Step 9. 전부 합친 완성판.

step1~8 의 흐름을 하나의 Streamlit 앱으로 묶는다. 사용자가 PDF 를
업로드하면 자동으로 청킹·임베딩·인덱싱하고, 질문이 들어오면 검색해서
LLM 답변과 출처를 같이 보여준다.

    streamlit run step9_app_complete.py
"""

import os
import tempfile

import streamlit as st

# .env 가 이미 로드되어 있도록 models 모듈을 먼저 import
from models import describe_backend, get_embeddings, get_llm
from ui_styles import apply_styles, brand, chip, empty, section, sidebar_section

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(
    page_title="사내 규정 챗봇",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()


# ── 핵심 파이프라인 ──────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def build_vector_db(file_bytes: bytes, file_name: str):
    """업로드된 PDF 바이트를 받아 FAISS 인덱스를 만든다.

    @st.cache_resource 가 (file_bytes, file_name) 조합 기준으로 결과를
    캐싱하므로 같은 PDF 에 대해서는 한 번만 인덱싱한다.
    """
    # PyPDFLoader 는 경로를 받기 때문에 임시 파일에 한 번 떨어뜨린다
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        pages = PyPDFLoader(tmp_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150,
        ).split_documents(pages)
        vector_db = FAISS.from_documents(chunks, get_embeddings())
        return vector_db, len(pages), len(chunks)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def generate_answer(vector_db, question: str, k: int = 3):
    """검색 + LLM. 답변과 함께 인용된 청크 리스트를 같이 돌려준다."""
    docs = vector_db.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[{i+1}] (p.{d.metadata.get('page','?')}) {d.page_content}"
        for i, d in enumerate(docs)
    )

    prompt = PromptTemplate.from_template(
        """당신은 사내 규정 안내 챗봇입니다. 반드시 아래 [검색된 문서] 만 참고하세요.
문서에 정답이 없다면 절대 지어내지 말고 "문서에서 찾을 수 없습니다" 라고 답하세요.
근거가 되는 출처 번호를 [1], [2] 형식으로 인용하세요.

[검색된 문서]
{context}

[사용자 질문]
{question}

[답변]"""
    )

    chain = prompt | get_llm(temperature=0)
    response = chain.invoke({"context": context, "question": question})
    return response.content, docs


# ── 사이드바 ─────────────────────────────────────────────────────────


with st.sidebar:
    brand("사내 규정 챗봇", "RAG 실습 완성판")

    sidebar_section("백엔드")
    st.caption(describe_backend())

    # 백엔드별로 어떤 키가 필요한지 확인하고 안내
    backend = os.getenv("BACKEND", "openai").lower()
    if backend == "openai":
        key_ok = bool(os.getenv("OPENAI_API_KEY"))
        st.markdown(
            chip("OPENAI_API_KEY", "active" if key_ok else "muted"),
            unsafe_allow_html=True,
        )
        if not key_ok:
            st.error(".env 의 OPENAI_API_KEY 가 비어 있습니다.")
            st.stop()
    else:
        key_ok = bool(os.getenv("HF_TOKEN"))
        st.markdown(
            chip("HF_TOKEN", "active" if key_ok else "muted"),
            unsafe_allow_html=True,
        )
        if not key_ok:
            st.error(".env 의 HF_TOKEN 이 비어 있습니다.")
            st.stop()

    sidebar_section("문서")
    uploaded_file = st.file_uploader("PDF 문서를 업로드", type=["pdf"])

    sidebar_section("작업")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── 메인 ─────────────────────────────────────────────────────────────


section("사내 규정 AI 챗봇", "업로드한 PDF 의 내용 안에서만 답변합니다.")


if "messages" not in st.session_state:
    st.session_state.messages = []


if uploaded_file is None:
    empty(
        "좌측 사이드바에서 PDF 를 먼저 업로드해 주세요."
        "<br>같은 폴더의 <code>출장 규정.pdf</code> 를 끌어다 놓아도 됩니다."
    )
    st.stop()


with st.spinner("PDF 분석 중 (첫 번째만 시간이 걸립니다)"):
    vector_db, n_pages, n_chunks = build_vector_db(
        uploaded_file.getvalue(), uploaded_file.name,
    )


# 상단에 인덱스 상태를 chip 으로 표시 — 메인 앱 사이드바 상태 영역과 같은 톤
cols = st.columns(4)
cols[0].metric("PDF", uploaded_file.name)
cols[1].metric("페이지", n_pages)
cols[2].metric("청크", n_chunks)
cols[3].metric("벡터 차원", vector_db.index.d)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if user_query := st.chat_input("질문을 입력하세요"):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("생각 중"):
            answer, source_docs = generate_answer(vector_db, user_query, k=3)
        st.markdown(answer)

        with st.expander(f"참고 청크 {len(source_docs)}개"):
            for i, doc in enumerate(source_docs, start=1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**[{i}] p.{page}**")
                body = doc.page_content
                if len(body) > 400:
                    body = body[:400] + "..."
                st.text(body)
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
