"""
Step 9 — 완성된 RAG 챗봇

목표: 단계 1~8을 모두 합쳐 하나의 Streamlit 앱으로 만든다.
사용자가 PDF를 업로드하면 자동으로 청킹·임베딩·인덱싱하고,
질문에 대해 검색 + LLM 답변을 생성한다.

실행:
    streamlit run step9_app_complete.py

데모 흐름:
1. 좌측 사이드바에서 PDF 업로드
2. 자동으로 인덱싱 진행 (한 번만)
3. 질문 입력 → 답변 + 출처 페이지 표시
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# .env 자동 로드 (OPENAI_API_KEY)
load_dotenv()

# LangChain 의존성
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ─── 페이지 설정 ──────────────────────────────────────────────────────
st.set_page_config(page_title="사내 규정 챗봇", page_icon="🤖", layout="wide")
st.title("사내 규정 AI 챗봇")


# ─── 핵심 RAG 파이프라인 ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_vector_db(file_bytes: bytes, file_name: str):
    """업로드된 PDF 바이트를 받아 FAISS 인덱스를 만든다.

    @st.cache_resource — 같은 파일에 대해서는 한 번만 인덱싱하고 캐시.
    파일이 바뀌면 (file_bytes, file_name 해시가 변경) 자동 재실행.
    """
    # 1. PDF 파일을 임시 디스크에 저장 (PyPDFLoader 가 경로를 받기 때문)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # 2. 페이지 로드
        pages = PyPDFLoader(tmp_path).load()
        # 3. 청킹
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        # 4. 임베딩 + FAISS
        embeddings = OpenAIEmbeddings()
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db, len(pages), len(chunks)
    finally:
        # 임시 파일 정리
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def generate_answer(vector_db, question: str, k: int = 3):
    """검색 + LLM. 답변과 함께 사용된 출처 청크 리스트를 반환."""
    docs = vector_db.similarity_search(question, k=k)

    # 검색 결과를 출처 번호와 함께 컨텍스트로 정리
    context = "\n\n".join(
        f"[{i+1}] (p.{d.metadata.get('page','?')}) {d.page_content}"
        for i, d in enumerate(docs)
    )

    # 엄격한 프롬프트 — 환각 방지, 출처 표기 유도
    prompt = PromptTemplate.from_template(
        """당신은 사내 규정 안내 챗봇입니다. 반드시 아래 [검색된 문서]만 참고하세요.
문서에 정답이 없다면 절대 지어내지 말고 "문서에서 찾을 수 없습니다"라고 답하세요.
근거가 되는 출처 번호를 [1], [2] 형식으로 인용하세요.

[검색된 문서]
{context}

[사용자 질문]
{question}

[답변]"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content, docs


# ─── 사이드바: 파일 업로드 + 상태 ─────────────────────────────────────

with st.sidebar:
    st.header("설정")

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        st.error("`OPENAI_API_KEY` 가 .env 에 없습니다.")
        st.code("cp .env.example .env\n# .env 편집 후 다시 실행")
        st.stop()
    else:
        st.success("✓ API 키 인식")

    # 파일 업로드
    uploaded_file = st.file_uploader(
        "PDF 문서를 업로드하세요",
        type=["pdf"],
    )

    # 대화 초기화
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─── 메인: RAG 동작 ───────────────────────────────────────────────────

# 대화 기록 저장소
if "messages" not in st.session_state:
    st.session_state.messages = []

# 파일이 없으면 안내만 보여주고 끝
if uploaded_file is None:
    st.info("👈 좌측 사이드바에서 PDF 파일을 먼저 업로드해주세요.")
    st.markdown(
        """
        ### 예시 질문
        - "해외 출장 갔을 때 일비는 얼마야?"
        - "출장 신청 절차가 어떻게 돼?"
        - "교통비는 어떻게 정산해?"
        """
    )
    st.stop()

# 파일이 있으면 인덱싱
with st.spinner("PDF 분석 중... (첫 번째만 시간이 걸립니다)"):
    vector_db, n_pages, n_chunks = build_vector_db(
        uploaded_file.getvalue(),
        uploaded_file.name,
    )

st.success(
    f"✓ '{uploaded_file.name}' 인덱싱 완료 — {n_pages}페이지 → {n_chunks}개 청크"
)
st.caption("이제 아래에서 질문하세요. 같은 PDF에 대한 재질문은 즉시 동작합니다.")

# 이전 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 새 입력
if user_query := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            answer, source_docs = generate_answer(vector_db, user_query, k=3)
        st.markdown(answer)

        # 출처 미리보기 (접힌 expander로)
        with st.expander(f"📚 참고 청크 {len(source_docs)}개", expanded=False):
            for i, doc in enumerate(source_docs, start=1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**[{i}] p.{page}**")
                st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
