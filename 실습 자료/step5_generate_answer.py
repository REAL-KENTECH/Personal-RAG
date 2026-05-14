"""
Step 5 — LLM으로 답변 생성

목표: 검색한 청크를 LLM에 함께 넘겨 자연어 답변을 받는다.
이게 RAG의 "Generation" 단계 (R = Retrieval + A = Augmented + G = Generation).

실행:
    python step5_generate_answer.py
"""

from dotenv import load_dotenv
load_dotenv()

# ─── 이전 단계: 인덱스 로드 + 검색 ────────────────────────────────────
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
INDEX_DIR = "my_faiss_index"

if not Path(INDEX_DIR).exists():
    raise SystemExit("⚠ my_faiss_index/ 가 없습니다. 먼저 step3을 실행하세요.")

vector_db = FAISS.load_local(
    INDEX_DIR, embeddings, allow_dangerous_deserialization=True,
)
print(f"✓ 인덱스 로드 ({vector_db.index.ntotal}개 벡터)")

# 검색 — 이번엔 k=3으로 받아 모두 컨텍스트로 사용
query = "해외 출장 갔을 때 돈 어떻게 받아?"
docs = vector_db.similarity_search(query, k=3)
print(f"✓ 관련 청크 {len(docs)}개 검색")

# ─── 이번 단계: LLM 호출 ──────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. LLM 모델 — gpt-4o-mini 가 저렴하고 빠름
#    더 좋은 답변이 필요하면 model="gpt-4o"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. 프롬프트 템플릿
#    {context}, {question} 두 변수에 우리가 값을 끼워 넣음
prompt = PromptTemplate.from_template(
    """당신은 사내 규정 안내 챗봇입니다. 아래 [검색된 문서]만 참고하여 답변하세요.

[검색된 문서]
{context}

[사용자 질문]
{question}

[답변]"""
)

# 3. 검색 결과 3개를 하나의 컨텍스트 문자열로 합치기
#    (앞 step 원본 코드는 docs[0]만 썼지만 k=3 다 활용하는 게 정답 품질에 더 좋음)
context = "\n\n---\n\n".join(
    f"(p.{d.metadata.get('page','?')})\n{d.page_content}"
    for d in docs
)

# 4. 체인 = 프롬프트 → LLM
chain = prompt | llm

# 5. 실행
print()
print("─" * 60)
print(f"❓ 질문: {query}")
print("─" * 60)
print("⏳ LLM 응답 생성 중...")
response = chain.invoke({"context": context, "question": query})
print()
print("💡 답변:")
print("─" * 60)
print(response.content)
print()
print("→ 다음 단계: step6_optimize_prompt.py")
