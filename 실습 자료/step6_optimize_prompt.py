"""
Step 6 — 프롬프트 최적화 (환각 방지)

목표: 같은 검색 결과라도 프롬프트를 어떻게 쓰느냐에 따라 답변이 크게 달라진다.
- 문서에 없는 질문도 LLM이 자기 지식으로 답해버리는 "환각" 차단
- 답변 형식 강제 (불릿, 길이 제한 등)

실행:
    python step6_optimize_prompt.py
"""

from dotenv import load_dotenv
load_dotenv()

# ─── 이전 단계 산출물 로드 ───────────────────────────────────────────
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

embeddings = OpenAIEmbeddings()
if not Path("my_faiss_index").exists():
    raise SystemExit("⚠ my_faiss_index/ 가 없습니다. step3을 먼저 실행하세요.")

vector_db = FAISS.load_local(
    "my_faiss_index", embeddings, allow_dangerous_deserialization=True,
)

# ─── 이번 단계: 두 프롬프트 비교 ─────────────────────────────────────

# Temperature=0 — 환각 가능성 최소화 (모델이 가장 가능성 높은 토큰만 선택)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 느슨한 프롬프트 (step5와 동일)
loose_prompt = PromptTemplate.from_template(
    """당신은 사내 규정 안내 챗봇입니다. 아래 [검색된 문서]를 참고하여 답변하세요.

[검색된 문서]
{context}

[사용자 질문]
{question}"""
)

# 엄격한 프롬프트 — 문서에 없으면 "없다"고 못 박기 + 형식 제약
strict_prompt = PromptTemplate.from_template(
    """당신은 사내 규정 안내 챗봇입니다. 반드시 아래 [검색된 문서]만 참고하세요.
문서에 정답이 없다면 절대 지어내지 말고 정확히 "문서에서 찾을 수 없습니다"라고만 답하세요.
답변은 불릿 포인트(•)로 3줄 이내로 요약하세요.

[검색된 문서]
{context}

[사용자 질문]
{question}"""
)


def ask(prompt_template, question):
    """주어진 프롬프트와 질문으로 답변 생성."""
    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    chain = prompt_template | llm
    return chain.invoke({"context": context, "question": question}).content


# 비교 1: 문서에 답이 있는 질문
q_in_doc = "해외 출장 갔을 때 돈 어떻게 받아?"
print("─" * 60)
print(f"❓ 질문 (문서에 답 있음): {q_in_doc}")
print("─" * 60)
print("[느슨한 프롬프트]")
print(ask(loose_prompt, q_in_doc))
print()
print("[엄격한 프롬프트]")
print(ask(strict_prompt, q_in_doc))
print()

# 비교 2: 문서에 없는 질문 — 환각 테스트
q_off_topic = "우리 회사 대표이사 이름이 뭐야?"
print("─" * 60)
print(f"❓ 질문 (문서에 답 없음 — 환각 테스트): {q_off_topic}")
print("─" * 60)
print("[느슨한 프롬프트]")
print(ask(loose_prompt, q_off_topic))
print()
print("[엄격한 프롬프트]")
print(ask(strict_prompt, q_off_topic))
print()

print("─" * 60)
print("📝 정리:")
print("  - temperature=0: 같은 입력에 항상 같은 출력 → 디버깅 ↑")
print("  - 강한 제약 프롬프트: 환각 줄어들고 형식 통일")
print("  - 트레이드오프: 너무 엄격하면 약간만 모호해도 '답 없음' 반환")
print()
print("→ 다음 단계: step7_chat_ui.py (Streamlit UI)")
