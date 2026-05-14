"""
Step 6. 프롬프트로 답변 통제하기.

같은 검색 결과라도 시스템 프롬프트를 어떻게 쓰느냐에 따라 답이
완전히 달라진다. 두 가지를 같은 질문에 돌려보고 차이를 직접 확인한다.

핵심 두 가지:
- 문서에 답이 없을 때 LLM 이 자기 지식으로 환각하는 걸 차단
- 답변 형식(불릿, 길이) 강제

    python step6_optimize_prompt.py
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from models import get_embeddings, get_llm


if not Path("my_faiss_index").exists():
    raise SystemExit("my_faiss_index/ 가 없다. step3 부터 다시.")

vector_db = FAISS.load_local(
    "my_faiss_index", get_embeddings(),
    allow_dangerous_deserialization=True,
)
llm = get_llm(temperature=0)


loose_prompt = PromptTemplate.from_template(
    """당신은 사내 규정 안내 챗봇입니다. 아래 [검색된 문서] 를 참고하여 답변하세요.

[검색된 문서]
{context}

[사용자 질문]
{question}"""
)

strict_prompt = PromptTemplate.from_template(
    """당신은 사내 규정 안내 챗봇입니다. 반드시 아래 [검색된 문서] 만 참고하세요.
문서에 정답이 없다면 절대 지어내지 말고 정확히 "문서에서 찾을 수 없습니다" 라고만 답하세요.
답변은 불릿 포인트(•)로 3줄 이내로 요약하세요.

[검색된 문서]
{context}

[사용자 질문]
{question}"""
)


def ask(prompt_template, question):
    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    chain = prompt_template | llm
    return chain.invoke({"context": context, "question": question}).content


# 1) 문서 안에 답이 있는 질문
q1 = "해외 출장 갔을 때 돈 어떻게 받아?"
print(f"[질문 1] {q1}")
print()
print("(느슨한 프롬프트)")
print(ask(loose_prompt, q1))
print()
print("(엄격한 프롬프트)")
print(ask(strict_prompt, q1))
print()
print("=" * 60)
print()

# 2) 문서에 없는 질문 — 환각 테스트
q2 = "우리 회사 대표이사 이름이 뭐야?"
print(f"[질문 2] {q2}")
print()
print("(느슨한 프롬프트)")
print(ask(loose_prompt, q2))
print()
print("(엄격한 프롬프트)")
print(ask(strict_prompt, q2))
print()

# 정리:
# - temperature=0: 같은 입력에 항상 같은 출력. 디버깅 편함.
# - 엄격한 프롬프트는 환각을 줄이고 형식을 통일하지만, 약간만 모호한 질문도
#   "답 없음" 으로 거절할 수 있다는 부작용이 있다.
