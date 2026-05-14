"""
Step 5. 검색 결과를 LLM 에 넘겨 답변 받기.

이제까지 검색만 했다. 검색해 온 청크들을 프롬프트의 context 자리에
끼워 넣고 LLM 한테 답을 받으면, 그게 RAG 의 G(Generation) 단계다.

    python step5_generate_answer.py
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from models import get_embeddings, get_llm


INDEX_DIR = "my_faiss_index"
if not Path(INDEX_DIR).exists():
    raise SystemExit(f"{INDEX_DIR}/ 가 없다. 먼저 step3 을 돌려라.")

vector_db = FAISS.load_local(
    INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True,
)
print(f"인덱스 로드 ({vector_db.index.ntotal}벡터)")


query = "해외 출장 갔을 때 돈 어떻게 받아?"
docs = vector_db.similarity_search(query, k=3)

# 검색 결과 3개를 하나의 context 문자열로 합친다.
# 원본 가이드 코드는 docs[0] 하나만 썼지만, 3개 다 활용하면 답 품질이
# 눈에 띄게 좋아진다.
context = "\n\n---\n\n".join(
    f"(p.{d.metadata.get('page','?')})\n{d.page_content}"
    for d in docs
)


prompt = PromptTemplate.from_template(
    """당신은 사내 규정 안내 챗봇입니다. 아래 [검색된 문서] 만 참고하여 답변하세요.

[검색된 문서]
{context}

[사용자 질문]
{question}

[답변]"""
)


llm = get_llm(temperature=0)
chain = prompt | llm

print(f"질문: {query}")
print("LLM 호출 중...")
response = chain.invoke({"context": context, "question": query})
print()
print("답변:")
print(response.content)
