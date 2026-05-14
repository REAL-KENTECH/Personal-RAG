"""
Step 4. 유사도 검색.

질문 문장을 같은 임베더로 벡터화해서 FAISS 안의 청크들과 거리를 비교,
가장 가까운 k 개를 가져온다. LLM 호출 없이 "검색만" 따로 확인하는
단계라 비용도 거의 안 든다.

    python step4_similarity_search.py
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import get_embeddings


INDEX_DIR = "my_faiss_index"
embeddings = get_embeddings()

if Path(INDEX_DIR).exists():
    # step3 산출물 재사용 — 임베딩 다시 안 돌려도 된다
    vector_db = FAISS.load_local(
        INDEX_DIR, embeddings,
        # 본인이 만든 인덱스니까 안전하다고 허용
        allow_dangerous_deserialization=True,
    )
    print(f"{INDEX_DIR}/ 에서 로드 ({vector_db.index.ntotal}벡터)")
else:
    # 없으면 step3 흐름을 새로 돈다
    print(f"{INDEX_DIR}/ 없음. 새로 만든다.")
    pages = PyPDFLoader("출장 규정.pdf").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(pages)
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_DIR)


query = "해외 출장 갔을 때 돈 어떻게 받아?"

# k=3 — 상위 3개. 너무 많이 가져오면 LLM context 가 길어지고 답이 흐려진다.
docs = vector_db.similarity_search(query, k=3)

print()
print(f"질문: {query}")
print()
for i, doc in enumerate(docs, start=1):
    page = doc.metadata.get("page", "?")
    preview = doc.page_content[:200].replace("\n", " ")
    print(f"[{i}위] p.{page}")
    print(f"    {preview}...")
    print()


# 점수까지 보고 싶으면 — 거리가 작을수록 더 비슷한 청크
print("점수 포함:")
for i, (doc, score) in enumerate(
    vector_db.similarity_search_with_score(query, k=3), start=1
):
    print(f"  {i}. distance={score:.4f}  p.{doc.metadata.get('page', '?')}")
