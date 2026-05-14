"""
Step 4 — 유사도 검색

목표: 질문을 임베딩해서 가장 가까운 청크 k개를 찾는다.
LLM 없이 "검색만" 해보는 단계 — 정답이 진짜 나올 만한 청크가 잡히는지 확인.

실행:
    python step4_similarity_search.py
"""

from dotenv import load_dotenv
load_dotenv()

# ─── 이전 단계 산출물 로드 (FAISS 인덱스가 디스크에 있으면 거기서 빠르게) ─
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
INDEX_DIR = "my_faiss_index"

if Path(INDEX_DIR).exists():
    # step3에서 저장한 인덱스 재사용 — 임베딩 비용 안 듦
    vector_db = FAISS.load_local(
        INDEX_DIR, embeddings,
        # 로컬에서 만든 안전한 pickle 이므로 허용
        allow_dangerous_deserialization=True,
    )
    print(f"✓ {INDEX_DIR}/ 에서 인덱스 로드 (재사용)")
else:
    # 인덱스가 없으면 step1-3 흐름을 새로 실행
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print("⚠ 저장된 인덱스 없음 — 새로 만듭니다 (step3 동작)")
    pages = PyPDFLoader("출장 규정.pdf").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(pages)
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_DIR)
    print(f"✓ 인덱스 생성 + 저장 ({vector_db.index.ntotal}개 벡터)")

# ─── 이번 단계: 검색 ──────────────────────────────────────────────────

# 1. 사용자 질문
query = "해외 출장 갔을 때 돈 어떻게 받아?"

# 2. 유사도 검색 — k=3 은 가장 가까운 조각 3개
docs = vector_db.similarity_search(query, k=3)

# 3. 결과 확인
print()
print("─" * 60)
print(f"❓ 질문: {query}")
print("─" * 60)
print()

for i, doc in enumerate(docs, start=1):
    page = doc.metadata.get("page", "?")
    preview = doc.page_content[:200].replace("\n", " ")
    print(f"📌 결과 {i}위 (p.{page}):")
    print(f"   {preview}...")
    print()

# 점수와 함께 보려면 — similarity_search_with_score 사용
print("─" * 60)
print("점수 포함 검색 (거리 작을수록 더 유사):")
print("─" * 60)
docs_with_score = vector_db.similarity_search_with_score(query, k=3)
for i, (doc, score) in enumerate(docs_with_score, start=1):
    print(f"  {i}. distance={score:.4f} (p.{doc.metadata.get('page','?')})")

print()
print("→ 다음 단계: step5_generate_answer.py")
