"""
Step 3. 임베딩 + FAISS 저장.

청크를 벡터로 바꿔서 FAISS 인덱스에 넣는다. 의미가 비슷한 텍스트는
벡터 공간에서 거리가 가깝게 매핑돼서, 나중에 질문 벡터로 가까운 청크를
찾아오게 된다.

여기서 처음 외부 API/모델이 들어간다. OPENAI 백엔드면 OpenAI 임베딩
API 호출, HUGGINGFACE 백엔드면 로컬 sentence-transformers (첫 실행
때만 가중치 다운로드 ~500MB).

    python step3_embed_store.py

한 번 돌리면 my_faiss_index/ 폴더에 저장돼서 step4~6 은 재사용한다.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import describe_backend, get_embeddings


print(describe_backend())

# step1~2 다시
pages = PyPDFLoader("출장 규정.pdf").load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150
).split_documents(pages)
print(f"입력: {len(pages)}페이지 -> {len(chunks)}청크")

# 임베더 받아서 FAISS 인덱스 빌드
embeddings = get_embeddings()
print("임베딩 중. 청크 수에 비례해서 시간이 걸린다...")
vector_db = FAISS.from_documents(chunks, embeddings)

# 디스크 저장. 다음 단계에서 매번 임베딩 다시 안 돌려도 되도록.
vector_db.save_local("my_faiss_index")

print()
print("완료.")
print(f"저장된 벡터 수: {vector_db.index.ntotal}")
print(f"차원: {vector_db.index.d}")
print("인덱스 위치: my_faiss_index/")
