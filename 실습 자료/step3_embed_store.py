"""
Step 3 — 임베딩 + 벡터 DB 저장

목표: 청크를 의미를 담은 숫자(벡터)로 변환해 FAISS 인덱스에 저장한다.
이 step은 OpenAI API를 처음 호출함 → .env 의 OPENAI_API_KEY 필요.

핵심 개념:
- 임베딩: 텍스트를 1536차원(text-embedding-ada-002 / 3-small) 벡터로 변환
- 의미가 비슷한 텍스트 → 벡터 공간에서 거리가 가까움
- FAISS: Meta가 만든 빠른 유사도 검색 라이브러리 (로컬, 무료)

실행:
    python step3_embed_store.py

이 step만 매번 돌리면 임베딩 API 호출 비용이 듭니다 (대략 ~₩1).
한 번 돌리고 나면 my_faiss_index/ 폴더에 저장되어 step4~6은 그걸 재사용.
"""

# .env 자동 로드 (OPENAI_API_KEY 환경변수에 주입)
from dotenv import load_dotenv
load_dotenv()

# ─── 이전 단계: PDF 로드 + 청크 분할 ─────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("출장 규정.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(pages)
print(f"✓ Step 1-2 완료: {len(pages)}페이지 → {len(chunks)}개 청크")

# ─── 이번 단계: 임베딩 + FAISS 저장 ──────────────────────────────────
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 임베딩 모델 설정
#    - OpenAI 기본 모델 (text-embedding-3-small, 비용 저렴)
#    - 다른 모델 쓰려면: OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = OpenAIEmbeddings()

# 2. 모든 청크를 임베딩하고 FAISS DB에 저장
#    — 이 라인이 OpenAI API에 청크 개수만큼 임베딩 요청을 보냅니다.
#    — 청크가 많으면 몇십 초 걸릴 수 있음.
print(f"\n⏳ {len(chunks)}개 청크를 임베딩 중... (OpenAI API 호출)")
vector_db = FAISS.from_documents(chunks, embeddings)
print("✓ 벡터 DB 생성 완료!")

# 3. 로컬 디스크에 저장 — 다음 step에서 재사용
vector_db.save_local("my_faiss_index")
print("✓ my_faiss_index/ 폴더에 저장됨 (재실행 시 임베딩 비용 절약)")

print()
print(f"📊 인덱스 통계:")
print(f"  - 저장된 벡터 수: {vector_db.index.ntotal}")
print(f"  - 차원 수: {vector_db.index.d}")
print()
print("→ 다음 단계: step4_similarity_search.py")
