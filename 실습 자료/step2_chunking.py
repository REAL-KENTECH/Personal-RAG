"""
Step 2 — 청크(Chunk) 분할

목표: 긴 페이지 텍스트를 검색하기 좋은 작은 조각으로 자른다.
이 step도 외부 API 호출 없음.

핵심 개념:
- LLM context는 한정적 → 페이지 통째로 던지면 비효율
- 너무 짧으면 문맥 손실, 너무 길면 검색 정확도 ↓
- 일반적으로 chunk_size = 500~1500자, overlap = 10~15%

실행:
    python step2_chunking.py
"""

# ─── 이전 단계: PDF 로드 ──────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("출장 규정.pdf")
pages = loader.load()
print(f"✓ Step 1: {len(pages)}페이지 로드 완료")

# ─── 이번 단계: 청크 분할 ─────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 시나리오에 따라 청크 크기를 조정합니다.
# 작게 자르면 — 정확한 매칭에 유리. FAQ 같은 짧은 문답형 문서에 적합.
splitter_small = RecursiveCharacterTextSplitter(
    chunk_size=300,    # 한 조각 ≈ 300자
    chunk_overlap=30,  # 인접 조각끼리 30자 겹침 (≈10%)
)

# 크게 자르면 — 문맥 보존에 유리. 줄글 보고서·규정 문서에 적합.
splitter_large = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,  # 약 15%
)

# 두 시나리오를 비교해보기 위해 둘 다 실행
chunks_small = splitter_small.split_documents(pages)
chunks_large = splitter_large.split_documents(pages)

print(f"\n📊 청크 분할 결과:")
print(f"  - 작게 분할 (300자): {len(chunks_small)}개 조각")
print(f"  - 크게 분할 (1000자): {len(chunks_large)}개 조각")

# 다음 단계에서 사용할 변수 — 출장 규정 같은 줄글에는 1000자가 적당
chunks = chunks_large

print(f"\n→ chunks 변수에 {len(chunks)}개 조각 저장 (다음 단계에서 사용)")
print()
print("─" * 60)
print(f"📄 첫 번째 청크 미리보기:")
print("─" * 60)
print(chunks[0].page_content[:300])
print("...")
print()
print(f"📋 메타데이터(원본 페이지 번호 포함): {chunks[0].metadata}")
print()
print("→ 다음 단계: step3_embed_store.py")
