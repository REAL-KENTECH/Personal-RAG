"""
Step 2. 청크 분할.

페이지 통으로 LLM 에 넣기엔 너무 길고, 한 문장씩 잘라 넣기엔 문맥이
날아간다. 적당한 크기로 자르고 인접한 조각끼리 살짝 겹쳐서 경계에서
끊어진 문맥을 보전한다.

대략 가이드:
- FAQ 같은 짧은 문답: 200~400자
- 줄글 보고서/규정: 800~1200자
- overlap 은 chunk_size 의 10~15%

    python step2_chunking.py
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# step1 다시 — pages 가 있어야 자를 게 있다
pages = PyPDFLoader("출장 규정.pdf").load()
print(f"step1 결과: {len(pages)}페이지")


# 두 가지 시나리오를 직접 비교해 보면 차이가 느껴진다.
small = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
large = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

chunks_small = small.split_documents(pages)
chunks_large = large.split_documents(pages)

print(f"300자 기준  -> {len(chunks_small)}개 청크")
print(f"1000자 기준 -> {len(chunks_large)}개 청크")
print()

# 출장규정처럼 줄글이 긴 문서엔 1000 쪽이 자연스럽다.
chunks = chunks_large

print(f"이후 단계엔 1000자 청크 {len(chunks)}개로 진행한다.")
print()
print("첫 청크 미리보기:")
print(chunks[0].page_content[:300])
print()
print(f"메타데이터(원본 페이지 번호가 그대로 남는다): {chunks[0].metadata}")
