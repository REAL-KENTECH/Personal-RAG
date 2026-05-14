"""
Step 1. PDF 읽어오기.

PDF 한 페이지를 한 덩어리(Document)로 받아오는 게 첫 단계다. 외부 API
호출이 없으니 키 없이도 돌아간다.

    python step1_load_document.py
"""

from langchain_community.document_loaders import PyPDFLoader

# 같은 폴더에 있는 PDF. 본인 파일로 바꾸려면 이 한 줄만 수정.
loader = PyPDFLoader("출장 규정.pdf")
pages = loader.load()

print(f"총 {len(pages)}페이지 로드.")
print()
print("1페이지 앞부분 200자:")
print(pages[0].page_content[:200])
print("...")
print()
print(f"메타데이터: {pages[0].metadata}")
