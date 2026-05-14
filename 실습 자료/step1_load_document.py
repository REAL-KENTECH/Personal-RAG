"""
Step 1 — PDF 문서 불러오기

목표: PDF 파일을 페이지 단위로 읽어 텍스트로 변환한다.
이 step은 외부 API 호출이 없어 인터넷·키 없이 실행 가능.

실행:
    python step1_load_document.py
"""

# 1. LangChain의 PDF 로더 도구 가져오기
from langchain_community.document_loaders import PyPDFLoader

# 2. 읽어들일 PDF 파일 지정
#    이 스크립트와 같은 폴더에 PDF가 있다는 가정.
#    본인 문서로 바꾸려면 아래 경로만 수정.
loader = PyPDFLoader("출장 규정.pdf")

# 3. 문서 로드 실행
#    pages는 LangChain의 Document 객체 리스트 — 한 페이지 = 한 Document.
pages = loader.load()

# 4. 결과 확인
print(f"✓ 성공! 총 {len(pages)}페이지를 불러왔습니다.")
print()
print("─" * 60)
print(f"📄 1페이지 미리보기 (앞 200자):")
print("─" * 60)
print(pages[0].page_content[:200])
print("...")
print()
print(f"📋 페이지 메타데이터: {pages[0].metadata}")
print()
print("→ 다음 단계: step2_chunking.py")
