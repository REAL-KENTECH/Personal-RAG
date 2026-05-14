# RAG 실습 — 처음부터 끝까지 따라가기

LangChain + OpenAI로 PDF 한 개를 받아 질문에 답하는 RAG 챗봇을 9단계로 직접 만들어 봅니다. 각 step 파일은 **혼자서 실행 가능**하도록 이전 단계 코드를 위에 포함했습니다. `step1` → `step9` 순서로 따라가면 됩니다.

## 사전 준비

### 1. Python 가상환경 + 의존성

```bash
cd "/home/inho20/Personal-RAG/실습 자료"

# 가상환경 (없으면 만들기)
python3 -m venv .venv
source .venv/bin/activate          # Linux/Mac
# Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. OpenAI API 키 발급 & 입력

1. https://platform.openai.com/api-keys → **Create new secret key**
2. 키를 `.env` 파일에 저장:
   ```bash
   cp .env.example .env
   # 그 다음 .env 파일을 열어 OPENAI_API_KEY=sk-... 부분에 본인 키 붙여넣기
   ```
3. 또는 쉘에서 직접:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

> 💡 OpenAI 계정에 결제 수단 등록 안 돼있으면 호출 거부됨. https://platform.openai.com/account/billing 에서 최소 \$5 충전 권장 (이 실습엔 \$0.10 정도밖에 안 씀).

### 3. PDF 준비

이 폴더에 `출장 규정.pdf`(3페이지)가 들어있습니다. 본인 문서로 바꾸려면 같은 폴더에 두고 step 파일 안의 파일명만 수정하세요.

## 단계별 실행

순서대로 한 파일씩 실행. 각 파일은 결과를 콘솔에 출력합니다.

| 단계 | 파일 | 무엇을 배우는가 |
|---|---|---|
| 1 | `step1_load_document.py` | PDF를 페이지별 텍스트로 로드 |
| 2 | `step2_chunking.py` | 긴 텍스트를 검색하기 쉬운 조각으로 분할 (청킹) |
| 3 | `step3_embed_store.py` | 조각을 임베딩(벡터)으로 변환해 FAISS DB에 저장 |
| 4 | `step4_similarity_search.py` | 질문과 의미적으로 가까운 조각 찾기 |
| 5 | `step5_generate_answer.py` | 검색 결과를 LLM에 넘겨 답변 생성 |
| 6 | `step6_optimize_prompt.py` | 프롬프트로 답변 품질 / 환각 통제 |
| 7 | `step7_chat_ui.py` | Streamlit으로 대화형 UI 만들기 (RAG 없이) |
| 8 | `step8_file_upload.py` | 파일 업로드 위젯 붙이기 |
| 9 | `step9_app_complete.py` | 전부 합쳐서 완성 — `streamlit run` |

### 실행 예시

```bash
# 단계 1~6: 일반 Python 스크립트
python step1_load_document.py
python step2_chunking.py
python step3_embed_store.py        # FAISS 인덱스를 my_faiss_index/ 에 저장
python step4_similarity_search.py
python step5_generate_answer.py
python step6_optimize_prompt.py

# 단계 7~9: Streamlit 앱
streamlit run step7_chat_ui.py
streamlit run step8_file_upload.py
streamlit run step9_app_complete.py
```

## 자주 막히는 곳

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'langchain_community'` | `pip install -r requirements.txt` 안 했거나 가상환경 미활성 |
| `openai.AuthenticationError` 또는 401 | `.env`에 키가 비었거나 `export OPENAI_API_KEY=` 안 됨 |
| `FileNotFoundError: 출장 규정.pdf` | 스크립트를 폴더 밖에서 실행 중. `cd "실습 자료"` 후 실행 |
| 한글 깨짐 (cp949 등) | 거의 PDF 자체가 잘못된 인코딩. UTF-8 가능한 다른 PDF로 시도 |
| 임베딩이 너무 느림 | step3을 한 번만 실행하면 `my_faiss_index/`에 저장됨. step4~6은 그걸 재사용해서 빠름 |

## 다음 단계

이 실습이 익숙해지면 한 단계 위 — **본 프로젝트의 메인 앱** (`../app.py` 119줄 + `auth/`, `data/`, `llm/`, `retrieval/` 등 모듈 패키지) 을 읽어보세요. 같은 RAG 흐름이지만 다음이 추가됨:

- 멀티프로바이더 (OpenAI · Anthropic · Fireworks · HF Router · DashScope · vLLM)
- 하이브리드 검색 (BM25 + Dense + RRF + Cross-encoder rerank)
- HyDE / Multi-query / Contextual rewrite
- 멀티모달 PDF 페이지 이미지
- Supabase 영속 저장 + 사용자 인증
- 에이전트 워크플로 (이메일/보고서/요약/분석/비교)

본 실습 9단계가 그 모든 기능의 "최소 핵심"이라고 생각하면 됩니다.

## (보너스) 오픈 모델 사용

상업 API 대신 오픈 모델을 쓰고 싶다면 Hugging Face Inference Router 등을 활용:

```python
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",       # 예시 — 본인이 쓰려는 오픈 모델
    huggingfacehub_api_token="hf_...",     # https://huggingface.co/settings/tokens
)
```

`step5`, `step6`, `step9`의 `ChatOpenAI(...)` 부분을 위와 같이 바꾸면 됩니다. 자세한 본 프로젝트의 멀티프로바이더 처리는 `../llm/params.py` 참고.
