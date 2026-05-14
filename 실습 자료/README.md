# RAG 실습 — 처음부터 따라가기

LangChain 으로 PDF 한 개를 받아 질문에 답하는 RAG 챗봇을 9단계에 걸쳐
만든다. 한 step 씩 따라가면 코드 한 줄이 어떤 역할을 하는지 자연스럽게
이해된다.

OpenAI(유료) 와 Hugging Face 오픈 모델(무료) 두 경로 모두 지원한다.
어느 쪽을 쓰든 step 코드는 그대로다. `.env` 의 `BACKEND` 값만 바꾸면
된다.

명령어만 빠르게 찾고 싶으면 [COMMANDS.md](COMMANDS.md) 참고.


## 사전 준비

### 1. 가상환경 + 의존성

```bash
cd "/home/inho20/Personal-RAG/실습 자료"
python3 -m venv .venv
source .venv/bin/activate          # Linux / Mac
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. API 키 발급

#### 옵션 A — OpenAI 경로 (기본, 유료)

가장 간단. 결제 수단 등록한 OpenAI 계정 하나면 끝.

- 키 발급: https://platform.openai.com/api-keys
- 결제: https://platform.openai.com/account/billing (최소 \$5 충전 권장. 본 실습은 다 합쳐 \$0.10 정도 쓴다)

#### 옵션 B — Hugging Face 경로 (무료)

오픈 모델을 HF Inference Providers 경유로 호출. 일부 provider 는 무료
크레딧을 준다.

- 키 발급: https://huggingface.co/settings/tokens (Fine-grained, "Make
  calls to Inference Providers" 권한 체크)
- Provider 활성화: https://huggingface.co/settings/inference-providers
- gated 모델(Llama 등) 쓰려면 모델 페이지에서 라이선스 약관 한 번 수락

임베딩은 sentence-transformers 로컬 모델로 처리한다 (첫 실행 때만 ~500MB
다운로드, 이후 오프라인 동작).


### 3. `.env` 만들기

```bash
cp .env.example .env
```

`.env` 를 열어서 본인 경로에 맞게 채운다.

**OpenAI 경로:**
```env
BACKEND=openai
OPENAI_API_KEY=sk-...
```

**Hugging Face 경로:**
```env
BACKEND=huggingface
HF_TOKEN=hf_...
```

(선택) 모델 바꾸려면 같은 파일에 `OPENAI_LLM_MODEL`, `HF_LLM_MODEL`,
`HF_EMBED_MODEL` 추가. 기본값은 `.env.example` 주석에 표시.


## 단계별 실행

| step | 무엇을 한다 | 실행법 | API 호출 |
|---|---|---|---|
| 1 | PDF 페이지별 로드 | `python step1_load_document.py` | 없음 |
| 2 | 텍스트 청크 분할 | `python step2_chunking.py` | 없음 |
| 3 | 임베딩 + FAISS 저장 | `python step3_embed_store.py` | 임베딩 |
| 4 | 유사도 검색 | `python step4_similarity_search.py` | 임베딩(질문 1회) |
| 5 | LLM 답변 생성 | `python step5_generate_answer.py` | LLM |
| 6 | 프롬프트 비교 (느슨 vs 엄격) | `python step6_optimize_prompt.py` | LLM x 4 |
| 7 | Streamlit 챗 UI 골격 | `streamlit run step7_chat_ui.py` | 없음 |
| 8 | 파일 업로드 위젯 | `streamlit run step8_file_upload.py` | 없음 |
| 9 | 전체 통합 챗봇 | `streamlit run step9_app_complete.py` | 임베딩+LLM |

step1~6 은 일반 파이썬 스크립트로 결과가 터미널에 찍힌다. step7~9 는
Streamlit 앱이라 브라우저로 띄운다 (보통 `http://localhost:8501`).


## 파일 구조

```
실습 자료/
├── README.md                 이 문서
├── COMMANDS.md               명령어 한눈 정리
├── requirements.txt          의존성
├── .env.example              키 템플릿
├── .gitignore                .env, my_faiss_index/ 등 제외
├── 출장 규정.pdf             샘플 PDF (3페이지)
├── models.py                 LLM/임베더 백엔드 스위치
├── step1_load_document.py
├── step2_chunking.py
├── step3_embed_store.py
├── step4_similarity_search.py
├── step5_generate_answer.py
├── step6_optimize_prompt.py
├── step7_chat_ui.py
├── step8_file_upload.py
└── step9_app_complete.py
```

`models.py` 가 LLM 과 임베더를 만드는 단일 지점이다. step3~9 는 여기서
`get_llm()`, `get_embeddings()` 만 호출하기 때문에 백엔드 교체에
영향받지 않는다.


## 자주 막히는 곳

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: langchain_community` | 가상환경 활성 안 됨, 또는 `pip install -r requirements.txt` 안 함 |
| `openai.AuthenticationError` 또는 401 | `.env` 의 `OPENAI_API_KEY` 가 비었거나 잘못된 키 |
| `Cannot find token` (HF 경로) | `.env` 의 `HF_TOKEN` 미설정, 또는 Inference Providers 권한 없음 |
| `FileNotFoundError: 출장 규정.pdf` | 스크립트를 폴더 밖에서 실행 중. `cd "실습 자료"` 후 실행 |
| 임베딩이 너무 느림 | step3 한 번 돌리고 나면 `my_faiss_index/` 에 저장돼서 step4~6 은 빠르다 |
| HF 첫 실행 때 500MB 다운로드 | 정상. sentence-transformers 가중치를 받는다. 이후엔 캐시 |
| `Port 8501 is already in use` | 다른 streamlit 이 떠 있다. `streamlit run ... --server.port 8502` 로 다른 포트 |


## 한 단계 위 — 메인 앱 보기

실습이 익숙해지면 이 repo 의 본 앱(`../app.py` 진입점 + `auth/`,
`data/`, `llm/`, `retrieval/` 등 모듈 패키지)을 읽어보면 좋다.
같은 RAG 흐름 위에 다음이 추가돼 있다.

- 멀티프로바이더 (OpenAI / Anthropic / Fireworks / HF / DashScope / vLLM)
- 하이브리드 검색 (BM25 + Dense + RRF + Cross-encoder rerank)
- HyDE / Multi-query / Contextual rewrite
- 멀티모달 PDF 페이지 이미지
- Supabase 영속 저장 + 사용자 인증
- 에이전트 워크플로 (이메일/보고서/요약/분석/비교)

실습 9단계가 그 모든 기능의 "최소 핵심" 이라고 생각하면 된다.
