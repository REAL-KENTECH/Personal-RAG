# Personal RAG (REAL Lab)

내 문서 기반 질의응답 + 업무 자동화 챗봇.

OpenAI 호환 LLM 멀티프로바이더 (OpenAI · Anthropic Claude · Fireworks AI · Hugging Face Router · DashScope · vLLM) · 하이브리드 검색 (Dense + BM25 + RRF) · Cross-encoder 재정렬 · HyDE / Multi-query / Contextual rewrite · 멀티모달 PDF 페이지 이미지 · 실시간 웹 검색 (DDG / Tavily / Brave) · 5종 에이전트 워크플로 (이메일 / 보고서 / 요약 / 분석 / 비교) · Supabase 영속 로깅 + pgvector 검색 · 사용자별 인증 · 인용 + 출처 추적.

## 빠른 시작

```bash
git clone https://github.com/REAL-KENTECH/Personal-RAG.git
cd Personal-RAG

# 가상환경 + 의존성 (자동화 스크립트 제공)
bash setup.sh

# 실행
./run.sh                # 기본: 127.0.0.1:8501
./run.sh --public       # 0.0.0.0:8501 (네트워크/외부 노출)
./run.sh --bg           # 백그라운드(nohup)
# 또는 직접:
streamlit run app.py
```

수동 설치라면:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-extras.txt   # 선택: PDF 구조화·멀티모달
cp .streamlit/secrets.toml.example .streamlit/secrets.toml  # 키 입력
streamlit run app.py
```

## 프로젝트 구조

진입점 한 파일 + 기능별 모듈 패키지. 처음 보는 사람은 `app.py`(119줄)부터 읽고 → 각 폴더의 `__init__.py` docstring으로 영역 파악:

```
Personal-RAG/
├── app.py                          # 진입점 (119줄) — 부트 시퀀스 + 뷰 라우터
├── config.py                       # 정적 상수 (프로바이더, 모델, 경로, CSS)
├── state.py                        # _init_state — 세션 기본값 60+
├── branding.py                     # 로고 base64 + favicon
│
├── auth/                           # 인증·사용자·prefs
│   ├── supabase_io.py              # Supabase client + RPC (signup/login/prefs)
│   ├── users.py                    # 사용자 디렉토리, _log_event, _auth_gate
│   └── prefs.py                    # _PERSIST_KEYS + load/save
│
├── data/                           # 영속화
│   ├── storage.py                  # 디스크 vector store + pgvector dual-write
│   └── sessions.py                 # 대화 세션 save/load/list/rename/delete
│
├── processing/                     # 문서 처리 파이프라인
│   ├── parsing.py                  # PDF (Docling/pypdf/OCR) · DOCX · CSV · HWPX
│   ├── chunking.py                 # 문단/문장 단위 (한국어 종결어 포함)
│   └── ingestion.py                # parse → chunk → embed → save → pgvector
│
├── retrieval/                      # 검색 파이프라인
│   ├── search.py                   # BM25 · dense (numpy + pgvector) · RRF · rerank
│   ├── expansion.py                # contextual rewrite · multi-query · HyDE
│   ├── web.py                      # DuckDuckGo / Tavily / Brave 어댑터
│   └── pipeline.py                 # retrieve_local + retrieve 디스패처
│
├── llm/                            # LLM 호출
│   ├── clients.py                  # OpenAI 클라(캐시), 임베더/리랭커 로더
│   ├── params.py                   # 프로바이더 판별, _build_completion_params
│   └── chat.py                     # build_messages, agentic loop, stream, handle_chat_turn
│
├── ui/                             # 공통 UI
│   ├── helpers.py                  # _chip, _section, _empty
│   ├── widgets.py                  # model_picker
│   └── sidebar.py                  # render_sidebar() — 브랜드/네비/세션/상태/유저
│
├── agents/                         # 업무 자동화 에이전트
│   ├── templates.py                # 5종 프롬프트 빌더 + AGENT_TASKS 레지스트리
│   └── runner.py                   # run_agent_task, 감사 로그
│
├── views/                          # 사이드바 메뉴별 페이지
│   ├── chat.py · docs.py · settings.py · cache.py · agents.py · about.py
│
├── requirements.txt                # Cloud 친화적 최소 의존성
├── requirements-extras.txt         # docling / pymupdf / tavily 등
├── packages.txt                    # Streamlit Cloud apt 패키지 (tesseract OCR)
├── db_schema*.sql                  # Supabase 스키마 (logging / users / prefs / pgvector / sessions)
├── setup.sh / run.sh               # 신규 서버 1회 셋업 + 일상 실행
├── install_systemd.sh              # 상시 데몬 등록 (백그라운드)
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example        # 실제 secrets.toml은 gitignored
├── logo/real_logo.png
├── .data/                          # 벡터 인덱스·세션 메타 (gitignored)
└── logs/                           # 대화·에이전트·이벤트 JSONL (gitignored)
```

데이터 플로우 요약:
```
유저 입력 → handle_chat_turn (llm/chat.py)
  → retrieve (retrieval/pipeline.py)
       └─ expand_queries → BM25 ∪ dense → RRF → rerank → per-doc balance
  → build_messages (system + history + context + 이미지)
  → 프로바이더 호출 (llm/params.py가 모델별 호환 처리)
  → render_assistant + 인용 expander + 영속화 (data/sessions.py)
```

## API 토큰 발급 가이드

### Hugging Face (`HF_TOKEN`)

HF Inference Router 경유 모든 모델 호출 (Gemma, DeepSeek, Qwen3, Llama 등).

발급: https://huggingface.co/settings/tokens → **New token → Fine-grained**

필요 권한:

| 권한 | 필수 | 설명 |
|---|---|---|
| **Make calls to Inference Providers** | ✓ | 라우터를 통한 LLM 호출 핵심 |
| Make calls to the serverless Inference API | 권장 | 일부 모델은 구 경로 사용 |
| Read access to public repositories | ✓ (자동) | 임베더/리랭커 가중치 다운로드 |
| Read access to selected gated repositories | 선택 | Llama·Gemma 등 license-gated 모델용 |

토큰 권한과 별개로 https://huggingface.co/settings/inference-providers 에서 사용하려는 provider (Together / Fireworks / Cerebras / Hyperbolic) 활성화 필요.

### OpenAI (`OPENAI_API_KEY`)

https://platform.openai.com/api-keys → Create new secret key.

지원 모델: gpt-5 / 5.x, gpt-4.1, gpt-4o, o3, o4-mini 등. *주의: `gpt-5-pro`, `o1-pro` 같은 Responses-API 전용 모델은 본 앱과 호환 안 됨 (자동 친화 모델로 swap 됨)*.

### Anthropic Claude (`ANTHROPIC_API_KEY`)

https://console.anthropic.com/settings/keys → Create Key.

지원 모델: claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5 등 (Claude 4.x). OpenAI-호환 엔드포인트 사용. *주의: Anthropic은 `top_p`, `temperature`, `presence_penalty`를 모델이 무시하거나 거부함 — 코드에서 자동 strip*.

### Fireworks AI (`FIREWORKS_API_KEY`)

https://fireworks.ai/account/api-keys.

오픈모델 빠른 추론 (Llama 3.3 70B, Qwen 2.5 72B, DeepSeek V3 등). 모델 ID는 항상 `accounts/fireworks/models/<name>` 풀 경로.

### DashScope / Qwen (`DASHSCOPE_API_KEY`, 선택)

Alibaba Cloud Model Studio (https://bailian.console.alibabacloud.com/) — 결제 등록 필요. Qwen 공식 API 쓸 때만.

### 웹 검색 (선택)

- **DuckDuckGo** — 키 불필요 (기본값)
- **Tavily** (`TAVILY_API_KEY`) — https://tavily.com, 월 1000건 무료
- **Brave Search** (`BRAVE_API_KEY`) — https://brave.com/search/api/, 무료 2000건/월

## 화면 구성 (사이드바 메뉴 6개)

| 메뉴 | 역할 | 모듈 |
|---|---|---|
| **대화** | 메시지 입력, 파일 inline 첨부, 답변 + 인용 expander, 모델 빠른 전환 | `views/chat.py` |
| **문서** | 업로드, 청킹·임베딩 진행 상황, 인덱스 관리, 검색 미리보기 | `views/docs.py` |
| **업무 도구(에이전트)** | 5종 워크플로 (이메일/보고서/요약/분석/비교) | `views/agents.py` |
| **설정** | 공급자/모델/API 키, 검색 파이프라인, 웹 검색, 응답/멀티모달, 고급 | `views/settings.py` |
| **캐시** | Supabase 영속 상태, 로컬 디스크 사용량, HF 모델 캐시 관리 | `views/cache.py` |
| **소개** | 검색 파이프라인 설명, 저장 구조 문서 | `views/about.py` |

## Streamlit Community Cloud 배포

1. https://share.streamlit.io → New app
2. Repo / branch / main file 선택:
   - **Repository**: `REAL-KENTECH/Personal-RAG` (또는 본인 fork)
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. **Advanced settings → Secrets** 에 키 입력:
   ```toml
   HF_TOKEN = "hf_..."
   OPENAI_API_KEY = "sk-..."
   ANTHROPIC_API_KEY = "sk-ant-..."        # 선택
   FIREWORKS_API_KEY = "fw_..."            # 선택
   SUPABASE_URL = "https://xxx.supabase.co"  # 영속 로깅용
   SUPABASE_KEY = "eyJ..."                   # anon public key
   ```
4. Deploy.

### Cloud 자원 한계

무료 티어 RAM ~1 GB. 기본 설정 일부는 무거움:

| 항목 | 메모리 | 권장 |
|---|---|---|
| BGE-M3 임베더 | ~2.2 GB | **MiniLM 다국어** (~470 MB) — 설정 → RAG 검색 → 임베딩 모델 |
| Reranker (bge-reranker-v2-m3) | ~580 MB | 자원 부족 시 OFF |
| Docling PDF 파싱 | 수 GB | `requirements.txt`에 없음 — 자동으로 pypdf 폴백 |
| 페이지 이미지 멀티모달 | pymupdf 필요 | 기본 OFF; 비전 모델 쓸 때만 활성화 |

Cloud 첫 접속 시:
1. 사이드바 **설정** 탭 → **RAG 검색** → 임베딩 모델을 `MiniLM`로 변경
2. **재정렬 모델** 체크 해제
3. 문서 업로드 → 질문

## 영속 데이터 위치

```
.data/{user_id}/{embedder}/{doc_hash}/        # 청크 + 임베딩 + 페이지 이미지
.data/{user_id}/sessions/{session_id}.json    # 대화 세션 메타
.data/{user_id}/preferences.json              # API 키, 모델, 검색 설정
logs/{user_id}/{session_id}.jsonl             # 대화 기록 (1줄 = 1턴)
logs/{user_id}/agents.jsonl                   # 에이전트 실행 기록
logs/{user_id}/events.jsonl                   # 로그인/문서/세션/LLM 에러 이벤트
```

**Streamlit Cloud는 파일시스템이 ephemeral** — redeploy / reboot / idle hibernation 시 위 파일이 전부 사라짐. 그래서 **Supabase 영속 로깅을 사실상 필수**로 권장.

## Supabase 영속 로깅 (Cloud 배포 시 필수)

5분 만에 설정됨. 안 하면 사용자 활동이 컨테이너 재시작과 함께 증발.

| 단계 | 작업 |
|---|---|
| 1 | https://supabase.com/dashboard → **New project** (region 가까운 곳, free 500 MB 충분) |
| 2 | **Project Settings → API** → Project URL + anon public key 복사 |
| 3 | **SQL Editor → New query** → 본 레포의 `db_schema.sql` 붙여넣고 **Run** |
| 4 | Streamlit Cloud 앱 **Settings → Secrets** 에 `SUPABASE_URL` + `SUPABASE_KEY` 추가 |
| 5 | **Manage app → Reboot** |

이후 모든 채팅 턴 / 에이전트 실행 / 이벤트가 `chat_turns` · `agent_runs` · `events` 테이블에 자동 INSERT. Supabase 대시보드 **Table Editor / SQL Editor** 에서 바로 조회·내보내기.

### 옵션: 회원가입 + 로그인 (멀티유저용)

| 단계 | 작업 |
|---|---|
| 1 | SQL Editor → `db_schema_users.sql` 실행 |
| 2 | (SUPABASE_URL/KEY가 이미 있으면 추가 작업 없음) |
| 3 | Reboot |

비밀번호는 Postgres `pgcrypto` (bcrypt)로 DB 안에서 해시·검증. 앱은 `signup_user` / `login_user` RPC만 호출 → password가 클라이언트에 절대 노출되지 않음. 첫 화면이 "로그인 / 회원가입" 두 탭으로 바뀜.

### 옵션: 대화 기록 멀티 디바이스 동기화

| 단계 | 작업 |
|---|---|
| 1 | SQL Editor → `db_schema_sessions.sql` 실행 |
| 2 | 다음 로그인부터 사이드바에 이전 모든 세션 자동 표시 |

`chat_turns` 테이블에 누적된 모든 턴을 `session_id`로 그룹화 (`list_user_sessions` RPC). 사이드바 클릭 시 그 세션의 모든 턴을 fetch → 시간순으로 session_state에 그대로 적용. 로컬 디스크 캐시도 함께 갱신.

### 옵션: API 키 / 설정 자동 동기화

| 단계 | 작업 |
|---|---|
| 1 | SQL Editor → `db_schema_preferences.sql` 실행 |
| 2 | (자동 감지) 로그인 후 설정 변경 → `user_preferences` 테이블에 upsert |
| 3 | 컨테이너 재시작 / 다른 기기 로그인 → 자동 복원 |

`get_prefs` / `set_prefs` RPC 2개만 anon에 노출 — 테이블 직접 접근 불가.

### 옵션: pgvector 청크 임베딩 영속화 (강력 추천)

문서 청크 임베딩까지 Supabase에 저장하면 컨테이너 재시작 후에도 인덱스가 살아남고, pgvector HNSW로 대규모 ANN 검색 가능.

| 단계 | 작업 |
|---|---|
| 1 | SQL Editor → `db_schema_pgvector.sql` 실행 |
| 2 | 문서 업로드 → 캐시 탭에서 `pgvector: 청크 임베딩 N/N 영속화 성공` 확인 |
| 3 | (선택) 설정 → RAG 검색 → **pgvector 의미 검색** 체크 → dense 검색을 Supabase 측 SQL RPC로 라우팅 |

내부 동작:
- `vector` 확장 활성화 + `public.doc_chunks` 테이블 (MiniLM 384 / BGE-M3 1024 두 차원 모두 지원)
- HNSW 인덱스로 cosine k-NN
- 인덱싱 시 dual-write (로컬 numpy + pgvector 동시 저장)
- `use_pgvector_search` ON이면 dense 검색이 `match_chunks_*` RPC 호출 → 실패 시 자동 로컬 폴백

### 자동 분기 우선순위

1. Supabase 연결됨 → Supabase users (회원가입 가능)
2. `secrets.[users]` 블록 있음 → 관리자 등록 사용자만 로그인
3. Cloud + 둘 다 없음 → 브라우저별 익명 UUID
4. 로컬 → `_local` 단일 사용자

## 시스템 데몬 등록 (선택)

상시 백그라운드 실행 / 부팅 시 자동 시작:

```bash
sudo ./install_systemd.sh           # systemd 서비스 등록 + 시작
sudo systemctl status personal-rag  # 상태 확인
sudo systemctl restart personal-rag # 재시작
sudo ./uninstall_systemd.sh         # 제거
```

## 보안 체크리스트

- [x] `.env` / `.streamlit/secrets.toml` 은 `.gitignore`에 포함
- [x] `logs/` / `.data/` gitignored (사용자 데이터·로그 포함)
- [x] API 키 코드 하드코딩 없음
- [x] `st.secrets` → `os.environ` 브리지로 secrets 안전 전달
- [x] 비밀번호는 Postgres bcrypt (pgcrypto) — 클라이언트 노출 없음
- [x] 사용자별 데이터 디렉토리 격리 (`.data/{user_id}/`, `logs/{user_id}/`)

## 라이선스 & 크레딧

- **LLM 엔드포인트**: 각 사용자 약관 적용 (OpenAI · Anthropic · HF Inference Providers · Fireworks · DashScope 등)
- **임베더 / 리랭커**: BAAI/bge-m3, BAAI/bge-reranker-v2-m3, sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **파싱**: Docling (선택), PyMuPDF (선택), pypdf, python-docx, Tesseract OCR
- **검색**: rank_bm25, ddgs (DuckDuckGo)
- **데이터**: Supabase (Postgres + pgvector + pgcrypto)
