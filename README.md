# Personal RAG (REAL Lab)

내 문서 기반 질의응답 + 업무 자동화 챗봇.

OpenAI 호환 LLM (Hugging Face Router · OpenAI · DashScope · vLLM) · 하이브리드 검색 (Dense+BM25+RRF) · 재정렬 · HyDE/Multi-query · 멀티모달 PDF · 실시간 웹 검색 · 5종 에이전트 워크플로 (이메일 / 보고서 / 요약 / 분석 / 비교) · 세션별 영속 저장 · 인용 + 출처 추적.

## 구조

```
chatbot_demo/
├── app.py                       # Streamlit 단일 파일 (≈3,200 라인)
├── requirements.txt             # Cloud 친화적 최소 의존성
├── requirements-extras.txt      # 풀 기능 (docling / pymupdf / tavily)
├── .streamlit/
│   └── secrets.toml.example     # 키 템플릿 (실제 secrets.toml은 gitignored)
├── logo/real_logo.png           # 브랜드 로고
├── .data/                       # 벡터 인덱스·세션 메타 (gitignored)
└── logs/                        # 대화·에이전트 실행 로그 (gitignored)
```

## 로컬 실행

```bash
git clone <your-repo-url>
cd chatbot_demo
pip install -r requirements.txt
pip install -r requirements-extras.txt   # 선택: PDF 구조화·멀티모달

# 옵션 A: 프로젝트 루트에 .env
cp .streamlit/secrets.toml.example .env
# .env 안의 키를 실제 값으로 채우고 TOML 문법을 KEY=VALUE 로 바꿔주세요

# 옵션 B: Streamlit secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# 위 파일은 자동 gitignore — 안전합니다

streamlit run app.py
```

## API 토큰 발급 가이드

### Hugging Face 토큰 (`HF_TOKEN`)

**용도:** Hugging Face Inference Router 경유 모든 모델 호출 (Gemma 4, DeepSeek, Qwen3, Llama 등).

**발급 위치:** https://huggingface.co/settings/tokens → **New token** → **Fine-grained** 추천 (least privilege)

**필요한 권한 (fine-grained 기준):**

| 권한 | 필수 여부 | 설명 |
|---|---|---|
| **Make calls to Inference Providers** | ✓ 필수 | 라우터를 통한 모든 LLM 호출의 핵심 권한 |
| **Make calls to the serverless Inference API** | 권장 | 일부 모델이 구 Inference API 경로 사용 |
| **Read access to public repositories** | ✓ 자동 포함 | 임베딩 모델 / reranker 가중치 다운로드 |
| **Read access to selected gated repositories** | 선택 | Llama, Gemma 등 라이선스 수락 후 접근하는 모델용. 해당 모델 페이지에서 라이선스 수락 후 토큰의 gated 권한에 추가 |
| **Write 권한** | ✗ 불필요 | 우리 앱은 push/comment 안 함 |

**Classic 토큰을 쓴다면:** `read` scope 하나면 동작. 단 권한이 광범위해 보안상 fine-grained 권장.

### Inference Provider 활성화 (별도)

토큰 권한과는 별개로, https://huggingface.co/settings/inference-providers 에서 **사용하려는 provider**(Together AI · Fireworks · Cerebras · Hyperbolic 등)를 활성화해야 합니다. 무료 크레딧 있는 provider도 많음. 모델 페이지 우측 "Inference Providers" 박스에서 어떤 provider가 그 모델을 서빙하는지 확인 가능.

### OpenAI API 키 (`OPENAI_API_KEY`)

**발급:** https://platform.openai.com/api-keys → **Create new secret key**

**권한:** OpenAI는 키 단위 권한이 단순합니다. 프로젝트 단위 권한 분리 원하면 "Project key" 사용 (특정 프로젝트 모델만 호출 가능).

### DashScope (Qwen 공식) 키 (`DASHSCOPE_API_KEY`, 선택)

**발급:** Alibaba Cloud Model Studio (https://bailian.console.alibabacloud.com/) — 결제 등록 필요. Qwen 공식 API 쓸 때만.

### Tavily / Brave 키 (웹 검색용, 선택)

- **Tavily:** https://tavily.com — 가입 시 월 1000건 무료
- **Brave Search API:** https://brave.com/search/api/ — 무료 티어 2000건/월
- **DuckDuckGo는 키 불필요** (기본값)

---

## Streamlit Community Cloud 배포

### 사전 준비

1. GitHub repo 준비 (아래 "GitHub 푸시" 섹션 참고).
2. 해당 repo가 public 이거나, Streamlit Cloud에 GitHub 권한 부여.

### 배포 단계

1. https://share.streamlit.io 접속 → "New app"
2. Repo / branch / main file 선택:
   - **Main file path**: `app.py`
   - **Branch**: `main`
3. **Advanced settings → Secrets** 에 키 입력:
   ```toml
   HF_TOKEN = "hf_..."
   OPENAI_API_KEY = "sk-..."
   ```
   필요한 것만. `.streamlit/secrets.toml.example` 참고.
4. Deploy 클릭.

### 자원 한계 주의

Streamlit Community Cloud 무료 티어는 RAM이 제한적입니다 (~1 GB 안팎). 기본 설정의 일부는 자원을 많이 씁니다:

| 항목 | 메모리 | 권장 |
|---|---|---|
| 임베딩 모델 BGE-M3 | ~2.2 GB | **MiniLM 다국어** 로 변경 (~470 MB). 설정 → 검색 → 임베딩 모델 |
| Reranker (BGE-reranker-v2-m3) | ~580 MB | 자원 부족 시 OFF |
| Docling (PDF 구조 파싱) | 모델 다운로드 ~수 GB | `requirements.txt`에 없음 → 자동으로 pypdf 폴백 |
| 페이지 이미지 멀티모달 | pymupdf 필요 | 기본 OFF, requirements-extras 설치 시 사용 |

**Cloud 첫 사용 시 권장 흐름:**
1. 배포 → 사이드바 `설정` 탭 진입
2. 임베딩 모델을 `paraphrase-multilingual-MiniLM-L12-v2`로 변경
3. `정확도 우선 (재정렬 모델 사용)` 체크 해제
4. 문서 업로드 (txt/md/작은 PDF)
5. 질문 시작

## GitHub 푸시

처음이라면:

```bash
cd /home/inho/etri/chatbot_demo

# 새 git repo 초기화 (이 폴더만 독립 repo로)
git init
git add .
git status                # 트래킹 대상 확인 (logs/, .data/, .env 안 보여야 OK)
git commit -m "Initial commit: Personal RAG chatbot"

# GitHub에서 빈 repo를 만든 뒤 (예: yourname/personal-rag):
git branch -M main
git remote add origin git@github.com:yourname/personal-rag.git
git push -u origin main
```

이후 변경분:
```bash
git add -A && git commit -m "..." && git push
```

## 영속 데이터 위치 (배포 시 주의)

- `./.data/{user}/{embedder}/{doc_hash}/` — 임베딩 / 청크 / 페이지 이미지
- `./.data/{user}/sessions/{id}.json` — 대화 세션 메타
- `./logs/{user}/{session_id}.jsonl` — 사용자 + 어시스턴트 대화 기록 (한 줄 = 한 턴)
- `./logs/{user}/agents.jsonl` — 에이전트 작업 실행 기록
- `./logs/{user}/events.jsonl` — 로그인 / 문서 / 세션 / LLM 에러 이벤트

**Streamlit Cloud 의 파일시스템은 ephemeral 합니다** — 모든 컨테이너 재시작(redeploy, reboot, idle hibernation) 시 위의 로컬 파일들은 통째로 사라집니다. 이 때문에 본 앱은 **Supabase Postgres 영속 로깅** 을 기본 지원합니다.

## 영속 로깅 설정 (Supabase) — Cloud 배포 시 사실상 필수

설정 안 하면 사용자 활동이 컨테이너 재시작과 함께 증발합니다. 5분이면 끝납니다.

| 단계 | 작업 |
|---|---|
| 1 | https://supabase.com/dashboard 에서 가입 → **New project** (region 가까운 곳, free 500 MB 충분) |
| 2 | **Project Settings → API** → "Project URL" 과 "anon public" 키 복사 |
| 3 | **SQL Editor → New query** → 본 레포의 `db_schema.sql` 내용 붙여넣고 **Run** |
| 4 | Streamlit Cloud 앱 **Settings → Secrets** 에 추가:<br>`SUPABASE_URL = "https://xxx.supabase.co"`<br>`SUPABASE_KEY = "eyJ..."` |
| 5 | **Manage app → Reboot** |

이후 자동으로 모든 채팅 턴 / 에이전트 실행 / 이벤트가 Supabase 의 `chat_turns`·`agent_runs`·`events` 테이블에 INSERT 됩니다. Supabase 대시보드의 **Table Editor** 또는 **SQL Editor** 에서 바로 조회/내보내기 가능.

설정 안 하면 앱은 그대로 동작하되 로컬 JSONL 만 씁니다 (개발 / 단일 사용자 OK, Cloud 배포 비추).

### 회원가입 / 로그인 — Supabase users (선택, 다중사용자 시 추천)

여러 사용자가 같은 앱을 쓰는 경우 회원가입 + 로그인을 활성화하려면:

| 단계 | 작업 |
|---|---|
| 1 | Supabase **SQL Editor** → 본 레포의 `db_schema_users.sql` 내용을 붙여넣고 **Run** |
| 2 | (이미 SUPABASE_URL / SUPABASE_KEY 가 secrets 에 있다면 추가 작업 없음) |
| 3 | Manage app → Reboot |

작동 방식:
- 비밀번호는 Postgres 의 `pgcrypto` (bcrypt) 로 DB 안에서 해시 + 검증
- 앱은 `signup_user` / `login_user` RPC 만 호출 → password_hash 가 클라이언트에 절대 노출 안 됨
- 첫 화면이 "로그인 / 회원가입" 2탭으로 바뀌고 누구나 자가 가입 가능

**활성화 우선순위 (자동 분기):**

1. Supabase 가 연결돼 있으면 → Supabase users (회원가입 가능)
2. 1번 아니고 secrets 에 `[users]` 블록이 있으면 → 관리자가 등록한 사용자만 로그인
3. 둘 다 없고 Cloud 면 → 브라우저별 익명 UUID
4. 로컬이면 → `_local` 단일 사용자

### pgvector 추가 — 청크 임베딩도 영속화 (선택, 강력 추천)

위 기본 로깅에 더해 **문서 청크 임베딩까지** 영속화하려면 한 번 더 SQL 실행. 그러면 사용자가 올린 PDF 의 임베딩이 컨테이너 재시작 후에도 살아남고, pgvector 의 HNSW 인덱스로 더 큰 규모의 ANN 검색이 가능합니다.

| 단계 | 작업 |
|---|---|
| 1 | Supabase **SQL Editor** → 본 레포의 `db_schema_pgvector.sql` 내용 통째로 붙여넣고 **Run** |
| 2 | 앱에서 문서 업로드 — 캐시 탭 가서 `pgvector: 청크 임베딩 N/N 건 영속화 성공` 초록 메시지 확인 |

내부적으로 어떻게 동작하는지:
- `vector` 확장 활성화 + `public.doc_chunks` 테이블 생성 (MiniLM 384 / BGE-M3 1024 두 차원 모두 지원)
- HNSW 인덱스로 cosine k-NN 빠르게
- 인덱싱 시 자동 dual-write (로컬 numpy + pgvector 동시 저장)
- 검색은 아직 로컬 numpy 가 기본 (Phase 2b 에서 토글로 pgvector 검색 전환 예정)

## 보안 체크리스트

- [x] `.env` / `.streamlit/secrets.toml` 은 `.gitignore`에 포함됨
- [x] `logs/`, `.data/` 도 gitignored (사용자 데이터·키 포함될 수 있음)
- [x] API 키 코드에 하드코딩 없음 (`grep -rE "sk-|hf_|tvly-"` 결과 비어있음)
- [x] `st.secrets` → `os.environ` 브리지로 secrets 안전 전달

## 라이선스 & 크레딧

- LLM endpoints: 각 사용자 약관 적용 (OpenAI · Hugging Face · DashScope 등).
- 모델: BAAI/bge-m3, BAAI/bge-reranker-v2-m3, sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- 파싱: Docling (선택), PyMuPDF (선택), pypdf
- 검색: rank_bm25, ddgs
