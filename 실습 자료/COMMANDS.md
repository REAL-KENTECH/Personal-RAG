# 명령어 정리 (수강생용)

순서대로 그대로 따라치면 된다. 막히면 [README.md](README.md) 의
"자주 막히는 곳" 표 확인.


## 0. 폴더 진입

```bash
cd "/home/inho20/Personal-RAG/실습 자료"
```

다른 경로에서 받았으면 그 경로로. 한글 + 공백 포함이라 반드시 큰따옴표.


## 1. 가상환경 + 의존성 (한 번만)

**Linux / Mac**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

설치 끝나면 프롬프트 앞에 `(.venv)` 가 보여야 한다. 안 보이면 activate 가
안 된 것.


## 2. `.env` 만들고 키 채워넣기 (한 번만)

```bash
cp .env.example .env
```

그 다음 `.env` 파일을 편집기로 열어서 하나 선택:

**OpenAI 경로**
```env
BACKEND=openai
OPENAI_API_KEY=sk-여기에본인키
```

**Hugging Face 경로**
```env
BACKEND=huggingface
HF_TOKEN=hf_여기에본인키
```

키 발급 링크:
- OpenAI: https://platform.openai.com/api-keys
- Hugging Face: https://huggingface.co/settings/tokens
  (Fine-grained > "Make calls to Inference Providers" 체크)


## 3. step 1~9 실행

매번 새 터미널을 열었다면 먼저 `source .venv/bin/activate` 부터.

```bash
# 외부 호출 없는 단계 — 빠르고 무료
python step1_load_document.py
python step2_chunking.py

# 임베딩 단계 — 한 번 돌리면 my_faiss_index/ 에 저장
python step3_embed_store.py

# step3 산출물 재사용 — 빠름
python step4_similarity_search.py
python step5_generate_answer.py
python step6_optimize_prompt.py

# Streamlit 앱 — 브라우저로 자동 이동
streamlit run step7_chat_ui.py
streamlit run step8_file_upload.py
streamlit run step9_app_complete.py
```

Streamlit 명령은 실행되면 보통 `http://localhost:8501` 이 열린다. 종료는
터미널에서 `Ctrl+C`.


## 4. 백엔드 바꾸기

`.env` 의 `BACKEND` 한 줄만 수정한 다음 다시 실행. step 코드는 그대로.

```env
BACKEND=openai          # → BACKEND=huggingface
```

이미 임베딩한 `my_faiss_index/` 는 백엔드별로 차원이 달라서 그대로 못
쓴다. 백엔드를 바꿨다면 한 번 삭제:

```bash
rm -rf my_faiss_index
python step3_embed_store.py
```


## 5. 자주 쓰는 보조 명령

**가상환경 다시 켜기**
```bash
source .venv/bin/activate
```

**파이썬 / 패키지 버전 확인**
```bash
python --version
pip list | grep -iE "langchain|streamlit|faiss"
```

**FAISS 인덱스만 지우고 다시 만들기**
```bash
rm -rf my_faiss_index
python step3_embed_store.py
```

**Streamlit 포트 충돌 시 다른 포트로**
```bash
streamlit run step9_app_complete.py --server.port 8502
```

**Streamlit 캐시 비우기 (`@st.cache_resource` 가 꼬였을 때)**

브라우저 우측 상단 햄버거 메뉴 → `Clear cache` → `Rerun`.


## 6. 메인 챗봇 (옵션)

상위 폴더에 있는 본 프로젝트 메인 앱을 돌려보고 싶으면:

```bash
cd ..               # /home/inho20/Personal-RAG
./run.sh            # 또는 streamlit run app.py
```

실습 9단계가 그 메인 앱의 "최소 핵심" 이라고 보면 된다.
