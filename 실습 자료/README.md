# RAG 실습

PDF 하나를 챗봇으로 만든다. 단계는 9개. step1 부터 순서대로 따라가면 된다.

OpenAI 키 쓰는 게 제일 편하지만 무료로 돌리려면 Hugging Face 토큰 받아서 `BACKEND=huggingface` 로만 바꿔도 같은 코드가 그대로 동작한다.

명령어만 빨리 보고 싶으면 [COMMANDS.md](COMMANDS.md).


## 한 번만 해두는 셋업

```bash
cd "/home/inho20/Personal-RAG/실습 자료"
python3 -m venv .venv
source .venv/bin/activate          # 윈도우: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

프롬프트 앞에 `(.venv)` 가 보이면 활성된 거다.

키는 둘 중 하나 받아둔다.

- OpenAI: https://platform.openai.com/api-keys 에서 키 발급 + 결제수단 등록. 실습 전체 합쳐도 $0.10 정도 쓴다.
- Hugging Face: https://huggingface.co/settings/tokens 에서 Fine-grained 토큰. "Make calls to Inference Providers" 권한 체크. 임베딩은 로컬 sentence-transformers 로 돌아가서 토큰 호출은 LLM 한정.

받았으면:

```bash
cp .env.example .env
```

`.env` 열어서 두 줄 채운다.

OpenAI 경우:
```
BACKEND=openai
OPENAI_API_KEY=sk-...
```

Hugging Face 경우:
```
BACKEND=huggingface
HF_TOKEN=hf_...
```


## 단계

step1~6 은 일반 파이썬 스크립트, step7~9 는 Streamlit.

| step | 무얼 한다 | 실행 |
|---|---|---|
| 1 | PDF 페이지별 로드 | `python step1_load_document.py` |
| 2 | 텍스트를 청크로 분할 | `python step2_chunking.py` |
| 3 | 임베딩 + FAISS 저장 | `python step3_embed_store.py` |
| 4 | 유사도 검색 | `python step4_similarity_search.py` |
| 5 | LLM 답변 생성 | `python step5_generate_answer.py` |
| 6 | 프롬프트 비교 | `python step6_optimize_prompt.py` |
| 7 | Streamlit 채팅 UI | `streamlit run step7_chat_ui.py` |
| 8 | 파일 업로드 위젯 | `streamlit run step8_file_upload.py` |
| 9 | 합친 최종 챗봇 | `streamlit run step9_app_complete.py` |

step3 을 한 번 돌리면 `my_faiss_index/` 에 인덱스가 저장돼서 step4~6 은 그걸 재사용한다. 그래서 step3 만 비싸고 나머지는 거의 공짜.

Streamlit 은 보통 http://localhost:8501 로 자동으로 열린다. 종료는 터미널에서 Ctrl+C.


## 막힐 만한 곳

**`ModuleNotFoundError: langchain_community`** — 가상환경이 안 켜졌거나 `pip install -r requirements.txt` 를 안 했다. 프롬프트 앞에 `(.venv)` 가 있는지 본다.

**`openai.AuthenticationError` 또는 401** — `.env` 의 키가 비었거나 잘못. 키 끝에 공백이 들어가는 일이 흔하다.

**`FileNotFoundError: 출장 규정.pdf`** — 실습 폴더 밖에서 실행한 경우. `cd "실습 자료"` 후 실행.

**임베딩이 너무 느림** — 정상이다. step3 한 번만 무겁다. step4~6 은 인덱스 재사용해서 빠르다.

**`Port 8501 is already in use`** — 다른 Streamlit 이 떠 있다. 끄거나 `--server.port 8502` 같이 다른 포트로.

**HF 첫 실행 때 ~500MB 다운로드** — 임베더 가중치(sentence-transformers). 한 번 받으면 캐시된다.


## 백엔드 바꿀 때

`.env` 의 `BACKEND` 한 줄만 바꾸면 step 코드는 그대로 돈다. 단 OpenAI 임베딩(1536차원)과 HF 임베딩(384/1024차원)이 다르기 때문에 `my_faiss_index/` 는 못 섞어 쓴다. 백엔드 바꾼 직후엔 한 번 지우고 다시:

```bash
rm -rf my_faiss_index
python step3_embed_store.py
```


## 다 끝났으면

상위 폴더의 메인 앱(`../app.py`) 을 열어봐도 좋다. 같은 RAG 흐름인데 멀티프로바이더, 하이브리드 검색, 재정렬, HyDE, 멀티모달, Supabase 영속화, 에이전트 워크플로가 다 붙어있다. 이 9단계가 그 앱의 최소 골격이다.
