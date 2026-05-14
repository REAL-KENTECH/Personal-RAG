# 명령어 모음

## 처음 한 번만

```bash
cd "/home/inho20/Personal-RAG/실습 자료"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env 열어서 OPENAI_API_KEY 또는 HF_TOKEN 채우기
```

윈도우는 `source .venv/bin/activate` 대신 `.venv\Scripts\Activate.ps1`.


## 매일 시작할 때 (새 터미널 열면)

```bash
cd "/home/inho20/Personal-RAG/실습 자료"
source .venv/bin/activate
```


## step 실행

```bash
python step1_load_document.py
python step2_chunking.py
python step3_embed_store.py
python step4_similarity_search.py
python step5_generate_answer.py
python step6_optimize_prompt.py

streamlit run step7_chat_ui.py
streamlit run step8_file_upload.py
streamlit run step9_app_complete.py
```

Streamlit 끝낼 때는 터미널에서 Ctrl+C.


## 백엔드 바꾸기

`.env` 의 `BACKEND` 한 줄 수정 후:

```bash
rm -rf my_faiss_index
python step3_embed_store.py
```


## 가끔 쓰는 것

```bash
# 포트가 이미 사용 중이면 다른 포트로
streamlit run step9_app_complete.py --server.port 8502

# 어떤 패키지 깔렸는지 확인
pip list | grep -iE "langchain|streamlit|faiss"

# 메인 앱 (상위 폴더, 옵션)
cd ..
./run.sh
```
