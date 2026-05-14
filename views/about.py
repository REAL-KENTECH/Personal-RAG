"""About view — brand lockup + architecture explainer."""

import streamlit as st

from branding import LOGO_URI


def view_about():
    if LOGO_URI:
        st.markdown(
            f'<div style="text-align:center; margin-bottom:18px;">'
            f'<img src="{LOGO_URI}" '
            f'style="width:100%; max-width:280px; height:auto; display:block; margin:0 auto 8px auto;" />'
            f'<div style="font-size:22px; font-weight:700; letter-spacing:-0.01em;">Personal RAG</div>'
            f'<div style="font-size:13px; color:rgba(128,128,128,0.95);">내 문서 기반 질의응답 시스템</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        내 문서를 기반으로 답하는 개인용 RAG (Retrieval-Augmented Generation) 시스템.

        #### 검색 파이프라인
        1. 문서 파싱 — PDF는 **Docling**으로 표·헤딩·리스트 구조를 보존하며 마크다운으로 변환, 페이지 메타를 청크별로 기록.
        2. 청킹 — 문단 단위로 묶고 size 한도를 넘으면 슬라이딩 윈도우로 분할.
        3. 임베딩 — **BGE-M3** (다국어 SOTA) 또는 MiniLM multilingual.
        4. 하이브리드 검색 — Dense 코사인 + BM25 의 **Reciprocal Rank Fusion**.
        5. (옵션) **HyDE** / **Multi-query** / **Contextual rewrite** 로 쿼리 보강.
        6. (옵션) Cross-encoder **bge-reranker-v2-m3** 로 top_n → top_k 재정렬.
        7. (옵션) PDF 페이지 이미지를 **멀티모달 LLM** 에 함께 전달.
        8. LLM 응답을 토큰 단위로 스트리밍, reasoning_content 와 본문을 분리.
        9. 답변 안의 `[N]` 인용 마커를 파싱하여 청크별로 출처 표시.

        #### 저장
        - `./.data/{embedder}/{doc_hash}/` — 임베더별 격리.
        - `meta.json` (청크, 페이지 메타, 원본 텍스트) + `embeddings.npy` + `pages/{page}.png`.
        - 같은 파일·청크 설정으로 다시 업로드 시 캐시에서 즉시 복원.
        - 사용자별 환경설정 (API 키, 모델, 검색 설정 등) 영속 저장: `./.data/{user}/preferences.json`. Streamlit Cloud의 idle 재연결 시 메모리가 초기화되더라도 다음 접속에서 자동 복원됩니다. (Cloud 컨테이너가 재시작되면 사라지므로 영구 보존이 필요하면 Streamlit Secrets 사용 권장.)
        - 대화 세션 메타: `./.data/{user}/sessions/{id}.json` (사이드바 대화 목록의 원천).
        - **대화 로그**: `./logs/{user}/{session_id}.jsonl` — 세션별로 한 파일, 한 줄당 한 턴. 분석·DB 친화적 구조.
          필드: session_id, turn_index, timestamp, user_message, assistant_message, reasoning, model/provider, elapsed_seconds, retrieved (rank·source·score·page·url), citation_numbers_used, query_variants, settings_snapshot (rerank/HyDE/multi-query/per-doc 등 모든 설정 스냅샷).
        - **에이전트 실행 로그**: `./logs/{user}/agents.jsonl` — 에이전트 워크플로(이메일/보고서/요약/분석/비교)의 실행 기록. task, inputs, output, retrieved, model, elapsed_seconds.
        - **이벤트 로그**: `./logs/{user}/events.jsonl` — 로그인/로그아웃, 문서 업로드·삭제, 세션 삭제, LLM 호출 실패 등 비-턴 이벤트. 각 줄에 event_type, timestamp, user_id, payload.
          pandas: `pd.concat([pd.read_json(f, lines=True) for f in glob.glob('logs/*/*.jsonl')])`.
          Postgres: `\\copy turns FROM '...' WITH (FORMAT json)` 또는 batched insert.

        #### 지원 LLM 엔드포인트
        Hugging Face Inference Router · OpenAI · Alibaba DashScope · vLLM / 로컬 OpenAI-호환 서버.

        #### 사용 라이브러리
        Streamlit, OpenAI SDK, sentence-transformers, rank_bm25, Docling, PyMuPDF, pypdf, ddgs.
        """
    )
