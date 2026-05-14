"""
Step 7 — Streamlit 챗 UI 골격 (아직 RAG 없음)

목표: 대화형 인터페이스를 만들고 대화 기록을 화면에 누적 표시한다.
이 단계는 LLM 호출 없이 "AI 자리에 가짜 응답"을 넣어 UI 흐름만 익힌다.

실행:
    streamlit run step7_chat_ui.py
"""

import streamlit as st

st.title("실습 챗 UI (단계 7)")
st.caption("아직 AI 응답은 가짜입니다. 단계 9에서 RAG 연결.")

# 1. 대화 기록 저장소 — session_state는 사용자 새로고침해도 유지되는 dict
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 지금까지의 대화 내역을 화면에 출력
#    st.chat_message("user" | "assistant") 가 말풍선 UI를 그림
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. 입력창 — 사용자가 엔터 누르면 prompt 변수에 텍스트 들어옴
if prompt := st.chat_input("질문을 입력하세요..."):
    # 3-1. 사용자 발화 화면에 띄우고 기록
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3-2. AI 응답 — 단계 9에서 RAG로 교체
    fake_response = f"(가짜 응답) 받은 질문: '{prompt}'"
    with st.chat_message("assistant"):
        st.markdown(fake_response)
    st.session_state.messages.append({"role": "assistant", "content": fake_response})

# 사이드바에 메타 정보
with st.sidebar:
    st.subheader("정보")
    st.write(f"누적 대화: {len(st.session_state.messages)}개 메시지")
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()
