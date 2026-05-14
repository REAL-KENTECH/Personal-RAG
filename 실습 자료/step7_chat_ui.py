"""
Step 7. Streamlit 으로 대화 UI 만들기.

여기까진 CLI 였다. 사용자가 실제로 쓸 인터페이스를 챗 형태로 띄워본다.
이 단계는 RAG 가 연결되어 있지 않아서 응답은 가짜로 돌려준다. UI 흐름을
먼저 익히는 게 목적이다.

    streamlit run step7_chat_ui.py
"""

import streamlit as st

from ui_styles import apply_styles, brand, sidebar_section


st.set_page_config(page_title="실습 챗 UI", layout="wide", initial_sidebar_state="expanded")
apply_styles()


with st.sidebar:
    brand("실습 챗봇", "step 7 — UI 골격")
    sidebar_section("상태")
    msg_count = len(st.session_state.get("messages", []))
    st.caption(f"누적 메시지: {msg_count}")

    sidebar_section("작업")
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.markdown(
    '<div class="section-title">실습 챗 UI</div>'
    '<div class="section-sub">RAG 는 step 9 에서 연결. 지금은 가짜 응답만 돌려준다.</div>',
    unsafe_allow_html=True,
)


# session_state 에 대화 기록을 누적. Streamlit 은 매 인터랙션마다 스크립트를
# 위에서 아래로 다시 돌리는데 session_state 만 그 사이에 유지된다.
if "messages" not in st.session_state:
    st.session_state.messages = []


# 지금까지 누적된 대화를 화면에 그린다
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 새 입력 — walrus(:=) 로 변수 할당하면서 None 체크
if prompt := st.chat_input("질문을 입력하세요"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # step9 에서는 이 자리에 RAG 답변이 들어간다.
    fake = f"(가짜 응답) 받은 질문: {prompt!r}"
    with st.chat_message("assistant"):
        st.markdown(fake)
    st.session_state.messages.append({"role": "assistant", "content": fake})
