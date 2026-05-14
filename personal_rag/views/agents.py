"""Agents view — task picker + form + result panel for each agent workflow."""

import datetime

import streamlit as st

from ..agents.runner import run_agent_task
from ..agents.templates import AGENT_TASKS
from ..llm.chat import _show_llm_error
from ..llm.clients import _active_api_key
from ..ui.helpers import _section


def view_agents():
    _section(
        '에이전트',
        '내 문서를 토대로 이메일/보고서/요약/데이터 분석/비교 분석을 자동 생성합니다. '
        '각 작업은 retrieval로 근거를 가져온 뒤 작업별 전용 프롬프트로 LLM에 요청합니다.',
    )

    # Task picker
    task_keys = list(AGENT_TASKS.keys())
    if 'agent_task_key' not in st.session_state:
        st.session_state['agent_task_key'] = task_keys[0]
    selected = st.radio(
        '작업 선택',
        task_keys,
        index=task_keys.index(st.session_state.get('agent_task_key', task_keys[0])),
        format_func=lambda k: AGENT_TASKS[k]['label'],
        horizontal=True,
    )
    st.session_state['agent_task_key'] = selected
    task = AGENT_TASKS[selected]
    st.caption(task['description'])

    if task.get('requires_docs') and not st.session_state['docs']:
        st.warning(
            '이 작업은 문서가 필요합니다. **문서** 탭에서 먼저 파일을 업로드해 주세요.'
        )

    # Input form
    inputs = {}
    selected_doc_ids = None
    with st.form(f'agent_form_{selected}'):
        # Per-task document subset selector. Shown whenever the user has
        # 2+ documents loaded — every agent task uses retrieved chunks, so
        # the choice "which docs do you want this draft to lean on" is
        # always meaningful even when requires_docs=False (e.g. email).
        if len(st.session_state['docs']) >= 2:
            doc_id_to_name = {d['id']: d['name'] for d in st.session_state['docs']}
            all_ids = list(doc_id_to_name.keys())
            selected_doc_ids = st.multiselect(
                '참고할 문서',
                options=all_ids,
                default=all_ids,
                format_func=lambda did: doc_id_to_name[did],
                help='이 작업이 검색·인용할 문서를 선택합니다. 기본은 전체 문서. '
                '특정 문서만 골라 좁히면 응답이 더 정확해집니다. '
                '문서가 1개라면 선택지가 없으니 자동 사용됩니다.',
            )

        for f in task['fields']:
            label = f['label']
            placeholder = f.get('placeholder', '')
            if f['type'] == 'text':
                inputs[f['key']] = st.text_input(label, placeholder=placeholder)
            elif f['type'] == 'textarea':
                inputs[f['key']] = st.text_area(label, height=120, placeholder=placeholder)
            elif f['type'] == 'select':
                inputs[f['key']] = st.selectbox(label, f['options'])
        submitted = st.form_submit_button('실행', use_container_width=True, type='primary')

    if submitted:
        if (not st.session_state['model']
                or not _active_api_key()
                or not st.session_state['base_url']):
            provider = st.session_state.get('provider', 'Hugging Face Router')
            st.error(
                f'설정 탭에서 공급자/모델/{provider} 용 API 키를 먼저 확인해 주세요.'
            )
            st.stop()
        if task.get('requires_docs') and not st.session_state['docs']:
            st.stop()
        if (task.get('requires_docs') and selected_doc_ids is not None
                and len(selected_doc_ids) == 0):
            st.error('최소 한 개 이상의 문서를 선택해 주세요.')
            st.stop()

        try:
            full_text, reasoning, retrieved, elapsed = run_agent_task(
                selected, inputs, doc_ids_filter=selected_doc_ids,
            )
        except Exception as e:
            _show_llm_error(e)
            st.stop()

        st.write('')
        _section('결과', f'생성 시간: {elapsed:.1f}s')
        if reasoning:
            with st.expander('추론 과정', expanded=False):
                st.markdown(reasoning)
        st.markdown(full_text or '*(빈 응답)*')

        if full_text:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            ext = task.get('output_ext', 'md')
            mime = 'text/markdown' if ext == 'md' else 'text/plain'
            st.download_button(
                '다운로드',
                data=full_text,
                file_name=f"{selected}_{stamp}.{ext}",
                mime=mime,
                use_container_width=True,
                key=f'dl_agent_{selected}_{stamp}',
            )

        if retrieved:
            with st.expander(f'사용된 근거 ({len(retrieved)}개)', expanded=False):
                for j, r in enumerate(retrieved, start=1):
                    if r.get('source') == 'web':
                        from urllib.parse import urlparse
                        host = urlparse(r.get('url', '')).netloc or '웹'
                        st.markdown(f"**[{j}] 웹 · {host}** — {r.get('doc', '')[:80]}")
                        if r.get('url'):
                            st.markdown(f"<{r.get('url')}>")
                    else:
                        pages = r.get('pages') or []
                        page_str = ''
                        if pages:
                            page_str = (f' p.{pages[0]}' if len(pages) == 1
                                        else f' pp.{pages[0]}-{pages[-1]}')
                        st.markdown(
                            f"**[{j}] {r.get('doc', '')}{page_str}** · "
                            f"chunk {r.get('chunk_idx', 0)} · "
                            f"score {r.get('score', 0):.3f}"
                        )
                    st.text((r.get('text', '') or '')[:600])
                    st.divider()
