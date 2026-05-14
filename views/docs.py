"""Documents view — upload, list, delete, reindex, search preview."""

import streamlit as st

from llm.chat import _citation_body, _citation_summary
from processing.ingestion import ingest_files, reindex_all, remove_doc
from retrieval.pipeline import retrieve_local
from ui.helpers import _empty, _section


def view_docs():
    _section(
        '문서 업로드',
        '.txt / .md / .pdf 파일을 업로드하면 자동으로 청킹·임베딩되어 디스크에 영속 저장됩니다. '
        '같은 파일을 다시 올려도 내용 해시가 일치하면 즉시 캐시에서 복원됩니다.',
    )
    uploaded = st.file_uploader(
        ' ',
        type=['txt', 'md', 'pdf', 'docx', 'csv', 'hwpx'],
        accept_multiple_files=True,
        label_visibility='collapsed',
    )
    if uploaded:
        # ingest_files() shows its own st.status with per-batch progress.
        added = ingest_files(uploaded)
        if added > 0:
            st.success(f'{added}개 새 문서 인덱싱 완료')
            st.rerun()

    st.write('')
    _section('인덱싱된 문서')

    if not st.session_state['docs']:
        _empty('아직 업로드된 문서가 없습니다. 위 영역에 파일을 끌어다 놓으세요.')
    else:
        total_chunks = sum(len(d['chunks']) for d in st.session_state['docs'])
        total_chars = sum(len(d.get('raw_text', '')) for d in st.session_state['docs'])
        st.caption(
            f"문서 {len(st.session_state['docs'])}개 · "
            f"청크 {total_chunks}개 · {total_chars:,}자"
        )
        for d in list(st.session_state['docs']):
            with st.container(border=True):
                row = st.columns([4, 1, 1, 1, 1])
                row[0].markdown(f"**{d['name']}**")
                row[0].caption(
                    f"id: `{d['id']}` · "
                    f"chunk_size={d.get('chunk_size', '?')} · "
                    f"overlap={d.get('chunk_overlap', '?')}"
                )
                row[1].metric('chunks', len(d['chunks']))
                row[2].metric('pages', d.get('page_count', 0) or '—')
                row[3].metric('images', 'on' if d.get('has_page_images') else 'off')
                with row[4]:
                    st.write('')
                    if st.button('삭제', key=f"del_doc_{d['id']}", use_container_width=True):
                        remove_doc(d['id'])
                        st.rerun()
        if st.button('모든 문서 삭제 (디스크 포함)', type='secondary'):
            for d in list(st.session_state['docs']):
                remove_doc(d['id'])
            st.rerun()

    st.write('')
    with st.expander('고급 청킹 설정'):
        st.caption(
            '값을 바꾸면 현재 보유한 모든 문서가 자동으로 재청킹·재임베딩됩니다. '
            '이전 설정의 캐시는 보존되어 되돌릴 수 있습니다.'
        )
        cs_cols = st.columns(2)
        with cs_cols[0]:
            new_size = st.number_input(
                '청크 크기 (문자)', 100, 4000,
                int(st.session_state['chunk_size']),
                help='하나의 청크에 들어가는 최대 글자 수.',
            )
        with cs_cols[1]:
            new_overlap = st.number_input(
                '청크 겹침 (문자)', 0, 1000,
                int(st.session_state['chunk_overlap']),
                help='이웃한 청크 사이에 중복으로 포함시킬 글자 수. 문맥 유실 방지.',
            )
        if (new_size, new_overlap) != (
            st.session_state['chunk_size'], st.session_state['chunk_overlap']
        ):
            st.session_state['chunk_size'] = new_size
            st.session_state['chunk_overlap'] = new_overlap
            if st.session_state['docs']:
                with st.spinner('재청킹 및 재임베딩 중...'):
                    reindex_all()
                st.rerun()

    st.write('')
    _section(
        '검색 미리보기',
        'LLM 호출 없이 검색 결과만 확인합니다. 어떤 청크가 어떤 점수로 잡히는지 점검하는 용도.',
    )
    preview_cols = st.columns([5, 1])
    with preview_cols[0]:
        preview_query = st.text_input(
            '검색어', key='preview_query', label_visibility='collapsed',
            placeholder='예: 매출 성장률 정의',
        )
    with preview_cols[1]:
        run_preview = st.button(
            '검색 실행',
            disabled=(not preview_query.strip() or not st.session_state['docs']),
            use_container_width=True,
        )
    if run_preview and preview_query.strip():
        with st.spinner('검색 중...'):
            results = retrieve_local(preview_query.strip())
        if not results:
            st.warning('결과 없음.')
        else:
            st.caption(f"{len(results)}개 청크 반환")
            for j, r in enumerate(results, start=1):
                with st.expander(_citation_summary(j, r), expanded=False):
                    _citation_body(r)
