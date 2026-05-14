"""Cache view — Supabase status, local disk usage, HF model cache management."""

import shutil

import streamlit as st

from auth.supabase_io import _supabase_client
from auth.users import _is_streamlit_cloud, _user_data_dir, _user_logs_dir
from ui.helpers import _section


def view_cache():
    _section(
        '데이터 & 저장소',
        '클라우드 영속 저장, 로컬 디스크, 모델 캐시를 한 곳에서 관리.',
    )

    user_logs = _user_logs_dir()
    user_dd = _user_data_dir()

    # ----- Compute compact stats for the overview row -----
    sb_connected = _supabase_client() is not None
    sb_attempts = st.session_state.get('_sb_attempts', 0)
    sb_failures = st.session_state.get('_sb_failures', 0)
    pgv_successes = st.session_state.get('_pgv_successes', 0)
    pgv_failures = st.session_state.get('_pgv_failures', 0)

    # Local disk usage (sum of all files under user_dd).
    local_bytes = 0
    if user_dd.exists():
        for p in user_dd.rglob('*'):
            if p.is_file():
                try:
                    local_bytes += p.stat().st_size
                except Exception:
                    pass

    # HF model cache size.
    hf_cached = []
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        hf_cached = [(r.repo_id, r.size_on_disk_str, str(r.repo_path), r.size_on_disk)
                     for r in info.repos]
    except Exception:
        hf_cached = []
    hf_total_bytes = sum(x[3] for x in hf_cached)

    def _fmt_size(b):
        if b < 1024:
            return f'{b} B'
        if b < 1024 * 1024:
            return f'{b / 1024:.1f} KB'
        if b < 1024 ** 3:
            return f'{b / 1024 / 1024:.1f} MB'
        return f'{b / 1024 ** 3:.2f} GB'

    # ----- Top overview: 4 metric cards -----
    with st.container(border=True):
        m = st.columns(4)
        if sb_connected:
            sb_label = '정상' if sb_failures == 0 else f'{sb_failures}건 실패'
            m[0].metric('영속 로깅', sb_label, delta=f'{sb_attempts} 시도' if sb_attempts else None)
        else:
            m[0].metric('영속 로깅', '미설정')
        if sb_connected and pgv_successes:
            pgv_label = f'{pgv_successes}건' if pgv_failures == 0 else f'{pgv_failures}건 실패'
            m[1].metric('pgvector', pgv_label)
        elif sb_connected:
            m[1].metric('pgvector', '대기')
        else:
            m[1].metric('pgvector', '미연결')
        m[2].metric('로컬 디스크', _fmt_size(local_bytes))
        m[3].metric('HF 모델 캐시', _fmt_size(hf_total_bytes))

    # ----- Tabbed body -----
    tab_supabase, tab_local, tab_hf = st.tabs(
        ['클라우드 영속 (Supabase)', '로컬 디스크', 'HF 모델 캐시']
    )

    # ============== Supabase ==============
    with tab_supabase:
        if not sb_connected:
            with st.container(border=True):
                if _is_streamlit_cloud():
                    st.warning(
                        '영속 로깅 미설정 — 아래 로컬 JSONL 은 컨테이너 재시작 시 사라집니다.'
                    )
                    st.markdown(
                        '**활성화 방법:** Manage app → Settings → Secrets 에 다음 추가:\n'
                        '```toml\n'
                        'SUPABASE_URL = "https://xxxx.supabase.co"\n'
                        'SUPABASE_KEY = "eyJ..."\n'
                        '```\n'
                        '그 후 SQL Editor 에서 `db_schema.sql` / `db_schema_pgvector.sql` / `db_schema_users.sql` 실행.'
                    )
                else:
                    st.info(
                        '로컬 개발 중에는 JSONL 만으로 충분합니다. '
                        'Cloud 배포 시 `SUPABASE_URL` / `SUPABASE_KEY` 설정 권장.'
                    )
        else:
            # Status cards
            with st.container(border=True):
                st.markdown('##### 로깅 (chat_turns / agent_runs / events)')
                successes = st.session_state.get('_sb_successes', 0)
                last_err = st.session_state.get('_sb_last_err')
                if sb_attempts == 0:
                    st.info('이 세션에서 INSERT 시도 없음. 아래 진단 버튼으로 즉시 테스트 가능.')
                elif sb_failures == 0:
                    st.success(f'INSERT {successes}/{sb_attempts} 성공. 컨테이너 재시작에도 보존.')
                else:
                    st.error(
                        f'INSERT {sb_failures}/{sb_attempts} 실패. '
                        'RLS / 스키마 누락 등 점검 필요.'
                    )
                    if last_err:
                        with st.expander('마지막 실패 메시지', expanded=True):
                            st.code(last_err)

                if st.button(
                    '연결 진단 (events 1행 INSERT + DELETE)',
                    key='sb_diagnose_btn',
                ):
                    import time as _t
                    client = _supabase_client()
                    probe = {
                        'event_type': 'diagnostic_probe',
                        'user_id': st.session_state.get('user_id', '_local'),
                        'payload': {'ts': _t.time(), 'note': '캐시 탭 진단 버튼'},
                    }
                    try:
                        ins = client.table('events').insert(probe).execute()
                        ins_id = (ins.data[0]['id']
                                  if getattr(ins, 'data', None) and ins.data else None)
                        st.success(f'INSERT 성공 (id={ins_id}). 정리 중...')
                        if ins_id is not None:
                            try:
                                client.table('events').delete().eq('id', ins_id).execute()
                                st.info(f'정리 완료 (id={ins_id} 삭제). DB 정상.')
                            except Exception as de:
                                st.warning(
                                    f'INSERT 됐는데 DELETE 실패 ({de}).'
                                )
                    except Exception as e:
                        st.error(f'INSERT 실패: {type(e).__name__}')
                        st.code(str(e)[:1500])
                        msg = str(e).lower()
                        if 'row-level security' in msg or 'rls' in msg or 'policy' in msg:
                            st.markdown(
                                '**원인: RLS 차단.** SQL Editor 에서 한 번 실행:\n\n'
                                '```sql\n'
                                'alter table public.events     disable row level security;\n'
                                'alter table public.chat_turns disable row level security;\n'
                                'alter table public.agent_runs disable row level security;\n'
                                '```'
                            )
                        elif 'does not exist' in msg or '42p01' in msg:
                            st.markdown('**원인: 테이블 없음.** `db_schema.sql` 실행 필요.')
                        elif '401' in msg or '403' in msg or 'unauthorized' in msg:
                            st.markdown('**원인: 키 권한.** anon public 키인지 확인.')

            # pgvector status
            with st.container(border=True):
                st.markdown('##### pgvector (청크 임베딩 영속화)')
                pgv_attempts = st.session_state.get('_pgv_attempts', 0)
                pgv_last_err = st.session_state.get('_pgv_last_err')
                if pgv_attempts == 0:
                    st.info(
                        '이 세션 인덱싱 없음. 문서 업로드 시 청크 임베딩이 '
                        '`doc_chunks` 테이블에 자동 저장됩니다. '
                        '(스키마: `db_schema_pgvector.sql`)'
                    )
                elif pgv_failures == 0:
                    st.success(
                        f'청크 임베딩 {pgv_successes}/{pgv_attempts} 영속화 성공.'
                    )
                else:
                    st.error(
                        f'영속화 {pgv_failures}/{pgv_attempts} 실패. '
                        '`db_schema_pgvector.sql` 적용 확인.'
                    )
                    if pgv_last_err:
                        with st.expander('마지막 pgvector 에러', expanded=False):
                            st.code(pgv_last_err)

                # Active retrieval source.
                if st.session_state.get('use_pgvector_search'):
                    n = st.session_state.get('_pgv_search_last_n')
                    err = st.session_state.get('_pgv_search_last_err')
                    if err:
                        st.warning('의미 검색 경로: pgvector (직전 호출 실패 → 로컬 폴백)')
                        with st.expander('검색 에러', expanded=False):
                            st.code(err)
                    elif n is None:
                        st.caption('의미 검색 경로: pgvector (아직 호출 없음)')
                    else:
                        st.caption(f'의미 검색 경로: pgvector — 직전 호출 {n}건')
                else:
                    st.caption('의미 검색 경로: 로컬 numpy (in-memory)')

    # ============== 로컬 디스크 ==============
    with tab_local:
        # Aggregate logs first (agents + events)
        with st.container(border=True):
            st.markdown('##### 통합 로그 파일')
            found_any = False
            for fname, label_text in (
                ('events.jsonl', '로그인 / 문서 / 세션 / LLM 에러 이벤트'),
                ('agents.jsonl', '에이전트 실행 기록'),
            ):
                fpath = user_logs / fname
                if not fpath.exists():
                    continue
                found_any = True
                try:
                    n_lines = sum(1 for _ in fpath.open('r', encoding='utf-8'))
                except Exception:
                    n_lines = '?'
                size_kb = fpath.stat().st_size / 1024
                unit = 'events' if 'events' in fname else 'runs'
                cols = st.columns([5, 1, 1])
                cols[0].markdown(f"**`{fname}`** · {label_text}")
                cols[1].caption(f'{n_lines} {unit} · {size_kb:.1f} KB')
                with cols[2]:
                    try:
                        st.download_button(
                            '다운로드', data=fpath.read_bytes(),
                            file_name=fname, mime='application/x-jsonlines',
                            key=f'dl_{fname}', use_container_width=True,
                        )
                    except Exception:
                        pass
            if not found_any:
                st.caption('아직 통합 로그 파일이 없습니다.')

        # Per-session chat logs
        _AGGREGATE_LOGS = {'agents.jsonl', 'events.jsonl'}
        jsonl_files = sorted(
            [p for p in user_logs.glob('*.jsonl') if p.name not in _AGGREGATE_LOGS],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        with st.container(border=True):
            st.markdown('##### 세션별 대화 로그')
            st.caption(f'경로: `{user_logs}`')
            if not jsonl_files:
                st.caption('대화 로그는 아직 없습니다.')
            else:
                for p in jsonl_files[:30]:
                    try:
                        n_lines = sum(1 for _ in p.open('r', encoding='utf-8'))
                    except Exception:
                        n_lines = '?'
                    size_kb = p.stat().st_size / 1024
                    cols = st.columns([5, 1, 1])
                    cols[0].markdown(f"`{p.name}`")
                    cols[1].caption(f'{n_lines} turns · {size_kb:.1f} KB')
                    with cols[2]:
                        try:
                            st.download_button(
                                '다운로드', data=p.read_bytes(),
                                file_name=p.name,
                                mime='application/x-jsonlines',
                                key=f'dl_jsonl_{p.name}',
                                use_container_width=True,
                            )
                        except Exception:
                            pass
                if len(jsonl_files) > 30:
                    st.caption(f'표시 30개 / 전체 {len(jsonl_files)}개')

        # Local vector store
        with st.container(border=True):
            st.markdown('##### 로컬 벡터 스토어')
            st.caption(f'경로: `{user_dd}` — 임베더 모델마다 하위 폴더.')
            if not user_dd.exists():
                st.caption('아직 저장된 벡터 인덱스가 없습니다.')
            else:
                rows = []
                for sub in sorted(user_dd.iterdir()):
                    if not sub.is_dir() or sub.name == 'sessions':
                        continue
                    doc_dirs = [p for p in sub.iterdir() if p.is_dir()]
                    total = 0
                    for p in sub.rglob('*'):
                        if p.is_file():
                            try:
                                total += p.stat().st_size
                            except Exception:
                                pass
                    rows.append((sub.name, len(doc_dirs), total))
                if not rows:
                    st.caption('아직 저장된 벡터 인덱스가 없습니다.')
                else:
                    for name, n, size in rows:
                        cols = st.columns([4, 1, 1])
                        cols[0].markdown(f"`{name}`")
                        cols[1].caption(f'{n} docs')
                        cols[2].caption(_fmt_size(size))

    # ============== HF 모델 캐시 ==============
    with tab_hf:
        with st.container(border=True):
            st.markdown('##### Hugging Face 다운로드 모델')
            st.caption(
                '임베더 / reranker / 자가호스팅 모델 등 로컬 다운로드된 가중치. '
                '사용하지 않는 항목은 삭제해 디스크 회수.'
            )
            if not hf_cached:
                st.caption('캐시된 모델이 없습니다.')
            else:
                for repo_id, size_str, path, _bytes in hf_cached:
                    cols = st.columns([4, 1, 1])
                    cols[0].markdown(f"`{repo_id}`")
                    cols[1].caption(size_str)
                    if cols[2].button(
                        '삭제', key=f'cache_{repo_id}',
                        use_container_width=True,
                    ):
                        try:
                            shutil.rmtree(path)
                            st.success(f'{repo_id} 삭제 완료')
                            st.rerun()
                        except Exception as e:
                            st.error(f'삭제 실패: {e}')
