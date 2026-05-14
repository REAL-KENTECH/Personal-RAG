"""Settings view — LLM provider, retrieval, web, response/multimodal, advanced."""

import streamlit as st

from auth.prefs import _save_user_prefs
from auth.supabase_io import _supabase_client
from config import EMBEDDER_CHOICES, PROVIDER_NAMES, PROVIDERS
from data.storage import load_all_for_current_embedder
from llm.clients import _active_api_key
from ui.helpers import _section
from ui.widgets import model_picker


def view_settings():
    _section(
        '설정',
        '모델·검색·웹·응답을 카테고리별로 정리. 변경은 즉시 저장됩니다.',
    )

    # ----- Top status snapshot -----
    active_p = st.session_state.get('provider', 'Hugging Face Router')
    model_short = (st.session_state.get('model') or '—')
    if len(model_short) > 30:
        model_short = model_short[:27] + '...'
    embedder_label = {
        'BAAI/bge-m3': 'BGE-M3',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': 'MiniLM',
    }.get(st.session_state.get('embedder_model', ''), '—')
    has_key = bool(_active_api_key())
    with st.container(border=True):
        m = st.columns(4)
        m[0].metric('공급자', active_p)
        m[1].metric('모델', model_short)
        m[2].metric('임베더', embedder_label)
        m[3].metric('API 키', '설정됨' if has_key else '없음')

    # ----- Tabbed sections -----
    tab_llm, tab_search, tab_web, tab_response, tab_advanced = st.tabs(
        ['LLM 공급자', 'RAG 검색', '웹 검색', '응답 / 멀티모달', '고급']
    )

    # ============== LLM ==============
    with tab_llm:
        with st.container(border=True):
            st.markdown('##### 공급자 선택')
            current_provider = st.session_state.get('provider', PROVIDER_NAMES[0])
            if current_provider not in PROVIDER_NAMES:
                current_provider = PROVIDER_NAMES[0]
            new_provider = st.selectbox(
                '공급자', PROVIDER_NAMES,
                index=PROVIDER_NAMES.index(current_provider),
                help='프리셋을 선택하면 엔드포인트 주소·기본 모델·환경변수에서 API 키를 자동으로 채웁니다.',
                label_visibility='collapsed',
            )
            if new_provider != current_provider:
                cfg = PROVIDERS[new_provider]
                if cfg['base_url']:
                    st.session_state['base_url'] = cfg['base_url']
                if cfg['default_model']:
                    st.session_state['model'] = cfg['default_model']
                st.session_state['provider'] = new_provider
                st.rerun()
            else:
                st.session_state['provider'] = new_provider
            model_picker('모델', key_prefix='settings')

        # API keys — staged via pending state. Typing into a field doesn't
        # touch session_state[<active_key>] until the user presses 저장.
        # That keeps every keystroke from triggering _save_user_prefs and
        # the disk/Supabase round-trips it carries, and gives the user a
        # clear "I'm done editing" moment.
        with st.container(border=True):
            st.markdown('##### API 키')
            st.caption(
                '공급자별로 따로 저장됩니다. "(사용 중)" 표시가 현재 공급자 키. '
                '입력 후 아래 **저장** 버튼을 눌러야 반영됩니다.'
            )

            def _key_label(name, owner):
                return f'{name} (사용 중)' if owner == active_p else name

            _key_specs = [
                ('hf_api_key',         'Hugging Face',         'Hugging Face Router',
                 'hf_...',
                 'HF Inference Router (Gemma / DeepSeek / Qwen 등). '
                 'fine-grained 토큰 권장: https://huggingface.co/settings/tokens'),
                ('openai_api_key',     'OpenAI',               'OpenAI',
                 'sk-...',
                 'gpt-4o / gpt-5 / o3 등. https://platform.openai.com/api-keys'),
                ('anthropic_api_key',  'Anthropic (Claude)',   'Anthropic (Claude)',
                 'sk-ant-...',
                 'Claude 4.x. https://console.anthropic.com/settings/keys'),
                ('fireworks_api_key',  'Fireworks AI',         'Fireworks AI',
                 'fw_...',
                 '오픈모델 빠른 추론. https://fireworks.ai/account/api-keys'),
                ('dashscope_api_key',  'DashScope (Qwen)',     'DashScope (Qwen)',
                 'sk-...',
                 'Qwen 공식 API.'),
                ('custom_api_key',     'Custom / vLLM',        'vLLM / local',
                 '(self-host endpoint key)',
                 'vLLM 등 셀프 호스팅 / Custom OpenAI-호환 endpoint.'),
            ]

            def _render_key_row(active_key, label_text, owner, ph, help_text):
                pending_key = f'_pending_{active_key}'
                # First-time render seeds the widget from the active value.
                # On subsequent reruns Streamlit retains the user's typing
                # via the widget key, so we don't re-seed.
                if pending_key not in st.session_state:
                    st.session_state[pending_key] = (
                        st.session_state.get(active_key, '') or ''
                    )
                st.text_input(
                    _key_label(label_text, owner),
                    type='password',
                    placeholder=ph,
                    help=help_text,
                    key=pending_key,
                )
                pending_val = st.session_state.get(pending_key, '') or ''
                active_val = st.session_state.get(active_key, '') or ''
                if pending_val != active_val:
                    bcols = st.columns([2, 1, 1])
                    bcols[0].caption('변경 대기 — 적용을 눌러야 호출에 반영됩니다.')
                    if bcols[1].button(
                        '취소', key=f'reset_{active_key}',
                        use_container_width=True,
                    ):
                        st.session_state[pending_key] = active_val
                        st.rerun()
                    if bcols[2].button(
                        '적용', key=f'apply_{active_key}',
                        type='primary', use_container_width=True,
                    ):
                        st.session_state[active_key] = pending_val
                        try:
                            _save_user_prefs()
                        except Exception:
                            pass
                        st.rerun()

            kc1, kc2 = st.columns(2)
            for i, spec in enumerate(_key_specs):
                col = kc1 if i % 2 == 0 else kc2
                with col:
                    _render_key_row(*spec)

            # Saved-count summary at the bottom.
            set_count = sum(
                1 for active_key, *_ in _key_specs
                if (st.session_state.get(active_key, '') or '').strip()
            )
            st.caption(f'현재 {set_count}/{len(_key_specs)} 개의 키가 저장돼 있습니다.')

    # ============== RAG 검색 ==============
    with tab_search:
        # Mode + embedder
        with st.container(border=True):
            st.markdown('##### 동작 모드')
            st.session_state['general_chat_mode'] = st.checkbox(
                '일반 대화 모드 (RAG / 웹 검색 끔)',
                value=st.session_state['general_chat_mode'],
                help='업로드 문서·웹 검색을 모두 건너뛰고 LLM 본연 지식으로 답합니다.',
            )
            prev_embedder = st.session_state['embedder_model']
            emb_labels = {
                'BAAI/bge-m3': 'BGE-M3 (한국어 강함, 2.2GB)',
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2':
                    'MiniLM 다국어 (가벼움, 470MB)',
            }
            emb_idx = (
                EMBEDDER_CHOICES.index(st.session_state['embedder_model'])
                if st.session_state['embedder_model'] in EMBEDDER_CHOICES else 0
            )
            st.session_state['embedder_model'] = st.selectbox(
                '임베딩 모델', EMBEDDER_CHOICES,
                index=emb_idx,
                format_func=lambda x: emb_labels.get(x, x),
                help='문서·질문을 벡터로 변환하는 모델.',
            )
            if st.session_state['embedder_model'] != prev_embedder:
                st.session_state['_loaded_for_embedder'] = None
                load_all_for_current_embedder()
                st.rerun()

        # Retrieval pipeline
        with st.container(border=True):
            st.markdown('##### 검색 파이프라인')
            mode_labels = {'hybrid': '하이브리드 (권장)',
                           'dense': '의미 기반만', 'bm25': '키워드만'}
            st.session_state['retrieval_mode'] = st.radio(
                '검색 방식',
                ['hybrid', 'dense', 'bm25'],
                index=['hybrid', 'dense', 'bm25'].index(
                    st.session_state['retrieval_mode']
                ),
                horizontal=True,
                format_func=lambda x: mode_labels[x],
            )
            cc1, cc2 = st.columns(2)
            with cc1:
                st.session_state['use_reranker'] = st.checkbox(
                    '재정렬 모델 사용 (Cross-encoder)',
                    value=st.session_state['use_reranker'],
                    help='정확도 향상, 응답 약간 느려짐.',
                )
                st.session_state['use_contextual_rewrite'] = st.checkbox(
                    '이어지는 질문 자동 보완',
                    value=st.session_state['use_contextual_rewrite'],
                    help='이전 대화 맥락으로 self-contained 질문 재작성.',
                )
                st.session_state['per_doc_balance'] = st.checkbox(
                    '여러 문서 균형 검색',
                    value=st.session_state['per_doc_balance'],
                    help='상위 결과가 한 문서에 쏠리지 않게 강제.',
                )
                st.session_state['comparison_autodetect'] = st.checkbox(
                    '비교 질문 자동 감지',
                    value=st.session_state['comparison_autodetect'],
                )
            with cc2:
                _pgv_available = _supabase_client() is not None
                st.session_state['use_pgvector_search'] = st.checkbox(
                    'pgvector 의미 검색 (Supabase)',
                    value=st.session_state['use_pgvector_search'],
                    help='Supabase pgvector 로 dense 검색. 미연결 시 자동 폴백.',
                    disabled=not _pgv_available,
                )
                if not _pgv_available:
                    st.caption('Supabase 미연동 — 비활성')
                st.session_state['use_agentic_search'] = st.checkbox(
                    '에이전트 검색 (LLM 추가 검색)',
                    value=st.session_state['use_agentic_search'],
                    help='LLM 이 도구 호출로 추가 검색 발행. function calling 지원 모델 필요.',
                )
                if st.session_state['use_agentic_search']:
                    st.session_state['agentic_max_iters'] = st.slider(
                        '최대 라운드', 1, 5,
                        int(st.session_state.get('agentic_max_iters', 3)),
                    )

    # ============== 웹 ==============
    with tab_web:
        with st.container(border=True):
            st.markdown('##### 실시간 웹 검색')
            st.caption(
                '질문 시 웹 검색 결과를 컨텍스트에 함께 포함. DuckDuckGo 는 API 키 불필요.'
            )
            st.session_state['web_enabled'] = st.checkbox(
                '웹 검색 사용',
                value=st.session_state['web_enabled'],
            )
            if st.session_state['web_enabled']:
                wp_labels = {
                    'duckduckgo': 'DuckDuckGo (키 불필요)',
                    'tavily': 'Tavily (LLM 최적화, 키 필요)',
                    'brave': 'Brave (키 필요)',
                }
                wc1, wc2 = st.columns(2)
                with wc1:
                    st.session_state['web_provider'] = st.selectbox(
                        '검색 제공자', ['duckduckgo', 'tavily', 'brave'],
                        index=['duckduckgo', 'tavily', 'brave'].index(
                            st.session_state['web_provider']
                        ),
                        format_func=lambda x: wp_labels[x],
                    )
                with wc2:
                    st.session_state['web_top_n'] = st.number_input(
                        '결과 수', 1, 20, int(st.session_state['web_top_n']),
                    )
                if st.session_state['web_provider'] == 'tavily':
                    st.session_state['tavily_key'] = st.text_input(
                        'Tavily API 키', st.session_state['tavily_key'],
                        type='password', placeholder='tvly-...',
                    )
                elif st.session_state['web_provider'] == 'brave':
                    st.session_state['brave_key'] = st.text_input(
                        'Brave API 키', st.session_state['brave_key'],
                        type='password',
                    )
            else:
                st.caption('웹 검색이 꺼져 있어 추가 설정이 보이지 않습니다.')

    # ============== 응답 / 멀티모달 ==============
    with tab_response:
        with st.container(border=True):
            st.markdown('##### 응답 동작')
            rc1, rc2 = st.columns(2)
            with rc1:
                st.session_state['stream'] = st.checkbox(
                    '스트리밍 응답', value=st.session_state['stream'],
                    help='응답을 토큰 단위로 실시간 표시.',
                )
            with rc2:
                st.session_state['enable_thinking'] = st.checkbox(
                    '추론 모드 사용', value=st.session_state['enable_thinking'],
                    help='지원 모델에서 reasoning 토큰을 분리 표시.',
                )

        with st.container(border=True):
            st.markdown('##### 멀티모달 (이미지)')
            st.caption(
                'PDF 페이지 이미지를 LLM 에 함께 전달해 표·차트·도식을 이해하게 합니다. '
                '비전 입력 지원 모델 필요.'
            )
            st.session_state['include_page_images'] = st.checkbox(
                'PDF 페이지 이미지 첨부',
                value=st.session_state['include_page_images'],
            )
            if st.session_state['include_page_images']:
                st.session_state['max_page_images'] = st.number_input(
                    '한 턴에 보낼 이미지 수', 1, 10,
                    int(st.session_state['max_page_images']),
                )

    # ============== 고급 ==============
    with tab_advanced:
        with st.container(border=True):
            st.markdown('##### 엔드포인트')
            st.session_state['base_url'] = st.text_input(
                '엔드포인트 주소', st.session_state['base_url'],
                help='OpenAI 호환 endpoint. {base_url}/chat/completions 가 호출됩니다.',
            )

        with st.container(border=True):
            st.markdown('##### 샘플링 / 응답 길이')
            ac1, ac2 = st.columns(2)
            with ac1:
                st.session_state['max_tokens'] = st.number_input(
                    '최대 응답 토큰', 16, 131072,
                    int(st.session_state['max_tokens']),
                )
                st.session_state['temperature'] = st.slider(
                    'temperature', 0.0, 2.0,
                    float(st.session_state['temperature']), 0.05,
                )
                st.session_state['top_p'] = st.slider(
                    'top_p', 0.0, 1.0, float(st.session_state['top_p']), 0.01,
                )
            with ac2:
                st.session_state['sampling_top_k'] = st.number_input(
                    'top_k', 1, 200, int(st.session_state['sampling_top_k']),
                )
                st.session_state['presence_penalty'] = st.slider(
                    'presence_penalty', 0.0, 2.0,
                    float(st.session_state['presence_penalty']), 0.1,
                )

        with st.container(border=True):
            st.markdown('##### 검색 정밀도')
            ad1, ad2 = st.columns(2)
            with ad1:
                st.session_state['retrieve_top_n'] = st.number_input(
                    '1차 후보 수', 1, 200,
                    int(st.session_state['retrieve_top_n']),
                    help='재정렬 전에 가져올 후보 청크 수.',
                )
                st.session_state['final_top_k'] = st.number_input(
                    '최종 청크 수', 1, 50,
                    int(st.session_state['final_top_k']),
                    help='LLM 에 컨텍스트로 전달할 최종 청크 수.',
                )
            with ad2:
                st.session_state['per_doc_reserve'] = st.number_input(
                    '문서당 최소 청크', 1, 5,
                    int(st.session_state['per_doc_reserve']),
                )
            st.session_state['use_multi_query'] = st.checkbox(
                '다중 쿼리 (paraphrase)',
                value=st.session_state['use_multi_query'],
                help='질문을 여러 표현으로 변형 후 합집합 검색.',
            )
            if st.session_state['use_multi_query']:
                st.session_state['n_paraphrases'] = st.number_input(
                    '변형 개수', 1, 8,
                    int(st.session_state['n_paraphrases']),
                )
            st.session_state['use_hyde'] = st.checkbox(
                'HyDE (가상 답안 검색)',
                value=st.session_state['use_hyde'],
            )
