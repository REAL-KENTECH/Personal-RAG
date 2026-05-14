"""Per-turn structured logging + first-turn auto-titler.

Both helpers straddle the chat / session boundary: they read the chat
state to compute settings snapshots and citation counts, but the writes
go to per-user JSONL files + the Supabase chat_turns table.
"""

import datetime
import json

import streamlit as st

from auth.supabase_io import _supabase_insert
from data.sessions import _session_jsonl_path
from llm.clients import get_openai_client
from llm.params import _build_completion_params, _thinking_off_extra_body

from .citations import parse_citations


def log_turn_structured(user_input: str, response_text: str, reasoning: str,
                        retrieved: list, model: str, elapsed_sec,
                        query_variants: list = None):
    """Append one JSON record per turn to ./logs/{session_id}.jsonl.
    Postgres-ready schema. One line per turn; analysis-friendly.

    Schema fields: session_id, session_title, turn_index, timestamp,
      user_message, assistant_message, reasoning,
      model, provider, base_url, elapsed_seconds,
      retrieved[] (rank/source/score/text/doc_id/doc_name/chunk_index/pages/url/title),
      n_retrieved, citation_numbers_used[], n_cited,
      query_variants[], settings_snapshot{} (all retrieval/sampling/feature flags).
    """
    sid = st.session_state.get('current_session_id')
    if not sid:
        return
    path = _session_jsonl_path(sid)

    # Normalize retrieved entries. We store only a 200-char preview plus a
    # reference (doc_id + chunk_index) so chat_turns rows stay light — the
    # full chunk body is in doc_chunks and can be joined back when needed.
    retrieved_records = []
    for j, r in enumerate(retrieved or [], start=1):
        rec = {
            'rank': j,
            'source': r.get('source'),
            'score': r.get('score'),
            'text_preview': (r.get('text') or '')[:200],
        }
        if r.get('source') == 'web':
            rec['title'] = r.get('doc')
            rec['url'] = r.get('url')
        else:
            rec['doc_id'] = r.get('doc_id')
            rec['doc_name'] = r.get('doc')
            rec['chunk_index'] = r.get('chunk_idx')
            rec['pages'] = r.get('pages') or []
        retrieved_records.append(rec)

    n_chunks = len(retrieved) if retrieved else 0
    cited_nums = sorted(parse_citations(response_text, n_chunks)) if n_chunks else []

    settings = {
        'embedder_model': st.session_state.get('embedder_model'),
        'retrieval_mode': st.session_state.get('retrieval_mode'),
        'retrieve_top_n': st.session_state.get('retrieve_top_n'),
        'final_top_k': st.session_state.get('final_top_k'),
        'use_reranker': st.session_state.get('use_reranker'),
        'use_pgvector_search': st.session_state.get('use_pgvector_search'),
        'use_agentic_search': st.session_state.get('use_agentic_search'),
        'general_chat_mode': st.session_state.get('general_chat_mode'),
        'agentic_max_iters': st.session_state.get('agentic_max_iters'),
        'use_hyde': st.session_state.get('use_hyde'),
        'use_multi_query': st.session_state.get('use_multi_query'),
        'n_paraphrases': st.session_state.get('n_paraphrases'),
        'use_contextual_rewrite': st.session_state.get('use_contextual_rewrite'),
        'per_doc_balance': st.session_state.get('per_doc_balance'),
        'per_doc_reserve': st.session_state.get('per_doc_reserve'),
        'comparison_autodetect': st.session_state.get('comparison_autodetect'),
        'chat_doc_filter': st.session_state.get('chat_doc_filter'),
        'web_enabled': st.session_state.get('web_enabled'),
        'web_provider': st.session_state.get('web_provider'),
        'web_top_n': st.session_state.get('web_top_n'),
        'include_page_images': st.session_state.get('include_page_images'),
        'max_page_images': st.session_state.get('max_page_images'),
        'temperature': st.session_state.get('temperature'),
        'top_p': st.session_state.get('top_p'),
        'sampling_top_k': st.session_state.get('sampling_top_k'),
        'presence_penalty': st.session_state.get('presence_penalty'),
        'max_tokens': st.session_state.get('max_tokens'),
        'stream': st.session_state.get('stream'),
        'enable_thinking': st.session_state.get('enable_thinking'),
        'chunk_size': st.session_state.get('chunk_size'),
        'chunk_overlap': st.session_state.get('chunk_overlap'),
    }

    record = {
        'session_id': sid,
        'session_title': st.session_state.get('current_session_title', ''),
        'turn_index': len(st.session_state['user_inputs']),
        'timestamp': datetime.datetime.now().isoformat(timespec='microseconds'),
        'user_id': st.session_state.get('user_id', '_local'),
        'user_message': user_input,
        'assistant_message': response_text or '',
        'reasoning': reasoning or '',
        'model': model,
        'provider': st.session_state.get('provider', ''),
        'base_url': st.session_state.get('base_url', ''),
        'elapsed_seconds': float(elapsed_sec) if elapsed_sec is not None else None,
        'retrieved': retrieved_records,
        'n_retrieved': len(retrieved_records),
        'citation_numbers_used': cited_nums,
        'n_cited': len(cited_nums),
        'query_variants': list(query_variants or []),
        'settings_snapshot': settings,
    }

    try:
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception as e:
        st.warning(f'구조화 로그 저장 실패: {e}')
    _supabase_insert('chat_turns', record)


def auto_title_session() -> str:
    """LLM-generate a short Korean title from the first turn.

    Falls back to a trimmed version of the user's first question when the LLM
    call fails or returns empty content (common with reasoning-style models
    like gpt-5 / o-series that can consume the whole max_tokens budget on
    internal thinking before emitting any visible output).
    """
    if not st.session_state['user_inputs'] or not st.session_state['generated_responses']:
        return ''
    user_msg = st.session_state['user_inputs'][0][:300]
    asst_msg = st.session_state['generated_responses'][0][:300]

    # Always-available fallback derived from the user's question itself.
    fallback = ' '.join((user_msg or '').split())[:16] or '새 대화'

    prompt = (
        '다음 대화 첫 턴을 보고 8글자 내외의 간결한 한국어 제목을 만드세요. '
        '큰따옴표 / 마침표 / 이모지 없이 제목만 한 줄로 출력하세요.\n\n'
        f'사용자: {user_msg}\n어시스턴트: {asst_msg}\n\n제목:'
    )
    try:
        client = get_openai_client()
        tparams = _build_completion_params(
            model=st.session_state['model'],
            messages=[{'role': 'user', 'content': prompt}],
            # Generous budget — reasoning models can spend hundreds of tokens
            # thinking before producing the 8-character title.
            max_tokens=400,
            temperature=0.3,
            extra_body=_thinking_off_extra_body(),
        )
        resp = client.chat.completions.create(**tparams)
        title = (resp.choices[0].message.content or '').strip()
        # If the model returned reasoning then the title on a final line,
        # keep only the last non-empty line.
        if '\n' in title:
            lines = [ln.strip() for ln in title.splitlines() if ln.strip()]
            if lines:
                title = lines[-1]
        title = title.strip('"').strip("'").strip('「').strip('」').strip()
        # Strip any "제목:" prefix the model might echo.
        for prefix in ('제목:', '제목 :', 'Title:'):
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        if len(title) > 30:
            title = title[:30]
        return title or fallback
    except Exception:
        return fallback
