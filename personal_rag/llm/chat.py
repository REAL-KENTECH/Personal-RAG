"""Chat orchestration — the glue between retrieval, the LLM call, and the UI.

One ``handle_chat_turn`` invocation does the full round trip:

1. Optional retrieval (``retrieve``) — local docs + web — with a chat-time
   document filter applied.
2. ``build_messages`` — system prompt + prior turns + the user's question
   with numbered Context blocks (and optional page-image attachments for
   multimodal models).
3. ``_build_completion_params`` to assemble provider-safe kwargs.
4. Either an agentic loop (``agentic_chat_pass`` — the LLM may invoke
   ``search_documents`` up to N times), a streaming call (``stream_chat``),
   or a non-streaming call (``non_stream_chat``).
5. ``render_assistant`` paints the answer + citations + reasoning expanders.
6. Persistence: ``save_current_session`` (disk + Supabase mirror),
   ``auto_title_session`` on the first turn, ``log_turn_structured`` for
   the audit trail.

``_show_llm_error`` lives here because every failure path in the chat
turn routes through it; it pattern-matches the common provider errors
and renders an actionable Korean message instead of a raw stack trace.
"""

import datetime
import json
import re
import time

import streamlit as st

from ..auth.supabase_io import _supabase_insert
from ..auth.users import _log_event
from ..data.sessions import (
    _new_session_id,
    _session_jsonl_path,
    save_current_session,
)
from ..data.storage import load_page_image_b64
from ..retrieval.pipeline import retrieve, retrieve_local
from .clients import _active_api_key, get_openai_client
from .params import (
    _build_completion_params,
    _is_dashscope_endpoint,
    _is_fireworks_endpoint,
    _is_hf_router_endpoint,
    _is_openai_endpoint,
    _provider_supports_top_k,
    _thinking_off_extra_body,
)


RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Ground every factual claim in the Context "
    "provided in the user's latest message. Also use the prior conversation "
    "to understand the user — resolve pronouns, follow-up references, and "
    "implied subjects from earlier turns. Each Context entry is marked with "
    "[N] and tagged as either a local document or a web result. Cite entry "
    "numbers like [1], [2] for every fact you take from the Context. If a "
    "factual answer is not supported by the Context or recent conversation, "
    "reply exactly: \"제공된 자료에서는 답을 찾을 수 없습니다.\""
)


def _format_pages(pages):
    if not pages:
        return ''
    if len(pages) == 1:
        return f' p.{pages[0]}'
    return f' pp.{pages[0]}-{pages[-1]}'


def _context_label(r: dict) -> str:
    if r.get('source') == 'web':
        url = r.get('url', '')
        return f'웹: {r.get("doc", "")} | {url}'
    pages = r.get('pages') or []
    page_part = _format_pages(pages)
    return f'로컬: {r.get("doc", "")} (chunk {r.get("chunk_idx", 0)}{page_part})'


def _collect_page_image_parts(retrieved: list, max_images: int) -> list:
    """Return a list of {type: image_url, image_url: {url: data:...}} parts
    for unique (doc_id, page) pairs that have rendered images on disk."""
    parts = []
    seen = set()
    embedder_id = st.session_state['embedder_model']
    for r in retrieved:
        if r.get('source') != 'doc':
            continue
        doc_id = r.get('doc_id')
        for p in (r.get('pages') or []):
            key = (doc_id, p)
            if key in seen:
                continue
            seen.add(key)
            b64 = load_page_image_b64(embedder_id, doc_id, p)
            if not b64:
                continue
            parts.append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/png;base64,{b64}'},
            })
            if len(parts) >= max_images:
                return parts
    return parts


def build_messages(user_input: str, retrieved: list) -> list:
    msgs = []
    if retrieved:
        msgs.append({'role': 'system', 'content': RAG_SYSTEM_PROMPT})
    for u, a in zip(
        st.session_state['user_inputs'], st.session_state['generated_responses']
    ):
        msgs.append({'role': 'user', 'content': u})
        msgs.append({'role': 'assistant', 'content': a})

    if retrieved:
        ctx = '\n\n'.join(
            f'[{i + 1}] ({_context_label(r)}) {r["text"]}'
            for i, r in enumerate(retrieved)
        )
        text_content = f'Context:\n{ctx}\n\nQuestion: {user_input}'
    else:
        text_content = user_input

    # Multimodal: attach unique page images for current turn only.
    image_parts = []
    if retrieved and st.session_state.get('include_page_images'):
        max_imgs = int(st.session_state.get('max_page_images', 3))
        image_parts = _collect_page_image_parts(retrieved, max_imgs)

    if image_parts:
        content = [{'type': 'text', 'text': text_content}] + image_parts
        msgs.append({'role': 'user', 'content': content})
    else:
        msgs.append({'role': 'user', 'content': text_content})
    return msgs


def split_thinking(text: str):
    s = text or ''
    m = re.search(r'<think>(.*?)</think>\s*(.*)', s, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(
        r'<\|channel\|?>thought\s*(.*?)<\|?channel\|>\s*(.*)',
        s, flags=re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return '', s.strip()


_CITE_PATTERN = r'\[((?:\d+\s*,\s*)*\d+)\]'


def parse_citations(answer: str, n_chunks: int) -> set:
    nums = set()
    for group in re.findall(_CITE_PATTERN, answer or ''):
        for piece in group.split(','):
            try:
                n = int(piece.strip())
                if 1 <= n <= n_chunks:
                    nums.add(n)
            except ValueError:
                pass
    return nums


def format_answer_with_citations(answer: str, n_chunks: int) -> str:
    def repl(m):
        inner = m.group(1)
        for piece in inner.split(','):
            try:
                n = int(piece.strip())
                if 1 <= n <= n_chunks:
                    return f'**[{inner}]**'
            except ValueError:
                pass
        return m.group(0)
    return re.sub(_CITE_PATTERN, repl, answer or '')


# -----------------------------------------------------------------------------
# Agentic RAG
# -----------------------------------------------------------------------------
# The assistant gets one tool — search_documents(query) — and may invoke it
# up to N times to fetch additional context before producing the final answer.
# Each tool result is appended back into the message history so subsequent
# turns of the loop see all prior searches.

_SEARCH_TOOL_DEF = {
    'type': 'function',
    'function': {
        'name': 'search_documents',
        'description': (
            "사용자가 업로드한 문서에서 추가로 검색합니다. 초기 컨텍스트만으로 "
            "답하기 어려운 경우 더 구체적인 쿼리로 다시 검색해 근거를 보강하세요. "
            "다음 상황에 유용합니다: (1) 질문에 여러 하위 토픽이 있어 각각 따로 "
            "검색이 필요할 때, (2) 초기 컨텍스트에 핵심 정보가 빠져 보일 때, "
            "(3) 비교/대조 질문에서 한쪽 정보만 충분한 경우. 한 번에 한 가지에 "
            "집중된 쿼리를 작성하세요."
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': '검색할 핵심 쿼리. 명확한 명사구 또는 self-contained 한 문장.',
                },
            },
            'required': ['query'],
        },
    },
}


def _format_tool_search_result(retrieved: list) -> str:
    """Compact representation of retrieved chunks for tool-message content."""
    if not retrieved:
        return '(검색 결과 없음)'
    lines = []
    for i, r in enumerate(retrieved, start=1):
        pages = r.get('pages') or []
        if len(pages) == 1:
            page_str = f' p.{pages[0]}'
        elif pages:
            page_str = f' pp.{pages[0]}-{pages[-1]}'
        else:
            page_str = ''
        src = r.get('doc', '') or r.get('url', '')
        lines.append(f"[검색 결과 {i}] ({src}{page_str}) {r.get('text', '')}")
    return '\n\n'.join(lines)


def agentic_chat_pass(client, params: dict, initial_retrieved: list,
                      max_iters: int = 3):
    """Run a chat completion loop where the assistant may invoke
    search_documents to refine context. Always non-streaming (tool-call
    streaming is brittle across providers). Returns
    (full_text, reasoning_text, augmented_retrieved).

    Falls back to a plain non-stream call without `tools` if the provider
    rejects tool calling (e.g. some HF Router-served small models)."""
    augmented = list(initial_retrieved)
    seen_keys = {(r.get('doc_id'), r.get('chunk_idx'))
                 for r in augmented if r.get('source') != 'web'}

    tools_params = {k: v for k, v in params.items() if k != 'stream'}
    tools_params['tools'] = [_SEARCH_TOOL_DEF]
    messages = list(tools_params['messages'])

    reasoning_acc = ''
    n_tool_calls_made = 0

    for iteration in range(max_iters):
        try:
            tools_params['messages'] = messages
            spinner_msg = (
                '생각 중…' if iteration == 0
                else f'추가 검색 결과 분석 중… (라운드 {iteration})'
            )
            with st.spinner(spinner_msg):
                resp = client.chat.completions.create(**tools_params)
        except Exception:
            # Provider/model rejects tools — fall back to plain non-stream.
            fallback = {k: v for k, v in params.items() if k not in ('stream',)}
            fallback['messages'] = messages
            with st.spinner('도구 호출 미지원 — 일반 모드로 재시도...'):
                resp = client.chat.completions.create(**fallback)
            msg = resp.choices[0].message
            full = msg.content or ''
            r = getattr(msg, 'reasoning_content', '') or ''
            return full, (reasoning_acc + r).strip(), augmented

        msg = resp.choices[0].message
        tool_calls = getattr(msg, 'tool_calls', None) or []
        r_part = getattr(msg, 'reasoning_content', '') or ''
        if r_part:
            reasoning_acc = (reasoning_acc + '\n\n' + r_part).strip()

        if not tool_calls:
            # Final answer — no more searches requested.
            return msg.content or '', reasoning_acc, augmented

        # Surface the search activity in the UI so the user sees the loop.
        st.info(f'추가 검색 라운드 {iteration + 1}/{max_iters}')

        # Append assistant turn with tool_calls intact (required by API).
        messages.append({
            'role': 'assistant',
            'content': msg.content or '',
            'tool_calls': [
                {
                    'id': tc.id,
                    'type': 'function',
                    'function': {
                        'name': tc.function.name,
                        'arguments': tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            if tc.function.name != 'search_documents':
                tool_text = f'(알 수 없는 도구 호출: {tc.function.name})'
            else:
                try:
                    args = json.loads(tc.function.arguments or '{}')
                except Exception:
                    args = {}
                q = (args.get('query') or '').strip()
                if not q:
                    tool_text = '(빈 쿼리)'
                else:
                    st.caption(f'→ "{q}"')
                    n_tool_calls_made += 1
                    new_chunks = retrieve_local(q)
                    tool_text = _format_tool_search_result(new_chunks)
                    for nc in new_chunks:
                        key = (nc.get('doc_id'), nc.get('chunk_idx'))
                        if key not in seen_keys:
                            augmented.append(nc)
                            seen_keys.add(key)

            messages.append({
                'role': 'tool',
                'tool_call_id': tc.id,
                'content': tool_text,
            })

    # Max iterations reached — make one final tool-less call to extract the
    # answer so we never return mid-loop.
    final_params = {k: v for k, v in params.items() if k != 'stream'}
    final_params['messages'] = messages
    with st.spinner('최종 답변 생성 중...'):
        resp = client.chat.completions.create(**final_params)
    msg = resp.choices[0].message
    r_part = getattr(msg, 'reasoning_content', '') or ''
    if r_part:
        reasoning_acc = (reasoning_acc + '\n\n' + r_part).strip()
    return msg.content or '', reasoning_acc, augmented


def _record_response_model(model_id: str) -> None:
    """Capture the model id the provider actually served the response from.
    Useful for verifying that requests went to the model the user picked —
    OpenAI chat completions often respond with a dated variant like
    'gpt-5-2025-08-07', so seeing that next to the configured 'gpt-5'
    confirms the routing without trusting the model's self-introduction."""
    if model_id:
        st.session_state['_last_response_model'] = str(model_id)


def stream_chat(client, params: dict):
    """Stream response. Returns (full_text, reasoning_text).

    Renders an immediate "생각 중…" placeholder so the user can tell the
    difference between "model is thinking" and "app is hung". The
    placeholder updates each second with an elapsed-time counter while
    we wait for the first visible token (reasoning models like
    HyperCLOVAX-Think can take 10s+ before they emit anything)."""
    placeholder = st.empty()
    placeholder.markdown('_생각 중…_')
    t0 = time.time()
    last_idle_tick = t0
    full_text = ''
    reasoning_text = ''
    try:
        stream = client.chat.completions.create(stream=True, **params)
        for chunk in stream:
            # Some providers stamp the model id on every chunk; capture
            # it once we see it.
            cm = getattr(chunk, 'model', None)
            if cm and not st.session_state.get('_stream_model_captured'):
                _record_response_model(cm)
                st.session_state['_stream_model_captured'] = True
            if not getattr(chunk, 'choices', None):
                continue
            delta = chunk.choices[0].delta
            rc = getattr(delta, 'reasoning_content', None) or ''
            c = getattr(delta, 'content', None) or ''
            if rc:
                reasoning_text += rc
            if c:
                full_text += c
            if full_text:
                placeholder.markdown(full_text)
            elif reasoning_text:
                elapsed = int(time.time() - t0)
                suffix = f' ({elapsed}초 경과)' if elapsed >= 2 else ''
                placeholder.markdown(
                    f'_생각 중{suffix}_\n\n> {reasoning_text}'
                )
            else:
                # No visible output yet — refresh the elapsed counter so
                # users can tell the connection isn't frozen.
                now = time.time()
                if now - last_idle_tick >= 1.0:
                    placeholder.markdown(
                        f'_생각 중… ({int(now - t0)}초 경과)_'
                    )
                    last_idle_tick = now
        placeholder.empty()
    except Exception:
        placeholder.empty()
        raise
    finally:
        st.session_state.pop('_stream_model_captured', None)
    return full_text, reasoning_text


def non_stream_chat(client, params: dict):
    resp = client.chat.completions.create(**params)
    _record_response_model(getattr(resp, 'model', None))
    choice = resp.choices[0]
    full_text = choice.message.content or ''
    reasoning = ''
    for attr in ('reasoning_content', 'reasoning'):
        val = getattr(choice.message, attr, None)
        if val:
            reasoning = val
            break
    return full_text, reasoning


def render_assistant(answer: str, reasoning: str, retrieved: list, turn_idx: int,
                     variants: list = None):
    if variants and len(variants) > 1:
        with st.expander(
            f'검색에 사용된 쿼리 변형 {len(variants)}개', expanded=False
        ):
            for i, v in enumerate(variants):
                label = '원본 질문' if i == 0 else f'변형 {i}'
                preview = v if len(v) <= 300 else v[:300] + '...'
                st.markdown(f'**{label}**: {preview}')
    if reasoning:
        with st.expander(f'추론 과정 (turn {turn_idx + 1})', expanded=False):
            st.markdown(reasoning)

    n = len(retrieved)
    pretty = format_answer_with_citations(answer, n) if n else (answer or '')
    st.markdown(pretty or '*(empty response)*')

    if not retrieved:
        return

    cited = parse_citations(answer, n)
    cited_items = [(i + 1, r) for i, r in enumerate(retrieved) if (i + 1) in cited]
    uncited_items = [(i + 1, r) for i, r in enumerate(retrieved) if (i + 1) not in cited]

    if cited_items:
        st.markdown(f"**출처** · 인용 {len(cited_items)} / 검색 {n}")
        for j, r in cited_items:
            with st.expander(_citation_summary(j, r), expanded=False):
                _citation_body(r)
        if uncited_items:
            with st.expander(
                f'인용되지 않은 검색 결과 {len(uncited_items)}개 보기', expanded=False
            ):
                for j, r in uncited_items:
                    st.markdown(f"**{_citation_summary(j, r)}**")
                    _citation_body(r)
                    st.divider()
    else:
        with st.expander(
            f'검색된 자료 {n}개 (모델이 [N] 인용 표기를 사용하지 않음)', expanded=False
        ):
            for i, r in enumerate(retrieved):
                st.markdown(f"**{_citation_summary(i + 1, r)}**")
                _citation_body(r)
                st.divider()


def _citation_summary(j: int, r: dict) -> str:
    from urllib.parse import urlparse
    if r.get('source') == 'web':
        host = urlparse(r.get('url', '')).netloc or '웹'
        title = (r.get('doc') or '').strip()
        return f"[{j}] 웹 · {host} — {title[:60]}"
    score = r.get('score', 0.0)
    pages = r.get('pages') or []
    page_part = _format_pages(pages)
    return (
        f"[{j}] {r.get('doc', '')}{page_part} · "
        f"chunk {r.get('chunk_idx', 0)} · score {score:.3f}"
    )


def _citation_body(r: dict):
    if r.get('source') == 'web':
        url = r.get('url', '')
        if url:
            st.markdown(f"[{url}]({url})")
    st.text(r.get('text', ''))


# -----------------------------------------------------------------------------
# Per-turn structured logging + auto-titler
# -----------------------------------------------------------------------------
# Both were originally in app.py because parse_citations / get_openai_client
# hadn't been split out yet. They live alongside the chat helpers now.

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


# -----------------------------------------------------------------------------
# LLM error → human-readable Korean explainer
# -----------------------------------------------------------------------------

def _show_llm_error(e: Exception):
    """Render a Korean, actionable error message for common LLM call failures.

    Pattern-matches the most frequent ones (HF Inference Providers permission
    missing, gated model, invalid token, quota, model not deployed, network)
    and falls back to the raw text otherwise."""
    err_str = str(e) or ''
    el = err_str.lower()

    # Persist the error to events.jsonl so failures aren't only visible
    # in the live UI. Best-effort — never raises.
    try:
        _log_event('llm_error', {
            'provider': st.session_state.get('provider', ''),
            'model': st.session_state.get('model', ''),
            'base_url': st.session_state.get('base_url', ''),
            'exception_type': type(e).__name__,
            'error': err_str[:1500],
        })
    except Exception:
        pass

    def show(headline, body_md):
        st.error(headline)
        st.markdown(body_md)
        with st.expander('원본 오류 메시지'):
            st.code(err_str[:1500] or repr(e))

    # HF Router — Inference Providers permission missing (403)
    if ('inference providers' in el
            and ('insufficient permissions' in el or 'does not have' in el
                 or 'this authentication method' in el)):
        show(
            'Hugging Face 토큰에 "Inference Providers" 권한이 없습니다.',
            """
**해결 방법:**

1. https://huggingface.co/settings/tokens 접속
2. 사용 중인 토큰 이름 클릭 → **Edit permissions** (또는 + Create new token → Fine-grained)
3. 다음 권한 체크:
   - ✅ **Make calls to Inference Providers** ← 필수
   - ✅ Make calls to the serverless Inference API (권장)
   - ✅ Read access to public repositories (자동 포함)
   - Llama / Gemma 같은 gated 모델 쓰면 → Read access to selected gated repositories 추가
4. **Save** → 설정 탭에서 Hugging Face 토큰 칸 갱신 (또는 .env / Cloud secrets 의 HF_TOKEN 갱신)
5. (Cloud 배포면) Manage app → Reboot
            """,
        )
        return

    # HF — gated model (Llama, Gemma 등) 라이선스 미수락 / gated 권한 누락
    if ('gated' in el or
            ('access to model' in el and ('granted' in el or 'requires' in el)) or
            ('is restricted' in el and 'license' in el)):
        show(
            'Gated 모델 접근 권한 없음.',
            """
**해결 방법:**

1. 해당 모델 페이지 (예: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 에서 라이선스 약관 수락
2. https://huggingface.co/settings/tokens → 토큰 권한에 **Read access to selected gated repositories** 추가하고 해당 모델 체크
3. 토큰 저장 → 앱에서 다시 시도
            """,
        )
        return

    # Fireworks AI — 모델 ID 오타 또는 계정 권한 부족
    if (_is_fireworks_endpoint() and
            ('not found' in el or 'inaccessible' in el or 'not deployed' in el)):
        show(
            'Fireworks 에서 이 모델을 찾을 수 없거나 계정에 접근 권한이 없습니다.',
            """
**해결 방법:**

1. **모델 ID 오타 확인** — Fireworks 모델 ID 는 항상 `accounts/fireworks/models/<name>` 풀 경로. 짧은 이름만 적으면 안 됩니다.
2. **계정 등급 제한** — Llama 405B 같은 일부 대형 모델은 유료 등급 / 신청 필요. 무료 계정에서는 70B 이하 권장.
3. **권장 동작 모델:** 설정 → Fireworks AI 모델 드롭다운에서:
   - `accounts/fireworks/models/llama-v3p3-70b-instruct` (기본)
   - `accounts/fireworks/models/qwen2p5-72b-instruct` (한국어 우수)
   - `accounts/fireworks/models/deepseek-v3` (강력)
4. 현재 본인 계정에서 호출 가능한 모델 전체 목록: https://fireworks.ai/models
            """,
        )
        return

    # Provider 미지원 / 모델 deploy 안 됨
    if 'not supported by any provider' in el or 'model_not_supported' in el:
        show(
            '이 모델을 서빙하는 활성 Provider가 없습니다.',
            """
**해결 방법:**

1. 설정 탭에서 다른 모델로 변경 (한국어가 강한 추천: `Qwen/Qwen3-Next-80B-A3B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`, `deepseek-ai/DeepSeek-V4-Pro`)
2. 또는 모델 카드 우측 "Inference Providers" 박스에서 서빙 가능한 provider 확인 후 https://huggingface.co/settings/inference-providers 에서 활성화 (Together AI / Cerebras / Hyperbolic 등)
            """,
        )
        return

    # OpenAI — Responses API 전용 모델 (gpt-5-pro, o1-pro 등) → Chat Completions 거부
    if 'v1/responses' in el or 'only supported in v1/responses' in el:
        show(
            '이 모델은 OpenAI 의 Responses API 전용 — 우리 앱은 호환되지 않습니다.',
            """
**원인:** `gpt-5-pro`, `o1-pro` 같은 "Pro" 등급 일부 모델은 OpenAI 의 새 `/v1/responses` 엔드포인트로만 제공됩니다. 본 앱은 표준 `/v1/chat/completions` 를 사용해 호출 자체가 거부됩니다.

**해결: 설정 → OpenAI 모델 변경.** 다음은 Chat Completions 로 정상 동작합니다:

- `gpt-5` — Pro 의 약 7할 성능, 같은 추론
- `gpt-5-mini` — 빠르고 저렴, 일반 RAG 충분
- `gpt-4.1` — 안정적인 차세대
- `o3` — 추론 강화 (논문 분석 / 수학에 강함)
- `o4-mini` — 추론 + 빠름
            """,
        )
        return

    # 토큰 자체가 무효 / 만료
    if ('invalid' in el and 'token' in el) or 'bad credentials' in el or '401' in el:
        show(
            'API 키 / 토큰 인증 실패 (401).',
            """
**해결 방법:**

1. 설정 탭에서 현재 공급자의 API 키가 비어있지 않은지 확인
2. 토큰이 만료됐다면 발급처에서 새로 만들기:
   - Hugging Face: https://huggingface.co/settings/tokens
   - OpenAI: https://platform.openai.com/api-keys
3. 새 키로 갱신 → 다시 시도
            """,
        )
        return

    # 결제 / 쿼터 초과
    if ('quota' in el or 'rate limit' in el or 'insufficient_quota' in el
            or '429' in err_str or '402' in err_str):
        show(
            '사용량 / 쿼터 초과 또는 결제 필요.',
            """
**해결 방법:**

- OpenAI: https://platform.openai.com/account/billing 에서 결제 / 한도 확인
- HF Inference Providers: provider 별로 무료 크레딧 한도 다름. 다른 provider 활성화 시도.
- 잠시 후 재시도하거나 더 작은 모델로 변경.
            """,
        )
        return

    # 네트워크 / 타임아웃
    if 'timeout' in el or 'timed out' in el or 'connection' in el:
        show(
            '네트워크 / 타임아웃.',
            '잠시 후 다시 시도해 주세요. 문제가 지속되면 다른 공급자로 변경.',
        )
        return

    # Fallback
    st.error(f'요청 실패: {e}')


# -----------------------------------------------------------------------------
# Top-level turn handler
# -----------------------------------------------------------------------------

def handle_chat_turn(user_input: str):
    """Process one user turn: retrieve, call LLM, render, persist to history."""
    if (not st.session_state['model']
            or not _active_api_key()
            or not st.session_state['base_url']):
        provider = st.session_state.get('provider', 'Hugging Face Router')
        st.error(
            f'설정 탭에서 모델 · 엔드포인트 · {provider} 용 API 키를 모두 입력해 주세요.'
        )
        return

    general_only = bool(st.session_state.get('general_chat_mode'))
    local_active = (not general_only) and bool(st.session_state['docs'])
    web_active = (not general_only) and bool(st.session_state['web_enabled'])
    rag_active = local_active or web_active

    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        retrieved = []
        if rag_active:
            if local_active and web_active:
                msg = '문서 및 웹 검색 중...'
            elif web_active:
                msg = '웹 검색 중...'
            else:
                msg = '문서 검색 중...'
            # Apply chat-time document filter (None = use all)
            all_docs_full = st.session_state['docs']
            chat_filter = st.session_state.get('chat_doc_filter')
            need_filter = (
                chat_filter is not None
                and 0 < len(chat_filter) < len(all_docs_full)
            )
            if need_filter:
                filt_set = set(chat_filter)
                st.session_state['docs'] = [
                    d for d in all_docs_full if d['id'] in filt_set
                ]
            try:
                with st.spinner(msg):
                    retrieved = retrieve(user_input)
            finally:
                if need_filter:
                    st.session_state['docs'] = all_docs_full

        messages = build_messages(user_input, retrieved)

        is_openai = _is_openai_endpoint()
        is_dashscope = _is_dashscope_endpoint()
        extra_body = {}
        if _provider_supports_top_k():
            extra_body['top_k'] = int(st.session_state['sampling_top_k'])
        if not st.session_state['enable_thinking']:
            if is_dashscope:
                extra_body['enable_thinking'] = False
            elif not is_openai and not _is_hf_router_endpoint():
                # vLLM / SGLang / self-hosted only. HF Router providers vary
                # and several (e.g. Cerebras-served Llama / gpt-oss) reject
                # chat_template_kwargs with HTTP 400.
                extra_body['chat_template_kwargs'] = {'enable_thinking': False}

        params = _build_completion_params(
            model=st.session_state['model'],
            messages=messages,
            max_tokens=st.session_state['max_tokens'],
            temperature=st.session_state['temperature'],
            top_p=st.session_state['top_p'],
            presence_penalty=st.session_state['presence_penalty'],
            extra_body=extra_body,
        )

        full_text, reasoning_text = '', ''
        elapsed_sec = None
        try:
            client = get_openai_client()
            _t0 = time.time()
            # Agentic mode: only meaningful when we have local docs to search
            # against. Web-only retrieval would need a different tool.
            if st.session_state.get('use_agentic_search') and local_active:
                max_iters = int(st.session_state.get('agentic_max_iters', 3))
                full_text, reasoning_text, retrieved = agentic_chat_pass(
                    client, params, retrieved, max_iters=max_iters,
                )
            elif st.session_state['stream']:
                full_text, reasoning_text = stream_chat(client, params)
            else:
                with st.spinner('생각 중…'):
                    full_text, reasoning_text = non_stream_chat(client, params)
            elapsed_sec = time.time() - _t0
        except Exception as e:
            _show_llm_error(e)
            full_text, reasoning_text = '', ''

        if full_text and not reasoning_text:
            rt, ct = split_thinking(full_text)
            if rt:
                reasoning_text = rt
                full_text = ct

        if full_text or reasoning_text:
            turn_variants = st.session_state.get('_last_variants') or [user_input]
            st.session_state['user_inputs'].append(user_input)
            st.session_state['generated_responses'].append(full_text)
            st.session_state['thinking_traces'].append(reasoning_text)
            st.session_state['retrieved_per_turn'].append(retrieved)
            st.session_state['query_variants_per_turn'].append(turn_variants)
            render_assistant(
                full_text, reasoning_text, retrieved,
                len(st.session_state['generated_responses']) - 1,
                variants=turn_variants,
            )
            # Persist this conversation. Create a session ID on first turn.
            if not st.session_state.get('current_session_id'):
                st.session_state['current_session_id'] = _new_session_id()
                st.session_state['current_session_created_at'] = (
                    datetime.datetime.now().isoformat()
                )
            save_current_session()
            # Auto-title once, after the very first turn.
            if (len(st.session_state['user_inputs']) == 1
                    and not st.session_state.get('current_session_title')):
                title = auto_title_session()
                if title:
                    st.session_state['current_session_title'] = title
                    save_current_session()
            # Append this turn to the per-session structured JSONL log.
            log_turn_structured(
                user_input=user_input,
                response_text=full_text,
                reasoning=reasoning_text,
                retrieved=retrieved,
                model=st.session_state['model'],
                elapsed_sec=elapsed_sec,
                query_variants=turn_variants,
            )
