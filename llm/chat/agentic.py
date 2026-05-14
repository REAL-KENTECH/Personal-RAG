"""Agentic RAG — let the LLM issue follow-up searches via a tool call.

The assistant gets one tool — ``search_documents(query)`` — and may invoke
it up to N times to fetch additional context before producing the final
answer. Each tool result is appended back into the message history so
subsequent turns of the loop see all prior searches.

Always non-streaming (tool-call streaming is brittle across providers).
Falls back to a plain non-stream call without ``tools`` if the provider
rejects tool calling (some HF Router-served small models do).
"""

import json

import streamlit as st

from retrieval.pipeline import retrieve_local


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
    search_documents to refine context. Returns
    (full_text, reasoning_text, augmented_retrieved)."""
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
