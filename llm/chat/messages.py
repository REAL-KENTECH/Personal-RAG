"""Chat message assembly — system prompt, history, context block, image attachments.

``build_messages`` is the single function chat code calls before sending a
turn to the LLM. It threads the conversation history, the retrieved
context with [N] markers, and (optionally) per-page PDF images into the
OpenAI ChatCompletions message format.
"""

import streamlit as st

from data.storage import load_page_image_b64


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
