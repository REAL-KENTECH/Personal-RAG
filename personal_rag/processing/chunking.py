"""Paragraph- and sentence-aware chunking with page-metadata preservation.

The chunker takes a list of ``{text, page}`` elements (produced by the
parsers) and groups them into chunks of approximately ``chunk_size``
characters. Adjacent short paragraphs are merged; a paragraph longer
than the chunk size is split on sentence boundaries — including the
most common Korean sentence endings — so chunks don't end mid-sentence.
"""

import re


_SENT_SPLIT_RE = re.compile(
    r'(?<=[.!?。!?])\s+(?=\S)|(?<=[다요죠음음됨함임함])\s*\n+\s*(?=\S)'
)


def _split_long_text_to_sentences(text: str) -> list:
    """Split a long block of text into rough sentence-like spans.

    Uses Latin sentence terminators (.!?) and the most common Korean sentence
    endings (다/요/죠/음/됨/함) followed by whitespace/newline. Output is a
    plain list of non-empty strings preserving original order; callers can
    glue them back together up to the size budget.
    """
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    # Some splits leave empty strings or leading/trailing whitespace.
    return [p.strip() for p in parts if p and p.strip()]


def _expand_into_paragraphs(elements: list) -> list:
    """Pre-process: explode any element whose text contains paragraph breaks
    into multiple smaller elements sharing the same page metadata.

    pypdf often returns one element per page = whole-page text glued together.
    Docling normally pre-splits at the layout level, so this is mostly a
    no-op for Docling input. The downstream chunker then groups paragraphs
    instead of being forced into raw char-step splits.
    """
    out = []
    for el in elements:
        text = (el.get('text') or '').strip()
        if not text:
            continue
        page = el.get('page')
        # Treat any run of newlines (with optional whitespace) as a boundary.
        paragraphs = re.split(r'\n\s*\n+', text)
        for p in paragraphs:
            p = p.strip()
            if p:
                out.append({'text': p, 'page': page})
    return out


def chunk_elements(elements: list, size: int, overlap: int):
    """Group elements (each {text, page}) into chunks of ~size chars.

    Paragraph-aware: adjacent short paragraphs are merged up to size; long
    paragraphs are split on sentence boundaries (not raw character offsets)
    so chunks don't end mid-sentence. Page metadata is preserved.

    Returns (chunks: list[str], chunk_pages: list[list[int]]).
    """
    if not elements:
        return [], []
    size = max(50, int(size))
    overlap = max(0, min(int(overlap), size - 1))

    # Pre-explode whole-page elements into paragraphs (no-op for already-fine
    # input — keeps Docling-style elements untouched).
    elements = _expand_into_paragraphs(elements)

    chunks, chunk_pages = [], []
    buf_text, buf_pages = '', set()

    def flush():
        nonlocal buf_text, buf_pages
        if buf_text:
            chunks.append(buf_text)
            chunk_pages.append(sorted(buf_pages))
            buf_text, buf_pages = '', set()

    for el in elements:
        text = (el.get('text') or '').strip()
        if not text:
            continue
        page = el.get('page')
        candidate = (buf_text + '\n\n' + text).strip() if buf_text else text
        if len(candidate) <= size:
            buf_text = candidate
            if page is not None:
                buf_pages.add(page)
            continue
        # candidate exceeds size: emit current buffer first.
        flush()
        if len(text) <= size:
            buf_text = text
            buf_pages = {page} if page is not None else set()
            continue
        # A single paragraph is longer than size — split by sentence boundary.
        sentences = _split_long_text_to_sentences(text)
        if not sentences:
            # Last-resort fallback: char-step (matches old behavior).
            step = max(1, size - overlap)
            for i in range(0, len(text), step):
                chunks.append(text[i:i + size])
                chunk_pages.append([page] if page is not None else [])
            continue
        sbuf = ''
        for s in sentences:
            cand = (sbuf + ' ' + s).strip() if sbuf else s
            if len(cand) <= size:
                sbuf = cand
                continue
            if sbuf:
                chunks.append(sbuf)
                chunk_pages.append([page] if page is not None else [])
                # Sentence-level overlap: keep tail of previous chunk if small.
                if overlap and len(sbuf) > overlap:
                    sbuf = sbuf[-overlap:] + ' ' + s
                else:
                    sbuf = s
            else:
                # A single sentence longer than size — char-step it.
                step = max(1, size - overlap)
                for i in range(0, len(s), step):
                    chunks.append(s[i:i + size])
                    chunk_pages.append([page] if page is not None else [])
                sbuf = ''
        if sbuf:
            chunks.append(sbuf)
            chunk_pages.append([page] if page is not None else [])
    flush()

    # Merge tiny trailing chunk into its predecessor.
    if len(chunks) >= 2 and len(chunks[-1]) < max(50, size // 5):
        chunks[-2] = (chunks[-2] + '\n\n' + chunks[-1])[: size + overlap]
        merged_pages = sorted(set(chunk_pages[-2]) | set(chunk_pages[-1]))
        chunk_pages[-2] = merged_pages
        chunks.pop()
        chunk_pages.pop()

    return chunks, chunk_pages


def chunk_text(text: str, size: int, overlap: int) -> list:
    """Backward-compatible: plain-text chunking returning list of strings only."""
    chunks, _pages = chunk_elements(
        [{'text': text, 'page': None}], size, overlap
    )
    return chunks
