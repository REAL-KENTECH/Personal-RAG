"""Chunking is pure — no Streamlit, no session_state — so this is the
easiest layer to test thoroughly."""

import pytest

from processing.chunking import (
    _expand_into_paragraphs,
    _split_long_text_to_sentences,
    chunk_elements,
    chunk_text,
)


# -----------------------------------------------------------------------------
# _split_long_text_to_sentences
# -----------------------------------------------------------------------------

def test_split_empty():
    assert _split_long_text_to_sentences('') == []


def test_split_english_periods():
    out = _split_long_text_to_sentences('First sentence. Second one! Third?')
    assert out == ['First sentence.', 'Second one!', 'Third?']


def test_split_korean_endings():
    # 다 / 요 / 음 followed by newline are sentence boundaries
    out = _split_long_text_to_sentences('이것은 첫 문장이다\n\n다음 문장입니다')
    assert len(out) >= 2
    assert any('첫 문장' in s for s in out)
    assert any('다음 문장' in s for s in out)


# -----------------------------------------------------------------------------
# _expand_into_paragraphs
# -----------------------------------------------------------------------------

def test_expand_single_paragraph_unchanged():
    el = [{'text': 'One paragraph only.', 'page': 1}]
    assert _expand_into_paragraphs(el) == [{'text': 'One paragraph only.', 'page': 1}]


def test_expand_double_newline_splits():
    el = [{'text': 'Para one.\n\nPara two.\n\nPara three.', 'page': 2}]
    out = _expand_into_paragraphs(el)
    assert len(out) == 3
    assert all(p['page'] == 2 for p in out)  # page metadata preserved
    assert [p['text'] for p in out] == ['Para one.', 'Para two.', 'Para three.']


def test_expand_drops_empty_paragraphs():
    el = [{'text': 'A\n\n   \n\nB', 'page': None}]
    out = _expand_into_paragraphs(el)
    assert [p['text'] for p in out] == ['A', 'B']


# -----------------------------------------------------------------------------
# chunk_elements
# -----------------------------------------------------------------------------

def test_chunk_empty():
    chunks, pages = chunk_elements([], size=100, overlap=10)
    assert chunks == [] and pages == []


def test_chunk_short_input_single_chunk():
    el = [{'text': 'short', 'page': 1}]
    chunks, pages = chunk_elements(el, size=200, overlap=20)
    assert chunks == ['short']
    assert pages == [[1]]


def test_chunk_preserves_page_metadata():
    el = [{'text': 'aaa', 'page': 1}, {'text': 'bbb', 'page': 2}]
    chunks, pages = chunk_elements(el, size=200, overlap=20)
    # Both fit in one chunk; union of pages is preserved
    assert chunks == ['aaa\n\nbbb']
    assert pages == [[1, 2]]


def test_chunk_splits_when_oversize():
    # Two paragraphs that together exceed size → produce 2 chunks
    el = [{'text': 'a' * 80, 'page': 1}, {'text': 'b' * 80, 'page': 2}]
    chunks, pages = chunk_elements(el, size=100, overlap=10)
    assert len(chunks) >= 2
    # Each chunk's page list points to the right source
    flat_pages = sorted({p for chunk_pgs in pages for p in chunk_pgs})
    assert flat_pages == [1, 2]


def test_chunk_size_floor_applied():
    # size < 50 gets bumped to 50 (chunker hard minimum)
    el = [{'text': 'x' * 200, 'page': 1}]
    chunks, _ = chunk_elements(el, size=10, overlap=0)
    # Should not produce 200 single-char chunks
    assert max(len(c) for c in chunks) >= 50


# -----------------------------------------------------------------------------
# chunk_text wrapper
# -----------------------------------------------------------------------------

def test_chunk_text_returns_strings_only():
    chunks = chunk_text('hello world', size=200, overlap=20)
    assert chunks == ['hello world']
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_empty_returns_empty():
    assert chunk_text('', size=100, overlap=10) == []
