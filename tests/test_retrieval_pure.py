"""Pure-function pieces of the retrieval pipeline — no Streamlit needed.

Streamlit-coupled scorers (dense_search, bm25_search, rerank,
retrieve_local) are exercised by manual smoke through the running app
since they depend on session state + loaded models.
"""

from retrieval.pipeline import _is_comparison_query, _select_with_per_doc_min
from retrieval.search import _tokenize_for_bm25, rrf_fuse


# -----------------------------------------------------------------------------
# _tokenize_for_bm25
# -----------------------------------------------------------------------------

def test_tokenize_ascii_lowercased():
    assert _tokenize_for_bm25('Hello World') == ['hello', 'world']


def test_tokenize_korean_preserved():
    out = _tokenize_for_bm25('안녕하세요 반가워요')
    assert '안녕하세요' in out and '반가워요' in out


def test_tokenize_mixed_script():
    out = _tokenize_for_bm25('Foo 한국어 123 中文')
    # Order preserved, all tokens captured
    assert out == ['foo', '한국어', '123', '中文']


def test_tokenize_punctuation_stripped():
    assert _tokenize_for_bm25('hello, world!') == ['hello', 'world']


def test_tokenize_empty():
    assert _tokenize_for_bm25('') == []


def test_tokenize_none():
    assert _tokenize_for_bm25(None) == []


# -----------------------------------------------------------------------------
# rrf_fuse — Reciprocal Rank Fusion
# -----------------------------------------------------------------------------

def _item(idx, score=0.0, doc_id='doc1', name='d', chunk_idx=0, text='t'):
    """Helper — build the (idx, score, meta, chunk) tuple shape that
    rankers produce."""
    return (idx, score, (doc_id, name, chunk_idx), text)


def test_rrf_single_ranking_preserves_order():
    ranking = [_item(1, 0.9), _item(2, 0.5), _item(3, 0.1)]
    out = rrf_fuse([ranking])
    assert [r[0] for r in out] == [1, 2, 3]


def test_rrf_two_rankings_merge():
    ra = [_item(1, 0.9), _item(2, 0.5)]
    rb = [_item(2, 0.8), _item(3, 0.4)]
    out = rrf_fuse([ra, rb])
    # All three idx present, idx=2 should rank first (appears in both)
    ids = [r[0] for r in out]
    assert set(ids) == {1, 2, 3}
    assert ids[0] == 2


def test_rrf_empty_rankings_returns_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


# -----------------------------------------------------------------------------
# _is_comparison_query
# -----------------------------------------------------------------------------

def test_comparison_korean_차이():
    assert _is_comparison_query('A와 B의 차이는?')


def test_comparison_korean_비교():
    assert _is_comparison_query('두 모델을 비교해줘')


def test_comparison_english_compare():
    assert _is_comparison_query('Compare A and B')


def test_comparison_english_vs():
    assert _is_comparison_query('GPT-4 vs Claude')


def test_comparison_negative_normal_query():
    assert not _is_comparison_query('이 문서의 핵심은?')
    assert not _is_comparison_query('What is RAG?')


def test_comparison_empty():
    assert not _is_comparison_query('')


# -----------------------------------------------------------------------------
# _select_with_per_doc_min
# -----------------------------------------------------------------------------

def test_select_per_doc_min_zero_returns_top_k():
    items = [_item(1), _item(2), _item(3), _item(4)]
    out = _select_with_per_doc_min(items, top_k=2, per_doc_min=0)
    assert out == items[:2]


def test_select_balances_across_docs():
    # Doc A dominates top of list. With per_doc_min=1, doc B must get at
    # least 1 slot before doc A fills everything.
    items = [
        _item(1, doc_id='A'),
        _item(2, doc_id='A'),
        _item(3, doc_id='A'),
        _item(4, doc_id='B'),
        _item(5, doc_id='B'),
    ]
    out = _select_with_per_doc_min(items, top_k=3, per_doc_min=1)
    doc_ids = [r[2][0] for r in out]
    # Each doc represented at least once
    assert 'A' in doc_ids and 'B' in doc_ids
    assert len(out) == 3


def test_select_top_k_zero_returns_empty():
    items = [_item(1), _item(2)]
    assert _select_with_per_doc_min(items, top_k=0, per_doc_min=1) == []
