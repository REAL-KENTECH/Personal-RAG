"""Citation parsing + reasoning extraction are pure regex on strings."""

from llm.chat.citations import (
    format_answer_with_citations,
    parse_citations,
    split_thinking,
)


# -----------------------------------------------------------------------------
# split_thinking
# -----------------------------------------------------------------------------

def test_split_no_thinking_tag():
    reasoning, content = split_thinking('Just a plain answer.')
    assert reasoning == ''
    assert content == 'Just a plain answer.'


def test_split_think_tag():
    reasoning, content = split_thinking('<think>internal reasoning</think>Final.')
    assert reasoning == 'internal reasoning'
    assert content == 'Final.'


def test_split_think_tag_multiline():
    src = '<think>line1\nline2</think>\n\nanswer body'
    reasoning, content = split_thinking(src)
    assert 'line1' in reasoning and 'line2' in reasoning
    assert content == 'answer body'


def test_split_empty_input():
    assert split_thinking('') == ('', '')


# -----------------------------------------------------------------------------
# parse_citations
# -----------------------------------------------------------------------------

def test_parse_single_citation():
    assert parse_citations('See [1] above.', n_chunks=3) == {1}


def test_parse_multiple_singles():
    assert parse_citations('Mix [1] and [2] together.', n_chunks=3) == {1, 2}


def test_parse_grouped():
    assert parse_citations('Together [1, 2, 3].', n_chunks=3) == {1, 2, 3}


def test_parse_out_of_range_filtered():
    # n_chunks=3 → [10] is out of range, [2] valid
    assert parse_citations('See [2] and [10].', n_chunks=3) == {2}


def test_parse_no_citations():
    assert parse_citations('No markers here.', n_chunks=3) == set()


def test_parse_empty_string():
    assert parse_citations('', n_chunks=5) == set()


def test_parse_none_chunks_disables():
    # With n_chunks=0, no marker can be valid
    assert parse_citations('See [1], [2], [3].', n_chunks=0) == set()


# -----------------------------------------------------------------------------
# format_answer_with_citations
# -----------------------------------------------------------------------------

def test_format_wraps_valid_markers():
    out = format_answer_with_citations('Then [1] proved it.', n_chunks=3)
    assert out == 'Then **[1]** proved it.'


def test_format_leaves_out_of_range_alone():
    out = format_answer_with_citations('See [99] reference.', n_chunks=3)
    assert out == 'See [99] reference.'


def test_format_grouped_marker_wrapped_when_any_valid():
    # Grouped form: if any index in range, wrap the whole marker
    out = format_answer_with_citations('See [1, 99].', n_chunks=3)
    assert out == 'See **[1, 99]**.'


def test_format_empty_returns_empty():
    assert format_answer_with_citations('', n_chunks=3) == ''


def test_format_none_returns_empty():
    assert format_answer_with_citations(None, n_chunks=3) == ''
