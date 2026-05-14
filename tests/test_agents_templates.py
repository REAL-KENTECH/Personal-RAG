"""Agent prompt builders are pure functions: take (inputs, retrieved) → messages."""

from agents.templates import (
    AGENT_TASKS,
    _agent_format_context,
    _analysis_build,
    _comparison_build,
    _email_build,
    _report_build,
    _summary_build,
)


# -----------------------------------------------------------------------------
# _agent_format_context
# -----------------------------------------------------------------------------

def test_format_context_empty():
    assert _agent_format_context([]) == '(검색된 근거 자료 없음)'


def test_format_context_web_entry():
    out = _agent_format_context([
        {'source': 'web', 'doc': 'Wikipedia', 'url': 'https://en.wikipedia.org/x',
         'text': 'hello'},
    ])
    assert '[1]' in out
    assert '웹:' in out
    assert 'Wikipedia' in out
    assert 'hello' in out


def test_format_context_local_doc():
    out = _agent_format_context([
        {'source': 'doc', 'doc': 'report.pdf', 'pages': [3], 'text': 'finding'},
    ])
    assert '[1]' in out
    assert '로컬:' in out
    assert 'report.pdf' in out
    assert 'p.3' in out


def test_format_context_local_doc_page_range():
    out = _agent_format_context([
        {'source': 'doc', 'doc': 'a.pdf', 'pages': [2, 3, 4], 'text': 't'},
    ])
    assert 'pp.2-4' in out


def test_format_context_mixed():
    out = _agent_format_context([
        {'source': 'doc', 'doc': 'a.pdf', 'pages': [1], 'text': 'A'},
        {'source': 'web', 'doc': 'site', 'url': 'http://x', 'text': 'B'},
    ])
    assert '[1]' in out and '[2]' in out


# -----------------------------------------------------------------------------
# Per-task builders — all return [{system}, {user}]
# -----------------------------------------------------------------------------

def _assert_messages_shape(msgs):
    assert isinstance(msgs, list)
    assert len(msgs) == 2
    assert msgs[0]['role'] == 'system'
    assert msgs[1]['role'] == 'user'
    assert msgs[0]['content'] and msgs[1]['content']


def test_email_build_returns_chat_messages():
    msgs = _email_build(
        {'recipient': '김부장님', 'subject_hint': 'Q1 결과', 'tone': '공식',
         'length': '보통', 'key_points': '매출 +10%'},
        [],
    )
    _assert_messages_shape(msgs)
    assert '이메일' in msgs[0]['content']
    assert '김부장님' in msgs[1]['content']


def test_report_build_returns_chat_messages():
    msgs = _report_build(
        {'topic': '시장 분석', 'audience': '임원', 'length': '중간', 'sections': ''},
        [{'source': 'doc', 'doc': 'a.pdf', 'pages': [1], 'text': 'data'}],
    )
    _assert_messages_shape(msgs)
    assert '시장 분석' in msgs[1]['content']
    assert '[1]' in msgs[1]['content']  # context block included


def test_summary_build_returns_chat_messages():
    msgs = _summary_build({'focus': '리스크', 'depth': '단락 요약'}, [])
    _assert_messages_shape(msgs)
    assert '리스크' in msgs[1]['content']


def test_analysis_build_returns_chat_messages():
    msgs = _analysis_build(
        {'question': '매출 추이는?', 'context_note': ''},
        [],
    )
    _assert_messages_shape(msgs)
    assert '매출 추이' in msgs[1]['content']


def test_comparison_build_returns_chat_messages():
    msgs = _comparison_build(
        {'item_a': '제품 A', 'item_b': '제품 B', 'criteria': ''},
        [],
    )
    _assert_messages_shape(msgs)
    assert '제품 A' in msgs[1]['content']
    assert '제품 B' in msgs[1]['content']


# -----------------------------------------------------------------------------
# AGENT_TASKS registry contract
# -----------------------------------------------------------------------------

EXPECTED_TASKS = {'email', 'report', 'summary', 'analysis', 'comparison'}
REQUIRED_KEYS = {
    'label', 'description', 'fields', 'requires_docs',
    'search_query', 'build_messages', 'output_ext',
}


def test_agent_tasks_complete():
    assert set(AGENT_TASKS.keys()) == EXPECTED_TASKS


def test_each_task_has_required_keys():
    for name, task in AGENT_TASKS.items():
        missing = REQUIRED_KEYS - set(task.keys())
        assert not missing, f'task {name} missing keys: {missing}'


def test_each_task_search_query_callable():
    for name, task in AGENT_TASKS.items():
        # Should accept the user's input dict and return a string
        q = task['search_query']({'topic': 't', 'question': 'q', 'focus': 'f',
                                  'subject_hint': 's', 'key_points': 'k',
                                  'item_a': 'A', 'item_b': 'B', 'sections': ''})
        assert isinstance(q, str), f'{name}.search_query did not return str'


def test_each_task_build_messages_callable():
    for name, task in AGENT_TASKS.items():
        msgs = task['build_messages']({}, [])
        _assert_messages_shape(msgs)
