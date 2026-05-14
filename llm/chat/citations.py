"""Reasoning extraction + [N] citation parsing.

All pure functions — no Streamlit, no I/O. Tested in tests/test_citations.py.
"""

import re


_CITE_PATTERN = r'\[((?:\d+\s*,\s*)*\d+)\]'


def split_thinking(text: str):
    """Separate reasoning/thinking content from the final answer.

    Handles two common formats:
    - <think>...</think>final answer  (Qwen3, DeepSeek-R1, etc.)
    - <|channel|>thought...<|channel|>final  (gpt-oss style)

    Returns (reasoning, content) — both strings, either may be empty.
    """
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


def parse_citations(answer: str, n_chunks: int) -> set:
    """Return the set of valid citation indices the model used in ``answer``.

    Accepts both single and grouped forms: [1], [2,3], [1, 2, 3]. Only
    indices in the range 1..n_chunks are returned (out-of-range markers
    in the model output are silently dropped).
    """
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
    """Bold-wrap [N] markers that point to a valid chunk index.

    Markers with at least one valid index inside are wrapped in **...**;
    invalid markers (all indices out of range) pass through unchanged.
    """
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
