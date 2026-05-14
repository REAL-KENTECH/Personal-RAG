"""Prompt builders + the AGENT_TASKS registry.

Each template is a pure function: takes ``(inputs, retrieved)`` and
returns a ``[{role, content}, ...]`` message list. The ``AGENT_TASKS``
dict declares everything the view layer needs to render the form and
the runner needs to execute the task (search query formula, build
function, whether docs are required, output extension).
"""


def _agent_format_context(retrieved: list) -> str:
    """Format retrieved chunks as numbered context lines for the LLM prompt."""
    if not retrieved:
        return '(검색된 근거 자료 없음)'
    lines = []
    for i, r in enumerate(retrieved, start=1):
        if r.get('source') == 'web':
            lines.append(f"[{i}] (웹: {r.get('doc', '')} {r.get('url', '')}) {r.get('text', '')}")
        else:
            pages = r.get('pages') or []
            page_str = ''
            if pages:
                page_str = (f' p.{pages[0]}' if len(pages) == 1
                            else f' pp.{pages[0]}-{pages[-1]}')
            lines.append(
                f"[{i}] (로컬: {r.get('doc', '')}{page_str}) {r.get('text', '')}"
            )
    return '\n\n'.join(lines)


def _email_build(inputs, retrieved):
    sys_msg = (
        '당신은 한국어 비즈니스 이메일 작성 도우미입니다. 사용자가 제공한 Context '
        '(사용자 문서에서 발췌)를 사실 근거로 사용해, 요청된 톤·길이·핵심 포인트에 '
        '맞춰 이메일 초안을 작성합니다. 형식은 "제목: ..." 한 줄 + 빈 줄 + 본문. '
        'Context에 사실이 부족하면 추측하지 말고 자연스럽게 일반론으로 처리하세요.'
    )
    user_msg = (
        f"작업: 이메일 초안 작성\n"
        f"수신자: {inputs.get('recipient') or '미지정'}\n"
        f"주제 힌트: {inputs.get('subject_hint') or '미지정'}\n"
        f"톤: {inputs.get('tone') or '공식'}\n"
        f"분량: {inputs.get('length') or '보통'}\n\n"
        f"꼭 포함할 핵심 요점:\n{inputs.get('key_points') or '(자유)'}\n\n"
        f"Context (참고 자료):\n{_agent_format_context(retrieved)}\n\n"
        f"이제 이메일 초안을 작성해 주세요."
    )
    return [{'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': user_msg}]


def _report_build(inputs, retrieved):
    sys_msg = (
        '당신은 보고서 작성 전문가입니다. 제공된 Context를 사실 근거로 사용해 마크다운 '
        '구조의 보고서를 작성합니다. 모든 사실 주장에는 출처 번호를 [1], [2] 형식으로 '
        '인용하세요. 섹션 제목은 ##, 소제목은 ###을 사용하세요. Context에 없는 사실은 '
        '추측하지 말고 "추가 자료 필요"로 표시하세요.'
    )
    user_msg = (
        f"작업: 보고서 작성\n"
        f"주제: {inputs.get('topic') or '미지정'}\n"
        f"대상 독자: {inputs.get('audience') or '일반'}\n"
        f"분량: {inputs.get('length') or '중간 (3~5 섹션)'}\n\n"
        f"포함해야 할 섹션·관점:\n{inputs.get('sections') or '(자동 구성)'}\n\n"
        f"Context (참고 자료):\n{_agent_format_context(retrieved)}\n\n"
        f"위 자료를 근거로 보고서를 작성해 주세요. 마크다운 구조."
    )
    return [{'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': user_msg}]


def _summary_build(inputs, retrieved):
    sys_msg = (
        '당신은 한국어 문서 요약 전문가입니다. 제공된 Context를 종합해 요청된 깊이·초점에 '
        '맞게 요약합니다. 출처는 [1], [2]로 인용하고, 핵심 포인트를 글머리 기호로 정리한 후 '
        '단락 형식의 본문 요약을 덧붙이세요.'
    )
    user_msg = (
        f"작업: 문서 요약\n"
        f"초점: {inputs.get('focus') or '전반'}\n"
        f"깊이: {inputs.get('depth') or '단락 요약'}\n\n"
        f"Context:\n{_agent_format_context(retrieved)}\n\n"
        f"요약해 주세요."
    )
    return [{'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': user_msg}]


def _analysis_build(inputs, retrieved):
    sys_msg = (
        '당신은 데이터/통찰 분석 도우미입니다. 제공된 Context에서 수치·표·핵심 사실을 '
        '추출하고, 사용자의 질문에 대해 마크다운 표 + 해석 + 통찰 순으로 답변합니다. '
        '모든 수치 인용에는 [1], [2] 형식의 출처 번호를 붙이세요. Context에 없는 추정은 '
        '"가정"으로 명시하세요.'
    )
    user_msg = (
        f"작업: 데이터/통찰 분석\n"
        f"분석 질문: {inputs.get('question') or '(미지정)'}\n"
        f"추가 맥락: {inputs.get('context_note') or '(없음)'}\n\n"
        f"Context:\n{_agent_format_context(retrieved)}\n\n"
        f"분석을 진행해 주세요. 구조: (1) 핵심 수치 표, (2) 해석, (3) 통찰/시사점."
    )
    return [{'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': user_msg}]


def _comparison_build(inputs, retrieved):
    sys_msg = (
        '당신은 비교 분석 도우미입니다. 두 대상에 대해 마크다운 비교 표를 만들고, '
        '각 항목 차이의 의미를 해석합니다. 모든 사실에 [N]으로 출처를 표기하세요. '
        'Context에 한쪽 정보가 부족하면 "정보 부족"으로 표시하세요.'
    )
    user_msg = (
        f"작업: 비교 분석\n"
        f"대상 A: {inputs.get('item_a') or '(미지정)'}\n"
        f"대상 B: {inputs.get('item_b') or '(미지정)'}\n"
        f"비교 기준 (필요시): {inputs.get('criteria') or '(자동 선택)'}\n\n"
        f"Context:\n{_agent_format_context(retrieved)}\n\n"
        f"비교 분석을 진행해 주세요. 구조: (1) 비교 표, (2) 주요 차이 해석, (3) 결론."
    )
    return [{'role': 'system', 'content': sys_msg},
            {'role': 'user', 'content': user_msg}]


AGENT_TASKS = {
    'email': {
        'label': '이메일 초안',
        'description': '문서 내용을 참고해 비즈니스 이메일 초안을 작성합니다.',
        'fields': [
            {'key': 'recipient', 'label': '수신자', 'type': 'text',
             'placeholder': '예: 박팀장님 / 거래처 ABC사 김부장님'},
            {'key': 'subject_hint', 'label': '주제 / 목적', 'type': 'text',
             'placeholder': '예: Q1 실적 공유 및 다음 분기 전략 미팅 일정 제안'},
            {'key': 'tone', 'label': '톤', 'type': 'select',
             'options': ['공식', '친근', '간결']},
            {'key': 'length', 'label': '분량', 'type': 'select',
             'options': ['짧게', '보통', '상세']},
            {'key': 'key_points', 'label': '꼭 포함할 핵심 요점', 'type': 'textarea',
             'placeholder': '한 줄에 한 가지씩 적어주세요.'},
        ],
        'requires_docs': False,
        'search_query': lambda i: f"{i.get('subject_hint', '')} {i.get('key_points', '')}".strip(),
        'build_messages': _email_build,
        'output_ext': 'md',
    },
    'report': {
        'label': '보고서 작성',
        'description': '문서를 근거로 구조화된 마크다운 보고서를 생성합니다.',
        'fields': [
            {'key': 'topic', 'label': '보고서 주제', 'type': 'text',
             'placeholder': '예: 2026 Q1 매출 분석 및 전략 제안'},
            {'key': 'audience', 'label': '대상 독자', 'type': 'text',
             'placeholder': '예: 임원진 / 실무팀 / 외부 파트너'},
            {'key': 'length', 'label': '분량', 'type': 'select',
             'options': ['짧게 (1~2 섹션)', '중간 (3~5 섹션)', '상세 (6+ 섹션)']},
            {'key': 'sections', 'label': '포함해야 할 섹션·관점', 'type': 'textarea',
             'placeholder': '비워두면 자동 구성됩니다.'},
        ],
        'requires_docs': True,
        'search_query': lambda i: f"{i.get('topic', '')} {i.get('sections', '')}".strip(),
        'build_messages': _report_build,
        'output_ext': 'md',
    },
    'summary': {
        'label': '문서 요약',
        'description': '업로드된 문서를 종합 요약합니다.',
        'fields': [
            {'key': 'focus', 'label': '요약 초점', 'type': 'text',
             'placeholder': '예: 매출과 마진 중심 / 리스크 위주 (비워두면 전반)'},
            {'key': 'depth', 'label': '깊이', 'type': 'select',
             'options': ['핵심만 (3줄)', '단락 요약', '상세 요약']},
        ],
        'requires_docs': True,
        'search_query': lambda i: i.get('focus') or '핵심 주제, 주요 논점, 결론',
        'build_messages': _summary_build,
        'output_ext': 'md',
    },
    'analysis': {
        'label': '데이터 분석',
        'description': '문서에서 수치를 뽑아 표 + 해석 + 통찰을 정리합니다.',
        'fields': [
            {'key': 'question', 'label': '분석 질문', 'type': 'text',
             'placeholder': '예: 분기별 매출 성장률과 마진 추이는?'},
            {'key': 'context_note', 'label': '추가 맥락 (선택)', 'type': 'textarea',
             'placeholder': '분석 시 고려할 추가 정보가 있으면 적어주세요.'},
        ],
        'requires_docs': True,
        'search_query': lambda i: f"{i.get('question', '')} 수치 통계 결과".strip(),
        'build_messages': _analysis_build,
        'output_ext': 'md',
    },
    'comparison': {
        'label': '비교 분석',
        'description': '두 대상을 문서 근거로 비교합니다 (여러 문서 균형 검색 자동 적용).',
        'fields': [
            {'key': 'item_a', 'label': '대상 A', 'type': 'text',
             'placeholder': '예: 자사 제품 A / 2025년 실적'},
            {'key': 'item_b', 'label': '대상 B', 'type': 'text',
             'placeholder': '예: 경쟁사 제품 B / 2026년 실적'},
            {'key': 'criteria', 'label': '비교 기준 (선택)', 'type': 'textarea',
             'placeholder': '비워두면 자동 선택.'},
        ],
        'requires_docs': True,
        'search_query': lambda i: f"{i.get('item_a', '')} {i.get('item_b', '')} 비교 차이",
        'build_messages': _comparison_build,
        'output_ext': 'md',
    },
}
