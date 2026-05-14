"""Chat orchestration — split into responsibility-specific submodules.

| submodule    | responsibility |
|--------------|---------------|
| messages     | RAG system prompt, build_messages, context formatting, page image attachments |
| citations    | split_thinking, parse_citations, format_answer_with_citations (pure) |
| agentic      | search_documents tool loop (agentic_chat_pass) |
| streaming    | stream_chat / non_stream_chat / _record_response_model |
| render       | render_assistant + citation expanders |
| session_io   | log_turn_structured + auto_title_session |
| errors       | _show_llm_error (provider-aware Korean explainer) |
| turn         | handle_chat_turn — top-level orchestrator |

External consumers (views/, agents/) keep doing
``from llm.chat import handle_chat_turn`` — every public name is re-
exported here for backward compatibility.
"""

from .agentic import _SEARCH_TOOL_DEF, _format_tool_search_result, agentic_chat_pass
from .citations import (
    _CITE_PATTERN,
    format_answer_with_citations,
    parse_citations,
    split_thinking,
)
from .errors import _show_llm_error
from .messages import (
    RAG_SYSTEM_PROMPT,
    _collect_page_image_parts,
    _context_label,
    _format_pages,
    build_messages,
)
from .render import _citation_body, _citation_summary, render_assistant
from .session_io import auto_title_session, log_turn_structured
from .streaming import _record_response_model, non_stream_chat, stream_chat
from .turn import handle_chat_turn


__all__ = [
    'RAG_SYSTEM_PROMPT',
    '_CITE_PATTERN',
    '_SEARCH_TOOL_DEF',
    '_citation_body',
    '_citation_summary',
    '_collect_page_image_parts',
    '_context_label',
    '_format_pages',
    '_format_tool_search_result',
    '_record_response_model',
    '_show_llm_error',
    'agentic_chat_pass',
    'auto_title_session',
    'build_messages',
    'format_answer_with_citations',
    'handle_chat_turn',
    'log_turn_structured',
    'non_stream_chat',
    'parse_citations',
    'render_assistant',
    'split_thinking',
    'stream_chat',
]
