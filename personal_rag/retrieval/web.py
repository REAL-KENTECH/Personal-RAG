"""External web search adapters (DuckDuckGo, Tavily, Brave).

Each returns the same normalized result shape that the chat builder
expects — ``{source: 'web', doc, url, chunk_idx, text, score}`` — so the
RAG context can include web hits alongside local document chunks.
"""

import streamlit as st


def web_search(query: str) -> list:
    """Return list of normalized web search results."""
    provider = st.session_state['web_provider']
    top_n = int(st.session_state['web_top_n'])
    try:
        if provider == 'duckduckgo':
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            results = list(DDGS().text(query, max_results=top_n))
            out = []
            for r in results:
                out.append({
                    'source': 'web',
                    'doc': r.get('title') or r.get('href', ''),
                    'url': r.get('href') or r.get('link', ''),
                    'chunk_idx': 0,
                    'text': r.get('body') or r.get('snippet', ''),
                    'score': 0.0,
                })
            return out

        if provider == 'tavily':
            key = st.session_state['tavily_key']
            if not key:
                st.warning(
                    'Tavily API 키가 비어 있습니다. '
                    '.env의 TAVILY_API_KEY에 추가하거나 사이드바에서 입력하세요. '
                    '키 발급: https://app.tavily.com'
                )
                return []
            try:
                from tavily import TavilyClient
            except ImportError:
                st.error(
                    'tavily-python 패키지가 설치되지 않았습니다. '
                    '터미널에서 `pip install tavily-python`를 실행해 주세요.'
                )
                return []
            client = TavilyClient(api_key=key)
            resp = client.search(query=query, max_results=top_n, search_depth='basic')
            out = []
            for r in resp.get('results', []):
                out.append({
                    'source': 'web',
                    'doc': r.get('title', ''),
                    'url': r.get('url', ''),
                    'chunk_idx': 0,
                    'text': r.get('content', ''),
                    'score': float(r.get('score', 0.0)),
                })
            return out

        if provider == 'brave':
            key = st.session_state['brave_key']
            if not key:
                st.warning('Brave API 키가 비어 있습니다 (.env BRAVE_API_KEY).')
                return []
            import requests
            r = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                params={'q': query, 'count': top_n},
                headers={
                    'X-Subscription-Token': key,
                    'Accept': 'application/json',
                },
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            out = []
            for item in data.get('web', {}).get('results', []):
                out.append({
                    'source': 'web',
                    'doc': item.get('title', ''),
                    'url': item.get('url', ''),
                    'chunk_idx': 0,
                    'text': item.get('description', ''),
                    'score': 0.0,
                })
            return out

    except Exception as e:
        st.warning(f'웹 검색 실패 ({provider}): {e}')
        return []
    return []
