"""LLM client + parameter-shaping helpers.

- ``clients`` builds and caches the OpenAI-compatible HTTP client, the
  embedder, and the reranker. Switching the provider/key swaps to a
  fresh client without reloading the underlying transformer weights.
- ``params`` knows the quirks of each provider (OpenAI gpt-5 / o-series,
  Anthropic OpenAI-compat, DashScope, HF Router, Fireworks, vLLM) and
  produces a ``chat.completions.create`` kwargs dict that won't trip
  any of their parameter restrictions.

The ``chat`` module that actually orchestrates a turn lands later (it
needs retrieval results and citation parsing too).
"""
