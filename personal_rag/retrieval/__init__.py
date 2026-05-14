"""Hybrid retrieval pipeline.

- ``search`` houses the primitive scorers — BM25, dense (numpy in-memory
  or Supabase pgvector RPC), Reciprocal Rank Fusion, and the cross-encoder
  reranker.
- ``expansion`` lets the LLM rewrite the user's question into multiple
  variants (contextual rewrite, multi-query paraphrases, HyDE).
- ``web`` wraps the optional external search providers (DuckDuckGo,
  Tavily, Brave).
- ``pipeline`` orchestrates everything: variants → per-query search →
  RRF → rerank → per-doc balancing → optional web append.
"""
