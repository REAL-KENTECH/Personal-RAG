"""Disk- and Supabase-backed persistence layers.

- ``storage`` owns the per-user vector store (chunks + numpy embeddings on
  disk, with an optional pgvector dual-write so embeddings survive Cloud
  container resets).
- ``sessions`` owns conversation history: per-session JSON blobs on disk and
  Supabase ``chat_turns`` mirrors that survive across devices.

Two log/title helpers — ``log_turn_structured`` and ``auto_title_session``
— are intentionally still in app.py for now because they straddle the
chat-helpers boundary; they will land in the chat module alongside
``handle_chat_turn`` in the next pass.
"""
