-- Personal RAG — pgvector schema for dense embedding storage.
-- Run this AFTER db_schema.sql, once per Supabase project.
--
-- Why: moves chunk embeddings out of the ephemeral container memory into
-- a real vector store. Survives restarts, scales beyond what fits in RAM,
-- and lets us run cosine k-NN at the DB layer (HNSW index).
--
-- Embedder dimension note: the app supports two embedders with different
-- vector dimensions, so we store one column per dimension and use the
-- `embedder` text column to tell them apart at query time:
--   MiniLM (paraphrase-multilingual-MiniLM-L12-v2)  → 384 dim
--   BGE-M3 (BAAI/bge-m3)                            → 1024 dim

-- 0) Enable the pgvector extension (idempotent).
create extension if not exists vector;

-- 1) Table — one row per (user, doc, chunk_idx).
create table if not exists public.doc_chunks (
    id           bigserial primary key,
    user_id      text not null,
    doc_id       text not null,
    doc_name     text,
    chunk_idx    int  not null,
    "text"       text,
    pages        jsonb default '[]'::jsonb,
    embedder     text not null,
    -- Exactly one of these is filled per row, matching `embedder`.
    embedding_minilm  vector(384),
    embedding_bgem3   vector(1024),
    created_at   timestamptz default now(),
    unique (user_id, doc_id, chunk_idx, embedder)
);

-- 2) Lookup indexes (filters before the ANN step).
create index if not exists doc_chunks_user_idx
    on public.doc_chunks (user_id);
create index if not exists doc_chunks_user_doc_idx
    on public.doc_chunks (user_id, doc_id);

-- 3) ANN indexes — HNSW with cosine distance.
--    Partial indexes so each only covers rows that actually have that dim.
create index if not exists doc_chunks_minilm_hnsw
    on public.doc_chunks using hnsw (embedding_minilm vector_cosine_ops)
    where embedding_minilm is not null;
create index if not exists doc_chunks_bgem3_hnsw
    on public.doc_chunks using hnsw (embedding_bgem3 vector_cosine_ops)
    where embedding_bgem3 is not null;

-- 4) Stored procedures — top-k cosine search per embedder.
--    The app calls these via supabase.rpc() because the SDK doesn't expose
--    raw SQL with vector parameters cleanly.
create or replace function public.match_chunks_minilm(
    p_user_id          text,
    p_query_embedding  vector(384),
    p_match_count      int,
    p_doc_ids          text[] default null
)
returns table (
    id          bigint,
    doc_id      text,
    doc_name    text,
    chunk_idx   int,
    "text"      text,
    pages       jsonb,
    score       float
) language sql stable as $$
    select id, doc_id, doc_name, chunk_idx, "text", pages,
           1 - (embedding_minilm <=> p_query_embedding) as score
      from public.doc_chunks
     where user_id = p_user_id
       and embedder = 'paraphrase-multilingual-MiniLM-L12-v2'
       and embedding_minilm is not null
       and (p_doc_ids is null or doc_id = any(p_doc_ids))
     order by embedding_minilm <=> p_query_embedding
     limit p_match_count;
$$;

create or replace function public.match_chunks_bgem3(
    p_user_id          text,
    p_query_embedding  vector(1024),
    p_match_count      int,
    p_doc_ids          text[] default null
)
returns table (
    id          bigint,
    doc_id      text,
    doc_name    text,
    chunk_idx   int,
    "text"      text,
    pages       jsonb,
    score       float
) language sql stable as $$
    select id, doc_id, doc_name, chunk_idx, "text", pages,
           1 - (embedding_bgem3 <=> p_query_embedding) as score
      from public.doc_chunks
     where user_id = p_user_id
       and embedder = 'bge-m3'
       and embedding_bgem3 is not null
       and (p_doc_ids is null or doc_id = any(p_doc_ids))
     order by embedding_bgem3 <=> p_query_embedding
     limit p_match_count;
$$;

-- 5) Permissions — let the app's anon key read & write.
grant insert, select, delete on public.doc_chunks to anon, authenticated;
grant usage,  select on all sequences in schema public to anon, authenticated;
grant execute on function public.match_chunks_minilm to anon, authenticated;
grant execute on function public.match_chunks_bgem3  to anon, authenticated;

-- 6) Optional: RLS. Same posture as the other logging tables — disable for
--    a single-app writer setup, or enable with permissive policies.
--    Default here matches the recommendation in db_schema.sql.
alter table public.doc_chunks disable row level security;
