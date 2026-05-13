-- Personal RAG — Supabase Postgres schema for persistent logging.
-- Run this once in your Supabase project's SQL Editor.
--
-- Why: Streamlit Community Cloud's filesystem is ephemeral. Every redeploy or
-- container restart wipes /mount/src/.../logs. These tables receive a copy of
-- every chat turn, agent run, and event so user activity survives restarts and
-- can be analyzed with regular SQL.

-- ---------------------------------------------------------------------------
-- 1) chat_turns — one row per (user, session, turn)
create table if not exists public.chat_turns (
    id                    bigserial primary key,
    session_id            text,
    session_title         text,
    turn_index            int,
    "timestamp"           timestamptz default now(),
    user_id               text,
    user_message          text,
    assistant_message     text,
    reasoning             text,
    model                 text,
    provider              text,
    base_url              text,
    elapsed_seconds       numeric,
    retrieved             jsonb,
    n_retrieved           int,
    citation_numbers_used jsonb,
    n_cited               int,
    query_variants        jsonb,
    settings_snapshot     jsonb
);

create index if not exists chat_turns_user_idx on public.chat_turns (user_id, "timestamp" desc);
create index if not exists chat_turns_session_idx on public.chat_turns (session_id, turn_index);

-- ---------------------------------------------------------------------------
-- 2) agent_runs — one row per agent task execution
create table if not exists public.agent_runs (
    id              bigserial primary key,
    kind            text default 'agent',
    task            text,
    "timestamp"     timestamptz default now(),
    user_id         text,
    inputs          jsonb,
    output          text,
    model           text,
    provider        text,
    elapsed_seconds numeric,
    retrieved       jsonb,
    n_retrieved     int
);

create index if not exists agent_runs_user_idx on public.agent_runs (user_id, "timestamp" desc);
create index if not exists agent_runs_task_idx on public.agent_runs (task);

-- ---------------------------------------------------------------------------
-- 3) events — login, logout, doc ingest/delete, session delete, llm errors
create table if not exists public.events (
    id          bigserial primary key,
    kind        text default 'event',
    event_type  text,
    "timestamp" timestamptz default now(),
    user_id     text,
    payload     jsonb
);

create index if not exists events_user_idx on public.events (user_id, "timestamp" desc);
create index if not exists events_type_idx on public.events (event_type, "timestamp" desc);

-- ---------------------------------------------------------------------------
-- Optional: Row-Level Security. By default, the anon key can INSERT/SELECT.
-- If you want to lock this down, enable RLS and add policies. For most setups
-- the anon-key + private project is enough since the app is the only writer.
--
-- alter table public.chat_turns enable row level security;
-- alter table public.agent_runs enable row level security;
-- alter table public.events     enable row level security;
--
-- create policy "app inserts" on public.chat_turns for insert with check (true);
-- create policy "app reads"   on public.chat_turns for select using (true);
-- (repeat for the other two tables)
