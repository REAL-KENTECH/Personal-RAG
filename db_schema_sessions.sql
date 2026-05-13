-- Personal RAG — session history restore from chat_turns.
-- Run this once after db_schema.sql.
--
-- The app already writes every chat turn into public.chat_turns. To rebuild
-- a user's sidebar conversation list across container restarts / devices we
-- need a fast "most-recent-N sessions" aggregation. Doing this via raw
-- PostgREST table reads forces us to pull thousands of rows and aggregate
-- client-side; an RPC keeps the GROUP BY in Postgres and returns one row
-- per session.

create or replace function public.list_user_sessions(
    p_user_id text,
    p_limit   int default 50
)
returns table (
    session_id text,
    title      text,
    model      text,
    updated_at timestamptz,
    n_turns    bigint
)
language sql
stable
security definer
set search_path = public, extensions
as $$
    select session_id,
           coalesce(max(session_title), '(제목 없음)') as title,
           max(model)             as model,
           max("timestamp")       as updated_at,
           count(*)               as n_turns
      from public.chat_turns
     where user_id = p_user_id
       and session_id is not null
     group by session_id
     order by updated_at desc
     limit p_limit;
$$;

grant execute on function public.list_user_sessions(text, int) to anon, authenticated;
