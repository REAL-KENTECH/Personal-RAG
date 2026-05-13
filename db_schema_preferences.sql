-- Personal RAG — per-user preferences sync.
-- Run this once in Supabase SQL Editor after db_schema_users.sql.
--
-- Stores each user's API keys, model selection, retrieval/sampling settings
-- as a single JSONB blob keyed by username. The app reads on login and
-- upserts whenever the user changes a setting, so the experience follows
-- the account across container restarts and devices.
--
-- Threat model: API keys are stored as plaintext inside Supabase. This is
-- acceptable when the anon key only lives in Streamlit Secrets (encrypted
-- at rest by Streamlit) and the deployment is single-tenant or
-- small-trusted-team. For larger deployments, migrate the auth layer to
-- Supabase Auth + RLS so per-user policies (auth.uid() = user_id) can
-- enforce isolation at the database level.

create table if not exists public.user_preferences (
    user_id     text primary key,
    prefs       jsonb default '{}'::jsonb,
    updated_at  timestamptz default now()
);

-- Bump updated_at on every UPDATE for audit trail.
create or replace function public.user_preferences_touch()
returns trigger
language plpgsql
as $$
begin
    new.updated_at := now();
    return new;
end;
$$;

drop trigger if exists user_preferences_updated on public.user_preferences;
create trigger user_preferences_updated
    before update on public.user_preferences
    for each row execute function public.user_preferences_touch();

-- ---------------------------------------------------------------------------
-- get_prefs(username) → jsonb (or NULL if no row yet)
-- ---------------------------------------------------------------------------
create or replace function public.get_prefs(p_user_id text)
returns jsonb
language sql
stable
security definer
set search_path = public, extensions
as $$
    select prefs from public.user_preferences where user_id = p_user_id;
$$;

-- ---------------------------------------------------------------------------
-- set_prefs(username, prefs jsonb) → void  (upsert)
-- ---------------------------------------------------------------------------
create or replace function public.set_prefs(p_user_id text, p_prefs jsonb)
returns void
language sql
security definer
set search_path = public, extensions
as $$
    insert into public.user_preferences (user_id, prefs)
    values (p_user_id, p_prefs)
    on conflict (user_id) do update
       set prefs = excluded.prefs;
$$;

grant execute on function public.get_prefs(text)        to anon, authenticated;
grant execute on function public.set_prefs(text, jsonb) to anon, authenticated;

-- The table itself is not granted to anon — only the RPCs are. Plaintext
-- API keys never leave the DB except through these two calls.
alter table public.user_preferences disable row level security;
