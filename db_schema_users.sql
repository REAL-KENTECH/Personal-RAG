-- Personal RAG — Supabase users table + bcrypt signup/login RPCs.
-- Run this once in your Supabase project's SQL Editor after the main
-- db_schema.sql. Enables self-signup so the app no longer needs the
-- admin-managed `[users]` block in Streamlit secrets.
--
-- Security posture:
-- - Passwords are hashed inside Postgres using pgcrypto's bcrypt
--   (crypt + gen_salt('bf')). Plain text never enters the table.
-- - We expose only two RPC functions to the anon role: signup_user
--   and login_user. The users table itself is NOT granted to anon,
--   so password_hash never leaves the database.
-- - The RPCs run as SECURITY DEFINER (function owner privilege) so
--   the table is reachable from inside the function even though the
--   caller doesn't have direct SELECT/INSERT on it.

create extension if not exists pgcrypto;

create table if not exists public.users (
    id              bigserial primary key,
    username        text unique not null
                    check (length(trim(username)) between 2 and 64),
    password_hash   text not null,
    display_name    text,
    created_at      timestamptz default now(),
    last_login_at   timestamptz
);

create index if not exists users_username_idx on public.users (username);

-- ---------------------------------------------------------------------------
-- signup_user(username, password) → (success, message, user_id)
-- ---------------------------------------------------------------------------
create or replace function public.signup_user(
    p_username text,
    p_password text
)
returns table(success boolean, message text, user_id bigint)
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
    new_id bigint;
    cleaned text := trim(p_username);
begin
    if length(cleaned) < 2 then
        return query select false, '아이디는 2자 이상이어야 합니다.'::text, null::bigint;
        return;
    end if;
    if length(p_password) < 6 then
        return query select false, '비밀번호는 6자 이상이어야 합니다.'::text, null::bigint;
        return;
    end if;
    if exists (select 1 from public.users where username = cleaned) then
        return query select false, '이미 사용 중인 아이디입니다.'::text, null::bigint;
        return;
    end if;
    insert into public.users (username, password_hash)
    values (cleaned, crypt(p_password, gen_salt('bf', 10)))
    returning id into new_id;
    return query select true, '회원가입 성공'::text, new_id;
end;
$$;

-- ---------------------------------------------------------------------------
-- login_user(username, password) → (success, message, user_id)
-- ---------------------------------------------------------------------------
create or replace function public.login_user(
    p_username text,
    p_password text
)
returns table(success boolean, message text, user_id bigint)
language plpgsql
security definer
set search_path = public, extensions
as $$
declare
    rec     record;
    cleaned text := trim(p_username);
begin
    select id, password_hash
      into rec
      from public.users
     where username = cleaned;

    if not found then
        return query select false, '존재하지 않는 아이디입니다.'::text, null::bigint;
        return;
    end if;
    if rec.password_hash <> crypt(p_password, rec.password_hash) then
        return query select false, '비밀번호가 일치하지 않습니다.'::text, null::bigint;
        return;
    end if;

    update public.users set last_login_at = now() where id = rec.id;
    return query select true, '로그인 성공'::text, rec.id;
end;
$$;

-- ---------------------------------------------------------------------------
-- Permissions: only the two RPCs are callable. The table stays private.
-- ---------------------------------------------------------------------------
grant execute on function public.signup_user(text, text)  to anon, authenticated;
grant execute on function public.login_user(text, text)   to anon, authenticated;

-- Deliberately NOT granting any rights on public.users to anon, so
-- password_hash is unreachable except through the RPCs.
-- Re-running this file is safe; everything is `create [or replace] ... if not exists`.
