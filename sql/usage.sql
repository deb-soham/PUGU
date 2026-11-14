create table if not exists public.ai_usage (
  google_id text    not null,
  day       date    not null default ((now() at time zone 'utc')::date),
  tokens    integer not null default 0,
  requests  integer not null default 0,
  primary key (google_id, day)
);
alter table public.ai_usage enable row level security;

create or replace function public.record_ai_usage(p_google_id text, p_tokens integer)
returns void
language plpgsql
as $$
declare
  v_day date := (now() at time zone 'utc')::date;
begin
  -- this user's row
  insert into public.ai_usage (google_id, day, tokens, requests)
  values (p_google_id, v_day, greatest(p_tokens, 0), 1)
  on conflict (google_id, day)
  do update set tokens   = public.ai_usage.tokens + excluded.tokens,
                requests = public.ai_usage.requests + 1;

  -- the row that counts everyone together
  insert into public.ai_usage (google_id, day, tokens, requests)
  values ('__all__', v_day, greatest(p_tokens, 0), 1)
  on conflict (google_id, day)
  do update set tokens   = public.ai_usage.tokens + excluded.tokens,
                requests = public.ai_usage.requests + 1;
end;
$$;

-- Only your server (the secret key) may call this function.
revoke all on function public.record_ai_usage(text, integer) from public, anon, authenticated;
grant execute on function public.record_ai_usage(text, integer) to service_role;