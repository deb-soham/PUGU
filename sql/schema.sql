create table if not exists public.user_settings (
  google_id     text primary key,
  email         text,
  chunk_minutes integer     not null default 25,
  font_choice   text        not null default 'OpenDyslexic',
  tts_enabled   boolean     not null default false,
  bg_color      text        not null default 'White',
  updated_at    timestamptz not null default now()
);
alter table public.user_settings enable row level security;