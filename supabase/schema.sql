-- Growfluence AI — Supabase schema
-- Run this entire file in Supabase SQL Editor.
create extension if not exists pgcrypto;

create table if not exists public.creator_profiles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null unique references auth.users(id) on delete cascade,
  display_name text not null default '',
  niche text not null default '',
  target_audience text not null default '',
  platforms text[] not null default array['Instagram']::text[],
  primary_goal text not null default 'Grow audience',
  brand_voice text not null default 'Friendly, useful and authentic',
  posts_per_week integer not null default 7 check (posts_per_week between 3 and 14),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.content_posts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  scheduled_for date not null,
  platform text not null,
  format text not null,
  topic text not null default '',
  hook text not null default '',
  caption text not null default '',
  cta text not null default '',
  hashtags text[] not null default '{}',
  objective text not null default '',
  status text not null default 'Idea' check (status in ('Idea','Draft','Ready','Published')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.creator_metrics (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  recorded_on date not null default current_date,
  platform text not null,
  followers bigint not null default 0 check (followers >= 0),
  reach bigint not null default 0 check (reach >= 0),
  engagements bigint not null default 0 check (engagements >= 0),
  saves bigint not null default 0 check (saves >= 0),
  created_at timestamptz not null default now()
);

alter table public.creator_profiles enable row level security;
alter table public.content_posts enable row level security;
alter table public.creator_metrics enable row level security;

drop policy if exists "Creators manage own profile" on public.creator_profiles;
create policy "Creators manage own profile" on public.creator_profiles for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
drop policy if exists "Creators manage own posts" on public.content_posts;
create policy "Creators manage own posts" on public.content_posts for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
drop policy if exists "Creators manage own metrics" on public.creator_metrics;
create policy "Creators manage own metrics" on public.creator_metrics for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create index if not exists content_posts_user_date_idx on public.content_posts(user_id, scheduled_for);
create index if not exists creator_metrics_user_date_idx on public.creator_metrics(user_id, recorded_on);

-- Supabase Auth email confirmation may be enabled. Keep RLS enabled.
