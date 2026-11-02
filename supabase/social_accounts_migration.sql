-- Run this migration in Supabase SQL Editor after the original schema.
create table if not exists public.creator_social_accounts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  platform text not null,
  profile_url text not null,
  region text not null default '',
  followers bigint not null default 0 check (followers >= 0),
  following bigint check (following is null or following >= 0),
  average_likes numeric(14,2) not null default 0 check (average_likes >= 0),
  average_comments numeric(14,2) not null default 0 check (average_comments >= 0),
  average_shares numeric(14,2) not null default 0 check (average_shares >= 0),
  average_views numeric(14,2) not null default 0 check (average_views >= 0),
  average_saves numeric(14,2) not null default 0 check (average_saves >= 0),
  updated_at timestamptz not null default now(),
  unique(user_id, platform)
);
alter table public.creator_social_accounts enable row level security;
drop policy if exists "Creators manage own social accounts" on public.creator_social_accounts;
create policy "Creators manage own social accounts" on public.creator_social_accounts
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
create index if not exists creator_social_accounts_user_idx on public.creator_social_accounts(user_id);
