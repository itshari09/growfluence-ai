# Growfluence AI — Creator Growth Studio

A multi-user creator workspace for audience-focused content strategy, weekly planning, AI-assisted content creation and growth tracking.

## Product features

- Private creator accounts powered by Supabase Auth.
- Per-user creator strategy: niche, audience, platforms, goals, voice and cadence.
- Seven-day content plan generation. Uses OpenAI when configured; otherwise provides a useful rule-based starter plan.
- Content studio for hooks, captions, short-form scripts, CTAs, hashtags and creative direction.
- Weekly post workflow: Idea → Draft → Ready → Published.
- Creator-reported analytics for followers, reach, engagements and saves.
- PostgreSQL persistence with Row Level Security (RLS) so each creator can only access their own records.

## Stack

- Python 3.11+
- Streamlit
- Supabase Auth + PostgreSQL + PostgREST
- OpenAI Chat Completions API (optional; set your own API key)
- Pandas

## Run locally

1. Clone the repository and enter it.
2. Create a virtual environment and install requirements:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

3. Create a Supabase project at https://supabase.com.
4. In Supabase → SQL Editor, run the complete `supabase/schema.sql` file.
5. Copy `.env.example` to `.env` and fill in your Supabase project URL and anon key. Add an OpenAI key to enable AI-generated plans and content.
6. Start the app:

   ```bash
   streamlit run app/app.py
   ```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_ANON_KEY` | Yes | Supabase publishable/anon key (never use service_role in the app) |
| `OPENAI_API_KEY` | No | Enables AI plan and content generation |
| `OPENAI_MODEL` | No | Defaults to `gpt-4o-mini` |

Set these in your hosting provider's secret/environment-variable settings. Never commit real keys.

## Deploy

Deploy as a Streamlit app (for example, Streamlit Community Cloud or a Python-capable host). Set the environment variables in the host's secrets manager. Vercel does not natively host a persistent Streamlit Python app as a standard Next.js deployment; use a Python-capable host or migrate the frontend/backend to a supported Vercel architecture.

## Security and product boundaries

- Supabase Row Level Security is enabled on all user-owned tables.
- The app uses the Supabase anon key plus the signed-in user's access token. Do not place a service-role key in client or app code.
- Analytics are creator-entered snapshots until official social-platform OAuth/API integrations are implemented and approved.
- Scheduling is an editorial plan; the app does not publish automatically to Instagram, YouTube, TikTok or other platforms.
- AI output is a draft for creator review, not a guarantee of reach, engagement or follower growth.

## Current scope / next production steps

This repository contains a working multi-user MVP foundation. Before a public paid launch, add email verification UX, password reset, billing, audit logging, rate limits, background jobs, automated tests, privacy/terms pages, monitoring, data export/deletion flows, and official social-platform integrations.
