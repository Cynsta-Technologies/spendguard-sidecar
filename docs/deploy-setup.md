# Deploy Setup

## Render Service

Use the repo-root `render.yaml` as the Blueprint definition for `cynsta-spendguard`.
Render auto-detects this file when creating a Blueprint service from GitHub.

Required environment variables:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `CAP_MODE=hosted`
- `CAP_STORE=supabase`
- `CAP_PRICING_SOURCE=auto` (or `supabase`)
- `CAP_PRICING_STRICT=true`
- `OPENAI_API_KEY` (and/or `XAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`)

Optional:

- `CAP_INGEST_URL`
- `CAP_INGEST_API_KEY`
- `CAP_PRICE_TABLE_JSON`
- `CAP_PRICE_TABLE_REFRESH_SECONDS` (default `300`)

## Database Migrations

Apply from `cynsta-db` in order:

1. `https://github.com/Cynsta-AI/cynsta-db/blob/main/infra/supabase/migrations/20260120123000_create_orgs_rbac_keys.sql`
2. `https://github.com/Cynsta-AI/cynsta-db/blob/main/infra/supabase/migrations/20260208180000_create_cap_tables.sql`
3. `https://github.com/Cynsta-AI/cynsta-db/blob/main/infra/supabase/migrations/20260214113000_create_cap_price_cards.sql`
4. `https://github.com/Cynsta-AI/cynsta-db/blob/main/infra/supabase/migrations/20260214114000_seed_cap_price_cards_defaults.sql`

## Branch Policy

`main` should require:

- Pull request review (1 approval)
- Passing status check: `tests`
