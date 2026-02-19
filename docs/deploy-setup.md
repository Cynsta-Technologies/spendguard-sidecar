# Deploy Setup

## Sidecar Runtime

`spendguard-sidecar` is local/BYOK runtime software and is not the hosted cloud deployment target.

- Use Docker (`docker compose up`) or local `uvicorn` as documented in `README.md`.
- Do not create a Render Blueprint from this repository.

## Hosted Cloud Deployment

Deploy the hosted pricing endpoint from:

- `https://github.com/Cynsta-AI/spendguard-cloud`
- Blueprint file: repo-root `render.yaml`

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
