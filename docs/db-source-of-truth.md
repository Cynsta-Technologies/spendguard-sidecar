# Database Source Of Truth

As of 2026-02-14, shared Supabase schema ownership moved to:

- `https://github.com/Cynsta-AI/cynsta-db`

Canonical migration folder:

- `https://github.com/Cynsta-AI/cynsta-db/tree/main/infra/supabase/migrations`

Hosted services (for example `spendguard-cloud`) consume this schema but do not own an
independent migration history.
