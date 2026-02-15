# Database Source Of Truth

As of 2026-02-14, shared Supabase schema ownership moved to:

- `https://github.com/Cynsta-Technologies/cynsta-db`

Canonical migration folder:

- `https://github.com/Cynsta-Technologies/cynsta-db/tree/main/infra/supabase/migrations`

`spendguard-server` consumes this schema but does not own an independent migration history.
