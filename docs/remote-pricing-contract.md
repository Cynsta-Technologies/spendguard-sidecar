# Sidecar <-> Cloud Pricing Contract

This deployment requires sidecar to fetch signed pricing from cloud.

## Cloud (`spendguard-cloud`)

Set:

- `CLOUD_PRICING_SIGNING_PRIVATE_KEY` (required; base64 raw Ed25519 private key)
- Optional: `CLOUD_PRICING_JSON` or `CLOUD_PRICING_FILE`
- Optional: `CLOUD_PRICING_TTL_SECONDS` (default `300`)
- Optional: `CLOUD_ENGINE_MIN_VERSION` / `CLOUD_ENGINE_MAX_VERSION`

Endpoint:

- `GET /v1/public/pricing`
- Supports `ETag` and `If-None-Match`

## Sidecar (`spendguard-sidecar`)

Set:

- `CAP_PRICING_SOURCE=remote`
- `CAP_PRICING_URL=https://<cloud-host>/v1/public/pricing`
- Optional override: `CAP_PRICING_SIGNING_PUBLIC_KEY=<cloud-public-key>`
- `CAP_PRICING_VERIFY_SIGNATURE=true`
- `CAP_PRICING_SCHEMA_VERSION=1`
- `CAP_PRICE_TABLE_REFRESH_SECONDS=300`

Behavior:

- Startup fails if pricing cannot be fetched/verified.
- No local default rate fallback is used for settlement.
- Sidecar caches last good pricing in memory and revalidates with `If-None-Match`.
