# SpendGuard Sidecar

Open-source local/BYOK SpendGuard wrapper service.

- Runtime mode: sidecar only
- Store: sqlite (default) or supabase
- Pricing source: remote Cynsta cloud pricing only (signed)
- Shared logic is imported from `spendguard-engine`

User quickstart: https://github.com/Cynsta-Technologies/spendguard-sdk/blob/main/docs/quickstart.md

## Hard Requirement

Sidecar does not use local default rates for settlement.
It must fetch a signed pricing document from cloud (`/v1/public/pricing`).
If pricing fetch/verification fails, startup fails.

## Run (Local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:CAP_STORE = "sqlite"
$env:CAP_SQLITE_PATH = ".\\cynsta-spendguard.db"

# Provider keys (BYOK)
$env:OPENAI_API_KEY = "sk-..."
$env:XAI_API_KEY = "xai-..."
# Optional: defaults to https://api.x.ai/v1
# $env:XAI_BASE_URL = "https://api.x.ai/v1"
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:GEMINI_API_KEY = "..."

# Cloud pricing (required)
$env:CAP_PRICING_SOURCE = "remote"
$env:CAP_PRICING_URL = "https://api.cynsta.com/v1/public/pricing"
# Optional override: defaults to Cynsta public key baked into sidecar
# $env:CAP_PRICING_SIGNING_PUBLIC_KEY = "<base64-raw-ed25519-public-key>"
$env:CAP_PRICING_VERIFY_SIGNATURE = "true"
$env:CAP_PRICING_SCHEMA_VERSION = "1"

uvicorn app.main:app --reload --port 8787
```

## Remote Pricing Contract

The sidecar expects this shape from cloud:

```json
{
  "schema_version": "1",
  "engine": { "name": "spendguard-engine", "min_version": "0.1.0", "max_version": "9.9.9" },
  "generated_at": "2026-02-15T12:00:00+00:00",
  "expires_at": "2026-02-15T12:05:00+00:00",
  "rates": { "openai": { "gpt-4o-mini": { "input_cents_per_1m": 30, "output_cents_per_1m": 120 } } },
  "signature": "<base64-ed25519-signature>"
}
```

- Signature algorithm: Ed25519 over canonical JSON of the document without `signature`.
- `ETag` and `If-None-Match` are supported for efficient refresh.

## API

- `POST /v1/agents`
- `GET /v1/agents`
- `POST /v1/agents/{agent_id}/budget`
- `GET /v1/agents/{agent_id}/budget`
- `POST /v1/agents/{agent_id}/runs`
- Provider endpoints under `/v1/agents/{agent_id}/runs/{run_id}/...`
  - `openai/chat/completions`, `openai/responses`
  - `grok/chat/completions`, `grok/responses` (xAI OpenAI-compatible API)
  - `gemini/generateContent`, `anthropic/messages`

This wrapper intentionally does not enforce hosted API key auth.

## License

MIT. See `LICENSE`.
