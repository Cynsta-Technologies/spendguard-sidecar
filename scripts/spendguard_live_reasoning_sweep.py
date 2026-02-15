from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _usd_from_microcents(microcents: int) -> float:
    # 1 USD = 100 cents = 100 * 1e6 microcents
    return float(microcents) / (100.0 * 1_000_000.0)


@dataclass(frozen=True)
class CallSpec:
    provider: str
    model: str
    endpoint: str
    payload: dict[str, Any]


def _fetch_ledger_row(db_path: Path, run_id: str) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "select provider, model, created_at, realized_cents, meta_json, input_tokens, output_tokens from cap_usage_ledger where run_id=? limit 1",
            (run_id,),
        ).fetchone()
        if not row:
            raise RuntimeError(f"Ledger row not found for run_id={run_id}")
        meta_raw = row["meta_json"]
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) and meta_raw else {}
        bb = meta.get("billing_breakdown") or {}
        totals = bb.get("totals") or {}
        realized_microcents = int(totals.get("realized_microcents") or 0)
        provider_request_id = meta.get("provider_request_id")
        if not provider_request_id and isinstance(meta.get("response"), dict):
            provider_request_id = meta["response"].get("id")

        return {
            "provider": str(row["provider"]),
            "model": str(row["model"]),
            "created_at": str(row["created_at"]),
            "run_id": run_id,
            "provider_request_id": provider_request_id,
            "tokens_in": row["input_tokens"],
            "tokens_out": row["output_tokens"],
            "realized_cents": int(row["realized_cents"]),
            "realized_microcents": realized_microcents,
            "realized_usd": round(_usd_from_microcents(realized_microcents), 8),
            "billing_breakdown": bb,
        }
    finally:
        conn.close()


def main() -> int:
    # Configure SpendGuard (must be set before importing app.main, which initializes store/rates).
    db_path = Path(f".\\spendguard-live-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db").resolve()
    os.environ["CAP_MODE"] = "sidecar"
    os.environ["CAP_STORE"] = "sqlite"
    os.environ["CAP_SQLITE_PATH"] = str(db_path)

    # Keep the sweep intentionally small (we're spending real money).
    agent_budget_cents = 2_500

    # Prompts structured to reliably consume output tokens without tool use.
    openai_prompt = (
        "Output the word 'zebra' exactly 400 times separated by single spaces. "
        "No punctuation. No extra text."
    )
    anthropic_prompt = (
        "Output the word 'zebra' exactly 1000 times separated by single spaces. "
        "No punctuation. No extra text."
    )
    gemini_prompt = (
        "Output the word zebra exactly 2500 times separated by single spaces. "
        "No punctuation. No extra text."
    )

    calls: list[CallSpec] = [
        CallSpec(
            provider="openai",
            model="gpt-5.2-pro",
            endpoint="openai/responses",
            payload={
                "model": "gpt-5.2-pro",
                "input": openai_prompt,
                "max_output_tokens": 400,
                "reasoning": {"effort": "high"},
            },
        ),
        CallSpec(
            provider="anthropic",
            model="claude-opus-4-6",
            endpoint="anthropic/messages",
            payload={
                "model": "claude-opus-4-6",
                "system": "Follow the user's instructions exactly.",
                "messages": [{"role": "user", "content": anthropic_prompt}],
                "max_tokens": 1100,
            },
        ),
        CallSpec(
            provider="gemini",
            model="gemini-3-pro-preview",
            endpoint="gemini/generateContent",
            payload={
                "model": "gemini-3-pro-preview",
                "prompt": gemini_prompt,
                "max_tokens": 2600,
            },
        ),
    ]

    start = _utc_now_iso()

    # Import after env config so it picks up sqlite path.
    from fastapi.testclient import TestClient  # type: ignore

    import app.main as spendguard  # type: ignore

    client = TestClient(spendguard.app)

    agent_id = client.post("/v1/agents", json={"name": "live-reasoning-sweep"}).json()["agent_id"]
    client.post(
        f"/v1/agents/{agent_id}/budget",
        json={"hard_limit_cents": agent_budget_cents, "topup_cents": agent_budget_cents},
    )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for spec in calls:
        run_id = str(uuid.uuid4())
        url = f"/v1/agents/{agent_id}/runs/{run_id}/{spec.endpoint}"
        try:
            r = client.post(url, json=spec.payload)
            if r.status_code >= 400:
                raise RuntimeError(f"{spec.provider}:{spec.model} HTTP {r.status_code}: {r.text}")
            results.append(_fetch_ledger_row(db_path, run_id))
        except Exception as exc:
            failures.append(
                {
                    "provider": spec.provider,
                    "model": spec.model,
                    "run_id": run_id,
                    "error": str(exc),
                }
            )

    end = _utc_now_iso()

    summary = {
        "db_path": str(db_path),
        "agent_id": agent_id,
        "window_start_utc": start,
        "window_end_utc": end,
        "results": results,
        "failures": failures,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())

