from __future__ import annotations

import json
import os
from typing import Any, Optional

from app.internal import CynstaTracer


class EvidenceEmitter:
    def __init__(self) -> None:
        self.enabled = bool(os.getenv("CAP_INGEST_URL")) and bool(
            os.getenv("CAP_INGEST_API_KEY") or os.getenv("INGEST_API_KEY")
        )
        self.capture = _env_bool("CAP_CAPTURE_PROMPT_RESPONSE", default=None)
        if self.capture is None:
            self.capture = (os.getenv("CAP_MODE") or "").strip().lower() == "hosted"
        self._tracer: CynstaTracer | None = None

    def _get_tracer(self, tenant_id: str | None, user_id: str | None) -> CynstaTracer:
        if self._tracer:
            return self._tracer
        self._tracer = CynstaTracer(
            mode="server_trust",
            ingest_endpoint=os.getenv("CAP_INGEST_URL")
            or os.getenv("CYNSTA_INGEST_URL")
            or "http://localhost:4318/v1/traces",
            api_key=os.getenv("CAP_INGEST_API_KEY")
            or os.getenv("CYNSTA_INGEST_API_KEY")
            or os.getenv("INGEST_API_KEY"),
            tenant_id=tenant_id,
            user_id=user_id,
            capture_prompt_response=self.capture,
            service_name=os.getenv("CAP_SERVICE_NAME") or "cynsta-spendguard",
        )
        return self._tracer

    def emit_llm(
        self,
        tenant_id: str | None,
        user_id: str | None,
        provider: str,
        model: str,
        prompt: Any,
        completion: str | None,
        tokens_in: int | None,
        tokens_out: int | None,
        meta: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        tracer = self._get_tracer(tenant_id, user_id)
        prompt_text = prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True)
        tracer.record_llm_interaction(
            model=f"{provider}:{model}",
            prompt=prompt_text,
            completion=completion or "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        tracer.record_tool_usage("cynsta.cap.settlement", meta, {"ok": True})

    def flush(self) -> None:
        if self._tracer:
            self._tracer.flush()


def _env_bool(name: str, default: Optional[bool]) -> Optional[bool]:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default
