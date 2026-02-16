from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from spendguard_engine import __version__ as ENGINE_VERSION
from spendguard_engine.pricing import RateCard, cost_cents, estimate_tokens_text

LOGGER = logging.getLogger(__name__)
DEFAULT_PRICING_SIGNING_PUBLIC_KEY_B64 = "1ITgfCZsRETBLvSKFUoPsP7p3RKNH4Nto6zKLlUD8nQ="


def _coerce_int(v: Any) -> int | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not v.is_integer():
            return None
        return int(v)
    if isinstance(v, str):
        raw = v.strip()
        if not raw:
            return None
        if raw[0] in {"+", "-"}:
            if len(raw) == 1:
                return None
            sign = raw[0]
            digits = raw[1:]
            if digits.isdigit():
                return int(sign + digits)
            return None
        if raw.isdigit():
            return int(raw)
    return None


def _coerce_float(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        raw = v.strip()
        if not raw:
            return None
        try:
            return float(raw)
        except Exception:
            return None
    return None


def _get_int(d: dict[str, Any], key: str) -> int | None:
    return _coerce_int(d.get(key))


def _get_float(d: dict[str, Any], key: str) -> float | None:
    return _coerce_float(d.get(key))


def _parse_rate_card(rates: dict[str, Any]) -> RateCard | None:
    inp = _get_int(rates, "input_cents_per_1m")
    out = _get_int(rates, "output_cents_per_1m")
    if inp is None or out is None or inp < 0 or out < 0:
        return None

    return RateCard(
        input_cents_per_1m=inp,
        output_cents_per_1m=out,
        cached_input_cents_per_1m=_get_int(rates, "cached_input_cents_per_1m"),
        uncached_input_cents_per_1m=_get_int(rates, "uncached_input_cents_per_1m"),
        reasoning_output_cents_per_1m=_get_int(rates, "reasoning_output_cents_per_1m"),
        cache_write_input_cents_per_1m=_get_int(rates, "cache_write_input_cents_per_1m"),
        cache_read_input_cents_per_1m=_get_int(rates, "cache_read_input_cents_per_1m"),
        grounding_cents_per_1k_queries=_get_int(rates, "grounding_cents_per_1k_queries"),
        web_search_cents_per_call=_get_int(rates, "web_search_cents_per_call"),
        file_search_cents_per_call=_get_int(rates, "file_search_cents_per_call"),
        context_cliff_threshold_tokens=_get_int(rates, "context_cliff_threshold_tokens"),
        context_cliff_input_multiplier=_get_float(rates, "context_cliff_input_multiplier"),
        context_cliff_output_multiplier=_get_float(rates, "context_cliff_output_multiplier"),
    )


def _parse_table(payload: dict[str, Any]) -> dict[str, dict[str, RateCard]]:
    parsed: dict[str, dict[str, RateCard]] = {}
    for provider, models in payload.items():
        if not isinstance(provider, str) or not isinstance(models, dict):
            continue
        provider_key = provider.strip().lower()
        if not provider_key:
            continue
        provider_models: dict[str, RateCard] = {}
        for model, rates in models.items():
            if not isinstance(model, str) or not isinstance(rates, dict):
                continue
            model_key = model.strip()
            if not model_key:
                continue
            card = _parse_rate_card(rates)
            if card is not None:
                provider_models[model_key] = card
        if provider_models:
            parsed[provider_key] = provider_models
    return parsed


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _load_signing_public_key() -> Ed25519PublicKey:
    raw = (os.getenv("CAP_PRICING_SIGNING_PUBLIC_KEY") or DEFAULT_PRICING_SIGNING_PUBLIC_KEY_B64).strip()
    if not raw:
        raise RuntimeError("No pricing signing public key configured")
    try:
        key_bytes = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(
            "CAP_PRICING_SIGNING_PUBLIC_KEY must be base64-encoded raw 32-byte Ed25519 public key"
        ) from exc
    if len(key_bytes) != 32:
        raise RuntimeError(
            "CAP_PRICING_SIGNING_PUBLIC_KEY must decode to exactly 32 bytes (Ed25519 public key)"
        )
    try:
        return Ed25519PublicKey.from_public_bytes(key_bytes)
    except ValueError as exc:
        raise RuntimeError("Invalid Ed25519 public key in CAP_PRICING_SIGNING_PUBLIC_KEY") from exc


def _verify_ed25519_signature(
    *, public_key: Ed25519PublicKey, signature: str, body: bytes
) -> None:
    try:
        signature_bytes = base64.b64decode(signature, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("Remote pricing signature must be base64-encoded Ed25519 bytes") from exc
    try:
        public_key.verify(signature_bytes, body)
    except InvalidSignature as exc:
        raise RuntimeError("Remote pricing signature verification failed") from exc


def _parse_iso8601_utc(value: str) -> datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_version(v: str) -> tuple[int, int, int] | None:
    core = v.strip().split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    nums: list[int] = []
    for part in parts:
        if not part or not part.isdigit():
            return None
        nums.append(int(part))
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def _is_version_compatible(current: str, min_v: str | None, max_v: str | None) -> bool:
    cur = _parse_version(current)
    if cur is None:
        return False
    if min_v:
        mn = _parse_version(min_v)
        if mn is None or cur < mn:
            return False
    if max_v:
        mx = _parse_version(max_v)
        if mx is None or cur > mx:
            return False
    return True


def _current_engine_version() -> str:
    return (os.getenv("CAP_ENGINE_VERSION") or ENGINE_VERSION).strip()


def _verify_and_parse_signed_pricing(payload: Any) -> dict[str, dict[str, RateCard]]:
    if not isinstance(payload, dict):
        raise RuntimeError("Remote pricing payload must be an object")

    expected_schema = (os.getenv("CAP_PRICING_SCHEMA_VERSION") or "1").strip()
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or schema_version.strip() != expected_schema:
        raise RuntimeError(
            f"Unsupported pricing schema_version: {schema_version!r}; expected {expected_schema!r}"
        )

    signature = payload.get("signature")
    if not isinstance(signature, str) or not signature.strip():
        raise RuntimeError("Remote pricing payload is missing signature")

    verify_signature = (os.getenv("CAP_PRICING_VERIFY_SIGNATURE") or "true").strip().lower()
    should_verify = verify_signature in {"1", "true", "yes", "on"}
    if should_verify:
        signing_public_key = _load_signing_public_key()
        unsigned = dict(payload)
        unsigned.pop("signature", None)
        _verify_ed25519_signature(
            public_key=signing_public_key,
            signature=signature.strip(),
            body=_canonical_json_bytes(unsigned),
        )

    expires_at = payload.get("expires_at")
    if not isinstance(expires_at, str) or not expires_at.strip():
        raise RuntimeError("Remote pricing payload is missing expires_at")
    expires_dt = _parse_iso8601_utc(expires_at)
    skew_seconds = max(0, int((os.getenv("CAP_PRICING_EXPIRY_SKEW_SECONDS") or "30").strip()))
    now = datetime.now(timezone.utc)
    if now.timestamp() > (expires_dt.timestamp() + skew_seconds):
        raise RuntimeError("Remote pricing payload has expired")

    generated_at = payload.get("generated_at")
    if isinstance(generated_at, str) and generated_at.strip():
        generated_dt = _parse_iso8601_utc(generated_at)
        if generated_dt.timestamp() > (now.timestamp() + skew_seconds):
            raise RuntimeError("Remote pricing payload generated_at is in the future")

    engine = payload.get("engine")
    if isinstance(engine, dict):
        min_version = engine.get("min_version") if isinstance(engine.get("min_version"), str) else None
        max_version = engine.get("max_version") if isinstance(engine.get("max_version"), str) else None
        if min_version or max_version:
            current_version = _current_engine_version()
            if not _is_version_compatible(current_version, min_version, max_version):
                raise RuntimeError(
                    "Remote pricing is not compatible with sidecar engine version "
                    f"{current_version} (allowed {min_version or '*'}..{max_version or '*'})"
                )

    rates_payload = payload.get("rates")
    if not isinstance(rates_payload, dict):
        raise RuntimeError("Remote pricing payload is missing rates object")
    rates = _parse_table(rates_payload)
    if not rates:
        raise RuntimeError("Remote pricing payload contained no valid rates")
    return rates


def _remote_pricing_url() -> str:
    return (os.getenv("CAP_PRICING_URL") or "").strip() or "https://api.cynsta.com/v1/public/pricing"


def _parse_pricing_source() -> str:
    raw = (os.getenv("CAP_PRICING_SOURCE") or "remote").strip().lower()
    if raw != "remote":
        LOGGER.warning("Unsupported CAP_PRICING_SOURCE '%s' in sidecar; forcing remote", raw)
    return "remote"


def load_rates_from_remote(
    *,
    url: str | None = None,
    etag: str | None = None,
) -> tuple[dict[str, dict[str, RateCard]] | None, str | None]:
    _ = _parse_pricing_source()
    target = url or _remote_pricing_url()
    if not target:
        raise RuntimeError("CAP_PRICING_URL is required")

    headers = {"Accept": "application/json"}
    if etag:
        headers["If-None-Match"] = etag

    req = Request(target, method="GET", headers=headers)
    timeout = max(1, int((os.getenv("CAP_PRICING_HTTP_TIMEOUT_SECONDS") or "10").strip()))

    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                raise RuntimeError("Remote pricing response was empty")
            parsed = json.loads(raw)
            rates = _verify_and_parse_signed_pricing(parsed)
            return rates, resp.headers.get("ETag")
    except HTTPError as exc:
        if exc.code == 304:
            return None, etag
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"Remote pricing request failed: HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Remote pricing request failed: {exc.reason}") from exc


def load_rates() -> dict[str, dict[str, RateCard]]:
    rates, _ = load_rates_from_remote()
    if rates is None:
        raise RuntimeError("Remote pricing returned 304, but no cached rates are available")
    return rates


class RateTableProvider:
    def __init__(self, refresh_seconds: int = 300) -> None:
        self.refresh_seconds = max(1, int(refresh_seconds))
        self._lock = threading.Lock()
        self._rates: dict[str, dict[str, RateCard]] | None = None
        self._loaded_at = 0.0
        self._etag: str | None = None

    @classmethod
    def from_env(cls) -> "RateTableProvider":
        raw = (os.getenv("CAP_PRICE_TABLE_REFRESH_SECONDS") or "300").strip()
        try:
            refresh = int(raw)
        except Exception:
            refresh = 300
        return cls(refresh_seconds=refresh)

    def _needs_refresh(self, now: float) -> bool:
        if self._rates is None:
            return True
        return (now - self._loaded_at) >= float(self.refresh_seconds)

    def _load_rates_cached_remote(self) -> dict[str, dict[str, RateCard]]:
        remote_rates, etag = load_rates_from_remote(etag=self._etag)
        if etag:
            self._etag = etag
        if remote_rates is not None:
            return remote_rates
        if self._rates is not None:
            return self._rates
        raise RuntimeError("Remote pricing returned 304, but no cached rates are available")

    def get_rates(self) -> dict[str, dict[str, RateCard]]:
        now = time.time()
        if not self._needs_refresh(now):
            return self._rates or {}
        with self._lock:
            now = time.time()
            if self._needs_refresh(now):
                self._rates = self._load_rates_cached_remote()
                self._loaded_at = now
        return self._rates or {}

    def force_refresh(self) -> dict[str, dict[str, RateCard]]:
        with self._lock:
            self._rates = self._load_rates_cached_remote()
            self._loaded_at = time.time()
            return self._rates

    def get_rate_card(self, provider: str, model: str) -> RateCard | None:
        provider_map = self.get_rates().get(provider) or {}
        return provider_map.get(model)
