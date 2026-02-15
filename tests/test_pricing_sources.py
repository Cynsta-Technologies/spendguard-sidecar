import hashlib
import hmac
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app.pricing import RateCard, RateTableProvider, load_rates, load_rates_from_remote


class _FakeResponse:
    def __init__(self, payload: dict, etag: str | None = None):
        self._raw = json.dumps(payload).encode("utf-8")
        self.headers = {"ETag": etag} if etag else {}

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _signed_payload(*, signing_key: str, rates: dict, ttl_seconds: int = 300) -> dict:
    now = datetime.now(timezone.utc)
    unsigned = {
        "schema_version": "1",
        "engine": {
            "name": "spendguard-engine",
            "min_version": "0.1.0",
            "max_version": "9.9.9",
        },
        "generated_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
        "rates": rates,
    }
    sig = hmac.new(signing_key.encode("utf-8"), _canonical_json_bytes(unsigned), hashlib.sha256).hexdigest()
    return {**unsigned, "signature": sig}


class TestPricingSources(unittest.TestCase):
    ENV_KEYS = [
        "CAP_PRICING_SOURCE",
        "CAP_PRICING_URL",
        "CAP_PRICING_HTTP_TIMEOUT_SECONDS",
        "CAP_PRICE_TABLE_REFRESH_SECONDS",
        "CAP_PRICING_SIGNING_KEY",
        "CAP_PRICING_VERIFY_SIGNATURE",
        "CAP_PRICING_SCHEMA_VERSION",
        "CAP_PRICING_EXPIRY_SKEW_SECONDS",
        "CAP_ENGINE_VERSION",
    ]

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self.ENV_KEYS}
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["CAP_PRICING_SOURCE"] = "remote"
        os.environ["CAP_PRICING_SIGNING_KEY"] = "test-signing-key"

    def tearDown(self):
        for key in self.ENV_KEYS:
            prior = self._saved.get(key)
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior

    def test_load_rates_from_remote_verifies_signature(self):
        payload = _signed_payload(
            signing_key="test-signing-key",
            rates={"openai": {"gpt-4o-mini": {"input_cents_per_1m": 77, "output_cents_per_1m": 155}}},
        )

        with patch("app.pricing.urlopen", return_value=_FakeResponse(payload, etag='"abc"')):
            rates, etag = load_rates_from_remote(url="https://example.local/pricing")

        self.assertEqual(etag, '"abc"')
        self.assertEqual(rates["openai"]["gpt-4o-mini"].input_cents_per_1m, 77)

    def test_load_rates_from_remote_rejects_invalid_signature(self):
        payload = _signed_payload(
            signing_key="test-signing-key",
            rates={"openai": {"gpt-4o-mini": {"input_cents_per_1m": 77, "output_cents_per_1m": 155}}},
        )
        payload["rates"]["openai"]["gpt-4o-mini"]["input_cents_per_1m"] = 99

        with patch("app.pricing.urlopen", return_value=_FakeResponse(payload)):
            with self.assertRaises(RuntimeError):
                load_rates_from_remote(url="https://example.local/pricing")

    @patch("app.pricing.load_rates_from_remote")
    def test_load_rates_requires_remote_rows(self, mock_remote):
        mock_remote.return_value = (None, None)
        with self.assertRaises(RuntimeError):
            load_rates()

    @patch("app.pricing.load_rates_from_remote")
    def test_rate_table_provider_caches_until_refresh(self, mock_remote):
        mock_remote.side_effect = [
            ({"openai": {"m1": RateCard(input_cents_per_1m=1, output_cents_per_1m=2)}}, '"etag-1"'),
            ({"openai": {"m1": RateCard(input_cents_per_1m=3, output_cents_per_1m=4)}}, '"etag-2"'),
        ]
        provider = RateTableProvider(refresh_seconds=300)

        first = provider.get_rate_card("openai", "m1")
        second = provider.get_rate_card("openai", "m1")
        self.assertEqual(mock_remote.call_count, 1)
        self.assertEqual(first.input_cents_per_1m, 1)
        self.assertEqual(second.input_cents_per_1m, 1)

        provider._loaded_at = 0.0
        with patch("app.pricing.time.time", return_value=1000.0):
            refreshed = provider.get_rate_card("openai", "m1")
        self.assertEqual(mock_remote.call_count, 2)
        self.assertEqual(refreshed.input_cents_per_1m, 3)

    @patch("app.pricing.load_rates_from_remote")
    def test_rate_table_provider_uses_cached_rates_on_304(self, mock_remote):
        mock_remote.side_effect = [
            ({"openai": {"m1": RateCard(input_cents_per_1m=1, output_cents_per_1m=2)}}, '"etag-1"'),
            (None, '"etag-1"'),
        ]
        provider = RateTableProvider(refresh_seconds=300)

        first = provider.get_rate_card("openai", "m1")
        self.assertEqual(first.input_cents_per_1m, 1)

        provider._loaded_at = 0.0
        with patch("app.pricing.time.time", return_value=1000.0):
            refreshed = provider.get_rate_card("openai", "m1")
        self.assertEqual(mock_remote.call_count, 2)
        self.assertEqual(refreshed.input_cents_per_1m, 1)


if __name__ == "__main__":
    unittest.main()
