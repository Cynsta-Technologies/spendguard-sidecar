import os
import sqlite3
import tempfile
import unittest


class _FakeMsg:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMsg(content)


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self):
        self.choices = [_FakeChoice("ok")]
        self.usage = _FakeUsage(prompt_tokens=10, completion_tokens=1)

    def model_dump(self):
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}


class TestOpenAIBudgetClampRespectsProviderCap(unittest.TestCase):
    def test_provider_cap_applied_before_reserve_and_logging(self):
        # Without the fix, WCEC could be computed from an enormous max_tokens value and reserve would fail (402)
        # even when the provider cap would have made the request affordable.
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "cap.db")
            os.environ["CAP_MODE"] = "sidecar"
            os.environ["CAP_STORE"] = "sqlite"
            os.environ["CAP_SQLITE_PATH"] = db_path
            os.environ["CAP_PRICING_SOURCE"] = "remote"
            os.environ["CAP_PRICING_SIGNING_KEY"] = "test-signing-key"
            os.environ["OPENAI_API_KEY"] = "test"
            os.environ["CAP_OPENAI_MAX_COMPLETION_TOKENS"] = "10"

            from fastapi.testclient import TestClient

            import app.pricing as pricing

            pricing.load_rates_from_remote = lambda **kwargs: (
                {"openai": {"gpt-4o-mini": pricing.RateCard(input_cents_per_1m=30, output_cents_per_1m=120)}},
                '"test-etag"',
            )

            import app.main as spendguard

            # Patch provider call to avoid real network.
            def _fake_call_openai_chat(*, max_tokens: int, **kwargs):
                self.assertEqual(max_tokens, 10)
                return _FakeResponse()

            spendguard.call_openai_chat = _fake_call_openai_chat
            spendguard.OpenAI = lambda api_key: object()

            client = TestClient(spendguard.app)
            agent_id = client.post("/v1/agents", json={"name": "smoke"}).json()["agent_id"]
            client.post(
                f"/v1/agents/{agent_id}/budget",
                json={"hard_limit_cents": 2, "topup_cents": 2},
            )

            r = client.post(
                f"/v1/agents/{agent_id}/runs/run-1/openai/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Say only: ok"}],
                    "max_tokens": 999999,
                    "stream": False,
                },
            )
            self.assertEqual(r.status_code, 200, r.text)

            conn = sqlite3.connect(db_path)
            try:
                meta_json = conn.execute(
                    "select meta_json from cap_usage_ledger where provider='openai' order by created_at desc limit 1"
                ).fetchone()[0]
            finally:
                conn.close()

            # meta_json is stored as a string in sqlite store.
            self.assertIn('"max_tokens": 10', meta_json)


if __name__ == "__main__":
    unittest.main()
