import importlib
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


class TestGrokOpenAICompat(unittest.TestCase):
    def test_grok_chat_uses_xai_base_url_and_logs_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "cap.db")
            os.environ["CAP_MODE"] = "sidecar"
            os.environ["CAP_STORE"] = "sqlite"
            os.environ["CAP_SQLITE_PATH"] = db_path
            os.environ["CAP_PRICING_SOURCE"] = "remote"
            os.environ["CAP_PRICING_VERIFY_SIGNATURE"] = "false"
            os.environ["XAI_API_KEY"] = "xai-test"
            os.environ["CAP_OPENAI_MAX_COMPLETION_TOKENS"] = "10"

            from fastapi.testclient import TestClient

            import app.pricing as pricing

            pricing.load_rates_from_remote = lambda **kwargs: (
                {"grok": {"grok-3": pricing.RateCard(input_cents_per_1m=300, output_cents_per_1m=1500)}},
                '"test-etag"',
            )

            import app.main as spendguard

            spendguard = importlib.reload(spendguard)
            captured_client_args: dict[str, str] = {}

            def _fake_openai(*, api_key: str, base_url: str | None = None):
                captured_client_args["api_key"] = api_key
                if base_url is not None:
                    captured_client_args["base_url"] = base_url
                return object()

            def _fake_call_openai_chat(*, max_tokens: int, **kwargs):
                self.assertEqual(max_tokens, 10)
                return _FakeResponse()

            spendguard.OpenAI = _fake_openai
            spendguard.call_openai_chat = _fake_call_openai_chat

            client = TestClient(spendguard.app)
            agent_id = client.post("/v1/agents", json={"name": "smoke"}).json()["agent_id"]
            client.post(
                f"/v1/agents/{agent_id}/budget",
                json={"hard_limit_cents": 5, "topup_cents": 0},
            )

            r = client.post(
                f"/v1/agents/{agent_id}/runs/run-1/grok/chat/completions",
                json={
                    "model": "grok-3",
                    "messages": [{"role": "user", "content": "Say only: ok"}],
                    "max_tokens": 999999,
                    "stream": False,
                },
            )
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(captured_client_args.get("api_key"), "xai-test")
            self.assertEqual(captured_client_args.get("base_url"), "https://api.x.ai/v1")

            conn = sqlite3.connect(db_path)
            try:
                provider = conn.execute(
                    "select provider from cap_usage_ledger order by created_at desc limit 1"
                ).fetchone()[0]
            finally:
                conn.close()

            self.assertEqual(provider, "grok")


if __name__ == "__main__":
    unittest.main()
