import importlib
import os
import tempfile
import unittest

from fastapi.testclient import TestClient


def _client_with_temp_store(db_path: str) -> TestClient:
    os.environ["CAP_MODE"] = "sidecar"
    os.environ["CAP_STORE"] = "sqlite"
    os.environ["CAP_SQLITE_PATH"] = db_path
    os.environ["CAP_PRICING_SOURCE"] = "remote"
    os.environ["CAP_PRICING_VERIFY_SIGNATURE"] = "false"

    import app.pricing as pricing

    pricing.load_rates_from_remote = lambda **kwargs: (
        {"openai": {"gpt-4o-mini": pricing.RateCard(input_cents_per_1m=30, output_cents_per_1m=120)}},
        '"test-etag"',
    )

    import app.main as spendguard

    spendguard = importlib.reload(spendguard)
    return TestClient(spendguard.app)


class TestAgentRoutes(unittest.TestCase):
    def test_get_rename_delete_agent_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "cap.db")
            client = _client_with_temp_store(db_path)

            created = client.post("/v1/agents", json={"name": "alpha"})
            self.assertEqual(created.status_code, 200, created.text)
            agent_id = created.json()["agent_id"]

            got = client.get(f"/v1/agents/{agent_id}")
            self.assertEqual(got.status_code, 200, got.text)
            self.assertEqual(got.json()["agent_id"], agent_id)
            self.assertEqual(got.json()["name"], "alpha")

            renamed = client.patch(f"/v1/agents/{agent_id}", json={"name": "beta"})
            self.assertEqual(renamed.status_code, 200, renamed.text)
            self.assertEqual(renamed.json()["name"], "beta")

            deleted = client.delete(f"/v1/agents/{agent_id}")
            self.assertEqual(deleted.status_code, 200, deleted.text)
            self.assertEqual(deleted.json(), {"agent_id": agent_id, "deleted": True})

            missing_get = client.get(f"/v1/agents/{agent_id}")
            self.assertEqual(missing_get.status_code, 404, missing_get.text)
            self.assertEqual(missing_get.json().get("detail"), "Agent not found")

            missing_delete = client.delete(f"/v1/agents/{agent_id}")
            self.assertEqual(missing_delete.status_code, 404, missing_delete.text)
            self.assertEqual(missing_delete.json().get("detail"), "Agent not found")

    def test_rename_agent_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "cap.db")
            client = _client_with_temp_store(db_path)

            created = client.post("/v1/agents", json={"name": "alpha"})
            self.assertEqual(created.status_code, 200, created.text)
            agent_id = created.json()["agent_id"]

            missing_name = client.patch(f"/v1/agents/{agent_id}", json={})
            self.assertEqual(missing_name.status_code, 400, missing_name.text)
            self.assertEqual(missing_name.json().get("detail"), "name is required (string)")

            non_string_name = client.patch(f"/v1/agents/{agent_id}", json={"name": 123})
            self.assertEqual(non_string_name.status_code, 400, non_string_name.text)
            self.assertEqual(non_string_name.json().get("detail"), "name is required (string)")


if __name__ == "__main__":
    unittest.main()
