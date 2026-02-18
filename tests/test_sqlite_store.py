import os
import sqlite3
import tempfile
import unittest

from app.core.store import BudgetError, SqliteCapStore


class TestSqliteStore(unittest.TestCase):
    def test_list_agents_scoped_by_org(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.db")
            store = SqliteCapStore(path)
            a1 = store.create_agent("org-a", name="agent-a1")
            _ = store.create_agent("org-a", name="agent-a2")
            _ = store.create_agent("org-b", name="agent-b1")

            rows = store.list_agents("org-a")
            ids = [row.agent_id for row in rows]
            names = [row.name for row in rows]
            self.assertEqual(len(rows), 2)
            self.assertIn(a1, ids)
            self.assertIn("agent-a1", names)
            self.assertNotIn("agent-b1", names)

    def test_reserve_and_settle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.db")
            store = SqliteCapStore(path)
            org = "local"
            agent_id = store.create_agent(org, name="a1")
            store.upsert_budget(org, agent_id, hard_limit_cents=100, topup_cents=100)
            run_id = "run-1"
            store.reserve(org, agent_id, run_id, wcec_cents=30)
            budget = store.get_budget(org, agent_id)
            self.assertEqual(budget.locked_run_id, run_id)
            store.settle(org, agent_id, run_id, realized_cents=10, meta={})
            budget2 = store.get_budget(org, agent_id)
            self.assertIsNone(budget2.locked_run_id)
            self.assertEqual(budget2.remaining_cents, 90)

    def test_reserve_insufficient_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.db")
            store = SqliteCapStore(path)
            org = "local"
            agent_id = store.create_agent(org, name=None)
            store.upsert_budget(org, agent_id, hard_limit_cents=10, topup_cents=10)
            with self.assertRaises(BudgetError) as ctx:
                store.reserve(org, agent_id, "run-1", wcec_cents=20)
            self.assertEqual(ctx.exception.status_code, 402)

    def test_get_rename_and_delete_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.db")
            store = SqliteCapStore(path)
            org = "local"
            agent_id = store.create_agent(org, name="alpha")

            agent = store.get_agent(org, agent_id)
            self.assertEqual(agent.name, "alpha")

            renamed = store.rename_agent(org, agent_id, "beta")
            self.assertEqual(renamed.agent_id, agent_id)
            self.assertEqual(renamed.name, "beta")

            store.upsert_budget(org, agent_id, hard_limit_cents=100, topup_cents=100)
            store.log_usage(
                {
                    "id": "usage-1",
                    "organization_id": org,
                    "agent_id": agent_id,
                    "run_id": "run-1",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "wcec_cents": 2,
                    "realized_cents": 1,
                    "termination_reason": None,
                    "meta_json": "{}",
                    "created_at": "2026-02-18T00:00:00+00:00",
                }
            )

            store.delete_agent(org, agent_id)

            with self.assertRaises(BudgetError) as get_ctx:
                store.get_agent(org, agent_id)
            self.assertEqual(get_ctx.exception.status_code, 404)

            with self.assertRaises(BudgetError) as budget_ctx:
                store.get_budget(org, agent_id)
            self.assertEqual(budget_ctx.exception.status_code, 404)

            conn = sqlite3.connect(path)
            try:
                row = conn.execute(
                    "select count(*) from cap_usage_ledger where organization_id=? and agent_id=?",
                    (org, agent_id),
                ).fetchone()
            finally:
                conn.close()
            self.assertEqual(int(row[0]), 0)

    def test_delete_agent_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cap.db")
            store = SqliteCapStore(path)
            with self.assertRaises(BudgetError) as ctx:
                store.delete_agent("local", "missing-agent")
            self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
