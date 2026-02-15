import os
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


if __name__ == "__main__":
    unittest.main()
