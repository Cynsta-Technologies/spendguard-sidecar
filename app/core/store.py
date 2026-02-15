from __future__ import annotations

import os
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.supabase import StorageError, SupabaseClient


class BudgetError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class AgentBudget:
    agent_id: str
    hard_limit_cents: int
    remaining_cents: int
    locked_run_id: str | None
    locked_cents: int
    locked_expires_at: str | None


@dataclass(frozen=True)
class AgentRecord:
    agent_id: str
    name: str | None
    created_at: str


class CapStore:
    def create_agent(self, organization_id: str, name: str | None) -> str:
        raise NotImplementedError

    def list_agents(self, organization_id: str) -> list[AgentRecord]:
        raise NotImplementedError

    def upsert_budget(
        self,
        organization_id: str,
        agent_id: str,
        hard_limit_cents: int,
        topup_cents: int,
    ) -> AgentBudget:
        raise NotImplementedError

    def get_budget(self, organization_id: str, agent_id: str) -> AgentBudget:
        raise NotImplementedError

    def reserve(
        self,
        organization_id: str,
        agent_id: str,
        run_id: str,
        wcec_cents: int,
        ttl_seconds: int = 300,
    ) -> AgentBudget:
        raise NotImplementedError

    def settle(
        self,
        organization_id: str,
        agent_id: str,
        run_id: str,
        realized_cents: int,
        meta: dict[str, Any],
    ) -> AgentBudget:
        raise NotImplementedError

    def log_usage(self, row: dict[str, Any]) -> None:
        raise NotImplementedError


class SqliteCapStore(CapStore):
    def __init__(self, path: str) -> None:
        self.path = path
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                create table if not exists cap_agents (
                  agent_id text primary key,
                  organization_id text not null,
                  name text,
                  created_at text not null
                )
                """
            )
            conn.execute(
                """
                create table if not exists cap_budgets (
                  agent_id text primary key,
                  organization_id text not null,
                  hard_limit_cents integer not null,
                  remaining_cents integer not null,
                  locked_run_id text,
                  locked_cents integer not null default 0,
                  locked_expires_at text
                )
                """
            )
            conn.execute(
                """
                create table if not exists cap_usage_ledger (
                  id text primary key,
                  organization_id text not null,
                  agent_id text not null,
                  run_id text not null,
                  provider text not null,
                  model text not null,
                  input_tokens integer,
                  output_tokens integer,
                  wcec_cents integer not null,
                  realized_cents integer not null,
                  termination_reason text,
                  meta_json text,
                  created_at text not null
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def create_agent(self, organization_id: str, name: str | None) -> str:
        agent_id = str(uuid.uuid4())
        conn = self._connect()
        try:
            conn.execute(
                "insert into cap_agents(agent_id, organization_id, name, created_at) values (?, ?, ?, ?)",
                (agent_id, organization_id, name, _utc_now()),
            )
            conn.commit()
            return agent_id
        finally:
            conn.close()

    def list_agents(self, organization_id: str) -> list[AgentRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "select agent_id, name, created_at from cap_agents where organization_id=? order by created_at asc",
                (organization_id,),
            ).fetchall()
            out: list[AgentRecord] = []
            for row in rows:
                out.append(
                    AgentRecord(
                        agent_id=str(row["agent_id"]),
                        name=row["name"],
                        created_at=str(row["created_at"]),
                    )
                )
            return out
        finally:
            conn.close()

    def upsert_budget(self, organization_id: str, agent_id: str, hard_limit_cents: int, topup_cents: int) -> AgentBudget:
        if hard_limit_cents <= 0:
            raise BudgetError(400, "hard_limit_cents must be > 0")
        if topup_cents < 0:
            raise BudgetError(400, "topup_cents must be >= 0")
        conn = self._connect()
        try:
            row = conn.execute(
                "select * from cap_budgets where organization_id=? and agent_id=?",
                (organization_id, agent_id),
            ).fetchone()
            if not row:
                remaining = min(hard_limit_cents, topup_cents if topup_cents else hard_limit_cents)
                conn.execute(
                    """
                    insert into cap_budgets(agent_id, organization_id, hard_limit_cents, remaining_cents, locked_cents)
                    values (?, ?, ?, ?, 0)
                    """,
                    (agent_id, organization_id, hard_limit_cents, remaining),
                )
            else:
                remaining = int(row["remaining_cents"]) + int(topup_cents)
                if remaining > hard_limit_cents:
                    remaining = hard_limit_cents
                conn.execute(
                    """
                    update cap_budgets
                    set hard_limit_cents=?, remaining_cents=?
                    where organization_id=? and agent_id=?
                    """,
                    (hard_limit_cents, remaining, organization_id, agent_id),
                )
            conn.commit()
            return self.get_budget(organization_id, agent_id)
        finally:
            conn.close()

    def get_budget(self, organization_id: str, agent_id: str) -> AgentBudget:
        conn = self._connect()
        try:
            row = conn.execute(
                "select * from cap_budgets where organization_id=? and agent_id=?",
                (organization_id, agent_id),
            ).fetchone()
            if not row:
                raise BudgetError(404, "Budget not found")
            return AgentBudget(
                agent_id=agent_id,
                hard_limit_cents=int(row["hard_limit_cents"]),
                remaining_cents=int(row["remaining_cents"]),
                locked_run_id=row["locked_run_id"],
                locked_cents=int(row["locked_cents"] or 0),
                locked_expires_at=row["locked_expires_at"],
            )
        finally:
            conn.close()

    def _release_expired_lock(self, conn: sqlite3.Connection, organization_id: str, agent_id: str) -> None:
        row = conn.execute(
            "select locked_run_id, locked_expires_at from cap_budgets where organization_id=? and agent_id=?",
            (organization_id, agent_id),
        ).fetchone()
        if not row:
            return
        locked_run_id = row["locked_run_id"]
        expires_at = row["locked_expires_at"]
        if not locked_run_id or not expires_at:
            return
        try:
            expires_epoch = int(expires_at)
        except Exception:
            return
        if int(time.time()) >= expires_epoch:
            conn.execute(
                """
                update cap_budgets
                set locked_run_id=null, locked_cents=0, locked_expires_at=null
                where organization_id=? and agent_id=? and locked_run_id=?
                """,
                (organization_id, agent_id, locked_run_id),
            )

    def reserve(self, organization_id: str, agent_id: str, run_id: str, wcec_cents: int, ttl_seconds: int = 300) -> AgentBudget:
        if wcec_cents <= 0:
            raise BudgetError(400, "wcec_cents must be > 0")
        conn = self._connect()
        try:
            conn.execute("begin immediate")
            self._release_expired_lock(conn, organization_id, agent_id)
            row = conn.execute(
                "select * from cap_budgets where organization_id=? and agent_id=?",
                (organization_id, agent_id),
            ).fetchone()
            if not row:
                raise BudgetError(404, "Budget not found")
            if row["locked_run_id"]:
                raise BudgetError(409, "Agent budget is locked by an in-flight run")
            remaining = int(row["remaining_cents"])
            if remaining < wcec_cents:
                raise BudgetError(402, "Insufficient budget")
            expires_epoch = int(time.time()) + int(ttl_seconds)
            conn.execute(
                """
                update cap_budgets
                set locked_run_id=?, locked_cents=?, locked_expires_at=?
                where organization_id=? and agent_id=? and locked_run_id is null
                """,
                (run_id, wcec_cents, str(expires_epoch), organization_id, agent_id),
            )
            conn.commit()
            return self.get_budget(organization_id, agent_id)
        finally:
            conn.close()

    def settle(self, organization_id: str, agent_id: str, run_id: str, realized_cents: int, meta: dict[str, Any]) -> AgentBudget:
        if realized_cents < 0:
            raise BudgetError(400, "realized_cents must be >= 0")
        conn = self._connect()
        try:
            conn.execute("begin immediate")
            row = conn.execute(
                "select * from cap_budgets where organization_id=? and agent_id=?",
                (organization_id, agent_id),
            ).fetchone()
            if not row:
                raise BudgetError(404, "Budget not found")
            if row["locked_run_id"] != run_id:
                raise BudgetError(409, "Run does not hold the budget lock")
            remaining = int(row["remaining_cents"])
            new_remaining = remaining - int(realized_cents)
            if new_remaining < 0:
                new_remaining = 0
            conn.execute(
                """
                update cap_budgets
                set remaining_cents=?, locked_run_id=null, locked_cents=0, locked_expires_at=null
                where organization_id=? and agent_id=? and locked_run_id=?
                """,
                (new_remaining, organization_id, agent_id, run_id),
            )
            conn.commit()
            return self.get_budget(organization_id, agent_id)
        finally:
            conn.close()

    def log_usage(self, row: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                insert into cap_usage_ledger(
                  id, organization_id, agent_id, run_id, provider, model,
                  input_tokens, output_tokens, wcec_cents, realized_cents,
                  termination_reason, meta_json, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["organization_id"],
                    row["agent_id"],
                    row["run_id"],
                    row["provider"],
                    row["model"],
                    row.get("input_tokens"),
                    row.get("output_tokens"),
                    row["wcec_cents"],
                    row["realized_cents"],
                    row.get("termination_reason"),
                    row.get("meta_json"),
                    row["created_at"],
                ),
            )
            conn.commit()
        finally:
            conn.close()


class SupabaseCapStore(CapStore):
    def __init__(self, client: SupabaseClient) -> None:
        self.client = client

    def create_agent(self, organization_id: str, name: str | None) -> str:
        rows = self.client.request_json(
            "POST",
            "/rest/v1/cap_agents",
            json_body=[{"organization_id": organization_id, "name": name}],
            headers={"Prefer": "return=representation"},
            expect=(200, 201),
        )
        if not rows:
            raise StorageError("Agent creation failed")
        return rows[0]["id"]

    def list_agents(self, organization_id: str) -> list[AgentRecord]:
        rows = self.client.request_json(
            "GET",
            "/rest/v1/cap_agents",
            params={
                "organization_id": f"eq.{organization_id}",
                "select": "id,name,created_at",
                "order": "created_at.asc",
            },
            expect=(200,),
        )
        out: list[AgentRecord] = []
        for row in rows:
            out.append(
                AgentRecord(
                    agent_id=row["id"],
                    name=row.get("name"),
                    created_at=row["created_at"],
                )
            )
        return out

    def upsert_budget(self, organization_id: str, agent_id: str, hard_limit_cents: int, topup_cents: int) -> AgentBudget:
        if hard_limit_cents <= 0:
            raise BudgetError(400, "hard_limit_cents must be > 0")
        if topup_cents < 0:
            raise BudgetError(400, "topup_cents must be >= 0")

        rows = self.client.request_json(
            "GET",
            "/rest/v1/cap_budgets",
            params={
                "organization_id": f"eq.{organization_id}",
                "agent_id": f"eq.{agent_id}",
                "select": "agent_id,hard_limit_cents,remaining_cents,locked_run_id,locked_cents,locked_expires_at",
                "limit": 1,
            },
            expect=(200,),
        )
        if not rows:
            remaining = min(hard_limit_cents, topup_cents if topup_cents else hard_limit_cents)
            self.client.request_json(
                "POST",
                "/rest/v1/cap_budgets",
                json_body=[
                    {
                        "organization_id": organization_id,
                        "agent_id": agent_id,
                        "hard_limit_cents": hard_limit_cents,
                        "remaining_cents": remaining,
                    }
                ],
                headers={"Prefer": "return=representation"},
                expect=(200, 201),
            )
        else:
            row = rows[0]
            remaining = int(row["remaining_cents"]) + int(topup_cents)
            if remaining > hard_limit_cents:
                remaining = hard_limit_cents
            self.client.request_json(
                "PATCH",
                "/rest/v1/cap_budgets",
                params={
                    "organization_id": f"eq.{organization_id}",
                    "agent_id": f"eq.{agent_id}",
                },
                json_body={
                    "hard_limit_cents": hard_limit_cents,
                    "remaining_cents": remaining,
                },
                headers={"Prefer": "return=representation"},
                expect=(200, 204),
            )
        return self.get_budget(organization_id, agent_id)

    def get_budget(self, organization_id: str, agent_id: str) -> AgentBudget:
        rows = self.client.request_json(
            "GET",
            "/rest/v1/cap_budgets",
            params={
                "organization_id": f"eq.{organization_id}",
                "agent_id": f"eq.{agent_id}",
                "select": "agent_id,hard_limit_cents,remaining_cents,locked_run_id,locked_cents,locked_expires_at",
                "limit": 1,
            },
            expect=(200,),
        )
        if not rows:
            raise BudgetError(404, "Budget not found")
        row = rows[0]
        return AgentBudget(
            agent_id=row["agent_id"],
            hard_limit_cents=int(row["hard_limit_cents"]),
            remaining_cents=int(row["remaining_cents"]),
            locked_run_id=row.get("locked_run_id"),
            locked_cents=int(row.get("locked_cents") or 0),
            locked_expires_at=row.get("locked_expires_at"),
        )

    def _clear_if_expired(self, organization_id: str, agent_id: str) -> None:
        rows = self.client.request_json(
            "GET",
            "/rest/v1/cap_budgets",
            params={
                "organization_id": f"eq.{organization_id}",
                "agent_id": f"eq.{agent_id}",
                "select": "locked_run_id,locked_expires_at",
                "limit": 1,
            },
            expect=(200,),
        )
        if not rows:
            return
        row = rows[0]
        locked_run_id = row.get("locked_run_id")
        expires_at = row.get("locked_expires_at")
        if not locked_run_id or not expires_at:
            return
        now_iso = _utc_now()
        if expires_at <= now_iso:
            self.client.request_json(
                "PATCH",
                "/rest/v1/cap_budgets",
                params={
                    "organization_id": f"eq.{organization_id}",
                    "agent_id": f"eq.{agent_id}",
                    "locked_run_id": f"eq.{locked_run_id}",
                },
                json_body={"locked_run_id": None, "locked_cents": 0, "locked_expires_at": None},
                headers={"Prefer": "return=representation"},
                expect=(200, 204),
            )

    def reserve(self, organization_id: str, agent_id: str, run_id: str, wcec_cents: int, ttl_seconds: int = 300) -> AgentBudget:
        if wcec_cents <= 0:
            raise BudgetError(400, "wcec_cents must be > 0")
        self._clear_if_expired(organization_id, agent_id)
        lock_until = datetime.fromtimestamp(
            datetime.now(tz=timezone.utc).timestamp() + int(ttl_seconds),
            tz=timezone.utc,
        ).isoformat()

        updated = self.client.request_json(
            "PATCH",
            "/rest/v1/cap_budgets",
            params={
                "organization_id": f"eq.{organization_id}",
                "agent_id": f"eq.{agent_id}",
                "locked_run_id": "is.null",
                "remaining_cents": f"gte.{wcec_cents}",
            },
            json_body={"locked_run_id": run_id, "locked_cents": wcec_cents, "locked_expires_at": lock_until},
            headers={"Prefer": "return=representation"},
            expect=(200, 204),
        )
        if not updated:
            budget = self.get_budget(organization_id, agent_id)
            if budget.locked_run_id:
                raise BudgetError(409, "Agent budget is locked by an in-flight run")
            raise BudgetError(402, "Insufficient budget")
        return self.get_budget(organization_id, agent_id)

    def settle(self, organization_id: str, agent_id: str, run_id: str, realized_cents: int, meta: dict[str, Any]) -> AgentBudget:
        budget = self.get_budget(organization_id, agent_id)
        if budget.locked_run_id != run_id:
            raise BudgetError(409, "Run does not hold the budget lock")
        new_remaining = budget.remaining_cents - int(realized_cents)
        if new_remaining < 0:
            new_remaining = 0
        self.client.request_json(
            "PATCH",
            "/rest/v1/cap_budgets",
            params={
                "organization_id": f"eq.{organization_id}",
                "agent_id": f"eq.{agent_id}",
                "locked_run_id": f"eq.{run_id}",
            },
            json_body={
                "remaining_cents": new_remaining,
                "locked_run_id": None,
                "locked_cents": 0,
                "locked_expires_at": None,
            },
            headers={"Prefer": "return=representation"},
            expect=(200, 204),
        )
        return self.get_budget(organization_id, agent_id)

    def log_usage(self, row: dict[str, Any]) -> None:
        meta = row.get("meta_json")
        if isinstance(meta, str):
            try:
                row["meta_json"] = json.loads(meta)
            except Exception:
                row["meta_json"] = {"raw": meta}
        self.client.request_json(
            "POST",
            "/rest/v1/cap_usage_ledger",
            json_body=[row],
            headers={"Prefer": "return=representation"},
            expect=(200, 201),
        )


def store_from_env() -> CapStore:
    mode = (os.getenv("CAP_STORE") or "").strip().lower()
    if not mode:
        mode = "supabase" if (os.getenv("CAP_MODE") or "").strip().lower() == "hosted" else "sqlite"
    if mode == "supabase":
        return SupabaseCapStore(SupabaseClient.from_env())
    if mode == "sqlite":
        path = os.getenv("CAP_SQLITE_PATH") or ".\\cynsta-spendguard.db"
        return SqliteCapStore(path)
    raise RuntimeError("CAP_STORE must be sqlite or supabase")
