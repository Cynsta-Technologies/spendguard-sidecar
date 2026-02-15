from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.core.supabase import StorageError, SupabaseClient


API_KEY_PATTERN = re.compile(r"^sk_cynsta_(live|test)_([0-9A-Za-z]{32})$")


class ApiKeyAuthError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class ApiKeyRecord:
    key_id: str
    organization_id: str
    scopes: list[str]


def hash_api_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def authenticate_api_key(x_api_key: Optional[str], required_scope: str) -> ApiKeyRecord:
    if not x_api_key:
        raise ApiKeyAuthError(status_code=401, detail="Missing API key")
    if not API_KEY_PATTERN.match(x_api_key):
        raise ApiKeyAuthError(status_code=401, detail="Invalid API key")
    try:
        client = SupabaseClient.from_env()
        rows = client.request_json(
            "GET",
            "/rest/v1/api_keys",
            params={
                "hash": f"eq.{hash_api_key(x_api_key)}",
                "revoked_at": "is.null",
                "select": "id,organization_id,scopes",
                "limit": 1,
            },
            expect=(200,),
        )
        if not rows:
            raise ApiKeyAuthError(status_code=401, detail="Invalid API key")
        row = rows[0]
        scopes = row.get("scopes") or []
        if required_scope not in scopes:
            raise ApiKeyAuthError(status_code=403, detail="Insufficient scope")
        try:
            client.request_json(
                "PATCH",
                "/rest/v1/api_keys",
                params={"id": f"eq.{row['id']}"},
                json_body={"last_used_at": datetime.now(tz=timezone.utc).isoformat()},
                headers={"Prefer": "return=representation"},
                expect=(200, 204),
            )
        except StorageError:
            pass
        return ApiKeyRecord(
            key_id=row["id"],
            organization_id=row["organization_id"],
            scopes=scopes,
        )
    except StorageError as exc:
        raise ApiKeyAuthError(status_code=500, detail=str(exc)) from exc

