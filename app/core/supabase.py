from __future__ import annotations

import json
import os
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class StorageError(Exception):
    pass


class SupabaseClient:
    def __init__(self, base_url: str, service_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_key = service_key

    @classmethod
    def from_env(cls) -> "SupabaseClient":
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise StorageError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
        return cls(url, key)

    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self.service_key,
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def request_json(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
        expect: Iterable[int] = (200, 201, 204),
    ) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        body = None
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
        request_headers = self._headers()
        if headers:
            request_headers.update(headers)
        request = Request(url, data=body, headers=request_headers, method=method)
        try:
            with urlopen(request) as response:
                if response.status not in expect:
                    raise StorageError(f"Supabase error {response.status}")
                payload = response.read()
                if not payload:
                    return None
                return json.loads(payload.decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise StorageError(f"Supabase error {exc.code}: {detail}") from exc

