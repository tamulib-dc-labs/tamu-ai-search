"""A very small Solr client: just the calls this service needs, over plain HTTP."""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx


class SolrError(RuntimeError):
    pass


class Solr:
    def __init__(self, base_url: str, core: str, timeout: float = 60.0):
        self.core_url = f"{base_url}/{core}"
        self.http = httpx.Client(timeout=timeout)

    def _check(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code >= 400:
            raise SolrError(f"{response.status_code} from {response.request.url}: {response.text[:500]}")
        return response.json()

    def select(self, params: dict[str, Any]) -> dict[str, Any]:
        # POST keeps long queries (vectors are 768 numbers) out of the URL.
        return self._check(self.http.post(f"{self.core_url}/select", data={**params, "wt": "json"}))

    def iter_docs(self, query: str, filters: list[str], fields: str, rows: int = 500) -> Iterator[dict[str, Any]]:
        """Page through every match with a cursor, which stays fast however deep it goes."""
        cursor = "*"
        while True:
            data = self.select({"q": query, "fq": filters, "fl": fields, "rows": rows,
                                "sort": "id asc", "cursorMark": cursor})
            yield from data["response"]["docs"]
            next_cursor = data["nextCursorMark"]
            if next_cursor == cursor:
                return
            cursor = next_cursor

    def add(self, docs: list[dict[str, Any]]) -> None:
        self._check(self.http.post(f"{self.core_url}/update", content=json.dumps(docs),
                                   headers={"Content-Type": "application/json"}))

    def delete_by_query(self, query: str) -> None:
        self._check(self.http.post(f"{self.core_url}/update", json={"delete": {"query": query}}))

    def commit(self) -> None:
        self._check(self.http.post(f"{self.core_url}/update", json={"commit": {}}))

    def schema(self, commands: dict[str, Any]) -> dict[str, Any]:
        return self._check(self.http.post(f"{self.core_url}/schema", json=commands))

    def schema_fields(self) -> dict[str, Any]:
        return self._check(self.http.get(f"{self.core_url}/schema"))["schema"]

    def set_user_property(self, name: str, value: str) -> None:
        self._check(self.http.post(f"{self.core_url}/config", json={"set-user-property": {name: value}}))
