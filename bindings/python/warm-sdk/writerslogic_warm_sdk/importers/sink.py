# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""WARM destination: wraps the SDK client with retries, probing and per-tenant clients."""

from __future__ import annotations

import re
import time
from urllib.parse import quote
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..client import WarmClient
from .common import register_secret, scrub
from .model import WarmDoc, ImporterError
from .inventory import ImportInventory

_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class BatchRejected(Exception):
    """The server refused a batch for a reason retrying cannot fix (4xx)."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


def _error_parts(res: httpx.Response) -> Tuple[str, str]:
    try:
        body = res.json()
        if not isinstance(body, dict):
            return "", "invalid error response"
        err = body if "code" in body else body.get("error", {})
        return str(err.get("code", "")), str(err.get("detail", err.get("message", "")))
    except (ValueError, AttributeError):
        return "", res.text[:200]


class WarmSink:
    """One WARM server. Clients are created per tenant on demand and share credentials."""

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        tenant: Optional[str] = None,
        zero_trust_key: str = "",
        timeout: float = 60.0,
        retries: int = 5,
        backoff: float = 0.5,
        transport: Optional[httpx.BaseTransport] = None,
        mask_salt: Optional[bytes] = None,
    ):
        register_secret(api_key)
        register_secret(zero_trust_key)
        self.url = url.rstrip("/")
        self.tenant = tenant
        self.retries = retries
        self.backoff = backoff
        self._args = dict(
            url=self.url,
            zero_trust_key=zero_trust_key,
            api_key=api_key,
            timeout=timeout,
            transport=transport,
            mask_salt=mask_salt,
        )
        self._clients: Dict[Optional[str], WarmClient] = {}
        self.split_policy = "reject"
        self.stale_parts_deleted = 0
        self._inventory: Optional[ImportInventory] = None
        self._identity_contract = False

    def client(self, tenant: Optional[str]) -> WarmClient:
        tenant = tenant or self.tenant
        got = self._clients.get(tenant)
        if got is None:
            got = self._clients[tenant] = WarmClient(tenant_id=tenant, **self._args)  # type: ignore[arg-type]
        return got

    def close(self) -> None:
        for c in self._clients.values():
            c.close()
        self._clients.clear()
        if self._inventory is not None:
            self._inventory.close()
            self._inventory = None

    # -- request plumbing -----------------------------------------------------------

    def _call(self, fn, *args, retry_safe=True, **kwargs):
        delay = self.backoff
        for attempt in range(self.retries + 1):
            try:
                return fn(*args, **kwargs)
            except httpx.HTTPStatusError as exc:
                res = exc.response
                if res.status_code == 401:
                    raise ImporterError("the WARM server rejected the API key (401)") from None
                code, message = _error_parts(res)
                if code == "mutation_committed":
                    raise ImporterError("WARM mutation committed; reconciliation required before retrying", code=code, committed=True) from None
                if res.status_code in _RETRY_STATUS and not retry_safe:
                    raise ImporterError(
                        f"WARM mutation outcome is uncertain (HTTP {res.status_code}); reconcile before retrying",
                        code="mutation_uncertain",
                    ) from None
                if res.status_code in _RETRY_STATUS and attempt < self.retries:
                    wait = float(res.headers.get("Retry-After", delay) or delay)
                    time.sleep(min(wait, 30.0))
                    delay = min(delay * 2, 30.0)
                    continue
                if res.status_code in _RETRY_STATUS:
                    raise ImporterError(
                        scrub(f"WARM server kept failing after {self.retries} retries: HTTP {res.status_code} {code}")
                    ) from None
                raise BatchRejected(res.status_code, code, scrub(message)) from None
            except httpx.TransportError as exc:
                if not retry_safe:
                    raise ImporterError(
                        f"WARM mutation outcome is uncertain ({type(exc).__name__}); reconcile before retrying",
                        code="mutation_uncertain",
                    ) from None
                if attempt < self.retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise ImporterError(scrub(f"cannot reach the WARM server at {self.url}: {type(exc).__name__}")) from None
        raise AssertionError("unreachable")

    # -- operations -----------------------------------------------------------------

    def probe(self) -> Dict[str, Any]:
        """Learn the server's input dimension and whether it embeds text.

        There is no capability endpoint; the dimension is read from the error a one-value
        query vector provokes, and text embedding from whether a text query is accepted.
        Both are read-only for the store (they do create the tenant's empty store)."""
        client = self.client(None)
        info: Dict[str, Any] = {"input_dim": None, "server_embeds": None}
        try:
            self._call(_post, client, "/api/v1/query", {"query_vector": [1.0], "top_k": 1})
        except BatchRejected as exc:
            m = re.search(r"exactly (\d+) values", exc.message)
            if m:
                info["input_dim"] = int(m.group(1))
        try:
            self._call(_post, client, "/api/v1/query", {"query": "warm-import capability probe", "top_k": 1})
            info["server_embeds"] = True
        except BatchRejected as exc:
            info["server_embeds"] = False if "on-device model" in exc.message else None
        return info

    def send(self, docs: List[WarmDoc]) -> None:
        if not self._identity_contract:
            try:
                spec = self._call(_get, self.client(None), "/openapi.json").json()
                properties = spec["components"]["schemas"]["DocIn"]["properties"]
                if "import_identity" not in properties or "import_adopt_legacy" not in properties:
                    raise KeyError("import_identity")
            except (BatchRejected, ValueError, KeyError, TypeError):
                raise ImporterError("destination lacks atomic import identity checks; upgrade warm-server before importing") from None
            self._identity_contract = True
        if self._inventory is None:
            self._inventory = ImportInventory()
        by_tenant: Dict[Optional[str], List[WarmDoc]] = {}
        for d in docs:
            by_tenant.setdefault(d.tenant, []).append(d)
        for tenant, group in by_tenant.items():
            client = self.client(tenant)
            key = tenant or self.tenant or "default"
            if key not in self._inventory.loaded:
                self._inventory.load(key, self.documents(tenant))
            obsolete = self._inventory.prepare(key, group, self.split_policy)
            self._call(client.add_documents, [d.payload() for d in group], retry_safe=False)
            for doc in group:
                self._inventory.record(key, doc.id, doc.metadata)
            self._inventory.db.commit()
            for ident, identity in obsolete:
                try:
                    self._call(_delete_owned, client, ident, identity, retry_safe=False)
                    self.stale_parts_deleted += 1
                except BatchRejected as exc:
                    if exc.status != 404:
                        raise
                self._inventory.remove(key, ident)
            self._inventory.db.commit()

    def documents(self, tenant: Optional[str] = None):
        client = self.client(tenant)
        cursor = None
        while True:
            params: Dict[str, Any] = {"limit": 1000}
            if cursor is not None:
                params["after"] = cursor
            page = self._call(_get, client, "/api/v1/documents", params).json()
            documents = page["documents"]
            yield from documents
            following = page.get("next_cursor")
            if following is None:
                return
            if not isinstance(following, str) or (cursor is not None and following <= cursor):
                raise ImporterError("destination returned a non-advancing document cursor")
            cursor = following

    def query(
        self,
        *,
        vector: Optional[List[float]] = None,
        text: Optional[str] = None,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        tenant: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        client = self.client(tenant)
        if vector is not None:
            res = self._call(client.query, vector, top_k, filter)
        else:
            body: Dict[str, Any] = {"query": text, "top_k": top_k}
            if filter:
                body["filter"] = filter
            res = self._call(_post, client, "/api/v1/query", body).json()
        return list(res.get("matches", []))

    def maintain_index(self, tenants: List[Optional[str]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for tenant in tenants:
            try:
                out[tenant or self.tenant or "default"] = self._call(self.client(tenant).maintain_document_index, retry_safe=False)
            except (BatchRejected, ImporterError) as exc:
                out[tenant or self.tenant or "default"] = {"error": scrub(str(exc))}
        return out


def _post(client: WarmClient, path: str, body: Dict[str, Any]) -> httpx.Response:
    res = client.client.post(f"{client.url}{path}", json=body)
    res.raise_for_status()
    return res


def _get(client: WarmClient, path: str, params: Optional[Dict[str, Any]] = None) -> httpx.Response:
    res = client.client.get(f"{client.url}{path}", params=params)
    res.raise_for_status()
    return res


def _delete_owned(client: WarmClient, ident: str, identity: str) -> None:
    res = client.client.delete(f"{client.url}/api/v1/documents/{quote(ident, safe='')}",
                               params={"import_identity": identity})
    res.raise_for_status()
