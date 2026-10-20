# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""HMS destination: wraps the SDK client with retries, probing and per-tenant clients."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..client import HolographicClient
from .common import register_secret, scrub
from .model import HmsDoc, ImporterError

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
        err = res.json().get("error", {})
        return str(err.get("code", "")), str(err.get("message", ""))
    except (ValueError, AttributeError):
        return "", res.text[:200]


class HmsSink:
    """One HMS server. Clients are created per tenant on demand and share credentials."""

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
        self._clients: Dict[Optional[str], HolographicClient] = {}

    def client(self, tenant: Optional[str]) -> HolographicClient:
        tenant = tenant or self.tenant
        got = self._clients.get(tenant)
        if got is None:
            got = self._clients[tenant] = HolographicClient(tenant_id=tenant, **self._args)  # type: ignore[arg-type]
        return got

    def close(self) -> None:
        for c in self._clients.values():
            c.close()
        self._clients.clear()

    # -- request plumbing -----------------------------------------------------------

    def _call(self, fn, *args, **kwargs):
        delay = self.backoff
        for attempt in range(self.retries + 1):
            try:
                return fn(*args, **kwargs)
            except httpx.HTTPStatusError as exc:
                res = exc.response
                if res.status_code == 401:
                    raise ImporterError("the HMS server rejected the API key (401)") from None
                code, message = _error_parts(res)
                if res.status_code in _RETRY_STATUS and attempt < self.retries:
                    wait = float(res.headers.get("Retry-After", delay) or delay)
                    time.sleep(min(wait, 30.0))
                    delay = min(delay * 2, 30.0)
                    continue
                if res.status_code in _RETRY_STATUS:
                    raise ImporterError(
                        scrub(f"HMS server kept failing after {self.retries} retries: HTTP {res.status_code} {code}")
                    ) from None
                raise BatchRejected(res.status_code, code, scrub(message)) from None
            except httpx.TransportError as exc:
                if attempt < self.retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise ImporterError(scrub(f"cannot reach the HMS server at {self.url}: {type(exc).__name__}")) from None
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
            self._call(_post, client, "/api/v1/query", {"query": "hms-import capability probe", "top_k": 1})
            info["server_embeds"] = True
        except BatchRejected as exc:
            info["server_embeds"] = False if "on-device model" in exc.message else None
        return info

    def send(self, docs: List[HmsDoc]) -> None:
        by_tenant: Dict[Optional[str], List[HmsDoc]] = {}
        for d in docs:
            by_tenant.setdefault(d.tenant, []).append(d)
        for tenant, group in by_tenant.items():
            client = self.client(tenant)
            self._call(client.add_documents, [d.payload() for d in group])

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
                out[tenant or self.tenant or "default"] = self._call(self.client(tenant).maintain_document_index)
            except (BatchRejected, ImporterError) as exc:
                out[tenant or self.tenant or "default"] = {"error": scrub(str(exc))}
        return out


def _post(client: HolographicClient, path: str, body: Dict[str, Any]) -> httpx.Response:
    res = client.client.post(f"{client.url}{path}", json=body)
    res.raise_for_status()
    return res
