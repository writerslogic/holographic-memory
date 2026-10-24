# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi==0.119.0", "httpx==0.28.1", "uvicorn==0.38.0"]
# ///
"""AML Cycle 2 Add/Search contract over a private hms-server instance."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import os
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request

FACT_BYTES = 60_000
FACT_WORDS = 4_000
MAX_BODY = 64 * 1024 * 1024
MAX_ID = 1_024
MAX_MESSAGES = 1_000


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > MAX_ID:
        raise ValueError(f"{name} must be a nonempty string of at most {MAX_ID} bytes")
    return value


def tenant_for(user_id: str) -> str:
    return "aml-" + hashlib.sha256(user_id.encode()).hexdigest()[:60]


def fact_id(user_id: str, request_id: str, message: int, part: int) -> str:
    key = f"{len(user_id)}:{user_id}{len(request_id)}:{request_id}:{message}:{part}"
    return "aml-" + hashlib.sha256(key.encode()).hexdigest()


def _word_count(text: str) -> int:
    return len(text.split())


def split_text(text: str) -> list[str]:
    """Partition text without changing any source byte or whitespace."""
    parts: list[str] = []
    offset = 0
    while offset < len(text):
        lo, hi = offset + 1, len(text)
        best = offset
        while lo <= hi:
            mid = (lo + hi) // 2
            part = text[offset:mid]
            if len(part.encode()) <= FACT_BYTES and _word_count(part) <= FACT_WORDS:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best == offset:
            raise ValueError("one character exceeds the fact limit")
        if best < len(text):
            cut = max(text.rfind(" ", offset, best), text.rfind("\n", offset, best))
            if cut > offset + (best - offset) // 2:
                best = cut + 1
        parts.append(text[offset:best])
        offset = best
    return parts


def add_facts(payload: Any) -> tuple[str, str, str, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ValueError("Add body must be an object")
    request_id = _nonempty(payload.get("request_id"), "request_id")
    user_id = _nonempty(payload.get("user_id"), "user_id")
    session_id = _nonempty(payload.get("session_id"), "session_id")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
        raise ValueError(f"messages must contain 1..{MAX_MESSAGES} entries")
    facts: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise ValueError(f"messages[{index}].role must be user or assistant")
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError(f"messages[{index}].content must be nonempty text")
        timestamp = message.get("timestamp")
        if timestamp is not None and (type(timestamp) is not int or not 0 <= timestamp <= 253402300799999):
            raise ValueError(f"messages[{index}].timestamp must be Unix milliseconds")
        text = f"{message['role']}: {content}"
        for part_index, part in enumerate(split_text(text)):
            fact: dict[str, Any] = {"id": fact_id(user_id, request_id, index, part_index), "text": part}
            if timestamp is not None:
                fact["eventTime"] = timestamp
            facts.append(fact)
    return request_id, user_id, session_id, facts


def search_input(payload: Any) -> tuple[str, str, int]:
    if not isinstance(payload, dict):
        raise ValueError("Search body must be an object")
    query = payload.get("query")
    if not isinstance(query, str) or not query or len(query.encode()) > 8192:
        raise ValueError("query must be nonempty text of at most 8192 bytes")
    user_id = _nonempty(payload.get("user_id"), "user_id")
    top_k = payload.get("top_k")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer from 1 to 100")
    options = payload.get("options")
    if options is not None and (not isinstance(options, list) or any(not isinstance(x, str) for x in options)):
        raise ValueError("options must be an array of strings")
    return query, user_id, top_k


def _token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    if auth.startswith("Token "):
        return auth[6:]
    return request.headers.get("x-api-key")


def create_app(hms_url: str, token: str, commit: Any = None) -> FastAPI:
    if not token:
        raise ValueError("AML_TOKEN is required")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    client = httpx.AsyncClient(base_url=hms_url, timeout=httpx.Timeout(1800))

    @app.on_event("shutdown")
    async def close_client() -> None:
        await client.aclose()

    async def require_auth(request: Request) -> None:
        supplied = _token(request)
        if supplied is None or not hmac.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="invalid API key")

    async def hms(op: str, tenant: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await client.post(f"/api/v1/memory/{op}", json=body, headers={"X-Tenant-ID": tenant})
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail="memory engine unavailable") from exc
        if result.status_code != 200:
            status = 409 if result.status_code == 409 else 503 if result.status_code >= 500 else 422
            raise HTTPException(status_code=status, detail=f"memory engine returned {result.status_code}")
        return result.json()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        try:
            result = await client.get("/readyz", timeout=5)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail="memory engine unavailable") from exc
        if result.status_code != 200:
            raise HTTPException(status_code=503, detail="memory engine unavailable")
        return {"status": "ok"}

    @app.post("/add")
    async def add(request: Request) -> dict[str, Any]:
        await require_auth(request)
        body = await request.body()
        if len(body) > MAX_BODY:
            raise HTTPException(status_code=413, detail="Add body exceeds 64 MiB")
        try:
            import json

            request_id, user_id, session_id, facts = add_facts(json.loads(body))
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        tenant = tenant_for(user_id)
        for start in range(0, len(facts), 100):
            await hms("remember", tenant, {"facts": facts[start : start + 100]})
        if commit is not None:
            await asyncio.to_thread(commit)
        return {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}

    @app.post("/search")
    async def search(request: Request) -> dict[str, Any]:
        await require_auth(request)
        try:
            payload = await request.json()
            query, user_id, top_k = search_input(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result = await hms("recall", tenant_for(user_id), {"query": query, "k": top_k})
        hits = result.get("results")
        if not isinstance(hits, list) or len(hits) > top_k:
            raise HTTPException(status_code=503, detail="invalid memory engine result count")
        data = []
        for hit in hits:
            content = hit.get("text")
            memory_id = hit.get("id")
            if not isinstance(content, str) or not content or not isinstance(memory_id, str) or not memory_id:
                raise HTTPException(status_code=503, detail="invalid memory engine result")
            item: dict[str, Any] = {"id": memory_id, "content": content}
            score = hit.get("score")
            if isinstance(score, (float, int)) and math.isfinite(score):
                item["score"] = score
            event_ms = hit.get("eventTime")
            if type(event_ms) is int and 0 <= event_ms <= 253402300799999:
                item["created_at"] = datetime.fromtimestamp(event_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
            data.append(item)
        return {"data": data}

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(os.environ["HMS_URL"], os.environ["AML_TOKEN"]), host="127.0.0.1", port=8000)
