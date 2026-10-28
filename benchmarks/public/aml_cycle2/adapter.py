# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["fastapi==0.119.0", "httpx==0.28.1", "uvicorn==0.38.0"]
# ///
"""AML Cycle 2 Add/Search contract over a private hms-server instance."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import heapq
import hmac
import json
import math
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request

FACT_BYTES = 60_000
FACT_WORDS = 4_000
MAX_BODY = 64 * 1024 * 1024
MAX_IMAGE = 10 * 1024 * 1024
MAX_IMAGES = 30 * 1024 * 1024
MAX_ID = 1_024
MAX_MESSAGES = 1_000
NATIVE_QUERY_BYTES = 8192
MAX_QUERY_BYTES = 1024 * 1024
MAX_QUERY_PARTS = MAX_QUERY_BYTES // (NATIVE_QUERY_BYTES // 2)
MAX_SEARCH_BODY = 8 * 1024 * 1024
MAX_ACTIVE_SEARCHES = 16
MAX_QUEUED_SEARCHES = 16
SEARCH_TIMEOUT = 1800


def query_parts(query: str) -> list[str]:
    encoded = query.encode()
    if not encoded or len(encoded) > MAX_QUERY_BYTES:
        raise ValueError('query exceeds the decoded query budget')
    parts = []
    offset = 0
    while offset < len(encoded):
        end = min(offset + NATIVE_QUERY_BYTES, len(encoded))
        while end < len(encoded) and encoded[end] & 0xC0 == 0x80:
            end -= 1
        part = encoded[offset:end].decode()
        if end < len(encoded):
            boundary = next((i + 1 for i in range(len(part) - 1, -1, -1) if part[i].isspace()), 0)
            prefix = part[:boundary].encode()
            if len(prefix) >= NATIVE_QUERY_BYTES // 2:
                end = offset + len(prefix)
                part = part[:boundary]
        parts.append(part)
        offset = end
    if len(parts) > MAX_QUERY_PARTS:
        raise ValueError('query exceeds the partition work budget')
    return parts


async def fused_recall(query: str, top_k: int, recall: Any) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    cache: dict[str, dict[str, Any]] = {}
    winners: list[str] = []
    for part in dict.fromkeys(query_parts(query)):
        result = await recall(part)
        if not isinstance(result, dict):
            raise HTTPException(status_code=503, detail='invalid memory engine result')
        hits = result.get('results')
        if not isinstance(hits, list) or len(hits) > top_k:
            raise HTTPException(status_code=503, detail='invalid memory engine result count')
        current = {}
        for hit in hits:
            if not isinstance(hit, dict):
                raise HTTPException(status_code=503, detail='invalid memory engine result')
            memory_id, text = hit.get('id'), hit.get('text')
            try:
                valid = (isinstance(memory_id, str) and bool(memory_id) and len(memory_id) <= 256
                         and len(memory_id.encode()) <= 256 and isinstance(text, str) and bool(text)
                         and len(text) <= 65536 and len(text.encode()) <= 65536)
            except UnicodeError:
                valid = False
            event_ms = hit.get('eventTime')
            if not valid or (type(event_ms) is int and not 0 <= event_ms <= 253402300799999):
                raise HTTPException(status_code=503, detail='invalid memory engine result bounds')
            if memory_id in current:
                continue
            current[memory_id] = {'id': memory_id, 'text': text}
            if type(event_ms) is int:
                current[memory_id]['eventTime'] = event_ms
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (60 + len(current))
        if len(scores) > MAX_QUERY_PARTS * 100:
            raise HTTPException(status_code=503, detail='fusion candidate budget exceeded')
        winners = heapq.nsmallest(top_k, scores, key=lambda key: (-scores[key], key))
        # Scores only increase: a new winner must occur in this batch, so only the
        # previous winners and this batch need to retain full memory content.
        retained = {}
        for memory_id in winners:
            hit = current.get(memory_id, cache.get(memory_id))
            if hit is None:
                raise HTTPException(status_code=503, detail='fusion candidate ownership invariant failed')
            retained[memory_id] = hit
        cache = retained
        del current, hits, result
        hit = text = None
    return [dict(cache[memory_id], score=scores[memory_id]) for memory_id in winners]


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


def _content_text(content: Any, name: str) -> str:
    if isinstance(content, str):
        if not content:
            raise ValueError(f"{name} must be nonempty")
        return content
    if not isinstance(content, list) or not content:
        raise ValueError(f"{name} must be nonempty text or ordered content parts")
    text: list[str] = []
    image_bytes = 0
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise ValueError(f"{name}[{index}] must be an object")
        if part.get("type") == "text":
            value = part.get("text")
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name}[{index}].text must be nonempty")
            text.append(value)
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            if not isinstance(url, str) or not any(url.startswith(f"data:image/{kind};base64,") for kind in ("jpeg", "png", "webp")):
                raise ValueError(f"{name}[{index}].image_url must be an inline JPEG, PNG, or WebP Data URI")
            encoded = url.split(",", 1)[1]
            if len(encoded) > (MAX_IMAGE + 2) // 3 * 4:
                raise ValueError(f"{name}[{index}].image_url exceeds 10 MiB")
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError(f"{name}[{index}].image_url is invalid Base64") from exc
            if len(decoded) > MAX_IMAGE:
                raise ValueError(f"{name}[{index}].image_url exceeds 10 MiB")
            image_bytes += len(decoded)
            if image_bytes > MAX_IMAGES:
                raise ValueError(f"{name} images exceed 30 MiB")
        else:
            raise ValueError(f"{name}[{index}].type must be text or image_url")
    return "\n".join(text) if text else "[image content omitted by text-only memory method]"


def add_facts(payload: Any, *, source_provenance: bool = False) -> tuple[str, str, str, list[dict[str, Any]]]:
    if type(source_provenance) is not bool:
        raise ValueError("source_provenance must be a boolean")
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
        content = _content_text(message.get("content"), f"messages[{index}].content")
        timestamp = message.get("timestamp")
        if timestamp is not None and (type(timestamp) is not int or not 0 <= timestamp <= 253402300799999):
            raise ValueError(f"messages[{index}].timestamp must be Unix milliseconds")
        text = f"{message['role']}: {content}"
        for part_index, part in enumerate(split_text(text)):
            fact: dict[str, Any] = {"id": fact_id(user_id, request_id, index, part_index), "text": part}
            if source_provenance:
                fact["metadata"] = {"aml_request_id": request_id, "aml_message_index": index,
                                    "aml_part_index": part_index}
            if timestamp is not None:
                fact["eventTime"] = timestamp
            facts.append(fact)
    return request_id, user_id, session_id, facts


def search_input(payload: Any) -> tuple[str, str, int, bool]:
    if not isinstance(payload, dict):
        raise ValueError("Search body must be an object")
    raw_query = payload.get("query")
    query = _content_text(raw_query, "query")
    if len(query) > MAX_QUERY_BYTES or len(query.encode()) > MAX_QUERY_BYTES:
        raise ValueError("query must be nonempty text of at most 1048576 UTF-8 bytes")
    user_id = _nonempty(payload.get("user_id"), "user_id")
    top_k = payload.get("top_k")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer from 1 to 100")
    options = payload.get("options")
    if options is not None and (not isinstance(options, list) or any(not isinstance(x, str) for x in options)):
        raise ValueError("options must be an array of strings")
    return query, user_id, top_k, isinstance(raw_query, list)


def _token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    if auth.startswith("Token "):
        return auth[6:]
    return request.headers.get("x-api-key")


def create_app(hms_url: str, token: str, commit: Any = None, *, source_provenance: bool = False) -> FastAPI:
    if type(source_provenance) is not bool:
        raise ValueError("source_provenance must be a boolean")
    if not token:
        raise ValueError("AML_TOKEN is required")
    client = httpx.AsyncClient(base_url=hms_url, timeout=httpx.Timeout(1800))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    search_active = asyncio.Semaphore(MAX_ACTIVE_SEARCHES)
    search_admitted: asyncio.Queue[None] = asyncio.Queue(MAX_ACTIVE_SEARCHES + MAX_QUEUED_SEARCHES)

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
        try:
            data = result.json()
        except (ValueError, RecursionError):
            raise HTTPException(status_code=503, detail='invalid memory engine JSON') from None
        if not isinstance(data, dict):
            raise HTTPException(status_code=503, detail='invalid memory engine response')
        return data

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

            request_id, user_id, session_id, facts = add_facts(json.loads(body), source_provenance=source_provenance)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        tenant = tenant_for(user_id)
        for start in range(0, len(facts), 100):
            await hms("remember", tenant, {"facts": facts[start : start + 100]})
        if commit is not None:
            await asyncio.to_thread(commit)
        return {"success": True, "request_id": request_id, "user_id": user_id, "session_id": session_id}

    async def perform_search(request: Request) -> dict[str, Any]:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_SEARCH_BODY:
                raise HTTPException(status_code=413, detail='Search body exceeds 8 MiB')
            body.extend(chunk)
        try:
            payload = json.loads(body)
        except (ValueError, RecursionError):
            raise HTTPException(status_code=422, detail='Search body must be valid JSON') from None
        try:
            query, user_id, top_k, multimodal = search_input(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if len(query.encode()) <= NATIVE_QUERY_BYTES:
            result = await hms("recall", tenant_for(user_id), {"query": query, "k": top_k})
            hits = result.get("results")
        else:
            async def recall(part: str) -> dict[str, Any]:
                return await hms('recall', tenant_for(user_id), {'query': part, 'k': top_k})
            hits = await fused_recall(query, top_k, recall)
        if not isinstance(hits, list) or len(hits) > top_k:
            raise HTTPException(status_code=503, detail="invalid memory engine result count")
        data = []
        for hit in hits:
            content = hit.get("text")
            memory_id = hit.get("id")
            if not isinstance(content, str) or not content or not isinstance(memory_id, str) or not memory_id:
                raise HTTPException(status_code=503, detail="invalid memory engine result")
            item: dict[str, Any] = {"id": memory_id, "content": [{"type": "text", "text": content}] if multimodal else content}
            score = hit.get("score")
            if isinstance(score, (float, int)) and math.isfinite(score):
                item["score"] = score
            event_ms = hit.get("eventTime")
            if type(event_ms) is int and 0 <= event_ms <= 253402300799999:
                item["created_at"] = datetime.fromtimestamp(event_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
            data.append(item)
        return {"data": data}

    @app.post('/search')
    async def search(request: Request) -> dict[str, Any]:
        await require_auth(request)
        try:
            search_admitted.put_nowait(None)
        except asyncio.QueueFull:
            raise HTTPException(status_code=503, detail='Search capacity unavailable') from None
        try:
            try:
                async with asyncio.timeout(SEARCH_TIMEOUT):
                    async with search_active:
                        return await perform_search(request)
            except TimeoutError:
                raise HTTPException(status_code=503, detail='Search request exceeded time budget') from None
        finally:
            search_admitted.get_nowait()
            search_admitted.task_done()

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(os.environ["HMS_URL"], os.environ["AML_TOKEN"]), host="127.0.0.1", port=8000)
