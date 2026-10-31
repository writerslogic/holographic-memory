# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["aiohttp>=3.9", "tiktoken>=0.7"]
# ///
"""Mem0 OSS REST surface over warm-server's memory API.

mem0ai/memory-benchmarks talks to a self-hosted Mem0 server through three calls
(benchmarks/common/mem0_client.py, ``--backend oss``):

    POST   /memories            {"messages": [{"role", "content"}], "user_id", "timestamp"?}
    POST   /search              {"query", "user_id", "limit"}
    DELETE /memories?user_id=   (not called by any run.py; answered 501)

This process serves those routes and forwards them to warm-server:

- add: the messages of one call become one fact (split only when it exceeds the
  engine's single-fact limit). Text is one ``"{role}: {content}"`` line per message,
  exactly the strings the harness sent. ``timestamp`` (unix seconds, the session
  date) becomes the fact's ``eventTime``. Optional configured embeddings use the
  same model and dimensions for stored facts and search queries.
- search: ``memory/recall`` with ``k = limit`` in the tenant mapped from ``user_id``;
  results come back best first as ``{"id", "memory", "score", "created_at"}``,
  ``created_at`` being the fact's event time as a naive UTC ISO-8601 string that the
  harness's ``_to_human_date`` parses.

The retrieval configuration is read from a JSON file fixed before the run; the
request never carries anything but the question text and the tenant.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, web

TENANT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Engine single-fact limits (docs/memory-api: 64 KiB and 4096 words); keep headroom.
MAX_FACT_BYTES = 60 * 1024
MAX_FACT_WORDS = 4000
ALLOWED_CONFIG_KEYS = {"recall", "max_k", "max_context_tokens", "embedding", "order_same_time"}
ALLOWED_RECALL_KEYS = {"lexicalWeight", "semanticWeight", "fusion", "candidateLimit", "history"}
EMBEDDING_MODEL = "text-embedding-3-large"
RUST_WHITESPACE = "\t\n\v\f\r \u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"


def tenant_for(user_id: str) -> str:
    """Map a harness user_id to an warm-server tenant id ([A-Za-z0-9_-]{1,64})."""
    if TENANT_RE.match(user_id):
        return user_id
    return "u-" + hashlib.sha256(user_id.encode()).hexdigest()[:60]


def fact_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def split_fact(text: str) -> list[str]:
    """Split text that exceeds the single-fact limit on word boundaries; else return it whole."""
    if len(text.encode()) <= MAX_FACT_BYTES and len(text.split()) <= MAX_FACT_WORDS:
        return [text]
    words: list[str] = []
    for word in text.split():
        while len(word.encode()) > MAX_FACT_BYTES:
            cut = MAX_FACT_BYTES // 4  # characters; at most 4 UTF-8 bytes each
            words.append(word[:cut])
            word = word[cut:]
        words.append(word)
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        wbytes = len(word.encode()) + 1
        if current and (size + wbytes > MAX_FACT_BYTES or len(current) >= MAX_FACT_WORDS):
            parts.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += wbytes
    if current:
        parts.append(" ".join(current))
    return parts


def iso_from_ms(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def fit_budget(hits: list, costs: list[int], budget: int) -> list:
    """Hits in rank order whose token costs fit the budget; a hit that would overflow it is skipped."""
    kept, used = [], 0
    for hit, cost in zip(hits, costs):
        if used + cost <= budget:
            kept.append(hit)
            used += cost
    return kept


def order_same_time(hits: list[dict]) -> list[dict]:
    positions: dict[int, list[int]] = {}
    for index, hit in enumerate(hits):
        event, ingested = hit.get("eventTime"), hit.get("ingestTime")
        if event is not None and ingested is not None:
            positions.setdefault(event, []).append(index)
    ordered = list(hits)
    for slots in positions.values():
        members = sorted((hits[index] for index in slots), key=lambda row: row["ingestTime"])
        for index, member in zip(slots, members):
            ordered[index] = member
    # The Mem0 client sorts by score again before its prompt sorts by event time.
    return [{**hit, "score": 1.0 / (index + 1)} for index, hit in enumerate(ordered)]


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    unknown = set(cfg) - ALLOWED_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    unknown = set(cfg.get("recall", {})) - ALLOWED_RECALL_KEYS
    if unknown:
        raise ValueError(f"unknown recall keys: {sorted(unknown)}")
    if type(cfg.get("order_same_time", False)) is not bool:
        raise ValueError("order_same_time must be boolean")
    embedding = cfg.get("embedding")
    if embedding is not None:
        if not isinstance(embedding, dict) or set(embedding) != {"model", "dimensions"}:
            raise ValueError("embedding requires model and dimensions")
        if embedding["model"] != EMBEDDING_MODEL:
            raise ValueError("unsupported embedding model")
        if type(embedding["dimensions"]) is not int or not 1 <= embedding["dimensions"] <= 3072:
            raise ValueError("embedding dimensions must be 1..=3072")
    return cfg


class EmbeddingClient:
    def __init__(self, session: ClientSession, dimensions: int, log_path: str | None):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required for hybrid recall")
        self.session = session
        self.key = key
        self.dimensions = dimensions
        self.log = open(log_path, "a", encoding="utf-8") if log_path else None

    def close(self) -> None:
        if self.log:
            self.log.close()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("embedding input must contain nonempty text")
        request = {"model": EMBEDDING_MODEL, "dimensions": self.dimensions,
                   "encoding_format": "float", "input": texts}
        for attempt in range(5):
            try:
                async with self.session.post(
                    "https://api.openai.com/v1/embeddings",
                    json=request,
                    headers={"Authorization": f"Bearer {self.key}"},
                ) as response:
                    if response.status in (429, 500, 502, 503, 504):
                        status = response.status
                    elif response.status != 200:
                        raise RuntimeError(f"embedding API returned HTTP {response.status}")
                    else:
                        payload = await response.json()
                        return self._parse(payload, len(texts))
            except (ClientError, asyncio.TimeoutError) as error:
                if attempt == 4:
                    raise RuntimeError("embedding API transport failed after five attempts") from error
            else:
                if attempt == 4:
                    raise RuntimeError(f"embedding API returned HTTP {status} after five attempts")
            await asyncio.sleep(min(2 ** attempt, 8))
        raise RuntimeError("embedding API retry invariant failed")

    def _parse(self, payload: dict[str, Any], count: int) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != count:
            raise ValueError("embedding response count mismatch")
        result: list[list[float] | None] = [None] * count
        for item in data:
            index = item.get("index") if isinstance(item, dict) else None
            vector = item.get("embedding") if isinstance(item, dict) else None
            if type(index) is not int or not 0 <= index < count or result[index] is not None:
                raise ValueError("embedding response index mismatch")
            if not isinstance(vector, list) or len(vector) != self.dimensions or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in vector
            ):
                raise ValueError("embedding response vector invalid")
            result[index] = [float(value) for value in vector]
        usage = payload.get("usage") or {}
        tokens = usage.get("total_tokens")
        if type(tokens) is not int or tokens < 0:
            raise ValueError("embedding response usage invalid")
        if self.log:
            self.log.write(json.dumps({"model": payload.get("model"), "inputs": count,
                                       "dimensions": self.dimensions, "tokens": tokens}) + "\n")
            self.log.flush()
        return [vector for vector in result if vector is not None]


class Shim:
    def __init__(self, warm_url: str, config: dict[str, Any], embedding_log: str | None = None):
        self.warm_url = warm_url.rstrip("/")
        self.recall_options = dict(config.get("recall", {}))
        self.max_k = int(config.get("max_k", 200))
        # Optional fixed token budget on the returned memory texts (o200k_base), filled in rank
        # order (memories that would overflow it are skipped): the same cap for every query.
        self.max_context_tokens = config.get("max_context_tokens")
        self.order_same_time = config.get("order_same_time", False)
        self.enc = None
        if self.max_context_tokens is not None:
            import tiktoken

            self.enc = tiktoken.get_encoding("o200k_base")
        self.session: ClientSession | None = None
        self.embedding_config = config.get("embedding")
        self.embedding_log = embedding_log
        self.embedder: EmbeddingClient | None = None
        self.counts: dict[str, int] = {}

    async def start(self, _app: web.Application) -> None:
        self.session = ClientSession(timeout=ClientTimeout(total=600))
        if self.embedding_config:
            self.embedder = EmbeddingClient(self.session, self.embedding_config["dimensions"], self.embedding_log)

    async def stop(self, _app: web.Application) -> None:
        if self.embedder:
            self.embedder.close()
        if self.session:
            await self.session.close()

    async def warm(self, op: str, tenant: str, body: dict[str, Any]) -> dict[str, Any]:
        assert self.session is not None
        delay = 0.05
        for _ in range(8):
            async with self.session.post(
                f"{self.warm_url}/api/v1/memory/{op}", json=body, headers={"X-Tenant-ID": tenant}
            ) as resp:
                if resp.status in (429, 503):
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 2.0)
                    continue
                payload = await resp.json(content_type=None)
                if resp.status == 409:
                    raise web.HTTPConflict(text="warm-server fact conflict")
                if resp.status != 200:
                    raise web.HTTPBadGateway(text=json.dumps({"warm_status": resp.status, "error": payload}))
                return payload
        raise web.HTTPServiceUnavailable(text="warm-server busy")

    async def remember_embedded(self, tenant: str, facts: list[dict[str, Any]]) -> dict[str, Any]:
        assert self.embedder is not None
        remembered = []
        for fact in facts:
            identity = json.dumps([fact["text"], fact.get("eventTime"), self.embedding_config],
                                  ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            fact["id"] = "mem0:" + hashlib.sha256(identity.encode()).hexdigest()
            existing = (await self.warm("get", tenant, {"id": fact["id"]}))["fact"]
            if existing is None:
                fact["embedding"] = (await self.embedder.embed([fact["text"]]))[0]
                try:
                    result = await self.warm("remember", tenant, {"facts": [fact]})
                except web.HTTPConflict:
                    existing = (await self.warm("get", tenant, {"id": fact["id"]}))["fact"]
                    if existing is None:
                        raise
                else:
                    if len(result.get("remembered", [])) != 1:
                        raise RuntimeError("warm-server returned an invalid remember result")
                    remembered.extend(result["remembered"])
                    continue
            # Native single-chunk text excludes surrounding Unicode White_Space.
            if (existing.get("text") != fact["text"].strip(RUST_WHITESPACE)
                    or existing.get("eventTime") != fact.get("eventTime")):
                raise RuntimeError("stored fact does not match its stable identity")
            remembered.append({"id": fact["id"], "created": False})
        return {"remembered": remembered}

    async def add(self, request: web.Request) -> web.Response:
        req = await request.json()
        messages = req.get("messages") or []
        user_id = req.get("user_id")
        if not user_id or not isinstance(messages, list) or not messages:
            raise web.HTTPBadRequest(text="messages and user_id are required")
        tenant = tenant_for(str(user_id))
        ts = req.get("timestamp")
        event_ms = int(ts) * 1000 if isinstance(ts, (int, float)) else None
        facts = []
        for part in split_fact(fact_text(messages)):
            fact: dict[str, Any] = {"text": part}
            if event_ms is not None:
                fact["eventTime"] = event_ms
            facts.append(fact)
        if self.embedder:
            out = await self.remember_embedded(tenant, facts)
            added = sum(bool(row.get("created")) for row in out["remembered"])
        else:
            out = await self.warm("remember", tenant, {"facts": facts})
            added = len(facts)
        self.counts[tenant] = self.counts.get(tenant, 0) + added
        created = iso_from_ms(event_ms)
        results = [
            {"id": r.get("id", ""), "memory": f["text"], "event": "ADD", **({"created_at": created} if created else {})}
            for r, f in zip(out.get("remembered", []), facts)
        ]
        return web.json_response({"results": results})

    async def search(self, request: web.Request) -> web.Response:
        req = await request.json()
        query, user_id = req.get("query"), req.get("user_id")
        if not isinstance(query, str) or not user_id:
            raise web.HTTPBadRequest(text="query and user_id are required")
        k = int(req.get("limit", 100))
        if not 1 <= k <= self.max_k:
            raise web.HTTPBadRequest(text=f"limit must be 1..={self.max_k}")
        tenant = tenant_for(str(user_id))
        recall = {"query": query, "k": k, **self.recall_options}
        if self.embedder:
            recall["embedding"] = (await self.embedder.embed([query]))[0]
        out = await self.warm("recall", tenant, recall)
        hits = out.get("results", [])
        if self.enc is not None:
            hits = fit_budget(hits, [len(self.enc.encode(r.get("text") or "")) for r in hits], self.max_context_tokens)
        if self.order_same_time:
            hits = order_same_time(hits)
        results = []
        for r in hits:
            entry: dict[str, Any] = {"id": r["id"], "memory": r.get("text") or "", "score": r.get("score", 0.0)}
            created = iso_from_ms(r.get("eventTime"))
            if created:
                entry["created_at"] = created
            results.append(entry)
        return web.json_response({"results": results})

    async def delete(self, _request: web.Request) -> web.Response:
        raise web.HTTPNotImplemented(text="delete is not used by the benchmark runners")

    async def stats(self, _request: web.Request) -> web.Response:
        return web.json_response({"facts_by_tenant": self.counts})


def build_app(shim: Shim) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(shim.start)
    app.on_cleanup.append(shim.stop)
    app.router.add_post("/memories", shim.add)
    app.router.add_post("/search", shim.search)
    app.router.add_delete("/memories", shim.delete)
    app.router.add_get("/_shim/stats", shim.stats)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--warm-url", required=True, help="warm-server base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--config", required=True, help="frozen retrieval config JSON")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--embedding-log", default=None)
    args = ap.parse_args()
    shim = Shim(args.warm_url, load_config(args.config), args.embedding_log)
    print(f"mem0-shim listening on 127.0.0.1:{args.port}", file=sys.stderr, flush=True)
    web.run_app(build_app(shim), host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
