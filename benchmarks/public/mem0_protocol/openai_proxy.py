# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["aiohttp>=3.9"]
# ///
"""Pass-through proxy for the OpenAI API that records call parameters and usage.

The memory-benchmarks LLM client builds ``openai.AsyncOpenAI()`` without a base URL,
so ``OPENAI_BASE_URL=http://127.0.0.1:<port>/v1`` routes it here unmodified. Each
call is forwarded byte for byte to https://api.openai.com and one JSON line is
appended to the log: requested model alias, the parameters that shape generation
(temperature, max_tokens/max_completion_tokens, reasoning_effort, response_format,
verbosity, seed), a hash and size of the prompt, the snapshot the API resolved,
usage (prompt, cached, completion, reasoning tokens), status and latency. The
Authorization header is forwarded and never written.

``--max-spend`` (USD, at the list prices in PRICES) is a hard stop: once the
logged spend reaches it, every further call is answered 402 without reaching
OpenAI, and the log says so. The harness then records empty answers, so a capped
run must be discarded, not scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

UPSTREAM = "https://api.openai.com"
# USD per 1M tokens: input, cached input, output (reasoning tokens bill as output).
PRICES = {"gpt-5": (1.25, 0.125, 10.0)}
SHAPING = ("model", "temperature", "max_tokens", "max_completion_tokens", "reasoning_effort",
           "response_format", "verbosity", "seed", "top_p", "n", "service_tier")
HOP = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}


def price_key(model: str) -> str | None:
    for key in sorted(PRICES, key=len, reverse=True):
        if model == key or model.startswith(key + "-20"):
            return key
    return None


def cost(model: str, usage: dict[str, Any]) -> float:
    key = price_key(model)
    if key is None:
        raise KeyError(f"no price for model {model!r}; add it to PRICES before spending")
    p_in, p_cached, p_out = PRICES[key]
    prompt = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    out = usage.get("completion_tokens", 0)
    return ((prompt - cached) * p_in + cached * p_cached + out * p_out) / 1e6


class Proxy:
    def __init__(self, log_path: str, max_spend: float):
        self.log_path = log_path
        self.max_spend = max_spend
        self.spend = 0.0
        self.session: ClientSession | None = None

    async def start(self, _app: web.Application) -> None:
        self.session = ClientSession(timeout=ClientTimeout(total=900), auto_decompress=True)

    async def stop(self, _app: web.Application) -> None:
        if self.session:
            await self.session.close()

    def log(self, record: dict[str, Any]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    async def forward(self, request: web.Request) -> web.Response:
        assert self.session is not None
        body = await request.read()
        record: dict[str, Any] = {"t": time.time(), "path": request.path}
        try:
            req = json.loads(body) if body else {}
        except json.JSONDecodeError:
            req = {}
        record["request"] = {k: req[k] for k in SHAPING if k in req}
        messages = req.get("messages") or []
        prompt_bytes = json.dumps(messages, sort_keys=True).encode()
        record["prompt_sha256"] = hashlib.sha256(prompt_bytes).hexdigest()
        record["prompt_chars"] = sum(len(str(m.get("content", ""))) for m in messages)
        if self.spend >= self.max_spend:
            record.update(status=402, blocked="spend cap reached", spend_usd=round(self.spend, 6))
            self.log(record)
            return web.json_response({"error": {"message": "local spend cap reached"}}, status=402)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        start = time.monotonic()
        async with self.session.request(
            request.method, UPSTREAM + request.path_qs, data=body, headers=headers
        ) as resp:
            payload = await resp.read()
            status = resp.status
            ctype = resp.headers.get("Content-Type", "application/json")
        record["latency_s"] = round(time.monotonic() - start, 3)
        record["status"] = status
        try:
            out = json.loads(payload)
        except json.JSONDecodeError:
            out = {}
        if status == 200 and isinstance(out, dict):
            usage = out.get("usage") or {}
            record["resolved_model"] = out.get("model")
            record["usage"] = usage
            record["finish_reason"] = [c.get("finish_reason") for c in out.get("choices", [])]
            record["completion_chars"] = sum(
                len((c.get("message") or {}).get("content") or "") for c in out.get("choices", [])
            )
            record["cost_usd"] = round(cost(req.get("model", ""), usage), 6)
            self.spend += record["cost_usd"]
        elif isinstance(out, dict):
            record["error"] = (out.get("error") or {}).get("code") or (out.get("error") or {}).get("type")
        record["spend_usd"] = round(self.spend, 6)
        self.log(record)
        return web.Response(body=payload, status=status, content_type=ctype.split(";")[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--log", required=True, help="JSONL call log")
    ap.add_argument("--max-spend", type=float, required=True, help="USD hard stop")
    args = ap.parse_args()
    proxy = Proxy(args.log, args.max_spend)
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    app.router.add_get("/_proxy/health", lambda _r: web.json_response({"spend_usd": proxy.spend}))
    app.router.add_route("*", "/{tail:.*}", proxy.forward)
    print(f"openai-proxy listening on 127.0.0.1:{args.port}", file=sys.stderr, flush=True)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
