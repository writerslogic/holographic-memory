# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["aiohttp>=3.9"]
# ///
"""Local GPT-5 proxy with durable pre-network reservations and no paid retries.

Positive-cost calls require a private ledger and final-run allocation. Uncertain
outcomes retain reservations and halt forwarding. Only this process owns the
upstream secret; clients authenticate with a separate local token.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import secrets
import sys
import time
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web

if __package__:
    from .budget_ledger import BudgetError, BudgetLedger, UNITS_PER_USD, usd_units
else:
    from budget_ledger import BudgetError, BudgetLedger, UNITS_PER_USD, usd_units

UPSTREAM = "https://api.openai.com"
SNAPSHOT = "gpt-5-2025-08-07"
MODELS = frozenset({"gpt-5", SNAPSHOT})
PRICES = {"gpt-5": (1.25, 0.125, 10.0)}
# https://developers.openai.com/api/docs/models/gpt-5: standard rates and model bounds.
CONTEXT_TOKENS, MAX_OUTPUT_TOKENS = 400_000, 128_000
INPUT_UNITS, CACHED_UNITS, OUTPUT_UNITS = 1250, 125, 10000
MAX_BODY_BYTES, MAX_RESPONSE_BYTES = 8 * 1024 * 1024, 16 * 1024 * 1024
SHAPING = ("model", "temperature", "max_tokens", "max_completion_tokens", "reasoning_effort",
           "response_format", "verbosity", "seed", "top_p", "n", "service_tier")
ALLOWED = frozenset(SHAPING) | {"messages", "stream"}


def price_key(model: str) -> str | None:
    return "gpt-5" if isinstance(model, str) and model in MODELS else None


def charge_units(model: str, usage: dict[str, Any], max_output=MAX_OUTPUT_TOKENS) -> int:
    if model not in MODELS or not isinstance(usage, dict):
        raise ValueError("unsupported model or missing usage")
    prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
    details, completion_details = usage.get("prompt_tokens_details"), usage.get("completion_tokens_details")
    if ((details is not None and not isinstance(details, dict))
            or (completion_details is not None and not isinstance(completion_details, dict))):
        raise ValueError("invalid token usage details")
    cached = (details or {}).get("cached_tokens", 0)
    reasoning = (completion_details or {}).get("reasoning_tokens", 0)
    if (any(type(n) is not int for n in (prompt, output, cached, reasoning))
            or not 0 < prompt <= CONTEXT_TOKENS or not 0 <= output <= max_output
            or not 0 <= cached <= prompt or not 0 <= reasoning <= output
            or prompt + output > CONTEXT_TOKENS):
        raise ValueError("usage exceeds the reserved model contract")
    if "total_tokens" in usage and (type(usage["total_tokens"]) is not int
                                    or usage["total_tokens"] != prompt + output):
        raise ValueError("invalid total token usage")
    return (prompt - cached) * INPUT_UNITS + cached * CACHED_UNITS + output * OUTPUT_UNITS


def cost(model: str, usage: dict[str, Any]) -> float:
    return charge_units(model, usage) / UNITS_PER_USD


def reservation(req: dict) -> int:
    if set(req) - ALLOWED or "max_tokens" in req:
        raise ValueError("unsupported request fields for finite-budget proxy")
    limit = req.get("max_completion_tokens")
    if type(limit) is not int or not 1 <= limit <= MAX_OUTPUT_TOKENS:
        raise ValueError("an explicit bounded max_completion_tokens is required")
    if (type(req.get("n", 1)) is not int or req.get("n", 1) != 1
            or req.get("stream", False) is not False
            or req.get("service_tier") not in (None, "default")):
        raise ValueError("only one nonstreaming standard-tier completion is supported")
    messages = req.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 4096:
        raise ValueError("bounded text messages are required")
    for message in messages:
        if (not isinstance(message, dict) or set(message) - {"role", "content", "name"}
                or message.get("role") not in {"system", "developer", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                or ("name" in message and (not isinstance(message["name"], str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", message["name"])))):
            raise ValueError("only text-only conversation messages are supported")
    if "response_format" in req and req["response_format"] not in ({"type": "json_object"}, {"type": "text"}):
        raise ValueError("unsupported response format")
    # Uncached full-context input is an upper bound without a local token estimate.
    return CONTEXT_TOKENS * INPUT_UNITS + limit * OUTPUT_UNITS


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _json(payload):
    def invalid_constant(_value):
        raise ValueError("nonfinite JSON number")

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON number")
        return number

    return json.loads(payload, object_pairs_hook=_unique_object, parse_constant=invalid_constant,
                      parse_float=finite_float)


class Proxy:
    def __init__(self, log_path: str, max_spend: float, *, ledger: BudgetLedger | None = None,
                 upstream_key: str | None = None, local_token: str | None = None):
        if type(max_spend) not in (int, float) or not math.isfinite(max_spend) or max_spend < 0:
            raise ValueError("max-spend must be a finite nonnegative USD amount")
        usd_units(max_spend)
        if max_spend > 0 and (ledger is None or not upstream_key or not local_token):
            raise ValueError("positive-cost forwarding requires a private ledger, upstream key and local token")
        if ledger is not None and usd_units(max_spend) != ledger.limit_units:
            raise ValueError("proxy allocation differs from its immutable ledger ceiling")
        self.log_path, self.max_spend, self.ledger = log_path, max_spend, ledger
        self.upstream_key, self.local_token = upstream_key, local_token
        self.session: ClientSession | None = None

    @property
    def spend(self):
        return self.ledger.snapshot()["spend_usd"] if self.ledger else 0.0

    async def start(self, _app: web.Application) -> None:
        if self.ledger:
            self.ledger.start()
        try:
            self.session = ClientSession(timeout=ClientTimeout(total=900), auto_decompress=True)
        except BaseException:
            if self.ledger:
                self.ledger.close()
            raise

    async def stop(self, _app: web.Application) -> None:
        try:
            if self.session:
                await self.session.close()
        finally:
            if self.ledger:
                self.ledger.close()

    def log(self, record: dict[str, Any]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def blocked(self, record: dict[str, Any], reason: str, status: int = 400) -> web.Response:
        record.update(status=status, blocked=reason, spend_usd=self.spend)
        try:
            self.log(record)
        except (OSError, ValueError):
            if self.ledger:
                self.ledger.halt("audit_write_failed")
            status, reason = 503, "local audit unavailable; forwarding halted"
        return web.json_response({"error": {"message": reason}}, status=status)

    async def health(self, _request):
        usage = self.ledger.snapshot() if self.ledger else {"spend_usd": 0, "unresolved_calls": 0, "halted": False}
        return web.json_response({"guard": "durable-openai-v1", **usage})

    async def forward(self, request: web.Request) -> web.Response:
        assert self.session is not None
        record: dict[str, Any] = {"t": time.time(), "path": request.path}
        if (request.method != "POST" or request.path != "/v1/chat/completions"
                or request.path_qs != request.path):
            return self.blocked(record, "unsupported endpoint for finite-budget proxy")
        if self.max_spend > 0 and not secrets.compare_digest(
                request.headers.get("Authorization", "").encode(), ("Bearer " + self.local_token).encode()):
            return self.blocked(record, "invalid local proxy authorization", 401)
        body = await request.read()
        if len(body) > MAX_BODY_BYTES:
            return self.blocked(record, "request body exceeds its limit", 413)
        try:
            req = _json(body)
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")
            if not isinstance(req.get("model"), str) or price_key(req["model"]) is None:
                raise ValueError("unsupported model for finite-budget proxy")
            record["request"] = {key: req[key] for key in SHAPING if key in req}
            if self.max_spend == 0:
                return self.blocked(record, "local zero-cost allocation", 402)
            amount = reservation(req)
        except (ValueError, TypeError, RecursionError, UnicodeDecodeError) as error:
            return self.blocked(record, str(error) if isinstance(error, ValueError) else "invalid bounded JSON request")
        record["prompt_sha256"] = hashlib.sha256(json.dumps(req["messages"], sort_keys=True).encode()).hexdigest()
        record["prompt_chars"] = sum(len(message["content"]) for message in req["messages"])
        forwarded = dict(req, model=SNAPSHOT, service_tier="default")
        upstream_body = json.dumps(forwarded, sort_keys=True, allow_nan=False).encode()
        request_hash = hashlib.sha256(upstream_body).hexdigest()
        record.update(request_sha256=request_hash, upstream_model=SNAPSHOT, upstream_service_tier="default")
        try:
            identifier = self.ledger.reserve(request_hash, amount)
        except BudgetError as error:
            return self.blocked(record, str(error), 402)
        except Exception:
            self.ledger.halt("reservation_uncertain")
            return self.blocked(record, "private budget ledger unavailable", 503)
        record["budget_call_id"] = identifier
        start = time.monotonic()
        try:
            async with self.session.request(
                    "POST", UPSTREAM + request.path, data=upstream_body,
                    headers={"Authorization": "Bearer " + self.upstream_key, "Content-Type": "application/json"},
                    allow_redirects=False) as response:
                payload = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    payload.extend(chunk)
                    if len(payload) > MAX_RESPONSE_BYTES:
                        raise ValueError("upstream response exceeds its bound")
                status = response.status
            record.update(latency_s=round(time.monotonic() - start, 3), status=status, upstream_status=status)
            out = _json(payload)
            if status != 200:
                if isinstance(out, dict) and isinstance(out.get("error"), dict):
                    code = out["error"].get("code") or out["error"].get("type")
                    if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", code):
                        record["error"] = code
                raise ValueError("upstream failure requires charge reconciliation")
            if not isinstance(out, dict) or out.get("model") != SNAPSHOT or out.get("service_tier") != "default":
                raise ValueError("upstream model or billing tier differs from the reservation")
            units = charge_units(SNAPSHOT, out.get("usage"), req["max_completion_tokens"])
            choices = out.get("choices")
            if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
                    or not isinstance(choices[0].get("message"), dict)):
                raise ValueError("invalid completion response")
            message = choices[0]["message"]
            if message.get("content") is not None and not isinstance(message["content"], str):
                raise ValueError("invalid text completion")
            record.update(resolved_model=out["model"], usage=out["usage"],
                          finish_reason=[choices[0].get("finish_reason")],
                          completion_chars=len(message.get("content") or ""),
                          cost_units=units, cost_usd=units / UNITS_PER_USD)
            self.ledger.settle(identifier, units)
            record["spend_usd"] = self.spend
            self.log(record)
        except asyncio.CancelledError:
            self.ledger.halt("cancelled", identifier)
            record["budget_uncertain"] = True
            self.blocked(record, "request cancelled; charge reconciliation required", 503)
            raise
        except Exception:
            self.ledger.halt("upstream_or_audit_uncertain", identifier)
            record["budget_uncertain"] = True
            return self.blocked(record, "forwarding halted; charge reconciliation required", 503)
        return web.Response(body=bytes(payload), status=status, content_type="application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--log", required=True, help="append-only call audit; no provider secret or balance")
    parser.add_argument("--max-spend", type=float, required=True, help="must equal the persisted final-run allocation")
    parser.add_argument("--budget-ledger", help="existing private SQLite ledger outside the repository")
    parser.add_argument("--budget-allocation", help="predeclared final-run allocation identifier")
    args = parser.parse_args()
    try:
        ledger = None
        if args.budget_ledger or args.budget_allocation:
            if not args.budget_ledger or not args.budget_allocation:
                raise ValueError("both ledger and allocation are required")
            ledger = BudgetLedger(args.budget_ledger, args.budget_allocation, args.max_spend)
        proxy = Proxy(args.log, args.max_spend, ledger=ledger,
                      upstream_key=os.environ.get("OPENAI_API_KEY"), local_token=os.environ.get("WARM_PROXY_TOKEN"))
    except Exception:
        parser.error("invalid private budget or proxy configuration")
    app = web.Application(client_max_size=MAX_BODY_BYTES)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    app.router.add_get("/_proxy/health", proxy.health)
    app.router.add_route("*", "/{tail:.*}", proxy.forward)
    print(f"guarded-openai-proxy listening on 127.0.0.1:{args.port}", file=sys.stderr, flush=True)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
