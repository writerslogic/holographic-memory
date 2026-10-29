# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Capped proxy rejection checks without network requests."""

import asyncio
import json

import pytest

from budget_ledger import BudgetLedger
from openai_proxy import Proxy, SNAPSHOT, charge_units, reservation
from run_mem0_protocol import summarize_proxy


class NoNetworkSession:
    calls = 0

    def request(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("a rejected request reached the upstream API")


class Request:
    method = "POST"

    def __init__(self, path, body):
        self.path = self.path_qs = path
        self.body = json.dumps(body).encode()
        self.headers = {"Authorization": "Bearer synthetic-do-not-log"}

    async def read(self):
        return self.body


@pytest.mark.parametrize("path,body", [
    ("/v1/embeddings", {"model": "text-embedding-3-large", "input": "source"}),
    ("/v1/responses", {"model": "gpt-5", "input": "source"}),
    ("/v1/chat/completions", {"model": "unpriced-model", "messages": []}),
    ("/v1/chat/completions", {"model": None, "messages": []}),
    ("/v1/chat/completions", ["invalid object"]),
])
def test_unsupported_spending_paths_fail_before_network_and_never_log_auth(tmp_path, path, body):
    proxy = Proxy(str(tmp_path / "calls.jsonl"), 0.)
    session = NoNetworkSession()
    proxy.session = session
    response = asyncio.run(proxy.forward(Request(path, body)))
    assert response.status == 400 and session.calls == 0 and proxy.spend == 0
    log = (tmp_path / "calls.jsonl").read_text()
    row = json.loads(log)
    assert row["blocked"] and row["status"] == 400
    assert "synthetic-do-not-log" not in log and "Authorization" not in log


def test_zero_budget_still_blocks_supported_chat_before_network(tmp_path):
    proxy = Proxy(str(tmp_path / "calls.jsonl"), 0.)
    session = NoNetworkSession()
    proxy.session = session
    request = Request("/v1/chat/completions", {"model": "gpt-5", "messages": []})
    response = asyncio.run(proxy.forward(request))
    assert response.status == 402 and session.calls == 0 and proxy.spend == 0


@pytest.mark.parametrize("budget", [-1, float("inf"), -float("inf"), float("nan"), True])
def test_proxy_requires_a_finite_nonnegative_budget(tmp_path, budget):
    with pytest.raises(ValueError, match="finite nonnegative"):
        Proxy(str(tmp_path / "calls.jsonl"), budget)
    assert not (tmp_path / "calls.jsonl").exists()


def request_body(text="synthetic source"):
    return {"model": "gpt-5", "messages": [{"role": "user", "content": text}],
            "max_completion_tokens": 4096}


def guarded(tmp_path, budget=1.):
    path = tmp_path / "budget.sqlite3"
    BudgetLedger.initialize(path, budget, {"final": budget})
    ledger = BudgetLedger(path, "final", budget)
    ledger.start()
    proxy = Proxy(str(tmp_path / "calls.jsonl"), budget, ledger=ledger,
                  upstream_key="synthetic-upstream-secret", local_token="synthetic-do-not-log")
    return proxy, ledger


def completion(**changes):
    value = {"model": SNAPSHOT, "service_tier": "default",
             "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101},
             "choices": [{"finish_reason": "stop", "message": {"content": "synthetic"}}]}
    value.update(changes)
    return value


class Response:
    status = 200

    def __init__(self, payload, gate=None, error=None):
        self.payload = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        self.gate, self.error, self.content = gate, error, self

    async def __aenter__(self):
        if self.error:
            raise self.error
        return self

    async def __aexit__(self, *args):
        return False

    async def iter_chunked(self, _size):
        if self.gate:
            self.gate[0].set()
            await self.gate[1].wait()
        yield self.payload


class Session:
    def __init__(self, response, ledger):
        self.response, self.ledger, self.calls, self.forwarded = response, ledger, 0, None

    def request(self, _method, url, **kwargs):
        assert self.ledger.snapshot()["reserved_usd"] > 0
        assert url == "https://api.openai.com/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer synthetic-upstream-secret"
        assert kwargs["allow_redirects"] is False
        self.calls += 1
        self.forwarded = json.loads(kwargs["data"])
        return self.response


def test_positive_cost_proxy_requires_existing_ledger_and_separate_credentials(tmp_path):
    with pytest.raises(ValueError, match="private ledger"):
        Proxy(str(tmp_path / "calls.jsonl"), 1.)


def test_upstream_credential_cannot_be_used_as_the_local_client_token(tmp_path):
    proxy, ledger = guarded(tmp_path)
    proxy.session = NoNetworkSession()
    request = Request("/v1/chat/completions", request_body())
    request.headers["Authorization"] = "Bearer synthetic-upstream-secret"
    response = asyncio.run(proxy.forward(request))
    assert response.status == 401 and proxy.session.calls == 0
    assert "synthetic-upstream-secret" not in (tmp_path / "calls.jsonl").read_text()
    ledger.close()


@pytest.mark.parametrize("body", [b'{"model":"gpt-5","model":"gpt-5"}',
                                  b'{"model":"gpt-5","temperature":NaN}',
                                  b'{"model":"gpt-5","temperature":1e999}'])
def test_ambiguous_or_nonfinite_json_never_reaches_upstream(tmp_path, body):
    proxy, ledger = guarded(tmp_path)
    proxy.session = NoNetworkSession()
    request = Request("/v1/chat/completions", request_body())
    request.body = body
    response = asyncio.run(proxy.forward(request))
    assert response.status == 400 and proxy.session.calls == 0
    ledger.close()


@pytest.mark.parametrize("changes", [
    {"model": "gpt-5-2099-01-01"}, {"model": "gpt-5-pro"}, {"max_completion_tokens": None},
    {"max_completion_tokens": 0}, {"max_completion_tokens": True}, {"max_completion_tokens": 128001},
    {"n": 2}, {"n": True}, {"stream": True}, {"service_tier": "priority"},
    {"tools": []}, {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
])
def test_unreservable_requests_never_reach_upstream(tmp_path, changes):
    proxy, ledger = guarded(tmp_path)
    proxy.session = NoNetworkSession()
    response = asyncio.run(proxy.forward(Request("/v1/chat/completions", dict(request_body(), **changes))))
    assert response.status == 400 and proxy.session.calls == 0
    assert ledger.snapshot()["reserved_usd"] == 0
    ledger.close()


def test_valid_call_pins_snapshot_and_tier_and_settles_actual_cost(tmp_path):
    proxy, ledger = guarded(tmp_path)
    proxy.session = Session(Response(completion()), ledger)
    response = asyncio.run(proxy.forward(Request("/v1/chat/completions", request_body())))
    assert response.status == 200
    assert proxy.session.forwarded["model"] == SNAPSHOT
    assert proxy.session.forwarded["service_tier"] == "default"
    assert ledger.snapshot()["settled_calls"] == 1 and ledger.snapshot()["reserved_usd"] == 0
    assert proxy.spend == .000135
    audit = (tmp_path / "calls.jsonl").read_text()
    assert "synthetic-upstream-secret" not in audit and "synthetic-do-not-log" not in audit
    duplicate = asyncio.run(proxy.forward(Request("/v1/chat/completions", request_body())))
    assert duplicate.status == 200 and proxy.session.calls == 2
    assert ledger.snapshot()["settled_calls"] == 2 and proxy.spend == .000270
    ledger.close()


def test_overlapping_calls_are_reserved_before_either_response_arrives(tmp_path):
    proxy, ledger = guarded(tmp_path)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        proxy.session = Session(Response(completion(), (entered, release)), ledger)
        first = asyncio.create_task(proxy.forward(Request("/v1/chat/completions", request_body("first"))))
        await entered.wait()
        second = await proxy.forward(Request("/v1/chat/completions", request_body("second")))
        assert second.status == 402 and proxy.session.calls == 1
        release.set()
        assert (await first).status == 200

    asyncio.run(run())
    ledger.close()


@pytest.mark.parametrize("payload", [b"not JSON", completion(usage=None), completion(model="unknown-snapshot"),
                                     completion(service_tier="priority"), completion(usage={"prompt_tokens": -1}),
                                     completion(usage={"prompt_tokens": 100, "completion_tokens": 4097})])
def test_uncertain_responses_keep_reservation_and_block_restarts(tmp_path, payload):
    proxy, ledger = guarded(tmp_path)
    proxy.session = Session(Response(payload), ledger)
    response = asyncio.run(proxy.forward(Request("/v1/chat/completions", request_body())))
    assert response.status == 503 and proxy.session.calls == 1
    assert ledger.snapshot()["halted"] and ledger.snapshot()["unresolved_calls"] == 1
    assert ledger.snapshot()["reserved_usd"] == .54096
    blocked = asyncio.run(proxy.forward(Request("/v1/chat/completions", request_body("new input"))))
    assert blocked.status == 402 and proxy.session.calls == 1
    ledger.close()


def test_timeout_and_cancellation_never_release_the_reservation(tmp_path):
    proxy, ledger = guarded(tmp_path)
    proxy.session = Session(Response(completion(), error=TimeoutError()), ledger)
    response = asyncio.run(proxy.forward(Request("/v1/chat/completions", request_body())))
    assert response.status == 503 and ledger.snapshot()["unresolved_calls"] == 1
    ledger.close()


def test_cancellation_retains_its_durable_reservation(tmp_path):
    proxy, ledger = guarded(tmp_path)

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        proxy.session = Session(Response(completion(), (entered, release)), ledger)
        task = asyncio.create_task(proxy.forward(Request("/v1/chat/completions", request_body())))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)

    asyncio.run(run())
    assert ledger.snapshot()["halted"] and ledger.snapshot()["reserved_usd"] == .54096
    ledger.close()


def test_reservation_write_failure_never_reaches_network(tmp_path, monkeypatch):
    proxy, ledger = guarded(tmp_path)
    proxy.session = NoNetworkSession()

    def failed(*_args):
        raise OSError("synthetic private storage failure")

    monkeypatch.setattr(ledger, "reserve", failed)
    response = asyncio.run(proxy.forward(Request("/v1/chat/completions", request_body())))
    assert response.status == 503 and proxy.session.calls == 0
    assert ledger.snapshot()["halted"]
    ledger.close()


def test_audit_failure_preserves_settled_charge_and_halts_forwarding(tmp_path, monkeypatch):
    proxy, ledger = guarded(tmp_path)
    proxy.session = Session(Response(completion()), ledger)

    def failed(_record):
        raise OSError("synthetic audit storage failure")

    monkeypatch.setattr(proxy, "log", failed)
    response = asyncio.run(proxy.forward(Request("/v1/chat/completions", request_body())))
    assert response.status == 503 and proxy.session.calls == 1
    assert ledger.snapshot()["halted"] and ledger.snapshot()["settled_calls"] == 1
    assert proxy.spend == .000135
    ledger.close()


def test_written_then_failed_audit_does_not_count_one_physical_charge_twice(tmp_path, monkeypatch):
    proxy, ledger = guarded(tmp_path)
    proxy.session = Session(Response(completion()), ledger)
    original_log = proxy.log
    writes = 0

    def write_then_fail_once(record):
        nonlocal writes
        original_log(record)
        writes += 1
        if writes == 1:
            raise OSError("synthetic failure after the audit record was written")

    monkeypatch.setattr(proxy, "log", write_then_fail_once)
    response = asyncio.run(proxy.forward(Request("/v1/chat/completions", request_body())))
    assert response.status == 503 and proxy.session.calls == 1 and writes == 2
    rows = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert rows[0]["budget_call_id"] == rows[1]["budget_call_id"]
    summary = summarize_proxy(tmp_path / "calls.jsonl")
    assert summary["spend_usd"] == ledger.snapshot()["spend_usd"] == .000135
    assert ledger.snapshot()["halted"]
    ledger.close()


@pytest.mark.parametrize("usage", [
    {}, {"prompt_tokens": True, "completion_tokens": 1},
    {"prompt_tokens": 100, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 101}},
    {"prompt_tokens": 100, "completion_tokens": 1, "completion_tokens_details": {"reasoning_tokens": 2}},
    {"prompt_tokens": 400000, "completion_tokens": 1},
    {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 102},
])
def test_usage_cannot_reduce_or_overrun_a_reservation(usage):
    with pytest.raises(ValueError):
        charge_units(SNAPSHOT, usage, 4096)


@pytest.mark.parametrize("limit", [1, 127999, 128000])
def test_explicit_output_limit_preserves_the_full_context_bound(limit):
    assert reservation(dict(request_body(), max_completion_tokens=limit)) == 500_000_000 + 10000 * limit
