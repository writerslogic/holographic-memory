# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Capped proxy rejection checks without network requests."""

import asyncio
import json

import pytest

from openai_proxy import Proxy


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
    proxy = Proxy(str(tmp_path / "calls.jsonl"), 1.)
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
