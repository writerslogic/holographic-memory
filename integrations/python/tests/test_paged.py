# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from holographic_integrations.core import HmsBackend, HmsConfig
from holographic_integrations.core.paged import PagedLog


@pytest.mark.parametrize("n", [9, 10, 11, 29, 30, 31])
def test_log_across_page_boundaries(server_factory, n):
    srv = server_factory(64, max_top_k=10)
    cfg = HmsConfig(url=srv.url, tenant_id=srv.tenant(), input_dim=64, max_fetch=10)
    with HmsBackend(cfg) as b:
        log = PagedLog(b, "k", page_size=10)
        items = [f"item {i:03d}" for i in range(n)]
        log.append(items[:4])
        log.append(items[4:])
        assert len(log) == n and log.read() == items
        assert log.read(limit=3) == items[-3:] and log.read(limit=n + 1) == items
        popped = log.pop()
        assert popped == items[-1]
        assert log.read() == items[:-1]
        log.append(["tail"])
        assert log.read(limit=1) == ["tail"]
        log.clear()
        assert log.read() == []
        popped = log.pop()
        assert popped is None
        assert len(log) == 0


def test_logs_with_different_keys_or_namespaces_do_not_mix(server):
    cfg = HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=server.dim)
    with HmsBackend(cfg) as b:
        a, c, d = PagedLog(b, "x"), PagedLog(b, "y"), PagedLog(b, "x", namespace="other")
        a.append(["a1"]), c.append(["c1"]), d.append(["d1"])
        assert a.read() == ["a1"] and c.read() == ["c1"] and d.read() == ["d1"]
        a.clear()
        assert a.read() == [] and c.read() == ["c1"] and d.read() == ["d1"]


def test_parts_limit_and_option_validation(server):
    cfg = HmsConfig(url=server.url, tenant_id=server.tenant(), input_dim=server.dim, max_fetch=20)
    with HmsBackend(cfg) as b:
        log = PagedLog(b, "p", chunk_chars=100, max_parts=3)
        for size in (299, 300):  # exactly max_parts chunks fits, one more character does not
            log.append(["z" * size])
        assert [len(t) for t in log.read()] == [299, 300]
        with pytest.raises(ValueError):
            log.append(["z" * 301])
        assert len(log) == 2
        with pytest.raises(ValueError):
            PagedLog(b, "q", page_size=21)
        with pytest.raises(ValueError):
            PagedLog(b, "q", chunk_chars=8001)
