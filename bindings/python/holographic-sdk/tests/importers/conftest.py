# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixtures: a real hms-server process and helpers for per-source round-trip tests.

Set HMS_SERVER_BIN to a binary built with `cargo build --release --features server --bin hms-server`.
"""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from holographic_sdk.importers import HmsSink, ImportOptions, ImporterError, run_import

DIM = 32
API_KEY = "import-test-key"
_DEFAULT_BIN = "/Volumes/A/.hms-target-import/release/hms-server"


def server_bin() -> Optional[str]:
    return os.environ.get("HMS_SERVER_BIN") or (_DEFAULT_BIN if os.path.exists(_DEFAULT_BIN) else None)


def vec(seed: int, dim: int = DIM) -> List[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(dim)]


class Server:
    def __init__(self, url: str):
        self.url = url
        self._n = 0

    def tenant(self, hint: str = "t") -> str:
        self._n += 1
        return f"{hint}{self._n}x{os.getpid()}"

    def sink(self, tenant: str) -> HmsSink:
        return HmsSink(self.url, api_key=API_KEY, tenant=tenant, backoff=0.05)


def start_server(tmp: Path, dim: int = DIM, extra_env: Optional[Dict[str, str]] = None):
    binary = server_bin()
    if not binary:
        pytest.skip("HMS_SERVER_BIN not set and no default hms-server build found")
    proc = subprocess.Popen(
        [binary, "--bind", "127.0.0.1:0", "--input-dim", str(dim), "--dim", "4096", "--data-dir", str(tmp), "--max-tenants", "100000"],
        stdout=subprocess.PIPE, text=True, env={**os.environ, "HMS_API_KEY": API_KEY, **(extra_env or {})},
    )
    line = proc.stdout.readline().strip()
    assert line.startswith("hms-server listening on "), line
    return proc, f"http://{line.rsplit(' ', 1)[1]}"


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    proc, url = start_server(tmp_path_factory.mktemp("hms-import-data"))
    try:
        yield Server(url)
    finally:
        proc.terminate()
        proc.wait(timeout=30)


@pytest.fixture
def run(server):
    """run(source, **opts) -> (report, tenant): imports into a fresh tenant and verifies every record."""

    def _run(source, sample: int = 1000, tenant: Optional[str] = None, **opts: Any):
        tenant = tenant or server.tenant()
        options = ImportOptions(verify_sample=sample, verify_k=5, **opts)
        sink = server.sink(tenant)
        try:
            with source:
                report = run_import(source, sink, options, tenant=tenant)
        finally:
            sink.close()
        return report, tenant

    return _run


def assert_faithful(report, expected_docs: int, *, vectors: bool = True) -> None:
    """The invariants every source's round trip must satisfy."""
    d = report.to_dict()
    assert d["complete"] and d["error"] is None
    assert d["rejected"]["total"] == 0, d["rejected"]
    assert d["counts"]["docs_acked"] == expected_docs, d["counts"]
    v = d["verification"]
    assert v["performed"], v
    assert v["roundtrip_mismatches"] == 0, v
    assert v["self_hit_rate"] == 1.0, v
    if vectors:
        assert v["recall_at_k_mean"] == 1.0, v


class CrashAfter(HmsSink):
    """A sink that fails after `crash_after` successful batches, to exercise resume."""

    def __init__(self, *a, crash_after, **kw):
        super().__init__(*a, **kw)
        self.left = crash_after

    def send(self, docs):
        if self.left <= 0:
            raise ImporterError("simulated crash")
        self.left -= 1
        super().send(docs)


def assert_resumable(make_source, server, tmp_path, expected_docs: int, batch_size: int = 5, **opts: Any) -> None:
    """Crash after two batches, resume from the checkpoint, and check nothing is lost.

    `make_source` builds a fresh source each call (a new connection, as a new process would)."""
    import json

    tenant = server.tenant("rs")
    ck = str(tmp_path / "resume-ck.json")
    options = ImportOptions(batch_size=batch_size, verify_sample=0, checkpoint_path=ck, **opts)
    src = make_source()
    try:
        with pytest.raises(ImporterError):
            run_import(src, CrashAfter(server.url, api_key=API_KEY, tenant=tenant, crash_after=2), options, tenant=tenant)
    finally:
        src.close()
    saved = json.load(open(ck))
    assert 0 < saved["position"] < expected_docs and not saved["complete"], saved
    src = make_source()
    sink = server.sink(tenant)
    try:
        report = run_import(src, sink, options, tenant=tenant)
    finally:
        sink.close()
        src.close()
    assert report.resumed_from == saved["position"], report.resumed_from
    assert report.complete and report.stats.counts["docs_acked"] == expected_docs, report.stats.counts
    assert report.stats.rejected_total == 0


try:  # the importer suite needs the `importers-test` group; the plain SDK test run skips it
    import numpy  # noqa: F401
    import pyarrow  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
