# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Server-mode invariants: server files never displace or mix with embedded files in the table,
their verdicts are drawn against warm-server, and a server's whole process tree is stopped."""

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from warmbench import registry, server_proc, server_registry, server_runner, table  # noqa: E402


def test_every_server_records_its_durability_class():
    assert server_runner.PROTOCOL["durability"] == server_registry.DURABILITY
    classes = server_runner.PROTOCOL["durability_class"]
    assert set(classes) == set(server_registry.DURABILITY) == set(server_registry.SERVER_SYSTEMS)
    assert set(classes.values()) == {"per-request-flush", "periodic", "undocumented"}


@pytest.mark.parametrize("system", ["pg-tsvector", "paradedb"])
def test_postgres_text_replacement_accepts_the_same_nul_text_as_ingest(tmp_path, system):
    psycopg = pytest.importorskip("psycopg")
    from warmbench.server_worker import TEXT

    encoded = []

    class Connection:
        def execute(self, statement, args):
            encoded.append([psycopg.sql.Literal(arg).as_string() for arg in args])

    adapter = TEXT[system](tmp_path, 1, "ops")
    adapter.db = Connection()
    adapter.replace_doc({"id": "doc/α", "text": "before\x00after"}, 7)
    assert encoded == [["'before after benchmarkrevision'", "'doc/α'"]]


def results_file(mode, reference, other, ref_qps, other_qps):
    def serve(qps):
        return {"configs": [{"params": {"ef": 64}, "recall_at_10": 0.96, "repeats": [{"qps": qps * (1 + 0.01 * i), "p50_us": 1, "p99_us": 2} for i in range(2)]}],
                "rss_steady": 2**20, "peak_rss_bytes": 2**21, "cold_start_s": 1.0}

    def system(qps):
        runs = [{"build_s": 2.0, "peak_rss_bytes": 2**22, "index_bytes": 2**20}] * 2
        return {"display": "x", "builds": {"b": {"build_runs": runs, "serve_rounds": [serve(qps * (1 + 0.01 * r)) for r in range(4)], "params": {}}}}

    data = {"schema_version": 1, "kind": "ann", "suite": "t", "commit": "0" * 40, "date": "2026-01-01T00:00:00",
            "hardware": {"class": "test", "cpu": "c", "logical_cores": 1, "memory_bytes": 2**30, "os": "o"},
            "dataset": {"name": "d", "n_train": 1, "n_test": 1}, "facts": {}, "timing_boundary": {}, "not_run": {}, "skipped": {},
            "systems": {reference: system(ref_qps), other: system(other_qps)}}
    if mode == "server":
        data |= {"mode": "server", "reference": reference}
    return data


def test_server_and_embedded_files_render_in_separate_sections(monkeypatch):
    embedded = (Path("embedded.json"), results_file("embedded", registry.WARM, "hnswlib", 2000, 1000))
    server = (Path("server.json"), results_file("server", "warm-server", "qdrant-server", 1000, 2000))
    # Same (kind, dataset, n_train, hardware, suite): the server file must not replace the embedded one.
    monkeypatch.setattr(table, "RESULTS", Path("/nonexistent"))
    found = {}
    for path, data in (embedded, server):
        key = (data["kind"], data["dataset"]["name"], 1, "test", "t", data.get("mode", "embedded"))
        found[key] = (path, data)
    assert len(found) == 2
    monkeypatch.setattr(table, "load_results", lambda: [server, embedded])
    markdown, cells, _, _ = table.build()
    results_at, server_at = markdown.index("## Results"), markdown.index("## Server mode results")
    embedded_title, server_title = markdown.index("### d (1 vectors, 1 queries)\n"), markdown.index("### d (1 vectors, 1 queries), server mode")
    assert results_at < embedded_title < server_at < server_title
    verdicts = {Path(c["file"]).name: c for c in cells if c.get("derived") and c["metric"] == "qps@recall0.95"}
    assert verdicts["embedded.json"]["warm"] == registry.WARM and verdicts["embedded.json"]["verdict"] == "WIN"
    assert verdicts["server.json"]["warm"] == "warm-server" and verdicts["server.json"]["other"] == "qdrant-server"
    assert verdicts["server.json"]["verdict"] == "LOSS"


def test_load_results_keeps_server_and_embedded_files_with_the_same_key(tmp_path, monkeypatch):
    import json
    for name, mode in (("a.json", "embedded"), ("b.json", "server")):
        data = results_file(mode, registry.WARM if mode == "embedded" else "warm-server", "hnswlib" if mode == "embedded" else "pgvector", 1, 1)
        (tmp_path / name).write_text(json.dumps(data))
    monkeypatch.setattr(table, "RESULTS", tmp_path)
    assert sorted(p.name for p, _ in table.load_results()) == ["a.json", "b.json"]


def test_stop_kills_the_whole_process_group(tmp_path):
    server = server_proc.Server("tree", ["sh", "-c", "sleep 300 & sleep 300 & wait"], tmp_path).start()
    pgid = server.proc.pid
    deadline = time.monotonic() + 5
    while len(server_proc._group_pids(pgid)) < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(server_proc._group_pids(pgid)) >= 3
    assert (tmp_path / "tree.pgid").exists()
    server.stop(grace=5)
    assert server_proc._group_pids(pgid) == []
    assert not (tmp_path / "tree.pgid").exists()


def test_orphaned_tree_is_killed_from_its_marker(tmp_path):
    server = server_proc.Server("orphan", ["sh", "-c", "sleep 300 & wait"], tmp_path).start()
    pgid = server.proc.pid
    server._stop.set()  # the client "dies" without stopping its server
    assert server_proc.kill_orphans(tmp_path) == [pgid]
    server.proc.wait(timeout=5)
    deadline = time.monotonic() + 5
    while server_proc._group_pids(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server_proc._group_pids(pgid) == []


def test_stable_ports_are_reused_while_free(tmp_path):
    path = tmp_path / "ports.txt"
    first = server_proc.stable_ports(path, 3)
    assert len(set(first)) == 3 and server_proc.stable_ports(path, 3) == first
    assert os.path.exists(path)
