# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pipeline invariants, exercised against a real hms-server."""

import csv
import json
import os
from pathlib import Path

import pytest

from holographic_sdk.importers import ImportOptions, ImporterError, run_import
from holographic_sdk.importers import cli as cli_mod
from holographic_sdk.importers.sources.base import ColumnMap, Source
from holographic_sdk.importers.sources.files import CsvSource, JsonlSource

from .conftest import API_KEY, DIM, CrashAfter, assert_faithful, vec


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HMS_IMPORT_STATE_DIR", str(tmp_path / "state"))


def write_jsonl(path, n, dim=DIM, **extra):
    with open(path, "w") as fh:
        for i in range(n):
            fh.write(json.dumps({"id": f"d{i}", "text": f"doc number {i}", "emb": vec(i, dim), **extra}) + "\n")
    return JsonlSource([str(path)], ColumnMap(["id=id", "text=text", "vector=emb"]))


def best_id(server, tenant, vector):
    sink = server.sink(tenant)
    try:
        return sink.query(vector=vector, top_k=1)[0]["id"]
    finally:
        sink.close()


def test_crash_then_resume_commits_every_record(tmp_path, server):
    src = write_jsonl(tmp_path / "a.jsonl", 100)
    tenant = server.tenant()
    opts = ImportOptions(batch_size=10, verify_sample=0, checkpoint_path=str(tmp_path / "ck.json"))
    crashing = CrashAfter(server.url, api_key=API_KEY, tenant=tenant, crash_after=3)
    with pytest.raises(ImporterError) as err:
        run_import(src, crashing, opts, tenant=tenant)
    assert err.value.report.error
    ck = json.loads((tmp_path / "ck.json").read_text())
    assert ck["position"] == 30 and not ck["complete"]

    sink = server.sink(tenant)
    report = run_import(src, sink, opts, tenant=tenant)
    sink.close()
    assert report.resumed_from == 30 and report.complete
    assert report.stats.counts["docs_acked"] == 100  # restored 30 plus 70 this run
    for i in range(100):
        assert best_id(server, tenant, vec(i)) == f"d{i}"


def test_rerun_is_idempotent(tmp_path, run, server):
    src = write_jsonl(tmp_path / "a.jsonl", 20)
    r1, tenant = run(src, batch_size=8)
    r2, _ = run(src, tenant=tenant, batch_size=8)
    assert_faithful(r1, 20)
    assert_faithful(r2, 20)
    sink = server.sink(tenant)
    ids = {m["id"] for m in sink.query(vector=vec(3), top_k=100)}
    sink.close()
    assert len(ids) == 20  # no duplicates after a second import


def test_dimension_mismatch_is_fatal_and_writes_nothing(tmp_path, server):
    src = write_jsonl(tmp_path / "a.jsonl", 10, dim=16)
    tenant = server.tenant()
    sink = server.sink(tenant)
    with pytest.raises(ImporterError, match=r"16 dimensions.*takes 32"):
        run_import(src, sink, ImportOptions(verify_sample=0), tenant=tenant)
    assert sink.query(vector=vec(1), top_k=5) == []
    sink.close()


def test_project_dry_run_reports_recall_below_one(tmp_path):
    src = write_jsonl(tmp_path / "a.jsonl", 300, dim=128)
    opts = ImportOptions(vectors="project", target_dim=32, dry_run=True, verify_sample=40, verify_k=5)
    report = run_import(src, None, opts)
    v = report.verification
    assert v["performed"] and v["basis"].startswith("dry run")
    assert 0.0 < v["recall_at_k_mean"] < 1.0, v
    assert report.stats.counts["vectors_projected"] == 300
    assert report.stats.counts["docs_planned"] == 300


def test_projection_is_deterministic_and_pads_when_growing(tmp_path, run, server):
    src = write_jsonl(tmp_path / "a.jsonl", 12, dim=8)
    report, tenant = run(src, vectors="project")
    assert_faithful(report, 12)  # zero padding preserves cosine exactly


def test_long_text_is_split_losslessly_and_parts_are_retrievable(tmp_path, run, server):
    words = [f"w{i}" for i in range(9000)]
    text = "  ".join(words) + "\n\ntail  "
    path = tmp_path / "long.jsonl"
    path.write_text(json.dumps({"id": "long", "text": text, "emb": vec(5)}) + "\n")
    src = JsonlSource([str(path)], ColumnMap(["id=id", "text=text", "vector=emb"]))
    report, tenant = run(src, sample=0)
    assert report.stats.counts["texts_split"] == 1 and report.stats.counts["text_parts_created"] == 3
    sink = server.sink(tenant)
    matches = sink.query(vector=vec(5), top_k=10, filter={"hms_parent_id": "long"})
    sink.close()
    parts = sorted(matches, key=lambda m: m["metadata"]["hms_part"])
    assert [m["metadata"]["hms_part"] for m in parts] == [0, 1, 2]
    # the server trims whitespace around each stored text, so compare the word sequence
    assert " ".join(m["text"] for m in parts).split() == text.split()


def test_metadata_oversize_reject_and_truncate(tmp_path, run):
    path = tmp_path / "m.jsonl"
    big = "x" * 80_000
    path.write_text(json.dumps({"id": "a", "text": "t", "emb": vec(1), "big": big, "keep": 1}) + "\n")
    mk = lambda: JsonlSource([str(path)], ColumnMap(["id=id", "text=text", "vector=emb"]))
    report, _ = run(mk(), sample=0)
    assert report.stats.rejects == {"metadata_too_large": 1}
    report, _ = run(mk(), sample=1, oversize_metadata="truncate")
    assert report.stats.rejected_total == 0 and report.stats.counts["metadata_keys_dropped"] == 1
    assert report.verification["roundtrip_mismatches"] == 0


def test_ids_namespaces_timestamps_and_reserved_keys(tmp_path, run, server):
    rows = [
        {"id": "x" * 300, "text": "long id", "emb": vec(1), "ns": "a", "ts": "2024-05-01T10:00:00Z"},
        {"id": "hms:reserved", "text": "reserved prefix", "emb": vec(2), "ns": "", "ts": 1_700_000_000_000},
        {"id": "same", "text": "in a", "emb": vec(3), "ns": "a", "hms_source": "mine", "ts": "not a date"},
        {"id": "same", "text": "in b", "emb": vec(4), "ns": "b", "ts": None},
    ]
    path = tmp_path / "n.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    src = JsonlSource([str(path)], ColumnMap(["id=id", "text=text", "vector=emb", "namespace=ns", "created_at=ts"]))
    report, tenant = run(src)
    assert_faithful(report, 4)
    assert report.stats.counts["ids_rewritten"] == 2 and report.stats.counts["timestamps_normalized"] == 2
    assert report.stats.counts["metadata_reserved_key_collisions"] == 1
    sink = server.sink(tenant)
    both = {m["id"]: m["metadata"] for m in sink.query(vector=vec(3), top_k=10, filter={"hms_source_id": "same"})}
    sink.close()
    assert set(both) == {"a::same", "b::same"}
    assert both["a::same"]["hms_orig_meta"] == {"hms_source": "mine"}
    assert both["a::same"]["hms_created_at_raw"] == "not a date"


def test_tenant_namespace_mode_isolates(tmp_path, run, server):
    rows = [{"id": f"i{i}", "text": f"t{i}", "emb": vec(i), "ns": "alpha" if i % 2 else "beta/x"} for i in range(6)]
    path = tmp_path / "n.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    src = JsonlSource([str(path)], ColumnMap(["id=id", "text=text", "vector=emb", "namespace=ns"]))
    tenant_prefix = f"nsm{os.getpid()}-"
    report, _ = run(src, namespace_mode="tenant", tenant_prefix=tenant_prefix, sample=100)
    assert_faithful(report, 6)
    sink = server.sink(None)
    got = sink.query(vector=vec(1), top_k=10, tenant=tenant_prefix + "alpha")
    sink.close()
    assert {m["id"] for m in got} == {"i1", "i3", "i5"}


def test_auto_never_mixes_embedding_spaces(tmp_path, run):
    path = tmp_path / "mix.jsonl"
    path.write_text(
        json.dumps({"id": "a", "text": "has vector", "emb": vec(1)}) + "\n"
        + json.dumps({"id": "b", "text": "no vector"}) + "\n"
    )
    src = JsonlSource([str(path)], ColumnMap(["id=id", "text=text", "vector=emb"]))
    report, _ = run(src, sample=0)
    assert report.stats.rejects == {"vector_missing_in_carry_run": 1}


def test_text_only_source_needs_an_encoder(tmp_path, server):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"id": "a", "text": "only text"}) + "\n")
    src = JsonlSource([str(path)], ColumnMap(["id=id", "text=text"]))
    sink = server.sink(server.tenant())
    with pytest.raises(ImporterError, match="no on-device embedder"):
        run_import(src, sink, ImportOptions(verify_sample=0))
    sink.close()


def test_client_side_embedder_encodes_text_only_source(tmp_path, run, monkeypatch):
    import sys
    import types

    mod = types.ModuleType("test_embedder_mod")
    mod.embed = lambda texts: [vec(sum(map(ord, t))) for t in texts]
    monkeypatch.setitem(sys.modules, "test_embedder_mod", mod)
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps({"id": f"a{i}", "text": f"text {i}"}) + "\n" for i in range(5)))
    src = JsonlSource([str(path)], ColumnMap(["id=id", "text=text"]))
    report, _ = run(src, embedder="test_embedder_mod:embed")
    assert_faithful(report, 5, vectors=False)
    assert report.stats.counts["vectors_encoded_client_side"] == 5


def test_resume_detects_a_reordered_source_and_restarts(tmp_path, server):
    path = tmp_path / "r.csv"
    rows = [(f"d{i}", f"text {i}", json.dumps(vec(i))) for i in range(30)]

    def write(order):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "text", "emb"])
            w.writerows(order)

    def source():
        return CsvSource([str(path)], ColumnMap(["id=id", "text=text", "vector=emb"]))

    write(rows)
    tenant = server.tenant()
    opts = ImportOptions(batch_size=10, verify_sample=0, checkpoint_path=str(tmp_path / "ck.json"))
    with pytest.raises(ImporterError):
        run_import(source(), CrashAfter(server.url, api_key=API_KEY, tenant=tenant, crash_after=2), opts, tenant=tenant)
    write(list(reversed(rows)))  # the source's order changes between runs
    sink = server.sink(tenant)
    report = run_import(source(), sink, opts, tenant=tenant)
    sink.close()
    assert report.server["resume_prefix_mismatch"] is True
    assert report.stats.counts["docs_acked"] == 30
    assert best_id(server, tenant, vec(7)) == "d7"


class Boom(Source):
    name = "boom"
    summary = "test"

    @classmethod
    def add_arguments(cls, parser):
        pass

    @classmethod
    def from_args(cls, args, env):
        from holographic_sdk.importers.common import secret_from, uri_secret

        uri_secret("postgres://user:s3cr3t-pw@db.example:5432/x")
        secret_from(None, {"BOOM_KEY": "sk-live-abcdef123456"}, "BOOM_KEY")
        return cls()

    def describe(self):
        return {}

    def records(self, cursor=None):
        raise RuntimeError("could not connect to postgres://user:s3cr3t-pw@db.example:5432/x with key sk-live-abcdef123456")
        yield


def test_credentials_never_reach_output_report_or_checkpoint(tmp_path, server, monkeypatch, capsys):
    monkeypatch.setattr(cli_mod, "source_class", lambda name: Boom)
    monkeypatch.setattr(cli_mod, "source_names", lambda: ["boom"])
    report_path = tmp_path / "rep.json"
    env = {"HMS_API_KEY": API_KEY}
    code = cli_mod.main(["boom", "--hms-url", server.url, "--report", report_path.as_posix()], env=env)
    assert code == 1
    cap = capsys.readouterr()
    blob = cap.out + cap.err + (report_path.read_text() if report_path.exists() else "")
    for root, _, files in os.walk(tmp_path):
        for f in files:
            blob += Path(root, f).read_text(errors="replace")
    assert API_KEY not in blob
    assert "Traceback" not in blob
    assert "s3cr3t-pw" not in blob and "sk-live-abcdef123456" not in blob
    assert "***" in cap.err


def test_cli_end_to_end(tmp_path, server, capsys):
    path = tmp_path / "docs.jsonl"
    write_jsonl(path, 15)
    report_path = tmp_path / "rep.json"
    code = cli_mod.main(
        ["jsonl", "--path", str(path), "--map", "id=id", "--map", "text=text", "--map", "vector=emb",
         "--hms-url", server.url, "--tenant", server.tenant("cli"), "--report", str(report_path), "--batch-size", "4"],
        env={"HMS_API_KEY": API_KEY},
    )
    assert code == 0, capsys.readouterr().err
    rep = json.loads(report_path.read_text())
    assert rep["counts"]["docs_acked"] == 15 and rep["verification"]["recall_at_k_mean"] == 1.0
    assert API_KEY not in report_path.read_text()
