# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""The table must be a pure function of the committed results files: every cell recomputes from
its cited file, every cited key path resolves, every cited file is tracked by git, and the
committed COMPARISON.md equals a fresh generation."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmsbench import analyze, registry, table  # noqa: E402
from hmsbench.common import ROOT  # noqa: E402

HAVE_RESULTS = bool(list(table.RESULTS.glob("*.json"))) and table.CELLS.exists()


def resolve(data, path):
    """Resolve `a.b[*].c[3]` or `a[flag]` against parsed JSON; `[*]` fans out. Returns the matched values."""
    nodes = [data]
    for part in re.findall(r"[^.\[\]]+|\[[^\]]*\]", path):
        nxt = []
        for node in nodes:
            if part == "[*]":
                nxt += list(node)
            elif part.startswith("[") and part[1:-1].isdigit():
                nxt.append(node[int(part[1:-1])])
            elif part.startswith("["):
                # `[flag]` keeps the list items whose `flag` is truthy (analyze.py
                # writes serve_rounds[graph_built] for rounds that built the graph).
                nxt += [item for item in node if isinstance(item, dict) and item.get(part[1:-1])]
            else:
                key = part
                if isinstance(node, dict) and key in node:
                    nxt.append(node[key])
                elif isinstance(node, list):
                    for item in node:
                        if isinstance(item, dict) and key in item:
                            nxt.append(item[key])
                else:
                    raise KeyError(f"{path}: {key}")
        nodes = nxt
    return nodes


def tracked(path):
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(path)], capture_output=True, text=True)
    return out.returncode == 0


@pytest.mark.skipif(not HAVE_RESULTS, reason="no committed competitive results yet")
def test_every_cell_recomputes_from_its_results_file():
    cells = json.loads(table.CELLS.read_text())
    markdown = table.COMPARISON.read_text()
    assert cells, "empty cell index"
    by_file = {}
    for cell in cells:
        by_file.setdefault(cell["file"], []).append(cell)
    for file, group in by_file.items():
        path = ROOT / file
        data = json.loads(path.read_text())
        metrics, _, _ = table.metrics_for(path, data)
        for cell in group:
            if cell.get("derived"):
                mine, other = metrics[cell["metric"]][cell["hms"]], metrics[cell["metric"]][cell["other"]]
                adv = table.advantage(cell["metric"], mine, other)
                if adv is None:  # one side not measured (n/m), e.g. a static index's replace: no factor is drawn
                    assert cell["factor"] is None and cell["ci"] is None and cell["text"] == "n/m"
                    continue
                assert (None if adv["factor"] == float("inf") else adv["factor"]) == cell["factor"]
                assert (adv["ci"] or None) == (cell["ci"] or None)
                assert cell["text"] == table.fx(adv["factor"])
                continue
            fresh = metrics[cell["metric"]][cell["system"]]
            assert fresh["value"] == cell["value"], (file, cell["metric"], cell["system"])
            assert fresh["n"] == cell["n"]
            assert fresh["source"] == cell["source"]
            assert resolve(data, cell["source"].split("#", 1)[1]) , f"{cell['source']} resolves to nothing"
        # Every number the table prints for this file appears as text in the markdown.
        for cell in group:
            printed = cell["text"] != "n/m" and (cell.get("derived") or cell["value"] is not None)
            if printed:
                assert cell["text"] in markdown, (file, cell["metric"], cell["text"])


@pytest.mark.skipif(not HAVE_RESULTS, reason="no committed competitive results yet")
def test_cited_files_are_committed_and_table_is_current():
    cited = {c["file"] for c in json.loads(table.CELLS.read_text())}
    for file in cited:
        assert (ROOT / file).exists()
        assert tracked(ROOT / file), f"{file} is cited but not committed"
    assert table.main(check=True) == 0, "run ./benchmarks/competitive/hms-bench table"


@pytest.mark.skipif(not HAVE_RESULTS, reason="no committed competitive results yet")
def test_published_tables_regenerate_byte_for_byte(tmp_path):
    paths = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "--", "README.md", "docs/benchmarks",
                                     "benchmarks/results/competitive", "benchmarks/competitive/README-block.md",
                                     "benchmarks/competitive/GAPS.json"], text=True).splitlines()
    expected = {path: (ROOT / path).read_bytes() for path in paths}
    for path, content in expected.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    harness = Path("benchmarks/competitive")
    shutil.copytree(ROOT / harness / "hmsbench", tmp_path / harness / "hmsbench", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / harness / "hms-bench", tmp_path / harness / "hms-bench")
    generated = subprocess.run([sys.executable, str(tmp_path / harness / "hms-bench"), "table"], cwd=tmp_path,
                               capture_output=True, text=True)
    assert generated.returncode == 0, generated.stderr
    for path, content in expected.items():
        assert (tmp_path / path).read_bytes() == content, f"stale generated file: {path}"


def fake_ann(hms_qps, other_qps, spread=0.01):
    def serve(qps):
        return {"configs": [{"params": {"ef": 64}, "recall_at_10": 0.96,
                             "repeats": [{"qps": qps * (1 + spread * i), "p50_us": 1, "p99_us": 2} for i in range(2)]}],
                "rss_steady": 2**20, "peak_rss_bytes": 2**21, "cold_start_s": 1.0, "index_bytes": 2**20, "build_s": 1.0, "graph_built": True}

    def system(qps, build_runs):
        return {"display": "x", "builds": {"b": {"build_runs": build_runs, "serve_rounds": [serve(qps * (1 + 0.01 * r)) for r in range(4)], "params": {}}}}

    runs = [{"build_s": 2.0, "peak_rss_bytes": 2**22, "index_bytes": 2**20}] * 2
    return {"schema_version": 1, "kind": "ann", "suite": "t", "commit": "0" * 40, "date": "2026-01-01T00:00:00", "hardware": {"class": "test"},
            "dataset": {"name": "d", "n_train": 1, "n_test": 1}, "systems": {registry.HMS: system(hms_qps, []), "hnswlib": system(other_qps, runs)}}


def test_verdicts_follow_the_interval_and_orientation():
    clear = fake_ann(2000, 1000)
    m = analyze.ann_metrics("x.json", clear, level=0.99)
    mine, other = m["qps@recall0.95"][registry.HMS], m["qps@recall0.95"]["hnswlib"]
    adv = table.advantage("qps@recall0.95", mine, other)
    assert table.verdict(adv) == "WIN" and adv["factor"] == pytest.approx(2.0, rel=0.05)
    # lower is better: a build twice as slow is a loss with the factor below 1.
    slow = {"value": 4.0, "ci": [4.0, 4.0], "n": 1, "source": "", "unit": "s", "note": ""}
    fast = {"value": 2.0, "ci": [2.0, 2.0], "n": 1, "source": "", "unit": "s", "note": ""}
    adv = table.advantage("build_seconds", slow, fast)
    assert adv["factor"] == 0.5 and table.verdict(adv) == "LOSS"
    tie = fake_ann(1000, 1000, spread=0.0)
    m = analyze.ann_metrics("x.json", tie, level=0.99)
    assert table.verdict(table.advantage("qps@recall0.95", m["qps@recall0.95"][registry.HMS], m["qps@recall0.95"]["hnswlib"])) == "TIE"


def test_unreached_recall_target_is_reported_not_dropped():
    data = fake_ann(1000, 1000)
    data["systems"]["hnswlib"]["builds"]["b"]["serve_rounds"][0]["configs"][0]["recall_at_10"] = 0.5
    for r in data["systems"]["hnswlib"]["builds"]["b"]["serve_rounds"]:
        r["configs"][0]["recall_at_10"] = 0.5
    cell = analyze.ann_metrics("x.json", data)["qps@recall0.95"]["hnswlib"]
    assert cell["value"] is None and "not reached" in cell["note"]


def test_selection_uses_pooled_timings_over_recall_qualified_rows():
    def rnd(a, b):
        return {"configs": [{"params": {"ef": 1}, "recall_at_10": 0.97, "repeats": [{"qps": a}]},
                            {"params": {"ef": 2}, "recall_at_10": 0.99, "repeats": [{"qps": b}]},
                            {"params": {"ef": 3}, "recall_at_10": 0.80, "repeats": [{"qps": 10 * b}]}]}
    builds = {"b": {"serve_rounds": [rnd(100, 50), rnd(300, 60), rnd(110, 55)]}}
    best, _ = analyze.pick_config(builds, 0.95)
    assert best[2] == 0  # ef=3 is faster but below the recall target; ef=1 beats ef=2 on pooled median
    assert analyze.pick_config(builds, 0.999)[0] is None


def test_a_comparator_that_ships_with_the_interpreter_draws_no_install_verdict():
    put = {}

    def collect(metric, system, c):
        put.setdefault(metric, {})[system] = c

    analyze._install_cells(collect, "sqlite-fts5", {"install": {"bytes": 0, "distributions": 0, "ships_with_interpreter": True}}, "x")
    analyze._install_cells(collect, "tantivy", {"install": {"bytes": 3 * 2**20, "distributions": 1}}, "x")
    name, best = table.best_other("install_mib", put["install_mib"])
    assert name == "tantivy" and best["value"] == 3.0
    assert put["install_mib"]["sqlite-fts5"]["informational"]


def test_raw_pass_timings_from_older_files_summarise_like_stored_summaries():
    from hmsbench.common import pass_summary

    raw = [[5_000, 1_000, 3_000, 9_000], [2_000, 2_000, 4_000, 8_000]]
    old = {"passes_ns": raw}
    new = {"passes": [pass_summary(ns) for ns in raw]}
    assert analyze._passes(old) == analyze._passes(new) == new["passes"]


@pytest.mark.parametrize("kind", ["ann", "text"])
@pytest.mark.parametrize("classes, expected", [
    ({"hms-server": "per-request-flush", "qdrant-server": "per-request-flush"}, "WIN"),
    ({"hms-server": "per-request-flush", "qdrant-server": "periodic"}, "n/v"),
    ({"hms-server": "per-request-flush", "qdrant-server": "undocumented"}, "n/v"),
    ({"hms-server": "per-request-flush"}, "n/v"),
    ({}, "n/v"),
])
def test_server_operation_verdicts_require_matching_recorded_classes(kind, classes, expected, monkeypatch):
    data = fake_ann(2000, 1000)
    data["systems"] = {"hms-server": data["systems"][registry.HMS], "qdrant-server": data["systems"]["hnswlib"]}
    data.update(kind=kind, mode="server", reference="hms-server", facts={}, protocol={"durability_class": classes})
    if not classes:
        data.pop("protocol")
    data["hardware"].update(cpu="c", logical_cores=1, memory_bytes=2**30, os="o")
    data["dataset"].update(documents=200, queries=1)
    for i, entry in enumerate(data["systems"].values(), 1):
        entry["ops"] = {name: {"supported": True, "per_op_ms_durable": i} for name in ("replace", "delete")}
        if kind == "text":
            entry.update(build_runs=[{"ingest_s": 1, "peak_rss_bytes": 2**20, "disk_bytes": 2**20}],
                         serve_rounds=[{"passes_ns": [[1000, 2000]], "peak_rss_bytes": 2**20, "cold_start_s": 1}])
    monkeypatch.setattr(table, "load_results", lambda: [(Path("server.json"), data)])
    markdown, cells, _, gaps = table.build()
    sec = table.section(Path("server.json"), data)
    _, tally = table.render_section(sec, [])
    operation_keys = {"replace_ms_per_op", "delete_ms_per_op"}
    derived = [c for c in cells if c.get("derived") and c["metric"] in operation_keys]
    assert len(derived) == 2
    assert {c["verdict"] for c in derived} == {expected}
    rows = [line for line in markdown.splitlines() if line.startswith(("| Replace one", "| Delete one"))]
    if expected == "WIN":
        assert all("†" not in line for line in rows)
        assert all(c["factor"] == 2 for c in derived)
        assert len([r for r in tally["WIN"] if r[0] in operation_keys]) == 2
    else:
        assert all("†" in line for line in rows)
        assert all(c["factor"] is None and c["ci"] is None for c in derived)
        assert not any(g["metric"] in operation_keys for g in gaps)
        assert not any(r[0] in operation_keys for status in ("WIN", "TIE", "LOSS", "n/m") for r in tally[status])
        assert "durability classes " + (classes.get("hms-server") or "unrecorded") in markdown
        assert "vs " + (classes.get("qdrant-server") or "unrecorded") + "; no verdict" in markdown
    data["systems"]["qdrant-server"].pop("ops")
    without_ops = table.section(Path("server.json"), data)
    assert sec["family"] == without_ops["family"] + (2 if expected == "WIN" else 0)
