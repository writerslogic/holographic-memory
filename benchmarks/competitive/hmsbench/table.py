# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generate docs/benchmarks/COMPARISON.md, the README block and the cell index.

Reads only the committed benchmarks/results/competitive/*.json (smoke runs excluded). Every
number in the output is produced by hmsbench.analyze from those files; docs/benchmarks/
comparison.cells.json lists every cell with the file and key path it came from, and
tests/test_table.py recomputes each one.
"""

import json
import re
import sys
from pathlib import Path

from . import analyze, registry
from .common import RESULTS, ROOT

DOCS = ROOT / "docs/benchmarks"
COMPARISON = DOCS / "COMPARISON.md"
CELLS = DOCS / "comparison.cells.json"
BLOCK = ROOT / "benchmarks/competitive/README-block.md"
GAPS_MD = DOCS / "GAPS.md"
# The gap ledger is a long generated table; keep it out of the docs search index so it
# does not crowd prose pages out of the top results (scripts/docs-qa/queries.txt).
GAPS_FRONT_MATTER = "---\nsearch:\n  exclude: true\n---\n\n"
GAPS_JSON = ROOT / "benchmarks/competitive/GAPS.json"
README = ROOT / "README.md"
START, END = "<!-- BENCH:START -->", "<!-- BENCH:END -->"

# metric key -> (label, unit label, decimals). Order is the display order.
ANN_METRICS = [
    ("qps@recall0.90", "QPS @ recall@10 >= 0.90", "QPS", 0), ("qps@recall0.95", "QPS @ recall@10 >= 0.95", "QPS", 0),
    ("qps@recall0.99", "QPS @ recall@10 >= 0.99", "QPS", 0),
    ("p50_latency@recall0.95", "p50 latency @ 0.95", "us", 0), ("p99_latency@recall0.95", "p99 latency @ 0.95", "us", 0),
    ("build_seconds", "Build time", "s", 2), ("build_peak_rss_mib", "Peak RSS, build", "MiB", 0),
    ("serving_rss_mib", "RSS, serving", "MiB", 0), ("index_bytes_mib", "Index size", "MiB", 1),
    ("cold_start_s", "Cold start to first answer", "s", 2),
    ("replace_ms_per_op", "Replace one vector (durable)", "ms", 3), ("delete_ms_per_op", "Delete one vector (durable)", "ms", 3),
    ("install_mib", "Install size", "MiB", 1), ("install_distributions", "Install distributions", "count", 0),
]
TEXT_METRICS = [
    ("ndcg@10", "nDCG@10", "", 4), ("recall@100", "recall@100", "", 4),
    ("qps", "QPS (single thread)", "QPS", 0), ("p50_latency", "p50 latency", "us", 0), ("p99_latency", "p99 latency", "us", 0),
    ("build_seconds", "Ingest/build time", "s", 2), ("build_peak_rss_mib", "Peak RSS, build", "MiB", 0),
    ("serving_rss_mib", "Peak RSS, serving", "MiB", 0), ("disk_mib", "On-disk size", "MiB", 1),
    ("cold_start_s", "Cold start to first answer", "s", 2),
    ("replace_ms_per_op", "Replace one document (durable)", "ms", 3), ("delete_ms_per_op", "Delete one document (durable)", "ms", 3),
    ("install_mib", "Install size", "MiB", 1), ("install_distributions", "Install distributions", "count", 0),
]
HEADLINE = {
    "ann": ["qps@recall0.95", "p99_latency@recall0.95", "build_seconds", "serving_rss_mib", "index_bytes_mib"],
    "text": ["ndcg@10", "qps", "p99_latency", "build_seconds", "serving_rss_mib", "disk_mib"],
}

# Candidate levers per metric. These are hypotheses for the coordinator, not measurements, except
# where a results file is named.
LEVERS = {
    "text:build_seconds": "unprofiled hypothesis: per-document text encoding in documents::prepare (src/core/encoding.rs) dominates ingest; profile it, then a lexical-only ingest path or a cheaper encoder is the lever",
    "text:build_peak_rss_mib": "unprofiled hypothesis: the arena and index are built in memory before commit; streaming commits per batch",
    "text:cold_start_s": "the store is read into memory at open (open_s in the raw file); lazy or memory-mapped loading of the arena and postings is the lever",
    "text:serving_rss_mib": "the whole store is resident after open; memory-mapped postings and file-backed text (ledger 2A/2B) are the open levers",
    "text:p99_latency": "tail queries are the long ones; MaxScore-style early termination for long queries (the ledger's impact-index work, category 1C) is the candidate",
    "text:p50_latency": "none identified; the median is within noise of the leader",
    "text:disk_mib": "compact postings and file-backed text (ledger 2B/2C) and removing preallocated arena slack",
    "text:recall@100": "none: a rounding-level difference in analyzer details (stemmer, stop words), not a lever",
    "text:replace_ms_per_op": "batching commits (memorize_batch) or a group commit; the per-op figure is one durable commit",
    "text:delete_ms_per_op": "batching commits (memorize_batch) or a group commit; the per-op figure is one durable commit",
    "qps@recall0.90": "per-query estimate count and memory traffic of the vertex graph (the vertex-graph rows under benchmarks/results)",
    "qps@recall0.95": "per-query estimate count and memory traffic of the vertex graph (ledger 1A; the neighbour-block fast-scan lever was killed on M4 in benchmarks/results/qgraph_block_scan_microbench.json, so the open levers are pruning and an x86 kernel)",
    "qps@recall0.99": "higher-recall operating points need more re-rank candidates; a cheaper re-rank (residual 4-bit, benchmarks/results/qgraph_bytes_paired_m4_glove_v1.json) or exact re-rank of fewer candidates",
    "p50_latency@recall0.95": "same as QPS at 0.95 (latency is the inverse of the single-thread rate here)",
    "p99_latency@recall0.95": "same as QPS at 0.95; tail latency comes from the slowest queries' expansion counts",
    "build_seconds": "graph construction cost: parallel build with a cheaper construction search (quantized_search) and a smaller build_ef, at a recall cost to be measured; for text, a lexical-only ingest path that skips dense-slot encoding",
    "build_peak_rss_mib": "stream the build instead of holding the f32 matrix plus graph (ledger 1A target: peak build RSS <= 2x final index bytes)",
    "serving_rss_mib": "persist the index memory-mapped and drop the build-time allocations; for text the serving V40 RSS campaign in the ledger (2A) is the open lever",
    "index_bytes_mib": "4-bit residual codes shrink the index 26% (benchmarks/results/qgraph_bytes_paired_m4_glove_v1.json) at a measured QPS cost; coarser id coding",
    "disk_mib": "compact postings and file-backed text (ledger 2B/2C work) and removing preallocated arena slack",
    "cold_start_s": "ANN: persist and mmap the VGraph (no persisted ANN index exists today); text: lazy-load the arena instead of reading it at open",
    "replace_ms_per_op": "batching commits (memorize_batch) or a group-commit log; the per-op figure is one durable commit",
    "delete_ms_per_op": "batching commits or a group-commit log",
    "install_mib": "strip symbols, split optional features out of the default binary",
    "install_distributions": "none needed",
}


def fmt(value, decimals):
    if value is None:
        return "n/m"
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def load_results():
    """(path, data) for every committed non-smoke results file, newest per key."""
    found = {}
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.startswith("smoke-"):
            continue
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1 or data.get("kind") not in ("ann", "text"):
            continue
        key = (data["kind"], data["dataset"].get("id") or data["dataset"].get("name"), data["dataset"].get("n_train"), data["hardware"]["class"], data["suite"],
               data.get("mode", "embedded"))
        if key not in found or data["date"] > found[key][1]["date"]:
            found[key] = (path, data)
    return sorted(found.values(), key=lambda pd: (pd[1]["kind"], str(pd[1]["dataset"]), pd[1]["suite"], pd[1]["hardware"]["class"]))


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def _has_ci(c):
    return c["n"] > 1 and c["ci"][0] < c["ci"][1]


def advantage(metric, mine, theirs):
    """HMS advantage factor (>1 means HMS is better) with a conservative interval, or None."""
    if mine["value"] is None or theirs["value"] is None:
        return None
    if analyze.comparison_note(metric, mine, theirs):
        return None
    higher = analyze.direction(metric) == analyze.HIGHER
    if mine["value"] <= 0 or theirs["value"] <= 0:
        if mine["value"] == theirs["value"]:
            return {"factor": 1.0, "ci": None}
        better = (mine["value"] > theirs["value"]) == higher
        return {"factor": float("inf") if better else 0.0, "ci": None}
    point = mine["value"] / theirs["value"] if higher else theirs["value"] / mine["value"]
    if _has_ci(mine) and _has_ci(theirs):
        (ml, mh), (tl, th) = mine["ci"], theirs["ci"]
        if min(ml, tl) <= 0:
            return {"factor": point, "ci": None}
        lo, hi = (ml / th, mh / tl) if higher else (tl / mh, th / ml)
        return {"factor": point, "ci": [lo, hi]}
    return {"factor": point, "ci": None}


def fx(factor):
    if factor == float("inf"):
        return "inf"
    return f"{factor:.2f}x"


def verdict(adv):
    if adv is None:
        return "n/m"
    if adv["ci"]:
        return "WIN" if adv["ci"][0] > 1 else "LOSS" if adv["ci"][1] < 1 else "TIE"
    return "WIN" if adv["factor"] > 1.05 else "LOSS" if adv["factor"] < 1 / 1.05 else "TIE"


def best_other(metric, cells):
    sign = 1 if analyze.direction(metric) == analyze.HIGHER else -1
    # The eager HMS row is reported, never a comparator.
    others = {s: c for s, c in cells.items()
              if registry.SYSTEMS[s]["family"] != "hms" and c["value"] is not None and not c.get("informational")}
    if not others:
        return None, None
    best = max(others, key=lambda s: sign * others[s]["value"])
    return best, others[best]


def reference(data):
    """The HMS row verdicts are drawn for: the library, or hms-server in a server-mode file."""
    return data.get("reference", registry.HMS)


def is_server(data):
    return data.get("mode") == "server"


def metrics_for(path, data):
    """Cells at a Bonferroni-adjusted level: the family is the verdict-bearing metrics of this
    results file (those where HMS and at least one comparator have a value)."""
    kind = data["kind"]
    fn = analyze.ann_metrics if kind == "ann" else analyze.text_metrics
    spec = ANN_METRICS if kind == "ann" else TEXT_METRICS
    first = fn(rel(path), data)
    ref = reference(data)
    family = sum(1 for k, *_ in spec if k not in analyze.NO_VERDICT and ref in first.get(k, {})
                 and first[k][ref]["value"] is not None and best_other(k, first[k])[0] is not None
                 and not analyze.comparison_note(k, first[k][ref], best_other(k, first[k])[1]))
    level = 1 - 0.05 / max(family, 1)
    return fn(rel(path), data, level=level), level, family


def section(path, data):
    kind = data["kind"]
    metrics, level, family = metrics_for(path, data)
    spec = ANN_METRICS if kind == "ann" else TEXT_METRICS
    hw = data["hardware"]
    ds = data["dataset"]
    title = f"{ds.get('id') or ds['name']}" + (f" ({ds['n_train']:,} vectors, {ds['n_test']:,} queries)" if kind == "ann" else f" ({ds['documents']:,} documents, {ds['queries']:,} queries)")
    if is_server(data):
        title += ", server mode"
    return {"path": path, "data": data, "metrics": metrics, "spec": spec, "title": title, "hw": hw, "kind": kind,
            "level": level, "family": family}


def timing_boundary_lines(data):
    boundary = data.get("timing_boundary", {})
    present = [s for s in data["systems"] if s in boundary]
    return ["Timing boundary (where each query is timed): " + "; ".join(f"{registry.SYSTEMS[s]['display']}: {boundary[s]}" for s in present) + ".", ""] if present else []


def render_section(sec, cells_out):
    data, metrics, spec, hw = sec["data"], sec["metrics"], sec["spec"], sec["hw"]
    path = sec["path"]
    lines = [f"### {sec['title']}", "",
             f"Suite `{data['suite']}`, hardware class `{hw['class']}` ({hw['cpu']}, {hw['logical_cores']} logical cores, {hw['memory_bytes'] / 2**30:.0f} GiB, " +
             f"{hw['os']}{', GitHub-hosted runner' if hw.get('hosted_runner') else ''}), commit `{data['commit'][:10]}`, {data['date'][:10]}. " +
             f"Raw results: [{path.name}](../../{rel(path)}). Intervals: {sec['level'] * 100:.2f}% bootstrap (Bonferroni family of {sec['family']} verdict metrics).", "",
             *timing_boundary_lines(data),
             f"| Metric | {'hms-server' if is_server(data) else 'HMS'} | Best other | Advantage (HMS vs best other) | Verdict |", "|---|---|---|---|---|"]
    tally = {"WIN": [], "TIE": [], "LOSS": [], "n/m": [], "n/v": []}
    footnotes = []

    def marker(key, mine, other, name):
        note = analyze.comparison_note(key, mine, other) if other else ""
        if not note:
            return ""
        footnote = f"{registry.SYSTEMS[reference(data)]['display']} vs {registry.SYSTEMS[name]['display']}: {note}."
        if footnote not in footnotes:
            footnotes.append(footnote)
        return "†"

    for key, label, unit, dec in spec:
        cells = metrics.get(key)
        if not cells:
            continue
        mine = cells.get(reference(data))
        name, other = best_other(key, cells)
        for system, c in cells.items():
            cells_out.append({"section": sec["title"], "file": rel(path), "hardware_class": hw["class"], "metric": key, "system": system,
                              "value": c["value"], "ci": c["ci"] if c["value"] is not None else None, "n": c["n"], "source": c["source"],
                              "text": fmt(c["value"], dec)})
        if mine is None:
            continue
        adv = advantage(key, mine, other) if other else None
        mark = marker(key, mine, other, name)
        v = "n/v" if key in analyze.NO_VERDICT or mark else verdict(adv) if other else "n/m"
        tally[v].append((key, label, name, adv))
        if other is not None:
            cells_out.append({"section": sec["title"], "file": rel(path), "hardware_class": hw["class"], "metric": key, "derived": True,
                              "hms": reference(data), "other": name, "factor": None if adv is None or adv["factor"] == float("inf") else adv["factor"],
                              "factor_infinite": bool(adv and adv["factor"] == float("inf")), "ci": adv["ci"] if adv else None, "verdict": v,
                              "text": "n/m" if adv is None else fx(adv["factor"])})
        adv_text = "n/m" if adv is None else fx(adv["factor"]) + (f" [{adv['ci'][0]:.2f}, {adv['ci'][1]:.2f}]" if adv["ci"] else " (no CI)")
        best_text = "n/m" if other is None else f"{registry.SYSTEMS[name]['display']} {fmt(other['value'], dec)}{mark}"
        note = f" ({mine['note']})" if mine["value"] is None and mine["note"] else ""
        lines.append(f"| {label}{' (' + unit + ')' if unit else ''} | {fmt(mine['value'], dec)}{mark}{note} | {best_text} | {adv_text} | **{v}** |")
    lines += ["", "<details><summary>All systems</summary>", ""]
    systems = list(data["systems"])
    head = ["Metric"] + [registry.SYSTEMS[s]["display"] for s in systems]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key, label, unit, dec in spec:
        cells = metrics.get(key)
        if not cells:
            continue
        best, _ = best_other(key, {**cells})
        sign = 1 if analyze.direction(key) == analyze.HIGHER else -1
        valued = {s: c["value"] for s, c in cells.items() if c["value"] is not None}
        mine = cells.get(reference(data))
        if mine and any(analyze.comparison_note(key, mine, cells[s]) for s in valued):
            valued = {}
        top = max(valued, key=lambda s: sign * valued[s]) if valued else None
        row = [label + (f" ({unit})" if unit else "")]
        for s in systems:
            c = cells.get(s)
            if c is None:
                row.append("")
            elif c["value"] is None:
                row.append(f"n/m ({c['note']})" if c["note"] else "n/m")
            else:
                text = fmt(c["value"], dec)
                if mine and s != reference(data):
                    text += marker(key, mine, c, s)
                row.append(f"**{text}**" if s == top else text)
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Bold marks the best value in the row. Operating points for the QPS rows: "
              + "; ".join(f"{registry.SYSTEMS[s]['display']}: {metrics['qps@recall0.95'][s]['note']}" for s in systems
                          if "qps@recall0.95" in metrics and s in metrics["qps@recall0.95"] and metrics["qps@recall0.95"][s]["note"]) + "." if sec["kind"] == "ann" else
              "Bold marks the best value in the row.", "", "</details>", ""]
    lines += [f"† {note}" for note in footnotes] + ([""] if footnotes else [])
    skipped = data.get("skipped")
    if skipped:
        lines += ["Systems that did not run in this file: " + "; ".join(f"`{k}`: {v[:160]}" for k, v in skipped.items()) + ".", ""]
    return lines, tally


def facts_table(results):
    seen = {}
    for _, data in results:
        for system in data["systems"]:
            seen.setdefault(system, data["facts"].get(system, {}))
    lines = ["| System | Service needed | Encrypted at rest | Basis |", "|---|---|---|---|"]
    for system, facts in seen.items():
        svc, enc = facts.get("service_needed", {}), facts.get("encrypted_at_rest", {})
        basis = "measured: " + enc["source"] if enc.get("basis") == "measured" else "documentation (not exercised)"
        lines.append(f"| {registry.SYSTEMS[system]['display']} | {svc.get('value', 'n/m')} | {enc.get('value', 'n/m')} | {basis} |")
    return lines


def versions_table(results):
    seen = {}
    for _, data in results:
        for system, entry in data["systems"].items():
            seen[system] = entry.get("version")
    lines = ["| System | Pinned version(s) |", "|---|---|"]
    for system, version in seen.items():
        lines.append(f"| {registry.SYSTEMS[system]['display']} | {', '.join(f'{k} {v}' for k, v in (version or {}).items())} |")
    return lines


def build():
    # Embedded sections first, then server-mode sections under their own heading: a server row is
    # never compared with an embedded one.
    results = sorted(load_results(), key=lambda pd: is_server(pd[1]))
    sections = [section(p, d) for p, d in results]
    cells = []
    body, server_body, summary, losses = [], [], [], []
    for sec in sections:
        lines, tally = render_section(sec, cells)
        if is_server(sec["data"]):
            server_body += lines
        else:
            body += lines
        counts = {k: len(v) for k, v in tally.items()}
        summary.append((sec, counts, tally))
        for key, label, name, adv in tally["LOSS"]:
            losses.append((sec, key, label, name, adv))
    out = ["# HMS comparison", "",
           "Generated by `./benchmarks/competitive/hms-bench table` from the committed files in `benchmarks/results/competitive/`; do not edit by hand. " +
           "Every cell is indexed with its source file and key path in [comparison.cells.json](comparison.cells.json). Method, caveats and how to reproduce: [README.md](README.md).", "",
           "Rows are never compared across hardware classes: each section is one dataset on one hardware class. " +
           "A verdict is WIN or LOSS only when the 95% bootstrap interval of the advantage factor excludes 1 (deterministic byte counts use a 5% band); otherwise TIE. " +
           "The advantage factor is always oriented so that more than 1 means HMS is better. Hosted-runner rows carry noise; trust the ratios, not the absolute values.", "",
           "## Scoreboard", "", "| Dataset | Hardware class | Suite | HMS wins | Ties | Losses | Not measured | Informational (no verdict) |", "|---|---|---|---|---|---|---|---|"]
    for sec, counts, _ in summary:
        out.append(f"| {sec['title']} | `{sec['hw']['class']}` | {sec['data']['suite']} | {counts['WIN']} | {counts['TIE']} | {counts['LOSS']} | {counts['n/m']} | {counts['n/v']} |")
    out += ["", "## Where HMS loses", ""]
    if losses:
        out += ["| Dataset | Metric | Best other | HMS disadvantage | Candidate lever (hypothesis unless a file is cited) |", "|---|---|---|---|---|"]
        for sec, key, label, name, adv in losses:
            out.append(f"| {sec['title']} ({sec['hw']['class']}) | {label} | {registry.SYSTEMS[name]['display']} | {'the comparator has 0' if adv['factor'] == 0 else format(1 / adv['factor'], '.2f') + 'x worse'} | {LEVERS.get(sec['kind'] + ':' + key) or LEVERS.get(key, '')} |")
    else:
        out.append("No metric in the committed results is lost.")
    out += ["", "## Results", ""] + body
    if server_body:
        out += ["## Server mode results", "",
                ("Every system here is a separate server process tree measured from one client over loopback; verdicts compare "
                 "comparators with `hms-server`, never with the embedded rows above. The protocol is recorded in each results file."), ""] + server_body
    out += ["## Operational facts", "", *facts_table(results), "",
                                           "## Versions", "", *versions_table(results), ""]
    nr = {}
    for _, data in results:
        nr.update(data.get("not_run", {}))
    out += ["## Not run", "", *[f"- `{k}`: {v}" for k, v in nr.items()], ""]
    block = readme_block(summary)
    return "\n".join(out), cells, block, gaps_from(summary)


CODE_PATHS = {
    "ann": {"default": "src/core/qgraph/vertex.rs (VGraph search/refine), src/bin/research/public-bench.rs (ann-qgraph)",
            "build_seconds": "src/core/qgraph (Graph::build_owned, VGraph::from_graph_with)",
            "cold_start_s": "no persisted VGraph exists; Graph::save/load only cache the structure (src/core/qgraph)",
            "serving_rss_mib": "src/core/qgraph/vertex.rs (index arrays), build-time allocations in Graph::build_owned"},
    "text": {"default": "src/core/documents.rs (prepare, search), src/core/documents/search.rs, src/core/engine/document_api.rs",
             "build_seconds": "src/core/documents.rs prepare + src/core/engine/document_api.rs memorize_documents",
             "cold_start_s": "src/core/storage.rs and src/core/engine/mod.rs (open: arena and document map load)",
             "serving_rss_mib": "src/core/storage.rs (arena), src/core/documents/index.rs, postings.rs",
             "disk_mib": "src/core/storage.rs (arena preallocation), src/core/documents/postings.rs",
             "replace_ms_per_op": "src/core/engine/mod.rs commit (one durable transaction per call), src/core/durable_file.rs",
             "delete_ms_per_op": "src/core/engine/mod.rs commit (one durable transaction per call), src/core/durable_file.rs"},
}


def gaps_from(summary):
    """Every metric where HMS loses or ties, with the numbers and the best hypothesis."""
    gaps = []
    for sec, _, tally in summary:
        for status in ("LOSS", "TIE"):
            for key, label, name, adv in tally[status]:
                cells = sec["metrics"][key]
                mine, other = cells[reference(sec["data"])], cells[name]
                paths = CODE_PATHS[sec["kind"]]
                gaps.append({
                    "status": status, "dataset": sec["title"], "hardware_class": sec["hw"]["class"], "suite": sec["data"]["suite"],
                    "metric": key, "label": label, "comparator": registry.SYSTEMS[name]["display"],
                    "hms": mine["value"], "comparator_value": other["value"], "unit": mine["unit"],
                    "advantage_factor": adv["factor"] if adv and adv["factor"] != float("inf") else None,
                    "advantage_ci": adv["ci"] if adv else None, "level": sec["level"], "file": rel(sec["path"]),
                    "hypothesis": LEVERS.get(sec["kind"] + ":" + key) or LEVERS.get(key, ""),
                    "code_path": paths.get(key, paths["default"]),
                })
    return gaps


def gaps_markdown(gaps):
    lines = ["# Gaps: where HMS loses or ties", "",
             "Generated with the table from the committed results; a gap stays here until a measured change closes it. " +
             "Factor is HMS advantage (below 1 = HMS worse). Hypotheses are unmeasured unless a file is cited.", "",
             "| Status | Dataset | Hardware class | Metric | Comparator | HMS | Comparator value | Factor [interval] | Hypothesis | Code path |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g in gaps:
        f = "n/m" if g["advantage_factor"] is None else f"{g['advantage_factor']:.2f}x" + (f" [{g['advantage_ci'][0]:.2f}, {g['advantage_ci'][1]:.2f}]" if g["advantage_ci"] else " (no CI)")
        lines.append(f"| {g['status']} | {g['dataset']} | `{g['hardware_class']}` | {g['label']} | {g['comparator']} | {g['hms']:.4g} {g['unit']} | {g['comparator_value']:.4g} | {f} | {g['hypothesis']} | {g['code_path']} |")
    return "\n".join(lines) + "\n"


def readme_block(summary):
    lines = [START, "", "_Generated from `benchmarks/results/competitive/` by `hms-bench table`; method and caveats in [docs/benchmarks/COMPARISON.md](docs/benchmarks/COMPARISON.md). " +
             "W = HMS ahead (95% interval excludes parity), T = tie, L = a comparator is ahead._", "",
             "| Dataset | Metric | HMS | Best other | Advantage | |", "|---|---|---|---|---|---|"]
    for sec, counts, tally in summary:
        if sec["data"]["suite"] == "smoke":
            continue
        spec = {k: (label, unit, dec) for k, label, unit, dec in sec["spec"]}
        short = sec["title"].split(" (")[0]
        for key in HEADLINE[sec["kind"]]:
            cells = sec["metrics"].get(key)
            ref = reference(sec["data"])
            if not cells or ref not in cells:
                continue
            label, unit, dec = spec[key]
            mine = cells[ref]
            name, other = best_other(key, cells)
            adv = advantage(key, mine, other) if other else None
            mark = {"WIN": "W", "TIE": "T", "LOSS": "L", "n/m": "-", "n/v": "-"}["n/v" if key in analyze.NO_VERDICT else verdict(adv) if other else "n/m"]
            best_text = "n/m" if other is None else f"{registry.SYSTEMS[name]['display']} {fmt(other['value'], dec)}"
            lines.append(f"| {short} | {label}{' (' + unit + ')' if unit else ''} | {fmt(mine['value'], dec)} | {best_text} | " +
                         f"{'n/m' if adv is None else fx(adv['factor'])} | {mark} |")
    lines += ["", END]
    return "\n".join(lines) + "\n"


def update_readme(block, path=README):
    if not path.exists():
        return False
    text = path.read_text()
    if START not in text or END not in text:
        return False
    new = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n?", lambda _: block, text, count=1, flags=re.S)
    if new != text:
        path.write_text(new)
    return True


def main(check=False):
    markdown, cells, block, gaps = build()
    index = json.dumps(cells, indent=1, allow_nan=False) + "\n"
    targets = {COMPARISON: markdown + "\n", CELLS: index, BLOCK: block, GAPS_MD: GAPS_FRONT_MATTER + gaps_markdown(gaps), GAPS_JSON: json.dumps(gaps, indent=1, allow_nan=False) + "\n"}
    if check:
        stale = [rel(p) for p, text in targets.items() if not p.exists() or p.read_text() != text]
        if stale:
            print("stale generated files: " + ", ".join(stale), file=sys.stderr)
            return 1
        return 0
    DOCS.mkdir(parents=True, exist_ok=True)
    for p, text in targets.items():
        p.write_text(text)
    update_readme(block)
    print(f"wrote {rel(COMPARISON)}, {rel(CELLS)}, {rel(BLOCK)}")
    return 0
