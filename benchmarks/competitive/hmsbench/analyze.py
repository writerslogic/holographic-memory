# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pure functions turning one committed results file into per-system metric cells.

A cell is {value, ci, n, source, unit, note}; `source` names the results file and key path the
number was computed from. The table generator and its traceability test both call these
functions, so a table cell can always be recomputed from the committed JSON. `level` is the
confidence level of every interval; the table passes a Bonferroni-adjusted level (the family
is the verdict-bearing metrics of one section).
"""

import statistics

from .common import bootstrap_ci, pass_summary

TARGETS = (0.90, 0.95, 0.99)
HIGHER = "higher"
LOWER = "lower"
# Rows that are informational: the two sides are not the same quantity, so no verdict is drawn.
NO_VERDICT = {"install_distributions", "index_bytes_mib"}


def cell(value, ci=None, n=1, source="", unit="", note=""):
    return {"value": value, "ci": ci if ci is not None else [value, value], "n": n, "source": source, "unit": unit, "note": note}


def _med(values):
    return statistics.median(values)


def pick_config(builds, target):
    """Choose (build_id, config index) by pooled-median QPS among rows whose recall reaches `target`.

    Selection pools every round so one lucky round cannot pick the row; the per-round values of
    the chosen row then carry the confidence interval. Returns (best, closest): `best` is
    (pooled qps, build id, index, recall) or None; `closest` is the highest recall seen.
    """
    best = None
    closest = (-1.0, None)
    for build_id, build in builds.items():
        rounds = build["serve_rounds"]
        if not rounds:
            continue
        for idx in range(len(rounds[0]["configs"])):
            recall = min(r["configs"][idx]["recall_at_10"] for r in rounds if idx < len(r["configs"]))
            if recall > closest[0]:
                closest = (recall, (build_id, idx))
            if recall < target:
                continue
            pooled = _med([x["qps"] for r in rounds for x in r["configs"][idx]["repeats"]])
            if best is None or pooled > best[0]:
                best = (pooled, build_id, idx, recall)
    return best, closest


def _install_cells(put, system, entry, src):
    inst = entry.get("install")
    if not inst:
        return
    if "bytes" in inst:
        # A comparator that ships with the interpreter installs nothing: 0 is not a result HMS can lose to.
        info = bool(inst.get("ships_with_interpreter"))
        note = "ships with the Python interpreter: not an install cost, no verdict drawn against it" if info else ""
        put("install_mib", system, cell(inst["bytes"] / 2**20, None, 1, f"{src}.install", "MiB", note or inst.get("note", "")) | {"informational": info})
        put("install_distributions", system, cell(inst.get("distributions", 0), None, 1, f"{src}.install", "count",
                                                  note or "Python distributions (wheels) the install pulls in") | {"informational": info})
        return
    binary = next(v for k, v in inst.items() if not k.startswith("_"))
    # Files written before the binaries were built with the `dist` profile record no symbols field.
    symbols = binary.get("symbols", "included")
    put("install_mib", system, cell(binary["bytes"] / 2**20, None, 1, f"{src}.install", "MiB", f"one statically linked binary, symbols {symbols}"))
    crates = inst.get("_crate_dependencies", {}).get("count", 0)
    put("install_distributions", system, cell(crates, None, 1, f"{src}.install._crate_dependencies", "count",
                                              "crates compiled into the binary (not separately installed); non-OS shared libraries: "
                                              f"{binary['non_system_dynamic_libraries']}"))


def ann_metrics(path, results, level=0.95):
    """Metric name -> {system -> cell} for an ANN results file."""
    out = {}

    def put(metric, system, c):
        out.setdefault(metric, {})[system] = c

    def ci(values):
        return bootstrap_ci(values, level=level)

    for system, entry in results["systems"].items():
        builds = entry["builds"]
        chosen = {}
        for t in TARGETS:
            best, closest = pick_config(builds, t)
            tag = f"{t:.2f}"
            if best is None:
                put(f"qps@recall{tag}", system, cell(None, source=path, note=f"not reached (max recall {closest[0]:.3f})"))
                continue
            _, build_id, idx, recall = best
            rounds = builds[build_id]["serve_rounds"]
            per_round = [_med([x["qps"] for x in r["configs"][idx]["repeats"]]) for r in rounds]
            src = f"{path}#systems.{system}.builds.{build_id}.serve_rounds[*].configs[{idx}]"
            put(f"qps@recall{tag}", system, cell(_med(per_round), ci(per_round), len(per_round), src, "QPS",
                                                 f"{build_id} {rounds[0]['configs'][idx]['params']} recall {recall:.3f}"))
            if all("p50_us" in x for r in rounds for x in r["configs"][idx]["repeats"]):
                for name in ("p50_us", "p99_us"):
                    vals = [_med([x[name] for x in r["configs"][idx]["repeats"]]) for r in rounds]
                    put(f"{name[:-3]}_latency@recall{tag}", system, cell(_med(vals), ci(vals), len(vals), src, "us"))
            chosen[t] = (build_id, idx)
        build_id = (chosen.get(0.95) or chosen.get(0.90) or (next(iter(builds)), 0))[0]
        build = builds[build_id]
        src = f"{path}#systems.{system}.builds.{build_id}"
        if build["build_runs"]:
            runs, runs_key = build["build_runs"], "build_runs"
            ib = runs[0]["index_bytes"]
            ib_note = "directory size of the persisted index (on disk)"
        else:
            runs, runs_key = [r for r in build["serve_rounds"] if r.get("graph_built")], "serve_rounds[graph_built]"
            ib = build["serve_rounds"][0]["index_bytes"]
            ib_note = "in-memory index_bytes reported by the binary (HMS has no persisted ANN index)"
        bt = [r["build_s"] for r in runs]
        put("build_seconds", system, cell(_med(bt), ci(bt), len(bt), f"{src}.{runs_key}", "s", build_id))
        pk = [r["peak_rss_bytes"] / 2**20 for r in runs]
        put("build_peak_rss_mib", system, cell(_med(pk), ci(pk), len(pk), f"{src}.{runs_key}", "MiB"))
        put("index_bytes_mib", system, cell(ib / 2**20, None, 1, src, "MiB", ib_note))
        steady = [r["rss_steady"] / 2**20 for r in build["serve_rounds"] if r.get("rss_steady")]
        if steady:
            put("serving_rss_mib", system, cell(_med(steady), ci(steady), len(steady), src + ".serve_rounds[*].rss_steady", "MiB",
                                                "fresh serving process"))
        cold = [r["cold_start_s"] for r in build["serve_rounds"] if r.get("cold_start_s") is not None]
        put("cold_start_s", system, cell(_med(cold) if cold else None, ci(cold) if cold else None, len(cold), src + ".serve_rounds[*].cold_start_s", "s",
                                         "" if cold else "not measured"))
        ops = entry.get("ops", {})
        for name in ("replace", "delete"):
            o = ops.get(name)
            if o and o.get("supported"):
                put(f"{name}_ms_per_op", system, cell(o["per_op_ms_durable"], None, 1, f"{path}#systems.{system}.ops.{name}", "ms",
                                                      "includes the explicit save the library requires" if o.get("flush_required") else ""))
            else:
                reason = (o or ops).get("reason") or (o or ops).get("error") or "not supported"
                put(f"{name}_ms_per_op", system, cell(None, None, 0, f"{path}#systems.{system}.ops", "ms", reason))
        _install_cells(put, system, entry, f"{path}#systems.{system}")
    return out


def _passes(serve_round):
    """Pass summaries of a serve round. Files written before summaries were stored carry the raw
    per-query nanoseconds as `passes_ns`; they are summarised the same way on read."""
    if "passes" in serve_round:
        return serve_round["passes"]
    return [pass_summary(ns) for ns in serve_round["passes_ns"]]


def text_metrics(path, results, level=0.95):
    out = {}

    def put(metric, system, c):
        out.setdefault(metric, {})[system] = c

    def ci(values):
        return bootstrap_ci(values, level=level)

    for system, entry in results["systems"].items():
        rounds = entry["serve_rounds"]
        src = f"{path}#systems.{system}"
        q = next((r["quality"] for r in rounds if r.get("quality")), None)
        if q:
            qsrc = src + ".serve_rounds[0].quality"
            per = q["per_query_ndcg_at_10"]
            put("ndcg@10", system, cell(q["ndcg_at_10"], bootstrap_ci(per, statistics.fmean, rounds=1000, level=level), q["queries"], qsrc, "nDCG"))
            put("recall@100", system, cell(q["recall_at_100"], bootstrap_ci(q["per_query_recall_at_100"], statistics.fmean, rounds=1000, level=level),
                                           q["queries"], qsrc, "recall"))
        qps, p50, p99 = [], [], []
        for r in rounds:
            for p in _passes(r):
                qps.append(p["queries"] / (p["total_ns"] / 1e9))
                p50.append(p["p50_ns"] / 1e3)
                p99.append(p["p99_ns"] / 1e3)
        put("qps", system, cell(_med(qps), ci(qps), len(qps), src + ".serve_rounds[*].passes", "QPS", results.get("timing_boundary", {}).get(system, "")))
        put("p50_latency", system, cell(_med(p50), ci(p50), len(p50), src + ".serve_rounds[*].passes", "us"))
        put("p99_latency", system, cell(_med(p99), ci(p99), len(p99), src + ".serve_rounds[*].passes", "us"))
        bt = [b["ingest_s"] for b in entry["build_runs"]]
        put("build_seconds", system, cell(_med(bt), ci(bt), len(bt), src + ".build_runs", "s"))
        pk = [b["peak_rss_bytes"] / 2**20 for b in entry["build_runs"]]
        put("build_peak_rss_mib", system, cell(_med(pk), ci(pk), len(pk), src + ".build_runs", "MiB"))
        sv = [r["peak_rss_bytes"] / 2**20 for r in rounds]
        put("serving_rss_mib", system, cell(_med(sv), ci(sv), len(sv), src + ".serve_rounds[*].peak_rss_bytes", "MiB", "peak RSS of the serving process"))
        put("disk_mib", system, cell(entry["build_runs"][0]["disk_bytes"] / 2**20, None, 1, src + ".build_runs[0].disk_bytes", "MiB", "allocated bytes (du)"))
        cs = [r["cold_start_s"] for r in rounds]
        put("cold_start_s", system, cell(_med(cs), ci(cs), len(cs), src + ".serve_rounds[*].cold_start_s", "s"))
        ops = entry.get("ops", {})
        for name in ("replace", "delete"):
            o = ops.get(name)
            if o and o.get("supported"):
                put(f"{name}_ms_per_op", system, cell(o["per_op_ms_durable"], None, 1, f"{src}.ops.{name}", "ms"))
            else:
                put(f"{name}_ms_per_op", system, cell(None, None, 0, f"{src}.ops", "ms", (o or ops).get("reason") or (o or ops).get("error") or "not supported"))
        _install_cells(put, system, entry, src)
    return out


DIRECTIONS = {"qps": HIGHER, "ndcg@10": HIGHER, "recall@100": HIGHER, **{f"qps@recall{t:.2f}": HIGHER for t in TARGETS}}


def direction(metric):
    return DIRECTIONS.get(metric, LOWER)
