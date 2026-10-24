# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import time

import numpy as np

from probe import PhaseSketch, generate, ordering, record, reservoir_scores, squared_l2, timed

HERE = Path(__file__).resolve().parent


def witnesses(data, count):
    if type(count) is not int or not 0 <= count <= len(data):
        raise ValueError("invalid witness count")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    center = data.mean(axis=0)
    first = int(np.argmin(((data - center) ** 2).sum(axis=1)))
    selected = []
    nearest = np.full(len(data), np.inf)
    current = first
    for _ in range(count):
        selected.append(current)
        distance = ((data - data[current]) ** 2).sum(axis=1)
        nearest = np.minimum(nearest, distance)
        nearest[selected] = -np.inf
        current = int(np.argmax(nearest))
    return np.array(selected)


def hybrid_order(witness_score, phase_score, fanout):
    if witness_score.shape != phase_score.shape or not 0 <= fanout <= witness_score.shape[1]:
        raise ValueError("invalid hybrid score geometry or fanout")
    witness_rank = ordering(witness_score)
    phase_rank = ordering(phase_score)
    result = []
    for first, second in zip(witness_rank, phase_rank):
        chosen = first[:math.ceil(fanout / 2)].tolist()
        for block in second:
            if len(chosen) == fanout:
                break
            if int(block) not in chosen:
                chosen.append(int(block))
        result.append(chosen)
    return np.array(result, dtype=np.int64).reshape(len(witness_rank), fanout)


def interior_case(n, seed):
    rng = np.random.default_rng(seed)
    queries = np.eye(32) * 12
    labels = np.arange(32) % 8
    counts = [n if b % 2 == 0 else n // 4 for b in range(8)]
    banks = []
    for block, count in enumerate(counts):
        own = np.flatnonzero(labels == block)
        preceding = np.flatnonzero(labels == (block - 1) % 8)
        needles = np.repeat(queries[own], 10, axis=0)
        ring_count = (count - 40) // 2
        rings = []
        for j, identifier in enumerate(own):
            size = ring_count // 4 + int(j < ring_count % 4)
            directions = rng.normal(size=(math.ceil(size / 2), 32))
            directions *= 3 / np.linalg.norm(directions, axis=1, keepdims=True)
            pairs = np.stack([directions, -directions], axis=1).reshape(-1, 32)[:size]
            rings.append(queries[identifier] + pairs)
        cloud_count = count - 40 - ring_count
        ids = np.resize(preceding, cloud_count)
        offset = rng.normal(size=(cloud_count, 32))
        offset *= 0.75 / np.linalg.norm(offset, axis=1, keepdims=True)
        banks.append(np.concatenate([needles, *rings, queries[ids] + offset]))
    return banks, queries


def make_case(scenario, n, seed):
    return interior_case(n, seed) if scenario == "interior_rare" else generate(scenario, n, seed)


def run_case(plan, scenario, n, seed):
    banks, queries = make_case(scenario, n, seed)
    source = np.concatenate(banks)
    counts = [len(bank) for bank in banks]
    labels = np.repeat(np.arange(8), counts)
    top = np.argsort(squared_l2(queries, source), axis=1, kind="stable")[:, :10]
    hits = np.array([np.bincount(labels[row], minlength=8) for row in top])
    if plan["phase_bytes"]["total"] >= source.nbytes:
        return {"scenario": scenario, "nominal_rows": n, "seed": seed,
                "skipped": "phase summary dominates raw vector storage", "raw_bytes": source.nbytes}
    begin = time.perf_counter_ns()
    witness_ids = [witnesses(bank, 48) for bank in banks]
    witness_build_ns = time.perf_counter_ns() - begin
    full = [bank[ids] for bank, ids in zip(banks, witness_ids)]
    small = [bank[ids[:36]] for bank, ids in zip(banks, witness_ids)]
    rng = np.random.default_rng(plan["reservoir_seed"] + seed)
    reservoir_ids = [rng.choice(len(bank), 48, replace=False) for bank in banks]
    reservoir = [bank[ids] for bank, ids in zip(banks, reservoir_ids)]
    center = np.vstack([bank.mean(axis=0) for bank in banks])
    methods = [record("oracle_hit_count", hits.astype(float), hits, plan["fanouts"])]
    for method, sample in [("geometry48", full), ("geometry36", small), ("reservoir48", reservoir)]:
        score, samples = timed(lambda: reservoir_scores(queries, sample))
        methods.append(record(method, score, hits, plan["fanouts"], samples,
                              bytes_total=sum(row.nbytes for row in sample),
                              score_fmas_per_query=sum(len(row) for row in sample) * 32))
    score, samples = timed(lambda: -squared_l2(queries, center))
    methods.append(record("centroid", score, hits, plan["fanouts"], samples, bytes_total=center.nbytes))
    for projection_seed in plan["projection_seeds"]:
        omega = np.random.default_rng(projection_seed).normal(size=(32, 64))
        sketch = PhaseSketch(omega, 8, 1.5)
        for block, bank in enumerate(banks):
            sketch.change(block, bank)
        phase_score, samples = timed(lambda: sketch.score(queries))
        methods.append(record("phase64", phase_score, hits, plan["fanouts"], samples,
                              projection_seed=projection_seed, bytes_total=sum(sketch.bytes().values())))
        def hybrid_scores():
            return reservoir_scores(queries, small), sketch.score(queries)
        (witness_score, phase_score), samples = timed(hybrid_scores)
        choices = {str(f): hybrid_order(witness_score, phase_score, f) for f in plan["fanouts"]}
        served_bytes = sum(sketch.bytes().values()) + sum(bank.nbytes for bank in small)
        assert served_bytes == plan["hybrid_total_bytes"] <= plan["total_budget_bytes"]
        cov = {str(f): float(np.take_along_axis(hits, choices[str(f)], axis=1).sum() / 320)
               for f in plan["fanouts"]}
        methods.append({"method": "hybrid", "projection_seed": projection_seed,
                        "coverage": cov, "selected_by_fanout": {key: value.tolist() for key, value in choices.items()},
                        "query_batch_ns_exploratory": samples, "bytes_total": served_bytes,
                        "score_fmas_per_query": 12288, "trigonometric_evaluations_per_query": 128})
    rarity = []
    if scenario == "interior_rare":
        for block, bank in enumerate(banks):
            distances = ((bank - bank.mean(axis=0)) ** 2).sum(axis=1)
            rarity.append({"block": block,
                           "needle_centroid_distance_percentiles": [float(np.mean(distances <= d)) for d in distances[:40]],
                           "needle_coordinates_selected_geometry36": int(np.sum(witness_ids[block][:36] < 40)),
                           "needle_coordinates_selected_geometry48": int(np.sum(witness_ids[block] < 40))})
    return {"scenario": scenario, "nominal_rows": n, "seed": seed, "counts": counts,
            "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            "query_sha256": hashlib.sha256(queries.tobytes()).hexdigest(), "top10_ids": top.tolist(),
            "block_top10_hits": hits.tolist(), "geometry_witness_ids": [ids.tolist() for ids in witness_ids],
            "reservoir_ids": [ids.tolist() for ids in reservoir_ids], "raw_bytes": source.nbytes,
            "phase_to_raw_bytes_ratio": plan["phase_bytes"]["total"] / source.nbytes,
            "witness_build_ns_exploratory": witness_build_ns, "interior_rarity": rarity, "methods": methods}


def decision(cases):
    differences = []
    structured = []
    by_scenario = {}
    for case in cases:
        if "skipped" in case:
            return {"eligible": False, "reason": "registered budget stop"}
        controls = {r["method"]: r["coverage"]["4"] for r in case["methods"] if r["method"] in ("geometry48", "reservoir48")}
        for row in case["methods"]:
            if row["method"] != "hybrid":
                continue
            value = row["coverage"]["4"]
            difference = value - max(controls.values())
            differences.append(difference)
            by_scenario.setdefault(case["scenario"], []).append(difference)
            if case["scenario"] != "mixed":
                structured.append(value)
    gains = {key: statistics.mean(values) for key, values in by_scenario.items()}
    return {"minimum_structured_coverage_at4": min(structured),
            "worst_delta_vs_better_control_at4": min(differences),
            "mean_delta_vs_better_control_by_scenario": gains,
            "eligible": min(structured) >= .9975 and min(differences) >= -.0025 and max(gains.values()) >= .02}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((HERE / "hybrid_PLAN.json").read_text())
    signal.alarm(plan["resource_bounds"]["deadline_seconds"])
    cases = []
    start = time.perf_counter()
    with (args.output / "cases.jsonl").open("w") as stream:
        for scenario in plan["scenarios"]:
            for n in plan["nominal_rows"]:
                for seed in plan["data_seeds"]:
                    case = run_case(plan, scenario, n, seed)
                    cases.append(case)
                    stream.write(json.dumps(case, allow_nan=False) + "\n")
                    stream.flush()
                    if platform.system() == "Darwin" and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > plan["resource_bounds"]["resident_bytes"]:
                        raise RuntimeError("registered resident limit exceeded")
                    print(f"complete {scenario} n={n} seed={seed}", flush=True)
    result = {"plan_sha256": hashlib.sha256((HERE / "hybrid_PLAN.json").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "base_probe_sha256": hashlib.sha256((HERE / "probe.py").read_bytes()).hexdigest(),
              "cases_sha256": hashlib.sha256((args.output / "cases.jsonl").read_bytes()).hexdigest(),
              "cases": len(cases), "decision": decision(cases),
              "elapsed_seconds_exploratory": time.perf_counter() - start,
              "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "ru_maxrss_units": "bytes" if platform.system() == "Darwin" else "KiB",
              "numpy": np.__version__, "official_data_accessed": False,
              "timing_scope": "Scoring only; block sorting excluded from all methods; local exploratory timing",
              "thread_environment": {name: os.environ.get(name) for name in
                                     ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")}}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["decision"]))


if __name__ == "__main__":
    main()
