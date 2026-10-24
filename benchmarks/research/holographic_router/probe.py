# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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

HERE = Path(__file__).resolve().parent


def rows(value, dim):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != dim or not np.isfinite(value).all():
        raise ValueError("expected finite rows with matching dimensions")
    return value


class PhaseSketch:
    def __init__(self, omega, blocks, sigma):
        omega = np.asarray(omega, dtype=np.float64)
        if omega.ndim != 2 or not all(omega.shape) or not np.isfinite(omega).all():
            raise ValueError("invalid frequency matrix")
        if type(blocks) is not int or blocks < 1 or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("invalid block count or bandwidth")
        self.omega = omega
        self.sigma = sigma
        self.cos = np.zeros((blocks, omega.shape[1]))
        self.sin = np.zeros_like(self.cos)
        self.counts = np.zeros(blocks, dtype=np.int64)

    def change(self, block, data, sign=1):
        data = rows(data, self.omega.shape[0])
        if type(block) is not int or not 0 <= block < len(self.counts) or sign not in (-1, 1):
            raise ValueError("invalid block or update sign")
        count = int(self.counts[block]) + sign * len(data)
        if count < 0:
            raise ValueError("negative population")
        phase = data @ self.omega / self.sigma
        self.cos[block] += sign * np.cos(phase).sum(axis=0)
        self.sin[block] += sign * np.sin(phase).sum(axis=0)
        self.counts[block] = count

    def score(self, queries, mean=False):
        queries = rows(queries, self.omega.shape[0])
        phase = queries @ self.omega / self.sigma
        score = (np.cos(phase) @ self.cos.T + np.sin(phase) @ self.sin.T) / self.omega.shape[1]
        if mean:
            score /= np.maximum(self.counts, 1)
        score[:, self.counts == 0] = -np.inf
        return score

    def bytes(self):
        return {"block_headers": self.cos.nbytes + self.sin.nbytes + self.counts.nbytes,
                "shared_projection": self.omega.nbytes}


def squared_l2(left, right):
    return np.maximum((left * left).sum(axis=1)[:, None]
                      + (right * right).sum(axis=1)[None, :] - 2 * left @ right.T, 0)


def ordering(score):
    if not np.isfinite(score).all():
        raise ValueError("nonfinite score in nonempty benchmark blocks")
    return np.argsort(-score, axis=1, kind="stable")


def coverage(order, hits, fanouts):
    ordered = np.take_along_axis(hits, order, axis=1)
    return {str(k): float(ordered[:, :k].sum() / hits.sum()) for k in fanouts}


def timed(function):
    function()
    durations = []
    for _ in range(5):
        begin = time.perf_counter_ns()
        result = function()
        durations.append(time.perf_counter_ns() - begin)
    return result, durations


def generate(scenario, n, seed, dim=32, blocks=8, queries=32):
    rng = np.random.default_rng(seed)
    counts = [n if b % 2 == 0 else n // 4 for b in range(blocks)]
    labels = np.arange(queries) % blocks
    if scenario == "coherent":
        centers = rng.normal(0, 3, (blocks, dim))
        banks = [centers[b] + rng.normal(0, 0.25, (count, dim)) for b, count in enumerate(counts)]
        query = centers[labels] + rng.normal(0, 0.25, (queries, dim))
    elif scenario == "multimodal":
        centers = rng.normal(size=(blocks, dim))
        centers *= 8 / np.linalg.norm(centers, axis=1, keepdims=True)
        banks = [np.concatenate([np.tile(centers[b], (count // 2, 1)),
                                 np.tile(-centers[b], (count - count // 2, 1))])
                 + rng.normal(0, 0.25, (count, dim)) for b, count in enumerate(counts)]
        signs = np.where(np.arange(queries) // blocks % 2 == 0, 1, -1)
        query = signs[:, None] * centers[labels] + rng.normal(0, 0.25, (queries, dim))
    elif scenario == "mixed":
        banks = [rng.normal(size=(count, dim)) for count in counts]
        query = rng.normal(size=(queries, dim))
    elif scenario == "rare":
        query = np.eye(dim)[:queries] * 12
        banks = []
        for b, count in enumerate(counts):
            own = np.flatnonzero(labels == b)
            preceding = np.flatnonzero(labels == (b - 1) % blocks)
            needles = np.repeat(query[own], 10, axis=0) + rng.normal(0, 0.001, (40, dim))
            ids = np.resize(preceding, count - len(needles))
            offsets = rng.normal(size=(len(ids), dim))
            offsets *= 0.75 / np.linalg.norm(offsets, axis=1, keepdims=True)
            clouds = query[ids] + offsets + rng.normal(0, 0.001, (len(ids), dim))
            banks.append(np.concatenate([needles, clouds]))
    else:
        raise ValueError("unknown scenario")
    return banks, query


def record(method, score, hits, fanouts, durations=None, **metadata):
    order = ordering(score)
    return {"method": method, "coverage": coverage(order, hits, fanouts),
            "order": order.tolist(), "query_batch_ns_exploratory": durations,
            **metadata}


def reservoir_scores(queries, representatives):
    return np.column_stack([-squared_l2(queries, block).min(axis=1) for block in representatives])


def run_case(plan, scenario, n, seed):
    banks, query = generate(scenario, n, seed)
    counts = np.array([len(b) for b in banks])
    source = np.concatenate(banks)
    dim, blocks = source.shape[1], len(banks)
    membership = np.repeat(np.arange(blocks), counts)
    distances = squared_l2(query, source)
    top = np.argsort(distances, axis=1, kind="stable")[:, :plan["top_k"]]
    hits = np.array([np.bincount(membership[row], minlength=blocks) for row in top])
    fanouts = plan["fanouts"]
    centers = np.vstack([block.mean(axis=0) for block in banks])
    centroid, elapsed = timed(lambda: -squared_l2(query, centers))
    methods = [record("centroid", centroid, hits, fanouts, elapsed,
                      bytes_total=centers.nbytes, score_fmas_per_query=blocks * dim)]
    random = np.random.default_rng(plan["seeds"]["random_order"] + seed).random((len(query), blocks))
    methods.append(record("random", random, hits, fanouts))
    methods.append(record("oracle_hit_count", hits.astype(float), hits, fanouts))
    offset = np.concatenate([[0], np.cumsum(counts)])
    for sigma in plan["bandwidths"]:
        kernel = np.exp(-distances / (2 * sigma * sigma))
        exact = np.column_stack([kernel[:, offset[b]:offset[b + 1]].sum(axis=1) for b in range(blocks)])
        for mean in (False, True):
            methods.append(record("exact_kernel_mean" if mean else "exact_kernel_sum",
                                  exact / counts if mean else exact, hits, fanouts, bandwidth=sigma))
    for width in plan["phase_widths"]:
        total_bytes = blocks * (2 * width * 8 + 8) + dim * width * 8
        representatives = max(1, (total_bytes // blocks - 8) // (dim * 8))
        rng = np.random.default_rng(plan["seeds"]["reservoir"] + seed)
        sample = [block[rng.choice(len(block), min(representatives, len(block)), replace=False)] for block in banks]
        score, elapsed = timed(lambda: reservoir_scores(query, sample))
        methods.append(record("reservoir", score, hits, fanouts, elapsed,
                              phase_width_budget=width, representatives_per_block=representatives,
                              bytes_total=sum(block.nbytes for block in sample) + blocks * 8,
                              phase_bytes_budget=total_bytes,
                              score_fmas_per_query=sum(len(block) for block in sample) * dim))
    mutation_cost = []
    for projection_seed in plan["seeds"]["projection"]:
        omega_full = np.random.default_rng(projection_seed).normal(size=(dim, max(plan["phase_widths"])))
        for width in plan["phase_widths"]:
            omega = np.ascontiguousarray(omega_full[:, :width])
            for sigma in plan["bandwidths"]:
                sketch = PhaseSketch(omega, blocks, sigma)
                start = time.perf_counter_ns()
                for b, block in enumerate(banks):
                    sketch.change(b, block)
                build_ns = time.perf_counter_ns() - start
                for mean in (False, True):
                    score, elapsed = timed(lambda: sketch.score(query, mean=mean))
                    methods.append(record("phase_mean" if mean else "phase_sum", score, hits, fanouts, elapsed,
                                          width=width, bandwidth=sigma, projection_seed=projection_seed,
                                          bytes=sketch.bytes(), build_ns_exploratory=build_ns,
                                          score_fmas_per_query=dim * width + 2 * blocks * width,
                                          trigonometric_evaluations_per_query=2 * width))
                old, new = banks[0][0:1], banks[0][1:2]
                def replacement():
                    sketch.change(0, old, -1)
                    sketch.change(0, new)
                    sketch.change(0, new, -1)
                    sketch.change(0, old)
                _, samples = timed(replacement)
                mutation_cost.append({"width": width, "bandwidth": sigma, "projection_seed": projection_seed,
                                      "two_replacements_ns_exploratory": samples,
                                      "one_replacement_fmas": 2 * dim * width,
                                      "one_replacement_trigonometric_evaluations": 4 * width})
    return {"scenario": scenario, "nominal_rows": n, "seed": seed, "counts": counts.tolist(),
            "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            "query_sha256": hashlib.sha256(query.tobytes()).hexdigest(), "top10_ids": top.tolist(),
            "block_top10_hits": hits.tolist(), "methods": methods, "mutation_cost": mutation_cost}


def capacity_probe(plan):
    config = plan["capacity_probe"]
    rng = np.random.default_rng(config["seed"])
    samples = {(n, d): [] for n in config["rows"] for d in config["widths"]}
    for _ in range(config["trials"]):
        cos = np.cos(rng.uniform(-np.pi, np.pi, (max(config["rows"]), max(config["widths"]))))
        for n, d in samples:
            samples[(n, d)].append(float(cos[:n, :d].sum() / d))
    return [{"rows": n, "width": d, "noise_samples": values,
             "noise_sd_measured": statistics.stdev(values), "noise_sd_iid_theory": math.sqrt(n / (2 * d)),
             "member_signal": 1.0, "member_snr_iid_theory": math.sqrt(2 * d / n)}
            for (n, d), values in samples.items()]


def decide(cases, plan):
    candidates = []
    for width in plan["phase_widths"]:
        for sigma in plan["bandwidths"]:
            for method in ("phase_sum", "phase_mean"):
                selected = [(case, row) for case in cases for row in case["methods"]
                            if row["method"] == method and row.get("width") == width
                            and row.get("bandwidth") == sigma]
                structured = [row["coverage"]["4"] for case, row in selected if case["scenario"] != "mixed"]
                gains = [row["coverage"]["4"] - next(x["coverage"]["4"] for x in case["methods"] if x["method"] == "centroid")
                         for case, row in selected if case["scenario"] == "multimodal"]
                candidates.append({"width": width, "bandwidth": sigma, "method": method,
                                   "minimum_structured_coverage_at4": min(structured),
                                   "mean_multimodal_gain_over_centroid_at4": statistics.mean(gains),
                                   "eligible_for_separate_development_plan": min(structured) >= 0.9975 and statistics.mean(gains) >= 0.10})
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((HERE / "PLAN.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    signal.alarm(180)
    beginning = time.perf_counter()
    cases = []
    with (args.output / "cases.jsonl").open("w") as stream:
        for scenario in plan["scenarios"]:
            for n in plan["rows_per_block"]:
                for seed in plan["seeds"]["data"]:
                    case = run_case(plan, scenario, n, seed)
                    cases.append(case)
                    stream.write(json.dumps(case, allow_nan=False) + "\n")
                    stream.flush()
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    if platform.system() == "Darwin" and rss > 512 * 1024**2:
                        raise RuntimeError("512MiB resident limit exceeded")
                    print(f"complete {scenario} n={n} seed={seed}", flush=True)
    result = {"plan_sha256": hashlib.sha256((HERE / "PLAN.json").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases_sha256": hashlib.sha256((args.output / "cases.jsonl").read_bytes()).hexdigest(),
              "cases": len(cases), "decisions": decide(cases, plan), "capacity": capacity_probe(plan),
              "elapsed_seconds_exploratory": time.perf_counter() - beginning,
              "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "ru_maxrss_units": "bytes" if platform.system() == "Darwin" else "KiB",
              "platform": {"system": platform.system(), "machine": platform.machine(), "numpy": np.__version__},
              "thread_environment": {name: os.environ.get(name) for name in
                                     ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")},
              "timing_limitation": "Local synthetic exploratory timings under concurrent work; not a measured production speedup",
              "official_data_accessed": False}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(cases), "eligible": sum(row["eligible_for_separate_development_plan"] for row in result["decisions"])}))


if __name__ == "__main__":
    main()
