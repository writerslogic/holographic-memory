#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded cost/accuracy audit of the existing Gaussian superposition experiment."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from superposition_floor_coded import block_amp, build_coupled, build_dense


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan_path = Path(__file__).with_name("PLAN.json")
    plan = json.loads(plan_path.read_text())
    files = [plan_path, Path(__file__), ROOT / "scripts/superposition_floor_coded.py"]
    environment = {
        "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "timings": "exploratory-concurrent; not a speed claim",
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    rows = []
    d, m, ell = plan["d"], plan["m"], plan["ell"]
    for seed in plan["seeds"]:
        labels = np.random.default_rng(seed + 10000).integers(0, ell, m)
        noise = np.random.default_rng(seed + 20000).standard_normal(d)
        for method in plan["methods"]:
            started = time.perf_counter()
            if method == "dense":
                problem = build_dense(d, m, ell, seed)
            else:
                width = plan["coupling_width"] if method == "coupled" else 1
                problem = build_coupled(d, m, ell, seed, plan["column_blocks"], width)
            problem["labels"] = labels
            signal = problem["A"][:, labels + ell * np.arange(m)].sum(axis=1)
            problem["s"] = signal
            build_seconds = time.perf_counter() - started
            metadata_bytes = sum(problem[key].nbytes for key in ("W", "row_block", "sec_col"))
            design_bytes = problem["A"].nbytes
            nonzero_bytes = int(np.count_nonzero(problem["A"])) * problem["A"].itemsize
            for sigma in plan["relative_awgn"]:
                query = dict(problem)
                query["s"] = signal + sigma * signal.std() * noise
                started = time.perf_counter()
                estimates, posterior = block_amp(query, n_iter=plan["iterations"])
                seconds = time.perf_counter() - started
                if not np.isfinite(posterior).all():
                    raise RuntimeError("non-finite posterior")
                row = {
                    "seed": seed, "method": method, "relative_awgn": sigma,
                    "d": d, "m": m, "ell": ell, "accuracy": float(np.mean(estimates == labels)),
                    "true_posterior_mean": float(posterior[np.arange(m), labels].mean()),
                    "design_bytes": design_bytes, "signal_bytes": signal.nbytes,
                    "decoder_metadata_bytes": metadata_bytes,
                    "persistent_array_bytes": design_bytes + signal.nbytes + metadata_bytes,
                    "nonzero_design_payload_bytes": nonzero_bytes,
                    "returned_posterior_bytes": posterior.nbytes,
                    "exact_labels_bits": m * (ell.bit_length() - 1),
                    "build_seconds": build_seconds, "decode_seconds": seconds,
                }
                rows.append(row)
                with (args.output / "cases.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"seed={seed} method={method} noise={sigma} accuracy={row['accuracy']:.4f}", flush=True)
    summary = []
    for method in plan["methods"]:
        for sigma in plan["relative_awgn"]:
            selected = [row for row in rows if row["method"] == method and row["relative_awgn"] == sigma]
            summary.append({
                "method": method, "relative_awgn": sigma,
                "mean_accuracy": statistics.mean(row["accuracy"] for row in selected),
                "min_accuracy": min(row["accuracy"] for row in selected),
                "max_accuracy": max(row["accuracy"] for row in selected),
                "persistent_array_bytes": selected[0]["persistent_array_bytes"],
                "nonzero_design_payload_bytes": selected[0]["nonzero_design_payload_bytes"],
            })
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
