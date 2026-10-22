# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Plot saved frozen points and export observed query-latency percentiles.

uv run --no-project --with matplotlib --with h5py --with numpy benchmarks/vibe/qag_plot.py
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results/query_aware_graph_v1")
    args = parser.parse_args()
    root = args.evidence_dir.resolve(strict=True)
    report = json.loads((root / "pareto.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True, constrained_layout=True)
    latencies = {"scope": "Observed saved fifth-pass query times; no confidence intervals or unmeasured interpolation", "rows": []}
    colors = {"baseline4": "#8b8b8b", "candidate8": "#006d77", "roargraph48": "#dc7f27",
              "roargraph96": "#b44a00", "mlann_rf10_depth14": "#5e548e", "mlann_rf80_depth13": "#a145aa"}
    handles = {}
    for column, dataset in enumerate(("llama-128-ip", "yi-128-ip")):
        for row, phase in enumerate(("official", "validation")):
            data = report["phases"][phase].get(dataset)
            ax = axes[row, column]
            if data is None:
                ax.set_title(f"{dataset} — {phase}: unavailable")
                continue
            for method, points in data["frontier_by_method"].items():
                points = sorted(points, key=lambda x: x["recall_ids_min"])
                artist = ax.errorbar(
                    [p["recall_ids_min"] for p in points], [p["qps_median"] for p in points],
                    yerr=[[p["qps_median"] - p["qps_min"] for p in points],
                          [p["qps_max"] - p["qps_median"] for p in points]],
                    fmt="o", markersize=4, capsize=3, color=colors.get(method), label=method,
                )
                handles[method] = artist
            for target in (.90, .95, .99):
                ax.axvline(target, color="#c9c9c9", linewidth=.8, linestyle="--", zorder=0)
            minimum = min(p["recall_ids_min"] for p in data["rows"])
            ax.set_xlim(max(0, minimum - .025), 1.003)
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=.2)
            ax.set_title(f"{dataset} — {phase}")
            ax.set_xlabel("Actual strict recall@100")
            ax.set_ylabel("QPS, median [observed min–max]")
            for point in data["rows"]:
                source = root / point["source_report"]
                original = json.loads(source.read_text())
                matching = [p for p in original["rows"] if p["arm"] == point["arm"]
                            and p["ef"] == point["ef"] and p["rerank"] == point["rerank"]
                            and p.get("patience", 0) == point.get("patience", 0)]
                if len(matching) != 1:
                    raise ValueError("ambiguous raw latency configuration")
                samples = []
                with h5py.File(source.parent / "raw.hdf5", "r") as raw:
                    for repetition in matching[0]["repetitions"]:
                        times = np.asarray(raw[repetition["raw_group"]]["times"][:], dtype=np.float64)
                        measured_qps = len(times) / times.sum(dtype=np.float64)
                        if not np.isclose(measured_qps, repetition["qps"], rtol=1e-12):
                            raise ValueError("saved latency times disagree with reported QPS")
                        quantiles = np.percentile(times * 1e6, [50, 90, 95, 99, 100])
                        samples.append({"repetition": repetition["repetition"], "queries": len(times),
                                        "microseconds": dict(zip(("p50", "p90", "p95", "p99", "max"), quantiles.tolist()))})
                latencies["rows"].append({"phase": phase, "dataset": dataset, "method": point["method"],
                                           "configuration": point["configuration"], "repetitions": samples})
    fig.suptitle("Frozen v1: measured Pareto points (markers only; no interpolation)\nCore Ultra 9 185H · Linux/WSL2 · one thread · three paired blocks · saved fifth pass", fontsize=12)
    fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(root / "pareto.svg")
    plt.close(fig)
    (root / "latency.json").write_text(json.dumps(latencies, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
