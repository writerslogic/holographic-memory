#!/usr/bin/env python3
"""
HMS Benchmark Visualization
============================
Generates publication-quality figures from HMS scaling and research benchmark data.

Usage:
    uv run python scripts/plot_benchmarks.py

Output:
    figures/capacity_wall.png
    figures/throughput_scaling.png
    figures/noise_tolerance.png
    figures/interference.png
    figures/sequence_encoding.png
    figures/compression.png
"""

import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SCALING_PATH = ROOT / "benchmark_scaling_results.json"
RESEARCH_PATH = ROOT / "research_bench_16384_256.json"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "grid.alpha": 0.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = [
    "#2D7DD2",
    "#F45B69",
    "#97CC04",
    "#EEC643",
    "#8B5CF6",
    "#14B8A6",
    "#F97316",
    "#EC4899",
    "#6366F1",
]

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h", "*"]


def dim_str(d):
    if d >= 1024:
        return f"{d // 1024}K"
    return str(d)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(SCALING_PATH) as f:
    scaling_data = json.load(f)

with open(RESEARCH_PATH) as f:
    research_data = json.load(f)

configs = scaling_data["scaling_benchmark"]


# ---------------------------------------------------------------------------
# Figure 1: Capacity Wall vs Density Denominator
# ---------------------------------------------------------------------------
def fig_capacity_wall():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    dim_groups = {}
    for c in configs:
        dim_groups.setdefault(c["dim"], []).append(c)

    dims_sorted = sorted(dim_groups.keys())
    color_map = {d: COLORS[i] for i, d in enumerate(dims_sorted)}
    marker_map = {d: MARKERS[i] for i, d in enumerate(dims_sorted)}

    all_denom = []
    all_wall = []
    all_dims_for_theory = []

    for dim_val in dims_sorted:
        group = dim_groups[dim_val]
        denoms = [c["density_denom"] for c in group]
        walls = [c["capacity_wall"]["wall_at_95_recall"] for c in group]

        ax.scatter(
            denoms, walls,
            c=color_map[dim_val],
            marker=marker_map[dim_val],
            s=90,
            zorder=5,
            label=f"D = {dim_str(dim_val)}",
            edgecolors="white",
            linewidth=0.6,
        )

        all_denom.extend(denoms)
        all_wall.extend(walls)
        all_dims_for_theory.extend([dim_val] * len(denoms))

    all_denom = np.array(all_denom, dtype=float)
    all_wall = np.array(all_wall, dtype=float)
    all_dims_arr = np.array(all_dims_for_theory, dtype=float)

    predictor = all_denom * np.log(all_dims_arr)
    alpha = np.sum(all_wall * predictor) / np.sum(predictor ** 2)

    for dim_val in dims_sorted:
        denom_range = np.array(sorted(set(c["density_denom"] for c in dim_groups[dim_val])))
        d_min = min(denom_range) * 0.8
        d_max = max(denom_range) * 1.2
        d_line = np.linspace(d_min, d_max, 100)
        theory_wall = alpha * d_line * np.log(dim_val)
        ax.plot(
            d_line, theory_wall,
            color=color_map[dim_val],
            linestyle="--",
            alpha=0.5,
            linewidth=1.2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Density Denominator (1/density)")
    ax.set_ylabel("Capacity Wall (items at 95% recall)")
    ax.set_title("Capacity Wall vs Sparsity")

    ax.annotate(
        f"Theory: wall = {alpha:.2f} $\\times$ denom $\\times$ ln(dim)",
        xy=(0.03, 0.95), xycoords="axes fraction",
        fontsize=9, color="#555555",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.legend(loc="lower right", ncol=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)

    fig.savefig(FIG_DIR / "capacity_wall.png")
    plt.close(fig)
    log.info("  -> %s", FIG_DIR / "capacity_wall.png")


# ---------------------------------------------------------------------------
# Figure 2: Throughput vs Active Indices
# ---------------------------------------------------------------------------
def fig_throughput_scaling():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    active_counts = [c["active_indices"] for c in configs]
    encode_ops = [c["throughput"]["encode_ops_per_sec"] for c in configs]
    dims = [c["dim"] for c in configs]
    denoms = [c["density_denom"] for c in configs]

    dim_set = sorted(set(dims))
    color_map = {d: COLORS[i] for i, d in enumerate(dim_set)}

    for i, c in enumerate(configs):
        color = color_map[c["dim"]]
        ax.scatter(
            active_counts[i], encode_ops[i],
            c=color,
            s=100,
            zorder=5,
            edgecolors="white",
            linewidth=0.6,
            marker=MARKERS[dim_set.index(c["dim"])],
        )
        label_text = f"({dim_str(dims[i])}, 1/{denoms[i]})"
        ax.annotate(
            label_text,
            (active_counts[i], encode_ops[i]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=7,
            color="#555555",
        )

    legend_handles = []
    for d in dim_set:
        h = ax.scatter([], [], c=color_map[d], marker=MARKERS[dim_set.index(d)],
                        s=60, label=f"D = {dim_str(d)}", edgecolors="white", linewidth=0.6)
        legend_handles.append(h)

    ax.set_xlabel("Active Index Count (k)")
    ax.set_ylabel("Encode Throughput (ops/sec)")
    ax.set_title("Encode Throughput vs Active Index Count")
    ax.set_yscale("log")

    active_groups = defaultdict(list)
    for i, c in enumerate(configs):
        active_groups[c["active_indices"]].append(encode_ops[i])

    for k, ops_list in active_groups.items():
        if len(ops_list) > 1:
            ymin, ymax = min(ops_list), max(ops_list)
            ax.plot([k, k], [ymin, ymax], color="#cccccc", linewidth=1.5, zorder=1)

    ax.legend(handles=legend_handles, loc="upper right")

    ax.annotate(
        "Vertical bars show throughput range\nacross dimensions at same k",
        xy=(0.03, 0.05), xycoords="axes fraction",
        fontsize=8, color="#777777",
        verticalalignment="bottom",
    )

    fig.savefig(FIG_DIR / "throughput_scaling.png")
    plt.close(fig)
    log.info("  -> %s", FIG_DIR / "throughput_scaling.png")


# ---------------------------------------------------------------------------
# Figure 3: Noise Tolerance Heatmap
# ---------------------------------------------------------------------------
def fig_noise_tolerance():
    corruption_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    labels = []
    jaccard_matrix = []
    hopfield_matrix = []

    for c in configs:
        label = f"D={dim_str(c['dim'])}\n1/{c['density_denom']}, k={c['active_indices']}"
        labels.append(label)

        noise = c["noise_tolerance"]["results"]
        j_row = [r["jaccard_accuracy"] for r in noise]
        h_row = [r["hopfield_accuracy"] for r in noise]
        jaccard_matrix.append(j_row)
        hopfield_matrix.append(h_row)

    jaccard_matrix = np.array(jaccard_matrix)
    hopfield_matrix = np.array(hopfield_matrix)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    corr_labels = [f"{int(c*100)}%" for c in corruption_levels]

    ax1.imshow(jaccard_matrix, cmap="Greens", vmin=0.0, vmax=1.0, aspect="auto")
    ax1.set_xticks(range(len(corr_labels)))
    ax1.set_xticklabels(corr_labels, fontsize=8)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Corruption Level")
    ax1.set_title("Jaccard Retrieval Accuracy")

    for i in range(len(labels)):
        for j in range(len(corr_labels)):
            val = jaccard_matrix[i, j]
            ax1.text(j, i, f"{val:.0%}", ha="center", va="center",
                     fontsize=7, color="white" if val > 0.5 else "#333333",
                     fontweight="bold")

    ax2.imshow(hopfield_matrix, cmap="Greens", vmin=0.0, vmax=1.0, aspect="auto")
    ax2.set_xticks(range(len(corr_labels)))
    ax2.set_xticklabels(corr_labels, fontsize=8)
    ax2.set_xlabel("Corruption Level")
    ax2.set_title("Hopfield Retrieval Accuracy")

    for i in range(len(labels)):
        for j in range(len(corr_labels)):
            val = hopfield_matrix[i, j]
            ax2.text(j, i, f"{val:.0%}", ha="center", va="center",
                     fontsize=7, color="white" if val > 0.5 else "#333333",
                     fontweight="bold")

    fig.suptitle("Noise Tolerance: Perfect Retrieval Across All Configs and Corruption Levels",
                 fontsize=13, y=1.02)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_tolerance.png")
    plt.close(fig)
    log.info("  -> %s", FIG_DIR / "noise_tolerance.png")


# ---------------------------------------------------------------------------
# Figure 4: Interference -- Individual vs Bundled
# ---------------------------------------------------------------------------
def fig_interference():
    interference = research_data["interference"]
    results = interference["results"]

    n_facts = [r["n_facts"] for r in results]
    individual_acc = [r["individual_accuracy"] for r in results]
    bundled_acc = [r["bundled_accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(
        n_facts, individual_acc,
        color=COLORS[0], marker="o", linewidth=2.0, markersize=7,
        label="Individual Composition",
        zorder=5,
    )
    ax.plot(
        n_facts, bundled_acc,
        color=COLORS[1], marker="s", linewidth=2.0, markersize=7,
        label="Bundled Bloom",
        zorder=5,
    )

    ax.set_xlabel("Number of Facts")
    ax.set_ylabel("Retrieval Accuracy")
    ax.set_title("Interference: Individual Composition vs Bundled Bloom")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xscale("log")

    ax.axhspan(0.95, 1.05, color=COLORS[0], alpha=0.06, zorder=0)
    ax.fill_between(n_facts, bundled_acc, alpha=0.10, color=COLORS[1], zorder=0)

    ax.legend(loc="center right", fontsize=10)

    ax.annotate(
        "Individual composition maintains\nperfect accuracy at all scales",
        xy=(200, 1.0), xytext=(200, 0.80),
        fontsize=9, color=COLORS[0],
        arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=1.2),
        ha="center",
    )
    ax.annotate(
        "Bundled Bloom degrades\nrapidly with fact count",
        xy=(50, 0.05), xytext=(10, 0.45),
        fontsize=9, color=COLORS[1],
        arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.2),
        ha="center",
    )

    fig.savefig(FIG_DIR / "interference.png")
    plt.close(fig)
    log.info("  -> %s", FIG_DIR / "interference.png")


# ---------------------------------------------------------------------------
# Figure 5: Sequence Encoding Accuracy vs Length
# ---------------------------------------------------------------------------
def fig_sequence_encoding():
    seq = research_data["sequence"]
    results = seq["results"]
    vocab_size = seq["vocab_size"]

    lengths = [r["sequence_length"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        lengths, accuracies,
        color=COLORS[0], marker="o", linewidth=2.0, markersize=8,
        zorder=5,
        label=f"EHV (D=16K, vocab={vocab_size})",
    )

    h2h_seq = research_data["head_to_head_vs_hrr"]["sequence_encoding"]
    hrr_lengths = [r["sequence_length"] for r in h2h_seq]
    hrr_acc = [r["hrr_accuracy"] for r in h2h_seq]

    ax.plot(
        hrr_lengths, hrr_acc,
        color=COLORS[1], marker="s", linewidth=2.0, markersize=8,
        linestyle="--",
        zorder=5,
        label="HRR (D=512, dense real)",
    )

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Retrieval Accuracy")
    ax.set_title("Sequence Encoding Accuracy vs Length")
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(0, 210)

    ax.axhline(y=1.0, color="#cccccc", linestyle=":", linewidth=1, zorder=0)

    ax.annotate(
        f"100% accuracy through length 200\n(vocab = {vocab_size})",
        xy=(150, 1.0), xytext=(120, 0.70),
        fontsize=9, color=COLORS[0],
        arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=1.2),
        ha="center",
    )

    ax.legend(loc="center right", fontsize=10)

    fig.savefig(FIG_DIR / "sequence_encoding.png")
    plt.close(fig)
    log.info("  -> %s", FIG_DIR / "sequence_encoding.png")


# ---------------------------------------------------------------------------
# Figure 6: Memory Compression Ratio
# ---------------------------------------------------------------------------
def fig_compression():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    labels = []
    compression_ratios = []
    sparse_bytes = []
    dense_bytes = []

    for c in configs:
        label = f"D={dim_str(c['dim'])}\n1/{c['density_denom']}"
        labels.append(label)
        compression_ratios.append(c["memory"]["compression_ratio"])
        sparse_bytes.append(c["memory"]["bytes_per_sparse_item"])
        dense_bytes.append(c["memory"]["bytes_per_dense_float32"])

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, dense_bytes, width,
           label="Dense float32", color=COLORS[1], alpha=0.85,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, sparse_bytes, width,
           label="Sparse binary", color=COLORS[0], alpha=0.85,
           edgecolor="white", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("Bytes per Item")
    ax.set_xlabel("Configuration")
    ax.set_title("Memory: Sparse Binary vs Dense float32 Representation")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)

    for i, ratio in enumerate(compression_ratios):
        ypos = max(dense_bytes[i], sparse_bytes[i]) * 1.4
        ax.text(x[i], ypos, f"{ratio:.0f}x",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=COLORS[4])

    ax.legend(loc="upper left", fontsize=10)

    ax.annotate(
        "Numbers above bars show compression ratio",
        xy=(0.98, 0.98), xycoords="axes fraction",
        fontsize=8, color="#777777",
        ha="right", va="top",
    )

    fig.savefig(FIG_DIR / "compression.png")
    plt.close(fig)
    log.info("  -> %s", FIG_DIR / "compression.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info("HMS Benchmark Visualization")
    log.info("  Scaling data: %d configurations", len(configs))
    log.info("  Output directory: %s", FIG_DIR)

    log.info("Generating figures:")
    fig_capacity_wall()
    fig_throughput_scaling()
    fig_noise_tolerance()
    fig_interference()
    fig_sequence_encoding()
    fig_compression()

    log.info("Done. All figures saved to figures/")
