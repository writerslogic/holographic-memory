# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = ["numpy==2.2.6"]
# ///
"""Train-only ANN failure decomposition; never opens benchmark test files."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/superiority_v2"
ARMS = ("residual16", "original16", "original64", "original_pool",
        "exact_traversal", "feedback_union", "double_ef", "flat")


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.rs"))
    paths.extend(ROOT / name for name in ("Cargo.toml", "Cargo.lock", "build.rs"))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths if path.is_file()}


def build(binary):
    before = source_hashes()
    command = ["cargo", "build", "--locked", "--release", "--features", "experimental", "--bin", "ann-failure-map"]
    with (OUT / "build-verified.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if source_hashes() != before:
        raise ValueError("source changed during build")
    write(OUT / "build_manifest.json", {"command": command, "source_sha256": before,
        "binary_sha256": sha(binary), "rustc": subprocess.check_output(["rustc", "-vV"], text=True),
        "cargo": subprocess.check_output(["cargo", "--version"], text=True),
        "RUSTFLAGS": os.environ.get("RUSTFLAGS"), "CARGO_ENCODED_RUSTFLAGS": os.environ.get("CARGO_ENCODED_RUSTFLAGS"),
        "cargo_metadata": json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"], text=True))})


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(dataset):
    destination = OUT / dataset
    destination.mkdir(exist_ok=True)
    manifest = destination / "inputs.json"
    if manifest.exists():
        info = json.loads(manifest.read_text())
        for name, digest in info["files"].items():
            if sha(destination / name) != digest:
                raise ValueError("existing input hash mismatch")
        return destination
    source = Path.home() / ".cache/hms-bench" / dataset
    meta = json.loads((source / "meta.json").read_text())
    data = np.memmap(source / "train.f32", dtype="<f4", mode="r").reshape(-1, meta["dim"])
    rng = np.random.default_rng(24301)
    ids = rng.choice(len(data), size=10200, replace=False)
    # Only nonzero training rows can become cosine queries; preserve zero database rows.
    nonzero = np.linalg.norm(data[ids].astype(np.float64), axis=1) > 0
    query_ids = ids[nonzero][-200:]
    query_set = set(query_ids.tolist())
    train_ids = np.array([i for i in ids if i not in query_set], dtype=np.int64)
    if len(query_ids) != 200 or len(train_ids) != 10000:
        raise ValueError("sample cannot supply preregistered split")
    for name, selected in (("train", train_ids), ("queries", query_ids)):
        np.asarray(data[selected], dtype="<f4").tofile(destination / f"{name}.f32")
        np.save(destination / f"{name}_original_ids.npy", selected, allow_pickle=False)
    names = ["train.f32", "queries.f32", "train_original_ids.npy", "queries_original_ids.npy"]
    write(manifest, {"dataset": dataset, "dim": meta["dim"], "seed": 24301,
                     "source_train_sha256": sha(source / "train.f32"),
                     "source_meta": meta, "benchmark_test_opened": False,
                     "files": {name: sha(destination / name) for name in names}})
    return destination


def run(dataset, binary):
    build_manifest = json.loads((OUT / "build_manifest.json").read_text())
    if source_hashes() != build_manifest["source_sha256"] or sha(binary) != build_manifest["binary_sha256"]:
        raise ValueError("executable/source differs from verified build")
    destination = prepare(dataset)
    manifest = json.loads((destination / "inputs.json").read_text())
    raw = destination / "raw.json"
    if raw.exists() or raw.with_suffix(".json.gz").exists():
        raise ValueError("refusing to overwrite an existing experiment")
    command = [str(binary.resolve()), "--data", str(destination), "--dim", str(manifest["dim"]),
               "--out", str(raw), "--repeats", str(manifest.get("repetitions", 3))]
    before = os.getloadavg()
    start = time.monotonic()
    with (destination / "run.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    sources = [ROOT / "src/bin/ann-failure-map.rs", ROOT / "src/core/qgraph/diagnostics.rs",
               ROOT / "src/core/qgraph/vertex.rs", ROOT / "src/core/qgraph/mod.rs",
               ROOT / "src/core/qgraph/kernels.rs", Path(__file__)]
    write(destination / "run.json", {"command": command, "exit_code": result.returncode,
        "elapsed_seconds": time.monotonic() - start, "load_before": before,
        "load_after": os.getloadavg(), "machine": platform.machine(),
        "platform": platform.platform(), "binary_sha256": sha(binary),
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "timing_gate_met": False, "timing_scope": "diagnostic under shared-host load, no exclusive timing lock",
        "python": platform.python_version(), "numpy": np.__version__})
    result.check_returncode()
    if source_hashes() != build_manifest["source_sha256"] or sha(binary) != build_manifest["binary_sha256"]:
        raise ValueError("source/executable changed during experiment")
    with raw.open("rb") as source, gzip.open(raw.with_suffix(".json.gz"), "wb") as target:
        while block := source.read(1 << 20):
            target.write(block)
    raw.unlink()


def interval(values):
    rng = np.random.default_rng(9913)
    resamples = rng.integers(0, len(values), size=(5000, len(values)))
    return np.quantile(np.mean(values[resamples], axis=1), [0.025, 0.975]).tolist()


def check(dataset):
    destination = OUT / dataset
    manifest = json.loads((destination / "inputs.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(destination / name) != digest:
            raise ValueError("input hash mismatch")
    train_ids = np.load(destination / "train_original_ids.npy", allow_pickle=False)
    query_ids = np.load(destination / "queries_original_ids.npy", allow_pickle=False)
    n = manifest.get("database_rows", 10000)
    repetitions = manifest.get("repetitions", 3)
    if type(n) is not int or type(repetitions) is not int or n not in (10000, 100000) or repetitions not in (1, 3):
        raise ValueError("shape outside frozen diagnostic budgets")
    if len(set(train_ids) & set(query_ids)) or len(set(train_ids)) != n or len(set(query_ids)) != 200:
        raise ValueError("split leakage or duplicate IDs")
    dim = manifest["dim"]
    train = np.fromfile(destination / "train.f32", dtype="<f4").reshape(-1, dim).astype(np.float64)
    queries = np.fromfile(destination / "queries.f32", dtype="<f4").reshape(-1, dim).astype(np.float64)
    if not np.isfinite(train).all() or not np.isfinite(queries).all():
        raise ValueError("nonfinite diagnostic input")
    norms = np.linalg.norm(train, axis=1)
    train /= np.where(norms > 0, norms, 1)[:, None]
    query_norms = np.linalg.norm(queries, axis=1)
    if np.any(query_norms == 0):
        raise ValueError("zero diagnostic query")
    queries /= query_norms[:, None]
    similarities = np.einsum("qd,nd->qn", queries, train, optimize=False)
    if not np.isfinite(similarities).all() or np.max(np.abs(similarities)) > 1 + 1e-12:
        raise ValueError("invalid independent cosine oracle")
    truth = np.argsort(-similarities, axis=1, kind="stable")[:, :10]
    threshold = np.take_along_axis(similarities, truth[:, -1:], axis=1)[:, 0]
    with gzip.open(destination / "raw.json.gz", "rt") as handle:
        raw = json.load(handle)
    if train.shape != (n, dim) or queries.shape != (200, dim) or (raw["n"], raw["dim"], raw["queries"]) != (n, dim, 200):
        raise ValueError("unexpected experiment shape")
    grouped = {(ef, arm): [] for ef in (32, 64, 128, 256) for arm in ARMS}
    seen, stable = set(), {}
    for row in raw["records"]:
        ef, arm, qi, rep = (row[key] for key in ("ef", "arm", "query", "repetition"))
        key = ef, arm, qi, rep
        if key in seen or (ef, arm) not in grouped or not 0 <= qi < 200 or not 0 <= rep < repetitions:
            raise ValueError("duplicate or invalid result key")
        seen.add(key)
        ids = row["ids"]
        if len(ids) != 10 or len(set(ids)) != 10 or any(type(i) is not int or not 0 <= i < n for i in ids):
            raise ValueError("invalid prediction IDs")
        if row["elapsed_ns"] <= 0 or row["evaluations"] <= 0:
            raise ValueError("invalid timing or evaluation count")
        if stable.setdefault(key[:3], ids) != ids:
            raise ValueError("nondeterministic predictions")
        native = float(np.mean(similarities[qi, ids] >= threshold[qi] - 0.001))
        strict = float(np.mean(similarities[qi, ids] >= threshold[qi] - 1e-6))
        overlap = len(set(ids) & set(truth[qi])) / 10
        if arm == "flat" and strict != 1:
            raise ValueError("independent float64 truth disagrees with flat oracle")
        coverage = None
        if arm in ("original16", "original64", "original_pool", "feedback_union", "double_ef"):
            pool = row["pool_ids"]
            expected = {"original16": 16, "original64": min(64, ef), "original_pool": ef, "double_ef": 2 * ef}.get(arm)
            if (expected is not None and len(pool) != expected) or len(set(pool)) != len(pool) or any(type(i) is not int or not 0 <= i < n for i in pool):
                raise ValueError("invalid pool")
            if arm == "feedback_union" and not ef <= len(pool) <= 2 * ef:
                raise ValueError("invalid union size")
            if not set(ids) <= set(pool):
                raise ValueError("prediction outside candidate pool")
            coverage = len(set(pool) & set(truth[qi])) / 10
            # Near-tied original vectors may change order after Graph's f32 normalization.
            selected_scores = sorted(similarities[qi, pool], reverse=True)[:10]
            if not np.allclose(sorted(similarities[qi, ids], reverse=True), selected_scores, rtol=0, atol=1e-6):
                raise ValueError("exact reranking fails candidate-pool oracle")
        grouped[ef, arm].append({"query": qi, "rep": rep, "native": native, "strict": strict,
                                "overlap": overlap, "coverage": coverage, **row})
    if len(seen) != 4 * len(ARMS) * 200 * repetitions:
        raise ValueError("missing experiment rows")
    by_key = {(r["ef"], r["arm"], r["query"], r["repetition"]): r for r in raw["records"]}
    for ef in (32, 64, 128, 256):
        for qi in range(200):
            for rep in range(repetitions):
                pool = by_key[ef, "original_pool", qi, rep]["pool_ids"]
                for arm, limit in (("original16", 16), ("original64", 64)):
                    if by_key[ef, arm, qi, rep]["pool_ids"] != pool[:limit]:
                        raise ValueError("prefix arm changed traversal pool")
                union = by_key[ef, "feedback_union", qi, rep]["pool_ids"]
                if not set(pool) <= set(union):
                    raise ValueError("feedback dropped original probe candidates")
                if ef < 256 and by_key[ef, "double_ef", qi, rep]["pool_ids"] != by_key[2 * ef, "original_pool", qi, rep]["pool_ids"]:
                    raise ValueError("wider control differs from same-width single probe")
    summary, contrasts = [], []
    for (ef, arm), rows in grouped.items():
        unique = sorted((r for r in rows if r["rep"] == 0), key=lambda r: r["query"])
        scores = np.array([r["overlap"] for r in unique])
        nanos = np.array([r["elapsed_ns"] for r in rows])
        summary.append({"ef": ef, "arm": arm, "id_overlap": float(scores.mean()),
            "id_overlap_ci95_exploratory": interval(scores),
            "native_recall_tolerance_0_001": float(np.mean([r["native"] for r in unique])),
            "strict_recall_tolerance_1e_6": float(np.mean([r["strict"] for r in unique])),
            "pool_id_coverage": float(np.mean([r["coverage"] for r in unique])) if arm == "original_pool" else None,
            "latency_us_p50_p95_p99": (np.quantile(nanos, [.5, .95, .99]) / 1000).tolist(),
            "mean_evaluations": float(np.mean([r["evaluations"] for r in rows])),
            "timing_claim_eligible": False})
    for ef in (32, 64, 128, 256):
        for a, b in (("original16", "residual16"), ("original_pool", "original16"),
                     ("exact_traversal", "original_pool"), ("feedback_union", "double_ef")):
            def quality(arm, metric):
                return np.array([r[metric] for r in sorted(grouped[ef, arm], key=lambda r: (r["rep"], r["query"])) if r["rep"] == 0])
            delta = quality(a, "overlap") - quality(b, "overlap")
            native_delta = quality(a, "native") - quality(b, "native")
            contrasts.append({"ef": ef, "a": a, "b": b, "delta_id_overlap": float(delta.mean()),
                              "paired_query_ci95_exploratory": interval(delta),
                              "delta_native_recall": float(native_delta.mean()),
                              "paired_native_ci95_exploratory": interval(native_delta)})
    write(destination / "summary.json", {"independent_truth_checked": True, "split_leakage": False,
        "rows_checked": len(seen), "benchmark_test_opened": False, "not_superiority_evidence": True,
        "raw_sha256": sha(destination / "raw.json.gz"), "checker_sha256": sha(Path(__file__)),
        "summary": summary, "contrasts": contrasts,
        "graph_build_seconds": raw["graph_build_seconds"], "encoding_seconds": raw["encoding_seconds"],
        "compact_index_bytes": raw["compact_index_bytes"], "original_vector_bytes": raw["original_vector_bytes"]})
    print(json.dumps({"dataset": dataset, "rows_checked": len(seen), "contrasts": contrasts}, indent=2))


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "check", "build"))
    parser.add_argument("dataset", choices=("glove-100-angular", "nytimes-256-angular"))
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/ann-failure-map")
    parser.add_argument("--out-root", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out_root.resolve()
    if args.action == "build":
        build(args.binary)
    elif args.action == "run":
        run(args.dataset, args.binary)
    elif args.action == "prepare":
        prepare(args.dataset)
    else:
        check(args.dataset)


if __name__ == "__main__":
    main()
