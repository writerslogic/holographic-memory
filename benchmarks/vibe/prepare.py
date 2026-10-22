# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare a bounded diagnostic from pinned VIBE embeddings; recompute its truth.

Run in an environment with numpy, h5py, fsspec, aiohttp and faiss-cpu installed.
The corpus is a prefix, not a random sample, and results are not full-scale VIBE results.
Official test queries retain their original distribution; train holdouts never replace them.
"""

import argparse
import hashlib
import json
from pathlib import Path

import faiss
import fsspec
import h5py
import numpy as np


REVISION = "07b387891a221b7b073b83d2f752b76462e5fa03"
DATASETS = {
    "arxiv-nomic-768-normalized": (768, "normalized", False),
    "inaturalist-resnet-2048-cosine": (2048, "cosine", False),
    "laion-clip-512-normalized": (512, "normalized", True),
    "imagenet-align-640-normalized": (640, "normalized", True),
    "agnews-mxbai-1024-euclidean": (1024, "euclidean", False),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(rows):
    value = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norms == 0):
        raise ValueError("nonfinite or zero vector")
    return np.ascontiguousarray(value / norms, dtype=np.float32)


def exact_truth(train, queries, metric):
    if metric == "cosine":
        train, queries = normalized(train), normalized(queries)
    index = faiss.IndexFlatL2(train.shape[1]) if metric == "euclidean" else faiss.IndexFlatIP(train.shape[1])
    index.add(np.ascontiguousarray(train))
    values, ids = index.search(np.ascontiguousarray(queries), 100)
    distances = np.sqrt(np.maximum(values, 0)) if metric == "euclidean" else 1 - values
    return ids, distances


def prepare(args):
    dim, metric, ood = DATASETS[args.dataset]
    if not 100 <= args.rows <= 2_000_000 or not 1 <= args.queries <= 1000:
        raise ValueError("rows must be in 100..2000000 and queries in 1..1000")
    out = args.out / f"{args.dataset}-n{args.rows}.hdf5"
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError(out)
    url = f"https://huggingface.co/datasets/vector-index-bench/vibe/resolve/{REVISION}/{args.dataset}.hdf5"
    source = str(args.source) if args.source else url
    print(f"reading {args.dataset}: {args.rows} corpus rows, {args.queries} original test queries", flush=True)
    with fsspec.open(source, "rb", block_size=8 << 20) as remote:
        with h5py.File(remote, "r") as handle:
            if handle.attrs["dimension"] != dim or handle.attrs["distance"] != metric:
                raise ValueError("source schema changed")
            if handle["train"].shape[0] < args.rows or handle["test"].shape[0] < args.queries:
                raise ValueError("requested prefix exceeds source")
            train = handle["train"][:args.rows].astype(np.float32)
            test = handle["test"][:args.queries].astype(np.float32)
            original_rows = int(handle["train"].shape[0])
            learn = handle["learn"][:128].astype(np.float32) if ood else None
    for rows in (train, test, learn):
        if rows is not None and (rows.shape[1] != dim or not np.isfinite(rows).all()):
            raise ValueError("invalid source vectors")
    if metric == "normalized":
        for rows in (train, test, learn):
            if rows is not None and not np.allclose(np.linalg.norm(rows, axis=1), 1, atol=2e-4):
                raise ValueError("normalized source contains non-unit rows")
    print("recomputing exact truth over the reduced corpus", flush=True)
    faiss.omp_set_num_threads(1)
    ids, distances = exact_truth(train, test, metric)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".part")
    with h5py.File(partial, "w") as handle:
        handle.attrs.update(dimension=dim, distance=metric, point_type="float")
        for name, rows in (("train", train), ("test", test), ("neighbors", ids), ("distances", distances)):
            handle.create_dataset(name, data=rows)
        if learn is not None:
            learn_ids, _ = exact_truth(train, learn, metric)
            handle.create_dataset("learn", data=learn)
            handle.create_dataset("learn_neighbors", data=learn_ids)
    partial.rename(out)
    manifest = {
        "schema_version": 1, "source": url, "source_revision": REVISION,
        "dataset": args.dataset, "original_rows": original_rows, "rows": args.rows,
        "queries": args.queries, "dimension": dim, "metric": metric, "ood": ood,
        "corpus_selection": "prefix", "query_selection": "original test prefix",
        "truth": "recomputed exact float32 FAISS flat; no inherited full-corpus neighbor IDs",
        "official_scale": args.rows == original_rows, "sha256": sha(out),
        "prepare_sha256": sha(Path(__file__)),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args())
