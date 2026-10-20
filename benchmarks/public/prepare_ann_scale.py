# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = ["numpy==2.2.6"]
# ///
"""Extend the verified 10k database to 100k while preserving its 200 train-only queries."""

import json
from pathlib import Path
import shutil

import numpy as np

from ann_failure_map import ROOT, sha, write


def main():
    previous = ROOT / "benchmarks/results/superiority_v2"
    out = previous / "ann_scale_100k"
    out.mkdir(exist_ok=False)
    write(out / "protocol.json", {
        "purpose": "Scale the candidate/ranking-loss decomposition, not performance or superiority evaluation",
        "database_rows": 100000, "query_rows": 200, "repetitions": 1,
        "extension_seed": 24302, "graph_seed": 24301,
        "comparison": "same 200 queries and original 10000-row database prefix as A1; add90000 uniformly sampled disjoint training IDs",
        "arms": "unchanged eight-arm runner and ef32/64/128/256; feedback remains rejected, retained only to reuse identical decomposition",
        "decision": "If original-pool minus residual16 or exact-traversal minus original-pool exceeds0.005 native recall, investigate that loss; otherwise prioritize topology/routing over ranking precision. Report paired-query intervals and every beam, no timing adoption.",
        "limits": "existing SSH host only; <=8GiB intended experiment RSS, <=4GiB artifacts; no benchmark test or QA data; no paid resources",
        "source_policy": "source train hash and old raw-input hashes must match; source IDs and query bytes preserved",
    })
    for dataset in ("glove-100-angular", "nytimes-256-angular"):
        old = previous / dataset
        manifest = json.loads((old / "inputs.json").read_text())
        for name, expected in manifest["files"].items():
            if sha(old / name) != expected:
                raise ValueError("old diagnostic input changed")
        source = Path.home() / ".cache/hms-bench" / dataset / "train.f32"
        if sha(source) != manifest["source_train_sha256"]:
            raise ValueError("original training corpus changed")
        dim = manifest["dim"]
        data = np.memmap(source, dtype="<f4", mode="r").reshape(-1, dim)
        original = np.load(old / "train_original_ids.npy", allow_pickle=False)
        queries = np.load(old / "queries_original_ids.npy", allow_pickle=False)
        for filename, ids in (("train.f32", original), ("queries.f32", queries)):
            if not np.array_equal(np.fromfile(old / filename, dtype="<f4").reshape(-1, dim), data[ids]):
                raise ValueError("old rows do not match original IDs")
        eligible = np.ones(len(data), dtype=bool)
        eligible[original] = False
        eligible[queries] = False
        additional = np.random.default_rng(24302).choice(np.flatnonzero(eligible), size=90000, replace=False)
        ids = np.concatenate((original, additional))
        target = out / dataset
        target.mkdir()
        np.asarray(data[ids], dtype="<f4").tofile(target / "train.f32")
        np.save(target / "train_original_ids.npy", ids, allow_pickle=False)
        for filename in ("queries.f32", "queries_original_ids.npy"):
            shutil.copyfile(old / filename, target / filename)
        names = ("train.f32", "queries.f32", "train_original_ids.npy", "queries_original_ids.npy")
        write(target / "inputs.json", {**manifest, "database_rows":100000, "repetitions":1,
            "extension_seed":24302, "parent_input_manifest_sha256":sha(old / "inputs.json"),
            "files":{name:sha(target / name) for name in names}})
        print(f"{dataset}:100000 database rows, same200 disjoint training queries", flush=True)


if __name__ == "__main__":
    main()
