# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dataset preparation. Sources come from benchmarks/public/prepare.py (cached under CACHE);
subsets are cut deterministically and their exact cosine truth recomputed, so every system and
WARM read byte-identical files."""

import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .common import CACHE, log

DATA = CACHE / "competitive" / "data"
ANN_URL = "https://ann-benchmarks.com/{name}.hdf5"
ANN_SETS = {"glove-100-angular": "angular", "sift-128-euclidean": "euclidean"}
SEED = 24731


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 22):
            digest.update(block)
    return digest.hexdigest()


def source_ann(name):
    """Full dataset directory as written by prepare.py (downloads it when missing)."""
    path = CACHE / name
    if not (path / "meta.json").exists():
        url = ANN_URL.format(name=name)
        hdf5 = CACHE / "downloads" / f"{name}.hdf5"
        if not hdf5.exists():
            log(f"downloading {url}")
            hdf5.parent.mkdir(parents=True, exist_ok=True)
            part = hdf5.with_suffix(".part")
            # The ann-benchmarks host rejects urllib's default User-Agent.
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "warm-bench/1.0"})) as response, part.open("wb") as out:
                while block := response.read(1 << 22):
                    out.write(block)
            part.rename(hdf5)
        subprocess.run(["uv", "run", "--no-project", "--python", "3.13", "--with", "h5py==3.16.0", "--with", "numpy==2.5.4", "python", "-I",
                        str(Path(__file__).with_name("h5_convert.py")), str(hdf5), str(path), name, url], check=True)
    return path


def _truth(train, test, metric, width):
    """Exact top-`width` neighbours, chunked so memory stays bounded."""
    out = np.empty((len(test), width), dtype="<i4")
    if metric == "angular":
        train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-30)
        test = test / np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-30)
    sq = (train * train).sum(1)
    for lo in range(0, len(test), 256):
        block = test[lo:lo + 256]
        score = block @ train.T if metric == "angular" else sq[None, :] - 2 * block @ train.T
        part = np.argpartition(-score if metric == "angular" else score, width, axis=1)[:, :width]
        rows = np.take_along_axis(score, part, 1)
        order = np.argsort(-rows if metric == "angular" else rows, axis=1, kind="stable")
        out[lo:lo + 256] = np.take_along_axis(part, order, 1)
    return out


def ann_subset(name, n_train, n_test):
    """Prepared directory with the first `n_train` train rows and `n_test` test rows (None = all)."""
    src = source_ann(name)
    meta = json.loads((src / "meta.json").read_text())
    if (n_train is None or n_train >= meta["n_train"]) and (n_test is None or n_test >= meta["n_test"]):
        return src, meta
    dim = meta["dim"]
    n_train = min(n_train or meta["n_train"], meta["n_train"])
    n_test = min(n_test or meta["n_test"], meta["n_test"])
    out = DATA / f"{name}-n{n_train}-q{n_test}"
    if (out / "meta.json").exists():
        return out, json.loads((out / "meta.json").read_text())
    log(f"cutting {out.name}")
    train = np.fromfile(src / "train.f32", dtype="<f4").reshape(-1, dim)[:n_train]
    test = np.fromfile(src / "test.f32", dtype="<f4").reshape(-1, dim)[:n_test]
    out.mkdir(parents=True, exist_ok=True)
    train.tofile(out / "train.f32")
    test.tofile(out / "test.f32")
    _truth(train, test, meta["metric"], meta["n_neighbors"]).tofile(out / "neighbors.i32")
    sub = {**meta, "n_train": int(n_train), "n_test": int(n_test), "subset_of": name,
           "source_sha256": meta["sha256"], "truth": "exact, recomputed by hmsbench.data",
           "train_sha256": _sha(out / "train.f32"), "test_sha256": _sha(out / "test.f32")}
    (out / "meta.json").write_text(json.dumps(sub, indent=2) + "\n")
    return out, sub


if __name__ == "__main__":
    print(ann_subset(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
