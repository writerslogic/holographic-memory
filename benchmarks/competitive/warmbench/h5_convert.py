# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""ann-benchmarks HDF5 -> the flat layout public-bench reads (same output as
benchmarks/public/prepare.py, without its sentence-transformers dependency).

Usage (inside an environment with h5py and numpy): h5_convert.py <in.hdf5> <out dir> <name> <url>
"""

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

src, out, name, url = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
with h5py.File(src, "r") as f:
    train = np.asarray(f["train"], dtype="<f4")
    test = np.asarray(f["test"], dtype="<f4")
    neighbors = np.asarray(f["neighbors"], dtype="<i4")
    metric = str(f.attrs.get("distance", "unknown"))
train.tofile(out / "train.f32")
test.tofile(out / "test.f32")
neighbors.tofile(out / "neighbors.i32")
digest = hashlib.sha256()
with src.open("rb") as handle:
    while block := handle.read(1 << 22):
        digest.update(block)
(out / "meta.json").write_text(json.dumps({
    "kind": "ann", "name": name, "source": url, "sha256": digest.hexdigest(), "metric": metric,
    "n_train": int(train.shape[0]), "n_test": int(test.shape[0]), "dim": int(train.shape[1]), "n_neighbors": int(neighbors.shape[1]),
}, indent=2))
