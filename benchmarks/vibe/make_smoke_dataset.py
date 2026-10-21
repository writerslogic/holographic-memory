# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Write a VIBE-format cosine dataset from an ann-benchmarks HDF5 file (exact ground truth).

VIBE ships no small dataset; this makes one for a correctness smoke of the wrapper:
    python make_smoke_dataset.py random-xs-20-angular.hdf5 data/random-xs-20-cosine.hdf5
"""
import sys

import h5py
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
K = 100
with h5py.File(src, "r") as f:
    train = f["train"][:].astype(np.float32)
    test = f["test"][:].astype(np.float32)
unit = train / np.linalg.norm(train, axis=1, keepdims=True)
qn = test / np.linalg.norm(test, axis=1, keepdims=True)
dist = np.clip(1 - qn @ unit.T, 0, 2)
neighbors = np.argsort(dist, axis=1, kind="stable")[:, :K]
distances = np.take_along_axis(dist, neighbors, axis=1)
with h5py.File(dst, "w") as f:
    f.attrs["distance"] = "cosine"
    f.attrs["dimension"] = train.shape[1]
    f.attrs["point_type"] = "float"
    f.create_dataset("train", data=train)
    f.create_dataset("test", data=test)
    f.create_dataset("neighbors", data=neighbors.astype(np.int32))
    f.create_dataset("distances", data=distances.astype(np.float32))
    f.create_dataset("avg_distances", data=dist.mean(axis=1).astype(np.float32))
print(train.shape, test.shape)
