# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Convert a verified diagnostic HDF5 file to the existing public-bench layout."""

import argparse
import json

import h5py
import numpy as np
from pathlib import Path

from prepare import sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.data.with_suffix(".json").read_text())
    if sha(args.data) != manifest["sha256"]:
        raise ValueError("source hash mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    with h5py.File(args.data, "r") as handle:
        metric = str(handle.attrs["distance"])
        if metric not in ("cosine", "normalized", "euclidean"):
            raise ValueError("unsupported metric")
        for name in ("train", "test"):
            with (args.out / f"{name}.f32").open("wb") as stream:
                for offset in range(0, len(handle[name]), 4096):
                    np.asarray(handle[name][offset:offset + 4096], dtype="<f4").tofile(stream)
        np.asarray(handle["neighbors"], dtype="<i4").tofile(args.out / "neighbors.i32")
        meta = {"name": manifest["dataset"], "metric": "euclidean" if metric == "euclidean" else "angular",
                "source_metric": metric, "dim": int(handle.attrs["dimension"]),
                "n_train": len(handle["train"]), "n_test": len(handle["test"]),
                "n_neighbors": int(handle["neighbors"].shape[1]), "input_manifest": manifest}
    meta["files"] = {path.name: sha(path) for path in args.out.iterdir()}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
