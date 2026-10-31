# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check the real compiled WARM wrapper and VIBE's own configuration loader."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vibe", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vibe.resolve()))
    from vibe.algorithms.warm.module import WARM
    from vibe.definitions import get_definitions

    rng = np.random.default_rng(683)
    records = []
    for dimension in (768, 2048):
        rows = rng.normal(size=(256, dimension)).astype(np.float32)
        rows /= np.linalg.norm(rows, axis=1)[:, None]
        original = rows.copy()
        for metric in ("cosine", "normalized"):
            definitions = [definition for definition in get_definitions(
                dimension, "float", metric, 100, str(args.vibe / "vibe/algorithms")) if definition.algorithm == "warm"]
            assert len(definitions) == 4 and all(len(d.query_argument_groups) == 33 for d in definitions)
            adapter = WARM(metric, 32, 100)
            adapter.fit(rows)
            adapter.set_query_arguments(256, 256)
            for k in (99, 100, 101, 255, 256, 257):
                ids = adapter.query(rows[0], k)
                assert ids.shape == (min(k, len(rows)),)
                assert len(np.unique(ids)) == len(ids) and ids[0] == 0
                assert np.all((ids >= 0) & (ids < len(rows)))
            assert np.array_equal(rows, original)
            assert f"metric={metric}" in str(adapter)
            records.append({"dimension": dimension, "metric": metric, "passed": True})
    for metric in ("euclidean", "ip"):
        try:
            WARM(metric, 32, 100)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"unsupported metric accepted: {metric}")
        assert not [definition for definition in get_definitions(
            768, "float", metric, 100, str(args.vibe / "vibe/algorithms")) if definition.algorithm == "warm"]
    args.out.write_text(json.dumps({"passed": True, "checks": records,
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256((args.vibe / "vibe/algorithms/warm/module.py").read_bytes()).hexdigest()}, indent=2) + "\n")
