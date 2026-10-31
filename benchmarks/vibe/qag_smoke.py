# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real paired extensions and comparator ABIs before benchmark builds."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vibe", type=Path, required=True)
    parser.add_argument("--python-tests", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vibe))
    from vibe.algorithms.qag_pair.module import QAGPair

    rng = np.random.default_rng(24301)
    base = rng.normal(size=(128, 128)).astype(np.float32)
    learn = rng.normal(size=(16384, 128)).astype(np.float32)
    neighbors = np.argsort(-(learn.astype(np.float64) @ base.astype(np.float64).T), axis=1)[:, :100]
    np.empty(0, dtype="<u4").tofile(args.reserved)
    os.environ["WARM_QAG_VALIDATION_IDS"] = str(args.reserved)
    pair = QAGPair("ip", 16, 64)
    pair.fit_ood(base, learn, np.ascontiguousarray(neighbors, dtype=np.uint32))
    rows = []
    for arm in range(5):
        pair.set_query_arguments(arm, 3 if arm == 4 else 128, 128, 0)
        ids = np.asarray(pair.query(learn[0], 100))
        if ids.shape != (100,) or len(set(ids.tolist())) != 100 or np.any(ids < 0) or np.any(ids >= 128):
            raise ValueError(f"invalid native smoke output for arm {arm}")
        rows.append({"arm": arm, "strict_recall": len(set(ids.tolist()) & set(neighbors[0].tolist())) / 100})
    baseline = pair._modules[0].VectorIndex("ip", degree=16, build_ef=64, threads=1)
    candidate = pair._modules[1].VectorIndex("ip", degree=16, build_ef=64, threads=1)
    for index in (baseline, candidate):
        index.fit(base)
        index.set_query_arguments(128, 128)
    if baseline.index_bytes != candidate.index_bytes:
        raise ValueError("default index allocation changed")
    for query in learn[:16]:
        np.testing.assert_array_equal(baseline.query(query, 100), candidate.query(query, 100))
    print(json.dumps({"native_arms": rows, "default_parity_queries": 16,
                      "resources": pair.get_additional()}), flush=True)
    sys.modules["writerslogic_warm"] = pair._modules[1]
    import pytest
    raise SystemExit(pytest.main(["-q", str(args.python_tests)]))


if __name__ == "__main__":
    main()
