# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare logical storage choices; measure recall before choosing smaller residuals."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from holographic_memory import VectorIndex


def main():
    rows = np.random.default_rng(17).normal(size=(128, 33)).astype(np.float32)
    with TemporaryDirectory() as directory:
        for residual_bits in (8, 4, 0):
            index = VectorIndex("euclidean", degree=16, build_ef=64, threads=1,
                                code_bits=8, residual_bits=residual_bits)
            index.fit(rows)
            index.set_query_arguments(ef=64, rerank=64)
            path = Path(directory) / f"residual-{residual_bits}.hms"
            index.save(path)
            reopened = VectorIndex("euclidean", threads=1)
            reopened.load(path)
            reopened.set_query_arguments(ef=64, rerank=64)
            np.testing.assert_array_equal(reopened.query(rows[0], 10), index.query(rows[0], 10))
            assert reopened.memory_breakdown == index.memory_breakdown
            assert sum(index.memory_breakdown.values()) == index.index_bytes
            print({"residual_bits": reopened.residual_bits,
                   "logical_index_bytes": index.index_bytes,
                   "components": index.memory_breakdown})


if __name__ == "__main__":
    main()
