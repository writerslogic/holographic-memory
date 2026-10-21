"""VIBE wrapper for HMS (holographic-memory), the quantized graph index.

Copy this directory to vibe/algorithms/hms/ in a VIBE checkout, build the image with
`./install.sh --algorithm hms`, and add HMS with its pinned revision to the README's
algorithm table.

The index is a proximity graph over unit vectors whose traversal reads 4-bit codes of the
randomly rotated rows; the `rerank` best candidates are re-scored with an 8-bit residual code.
It ranks by cosine, so it serves the `cosine` and `normalized` datasets.
"""
import numpy as np
from holographic_memory import VectorIndex

from ..base.module import BaseANN


class HMS(BaseANN):
    def __init__(self, metric, degree, build_ef):
        if metric not in ("cosine", "normalized"):
            raise NotImplementedError(f"hms supports cosine and normalized data, not {metric}")
        self.degree = degree
        self.build_ef = build_ef
        self.ef = None
        self.rerank = None
        self.index = None

    def fit(self, X):
        self.index = VectorIndex("angular", degree=self.degree, build_ef=self.build_ef, threads=1)
        self.index.fit(np.ascontiguousarray(X, dtype=np.float32))

    def set_query_arguments(self, ef, rerank):
        self.ef = ef
        self.rerank = rerank
        self.index.set_query_arguments(ef, rerank)

    def query(self, v, n):
        return self.index.query(v, n)

    def __str__(self):
        return f"HMS(degree={self.degree}, build_ef={self.build_ef}, ef={self.ef}, rerank={self.rerank})"
