"""VIBE wrapper for HMS (holographic-memory), the quantized graph index.

The index is a proximity graph whose traversal reads 4-bit (default) or 8-bit codes of the mean-centered rows;
the `rerank` best candidates are re-scored with an 8-bit residual code. Cosine and normalized
data are searched by angle, euclidean data by Euclidean distance, and ip data by inner product
(reduced to Euclidean search by appending one coordinate to each row).
"""
import numpy as np
from holographic_memory import VectorIndex

from ..base.module import BaseANN

METRICS = {"cosine": "angular", "normalized": "angular", "euclidean": "euclidean", "ip": "ip"}


class HMS(BaseANN):
    def __init__(self, metric, degree, build_ef, code_bits=4, query_aware=False):
        if metric not in METRICS:
            raise NotImplementedError(f"hms does not support the {metric} distance")
        self.metric = metric
        self.degree = degree
        self.build_ef = build_ef
        if code_bits not in (4, 8):
            raise ValueError("code_bits must be 4 or 8")
        self.code_bits = code_bits
        self.query_aware = bool(query_aware)
        self.ef = None
        self.rerank = None
        self.index = None

    def fit(self, X):
        if self.query_aware:
            raise ValueError("query-aware HMS requires the separate learn split through fit_ood")
        self._fit(X, None)

    def fit_ood(self, X, X_learn, X_learn_neighbors):
        self._fit(X, X_learn if self.query_aware else None)

    def _fit(self, X, queries):
        self.index = VectorIndex(METRICS[self.metric], degree=self.degree, build_ef=self.build_ef,
                                 threads=1, code_bits=self.code_bits)
        learning = None if queries is None else np.ascontiguousarray(queries, dtype=np.float32)
        self.index.fit(np.ascontiguousarray(X, dtype=np.float32), training_queries=learning)

    def set_query_arguments(self, ef, rerank=None):
        if rerank is None:
            ef, rerank = ef
        self.ef = ef
        self.rerank = rerank
        self.index.set_query_arguments(ef, rerank)

    def query(self, v, n):
        return self.index.query(v, n)

    def __str__(self):
        precision = "" if self.code_bits == 4 else f", code_bits={self.code_bits}"
        training = ", query_aware=True" if self.query_aware else ""
        return f"HMS(metric={self.metric}, degree={self.degree}, build_ef={self.build_ef}, ef={self.ef}, rerank={self.rerank}{precision}{training})"
