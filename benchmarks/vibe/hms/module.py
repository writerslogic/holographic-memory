"""VIBE wrapper for HMS (holographic-memory), the quantized graph index.

The index is a proximity graph whose traversal reads 4-bit codes of the mean-centered rows;
the `rerank` best candidates are re-scored with an 8-bit residual code. Cosine and normalized
data are searched by angle, euclidean data by Euclidean distance, and ip data by inner product
(reduced to Euclidean search by appending one coordinate to each row).
"""
import numpy as np
from holographic_memory import VectorIndex

from ..base.module import BaseANN

METRICS = {"cosine": "angular", "normalized": "angular", "euclidean": "euclidean", "ip": "ip"}


class HMS(BaseANN):
    def __init__(self, metric, degree, build_ef):
        if metric not in METRICS:
            raise NotImplementedError(f"hms does not support the {metric} distance")
        self.metric = metric
        self.degree = degree
        self.build_ef = build_ef
        self.ef = None
        self.rerank = None
        self.index = None

    def fit(self, X):
        self.index = VectorIndex(METRICS[self.metric], degree=self.degree, build_ef=self.build_ef, threads=1)
        self.index.fit(np.ascontiguousarray(X, dtype=np.float32))

    def set_query_arguments(self, ef, rerank):
        self.ef = ef
        self.rerank = rerank
        self.index.set_query_arguments(ef, rerank)

    def query(self, v, n):
        return self.index.query(v, n)

    def __str__(self):
        return f"HMS(metric={self.metric}, degree={self.degree}, build_ef={self.build_ef}, ef={self.ef}, rerank={self.rerank})"
