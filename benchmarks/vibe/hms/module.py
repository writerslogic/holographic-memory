"""VIBE wrapper for HMS (holographic-memory), the quantized graph index.

The index is a proximity graph whose traversal reads 8-bit (default) or 4-bit codes of the mean-centered rows;
the `rerank` best candidates are re-scored with an 8-bit residual code. Cosine and normalized
data are searched by angle, euclidean data by Euclidean distance, and ip data by inner product
(reduced to Euclidean search by appending one coordinate to each row).
"""
import os

# Back the index arrays with transparent huge pages (a no-op where the kernel refuses them).
# Set before the import so revisions where the advice is off by default take it too.
os.environ.setdefault("HMS_QGRAPH_HUGEPAGES", "1")

import numpy as np
from holographic_memory import VectorIndex

from ..base.module import BaseANN

METRICS = {"cosine": "angular", "normalized": "angular", "euclidean": "euclidean", "ip": "ip"}


class HMS(BaseANN):
    def __init__(self, metric, degree, build_ef, code_bits=8, query_aware=True):
        if metric not in METRICS:
            raise NotImplementedError(f"hms does not support the {metric} distance")
        self.metric = metric
        self.degree = degree
        self.build_ef = build_ef
        if code_bits not in (4, 8):
            raise ValueError("code_bits must be 4 or 8")
        self.code_bits = code_bits
        self.query_aware = bool(query_aware)
        self.trained = False
        self.ef = None
        self.rerank = None
        self.patience = 0
        self.index = None

    def fit(self, X):
        # In-distribution sets have no learn split: the plain graph.
        self.trained = False
        self._fit(X, None)

    def fit_ood(self, X, X_learn, X_learn_neighbors):
        # Out-of-distribution sets: the dataset's official learn split routes the graph (never
        # the test queries).
        self.trained = self.query_aware and X_learn is not None and len(X_learn) > 0
        self._fit(X, X_learn if self.trained else None)

    def _fit(self, X, queries):
        self.index = VectorIndex(METRICS[self.metric], degree=self.degree, build_ef=self.build_ef,
                                 threads=1, code_bits=self.code_bits)
        learning = None if queries is None else np.ascontiguousarray(queries, dtype=np.float32)
        self.index.fit(np.ascontiguousarray(X, dtype=np.float32), training_queries=learning)

    def set_query_arguments(self, ef, rerank=None, patience=0):
        if rerank is None:
            arguments = tuple(ef)
            if len(arguments) == 2:
                ef, rerank = arguments
            elif len(arguments) == 3:
                ef, rerank, patience = arguments
            else:
                raise ValueError("query arguments must contain ef, rerank and optional patience")
        if patience == 0:
            self.index.set_query_arguments(ef, rerank)
        else:
            self.index.set_query_arguments(ef, rerank, patience)
        self.ef = ef
        self.rerank = rerank
        self.patience = patience

    def query(self, v, n):
        return self.index.query(v, n)

    def __str__(self):
        precision = "" if self.code_bits == 4 else f", code_bits={self.code_bits}"
        training = ", query_aware=True" if self.trained else ""
        early_stop = f", patience={self.patience}" if self.patience > 0 else ""
        return f"HMS(metric={self.metric}, degree={self.degree}, build_ef={self.build_ef}, ef={self.ef}, rerank={self.rerank}{precision}{training}{early_stop})"
