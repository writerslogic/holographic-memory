"""ann-benchmarks wrapper for HMS (holographic-memory), the quantized graph index.

Copy this directory to ann_benchmarks/algorithms/hms/ in an ann-benchmarks checkout and add
`- hms` to the library matrix in .github/workflows/benchmarks.yml. Build the image with
`python install.py --algorithm hms`.
"""

import numpy as np
from holographic_memory import VectorIndex

from ..base.module import BaseANN


class HMS(BaseANN):
    """HMS (holographic-memory) quantized graph index.

    A proximity graph whose traversal reads a `code_bits`-bit (8 or 4) code of each row; the
    `rerank` best candidates are re-scored with an 8-bit residual code. Build arguments: graph
    out-degree, construction search width and traversal code width. Query arguments: candidate
    pool size `ef` and re-rank depth.
    """

    def __init__(self, metric, method_param):
        if metric not in ("angular", "euclidean"):
            raise NotImplementedError(f"hms does not support metric {metric}")
        self.metric = metric
        self.degree = method_param["degree"]
        self.build_ef = method_param["build_ef"]
        self.code_bits = method_param.get("code_bits", 8)
        self.name = f"hms(degree={self.degree}, build_ef={self.build_ef}, code_bits={self.code_bits})"
        self.index = None

    def fit(self, X):
        self.index = VectorIndex(
            self.metric, degree=self.degree, build_ef=self.build_ef, code_bits=self.code_bits
        )
        self.index.fit(np.ascontiguousarray(X, dtype=np.float32))

    def set_query_arguments(self, ef, rerank):
        self.index.set_query_arguments(ef, rerank)
        self.name = (
            f"hms(degree={self.degree}, build_ef={self.build_ef}, code_bits={self.code_bits}, "
            f"ef={ef}, rerank={rerank})"
        )

    def query(self, v, n):
        return self.index.query(v, n)

    def batch_query(self, X, n):
        self.res = self.index.batch_query(np.ascontiguousarray(X, dtype=np.float32), n)

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        return {"index_bytes": self.index.index_bytes}

    def done(self):
        self.index = None
