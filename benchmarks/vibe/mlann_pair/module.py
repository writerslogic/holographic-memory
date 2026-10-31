# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""One RF80/depth13 MLANN comparator using the paired harness's query path."""

import hashlib
import operator
import os
from pathlib import Path
import time

import numpy as np

from ..qag_pair.module import QAGPair, _aligned_f32_matrix, _matrix, _training_query_file


class MLANNPair(QAGPair):
    def __init__(self, metric, trees=80, depth=13):
        trees, depth = map(operator.index, (trees, depth))
        if (trees, depth) != (80, 13):
            raise ValueError("this frozen comparator requires RF80, depth13")
        super().__init__(metric, 64, 500)
        self.trees, self.depth = trees, depth

    def fit(self, X):
        raise ValueError("MLANNPair requires the separate construction learn split")

    def fit_ood(self, X, X_learn, X_learn_neighbors):
        rows, learn = _matrix(X, "base vectors"), _matrix(X_learn, "construction queries")
        if rows.shape[1] != learn.shape[1] or len(learn) != 32768:
            raise ValueError("requires exactly32768 construction queries matching base dimension")
        if len(learn) < 1 << self.depth:
            raise ValueError("forest depth exceeds the construction query count")
        neighbors = np.asarray(X_learn_neighbors)
        if (neighbors.shape != (32768, 100) or neighbors.dtype.kind not in "iu"
                or np.any(neighbors < 0) or np.any(neighbors >= len(rows))):
            raise ValueError("requires100 integer in-range neighbors per construction query")
        ordered = np.sort(neighbors, axis=1)
        if np.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError("construction neighbor IDs must be unique per row")
        reserved_value = os.environ.get("WARM_QAG_VALIDATION_IDS")
        if not reserved_value:
            raise ValueError("requires an explicit empty reserved-ID file for the separated learn pool")
        reserved = Path(reserved_value).expanduser().resolve(strict=True)
        if not reserved.is_file() or reserved.stat().st_size != 0:
            raise ValueError("already-separated construction queries require empty reserved IDs")
        training, training_path, training_sha = _training_query_file(rows.shape[1], len(learn))
        learn_sha = hashlib.sha256(memoryview(np.ascontiguousarray(learn, dtype="<f4")).cast("B")).hexdigest()
        if training is None or not np.array_equal(training, learn) or learn_sha != training_sha:
            raise ValueError("construction file must match the exact HDF5 learning query pool")
        construction_neighbors = np.ascontiguousarray(neighbors, dtype=np.uint32)
        import mlann

        started = time.perf_counter()
        base, base_owner = _aligned_f32_matrix(rows, "MLANN base vectors")
        construction, learn_owner = _aligned_f32_matrix(learn, "MLANN construction queries")
        query, query_owner = _aligned_f32_matrix(
            np.zeros((1, rows.shape[1]), dtype=np.float32), "MLANN query buffer"
        )
        if self.metric == "cosine":
            for matrix in (base, construction):
                norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise ValueError("MLANN cosine vectors must have nonzero norm")
                matrix /= norms
        index = mlann.MLANNIndex(base, "RF")
        index.build(construction, construction_neighbors, self.trees, self.depth)
        elapsed = time.perf_counter() - started
        self._indexes = {4: index}
        self._build_seconds = {4: elapsed}
        self._query = None
        self._validation_count = 0
        self._construction_count = len(learn)
        self._training_count = 0
        self._training_path, self._training_sha256 = training_path, training_sha
        self._mlann_owners = (base_owner, learn_owner, query_owner)
        self._mlann_query = query[0]
        self._mlann_base_count = len(rows)
        self._mlann_distance = mlann.IP if self.metric in ("cosine", "normalized", "ip") else mlann.L2
        self._mlann_threshold = None
        self._candidate_patience = 0
        self.arm = self.ef = self.rerank = self.repetition = None
        self.patience = 0

    def get_additional(self):
        attrs = super().get_additional()
        for name in ("qag_mlann_rf10_depth14_build_seconds", "qag_mlann_rf10_depth14_index_bytes"):
            attrs.pop(name, None)
        if 4 in self._build_seconds:
            attrs["qag_mlann_rf80_depth13_build_seconds"] = self._build_seconds[4]
            if hasattr(self._indexes[4], "index_bytes"):
                attrs["qag_mlann_rf80_depth13_index_bytes"] = int(self._indexes[4].index_bytes)
        attrs.update(
            qag_mlann_construction="RF, n_trees=80, depth=13; same already-separated32768 construction queries",
            qag_resource_scope="only MLANN built; VIBE USS delta is per-arm; process peak RSS includes input buffers and loaded WARM libraries",
            qag_warm_indexes_built=0,
        )
        if self._training_sha256 is not None:
            attrs["qag_construction_query_sha256"] = self._training_sha256
            attrs["qag_construction_query_file"] = self._training_path
        return attrs

    def __str__(self):
        return (f"MLANNPair(metric={self.metric}, RF={self.trees}, depth={self.depth}, "
                f"effort={self.ef}, repetition={self.repetition})")
