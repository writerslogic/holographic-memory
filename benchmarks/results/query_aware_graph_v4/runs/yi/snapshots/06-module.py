# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Four validation arms sharing two built indexes in the pinned VIBE harness."""

import operator
import os
import resource
import time

from vibe.algorithms.base.module import BaseANN
from vibe.algorithms.qag_pair.module import (
    _extension_path,
    _load_extension,
    _matrix,
    _training_query_file,
)


ARMS = {0: "lifted", 1: "lifted+reservoir", 2: "direct", 3: "direct+reservoir"}


class QAGServing(BaseANN):
    def __init__(self, metric, degree, build_ef):
        if metric != "ip":
            raise ValueError("serving comparison requires original inner product")
        self._module = _load_extension(
            _extension_path("HMS_QAG_CANDIDATE_SO"), "_hms_qag_serving"
        )
        self.degree, self.build_ef = degree, build_ef
        self._indexes, self._seconds = {}, {}
        self.arm = self.ef = self.rerank = self.repetition = None
        self._query = None
        self._training_path = self._training_sha256 = None

    def fit(self, X):
        raise ValueError("serving comparison requires the frozen construction view")

    def fit_ood(self, X, X_learn, X_learn_neighbors):
        rows = _matrix(X, "base")
        learn = _matrix(X_learn, "construction queries")
        if learn.shape != (32768, rows.shape[1]):
            raise ValueError("expected the frozen 32768 construction queries")
        training, path, checksum = _training_query_file(rows.shape[1], len(learn))
        if training is None or training.shape != learn.shape or not (training == learn).all():
            raise ValueError("construction file differs from the frozen view")
        self._training_path, self._training_sha256 = path, checksum
        for direct in (False, True):
            started = time.perf_counter()
            index = self._module.VectorIndex(
                "ip", degree=self.degree, build_ef=self.build_ef, threads=1,
                code_bits=8, direct_ip=direct,
            )
            index.fit(rows, training_queries=training)
            self._indexes[direct] = index
            self._seconds[direct] = time.perf_counter() - started

    def set_query_arguments(self, arm, ef=None, rerank=None, repetition=None):
        if ef is None and rerank is None and repetition is None:
            arm, ef, rerank, repetition = arm
        arm, ef, rerank, repetition = map(operator.index, (arm, ef, rerank, repetition))
        if arm not in ARMS or ef not in (100, 256, 512) or rerank != 400 or repetition not in (0, 1, 2):
            raise ValueError("query arguments differ from the declared serving comparison")
        index = self._indexes[arm >= 2]
        index.set_query_arguments(ef, rerank, patience=0, reservoir=400 if arm % 2 else 0)
        self.arm, self.ef, self.rerank, self.repetition = arm, ef, rerank, repetition
        self._query = index.query

    def query(self, v, n):
        return self._query(v, n)

    def get_additional(self):
        attrs = {
            "qag_arm": self.arm, "qag_repetition": self.repetition, "qag_patience": 0,
            "qag_candidate_code_bits": 8, "qag_candidate_training_queries": 32768,
            "qag_candidate_training_query_file": self._training_path,
            "qag_candidate_training_query_sha256": self._training_sha256,
            "qag_serving_method": ARMS[self.arm],
            "qag_reservoir": 400 if self.arm % 2 else 0,
            "qag_thread_environment": ",".join(f"{k}={os.environ.get(k, '')}" for k in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")),
            "qag_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "qag_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "qag_resource_scope": "two representations co-resident; RSS/USS are aggregate, not per-index",
        }
        for direct, index in self._indexes.items():
            name = "direct" if direct else "lifted"
            attrs[f"qag_{name}_index_bytes"] = int(index.index_bytes)
            attrs[f"qag_{name}_build_seconds"] = self._seconds[direct]
        return attrs

    def __str__(self):
        return f"QAGServing(arm={self.arm}, ef={self.ef}, rerank={self.rerank}, repetition={self.repetition})"
