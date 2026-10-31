# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Validation-only paired storage comparison; both libraries use the same source."""

import time

from vibe.algorithms.qag_pair.module import QAGPair, METRICS, _matrix, _training_query_file


class QAGLayout(QAGPair):
    def __init__(self, metric, degree, build_ef, candidate_code_bits=8, legacy_reference=False,
                 revision_comparison=False):
        super().__init__(metric, degree, build_ef, candidate_code_bits)
        self._legacy_reference = bool(legacy_reference)
        self._revision_comparison = bool(revision_comparison)
        if self._legacy_reference and self._revision_comparison:
            raise ValueError("legacy and revision comparisons are mutually exclusive")
        if self._legacy_reference and candidate_code_bits != 4:
            raise ValueError("the original reference uses four-bit codes")

    def _fit_warm(self, rows, training_queries=None):
        self._indexes.clear()
        self._build_seconds.clear()
        self._training_count = 0 if training_queries is None else len(training_queries)
        for arm, module in enumerate(self._modules):
            started = time.perf_counter()
            options = dict(degree=self.degree, build_ef=self.build_ef, threads=1)
            if arm == 1 or not self._legacy_reference:
                options.update(code_bits=self._candidate_code_bits,
                               compact_adjacency=bool(arm) and not self._revision_comparison)
            index = module.VectorIndex(METRICS[self.metric], **options)
            if training_queries is None:
                index.fit(rows)
            else:
                index.fit(rows, training_queries=training_queries)
            self._build_seconds[arm] = time.perf_counter() - started
            self._indexes[arm] = index

    def fit_ood(self, X, X_learn, X_learn_neighbors):
        rows, learn = _matrix(X, "base"), _matrix(X_learn, "construction queries")
        if learn.shape != (32768, rows.shape[1]):
            raise ValueError("layout comparison requires the frozen 32768 construction rows")
        training, path, checksum = _training_query_file(rows.shape[1], len(learn))
        if training is None or training.shape != learn.shape or not (training == learn).all():
            raise ValueError("layout training file differs from the permitted construction view")
        self._fit_warm(rows, training)
        self._construction_count = len(learn)
        self._training_path, self._training_sha256 = path, checksum

    def get_additional(self):
        attrs = super().get_additional()
        attrs.update(qag_layout_comparison=True, qag_reference_code_bits=self._candidate_code_bits,
                     qag_legacy_reference=self._legacy_reference,
                     qag_revision_comparison=self._revision_comparison,
                     qag_layout="compact" if self.arm == 1 and not self._revision_comparison else "fixed")
        return attrs
