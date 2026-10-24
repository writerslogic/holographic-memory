# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Reference ranking and frozen selection checks for the OOD experiment."""
import unittest

import numpy as np
from ood_ivf_ab import choose, exact_top, normalized


class OodSelectionTests(unittest.TestCase):
    def test_exact_scores_use_ids_for_boundary_ties(self):
        base = np.array([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.float32)
        queries = np.array([[1, 0], [0, 0]], dtype=np.float32)
        for k in (1, 3, 4):
            actual = exact_top(base, queries, k)
            expected = [sorted(range(len(base)), key=lambda i: (-sum(float(x) * float(y) for x, y in zip(q, base[i])), i))[:k] for q in queries]
            np.testing.assert_array_equal(actual, expected)
        for k in (0, 5):
            with self.assertRaises(ValueError):
                exact_top(base, queries, k)

    def test_selection_requires_recall_and_uses_median(self):
        rows = [
            {"arm": "lowrank", "val_recall": .9499, "qps": [999, 999, 999], "build": "a", "args": [8, 100]},
            {"arm": "lowrank", "val_recall": .95, "qps": [8, 8, 1], "build": "b", "args": [8, 100]},
            {"arm": "lowrank", "val_recall": .9501, "qps": [7, 7, 99], "build": "c", "args": [8, 100]},
        ]
        self.assertEqual(choose(rows, "lowrank"), 1)
        self.assertIsNone(choose(rows, "graph"))

    def test_normalization_preserves_zero_and_extreme_direction(self):
        rows = np.array([[0, 0], [3, 4], [3e30, 4e30]], dtype=np.float32)
        actual = normalized(rows)
        np.testing.assert_allclose(actual, [[0, 0], [.6, .8], [.6, .8]], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
