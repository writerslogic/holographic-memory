# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
from probe import PhaseSketch, coverage, generate, ordering, reservoir_scores, squared_l2


class PhaseTests(unittest.TestCase):
    def test_correlation_matches_complex_phase_readout(self):
        omega = np.array([[0.2, -1.0, 2.0], [1.0, 0.3, -0.1]])
        data = np.array([[0.5, 1.0], [-0.25, 3.0]])
        query = np.array([[1.0, -0.5], [0.0, 0.0]])
        sketch = PhaseSketch(omega, 1, 0.75)
        sketch.change(0, data)
        reference = (np.exp(1j * (query @ omega / 0.75))
                     @ np.exp(1j * (data @ omega / 0.75)).sum(axis=0).conj()).real / 3
        np.testing.assert_allclose(sketch.score(query)[:, 0], reference, rtol=0, atol=1e-14)

    def test_delete_replace_reuse_matches_full_resummation(self):
        rng = np.random.default_rng(1209)
        omega = rng.normal(size=(3, 64))
        bank = {i: (i % 3, rng.normal(size=3)) for i in range(33)}
        sketch = PhaseSketch(omega, 4, 1.5)
        for block, row in bank.values(): sketch.change(block, row[None])
        for identifier in (0, 16, 32):
            block, row = bank.pop(identifier)
            sketch.change(block, row[None], -1)
        for identifier in (1, 15, 31):
            block, old = bank[identifier]
            new = rng.normal(size=3)
            sketch.change(block, old[None], -1)
            sketch.change(block, new[None])
            bank[identifier] = (block, new)
        for identifier in (0, 16, 32):
            row = np.array([1.0, 2.0, 3.0])
            block = (identifier + 1) % 3
            bank[identifier] = (block, row)
            sketch.change(block, row[None])
        reference = PhaseSketch(omega, 4, 1.5)
        for block in range(4):
            data = np.array([row for owner, row in bank.values() if owner == block]).reshape(-1, 3)
            reference.change(block, data)
        np.testing.assert_array_equal(sketch.counts, reference.counts)
        np.testing.assert_allclose(sketch.cos, reference.cos, rtol=0, atol=2e-14)
        np.testing.assert_allclose(sketch.sin, reference.sin, rtol=0, atol=2e-14)
        query = rng.normal(size=(12, 3))
        np.testing.assert_allclose(sketch.score(query)[:, :3], reference.score(query)[:, :3], rtol=0, atol=2e-14)
        self.assertTrue(np.isneginf(sketch.score(query)[:, 3]).all())

    def test_population_boundaries_and_failed_update_are_atomic(self):
        sketch = PhaseSketch(np.eye(2), 1, 1)
        before = sketch.cos.copy()
        with self.assertRaises(ValueError): sketch.change(0, np.zeros((1, 2)), -1)
        np.testing.assert_array_equal(sketch.cos, before)
        for size in (0, 1, 2):
            data = np.arange(size * 2).reshape(size, 2)
            sketch.change(0, data)
            self.assertEqual(sketch.counts[0], size)
            sketch.change(0, data, -1)
            self.assertEqual(sketch.counts[0], 0)
            np.testing.assert_allclose(sketch.cos, 0, rtol=0, atol=1e-15)

    def test_invalid_values_are_rejected_before_update(self):
        sketch = PhaseSketch(np.eye(2), 1, 1)
        for bad in (np.array([[1.0, np.nan]]), np.zeros((1, 3)), np.ones(2)):
            with self.assertRaises(ValueError): sketch.change(0, bad)
        for sigma in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): PhaseSketch(np.eye(2), 1, sigma)
        self.assertEqual(sketch.counts[0], 0)

    def test_density_can_reverse_nearest_neighbor_ranking(self):
        query = np.array([[0.0]])
        blocks = [np.array([[0.0]]), np.full((10, 1), 0.5)]
        nearest = reservoir_scores(query, blocks)
        density = np.column_stack([np.exp(-squared_l2(query, b) / 2).sum(axis=1) for b in blocks])
        self.assertEqual(ordering(nearest)[0, 0], 0)
        self.assertEqual(ordering(density)[0, 0], 1)

    def test_oracle_coverage_and_ties(self):
        hits = np.array([[3, 3, 4], [0, 10, 0]])
        order = ordering(hits.astype(float))
        np.testing.assert_array_equal(order, [[2, 0, 1], [1, 0, 2]])
        self.assertEqual(coverage(order, hits, [1, 2, 3]), {"1": 0.7, "2": 0.85, "3": 1.0})

    def test_rare_case_contains_its_registered_true_neighbors(self):
        banks, query = generate("rare", 256, 17)
        source = np.concatenate(banks)
        labels = np.repeat(np.arange(8), [len(b) for b in banks])
        top = np.argsort(squared_l2(query, source), axis=1, kind="stable")[:, :10]
        np.testing.assert_array_equal(labels[top], np.repeat((np.arange(32) % 8)[:, None], 10, axis=1))
        self.assertEqual([len(b) for b in banks], [256, 64] * 4)


if __name__ == "__main__": unittest.main()
