# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
from hybrid_probe import hybrid_order, interior_case, witnesses
from probe import squared_l2


class HybridTests(unittest.TestCase):
    def test_witness_boundaries_ties_and_uniqueness(self):
        rows = np.array([[0., 0.], [0., 0.], [2., 0.], [-2., 0.]])
        self.assertEqual(witnesses(rows, 0).tolist(), [])
        self.assertEqual(witnesses(rows, 1).tolist(), [0])
        self.assertEqual(witnesses(rows, 2).tolist(), [0, 2])
        self.assertEqual(witnesses(rows, 4).tolist(), [0, 2, 3, 1])
        for bad in [-1, 5]:
            with self.assertRaises(ValueError): witnesses(rows, bad)

    def test_reserved_witness_slots_and_phase_fill(self):
        witness = np.array([[4., 3., 2., 1., 0.]])
        phase = np.array([[0., 1., 2., 3., 4.]])
        self.assertEqual(hybrid_order(witness, phase, 0).tolist(), [[]])
        self.assertEqual(hybrid_order(witness, phase, 1).tolist(), [[0]])
        self.assertEqual(hybrid_order(witness, phase, 2).tolist(), [[0, 4]])
        self.assertEqual(hybrid_order(witness, phase, 3).tolist(), [[0, 1, 4]])
        self.assertEqual(hybrid_order(witness, phase, 4).tolist(), [[0, 1, 4, 3]])
        self.assertEqual(sorted(hybrid_order(witness, phase, 5)[0]), list(range(5)))

    def test_interior_neighbors_and_antipodal_shell(self):
        banks, queries = interior_case(256, 59)
        source = np.concatenate(banks)
        labels = np.repeat(np.arange(8), [len(b) for b in banks])
        top = np.argsort(squared_l2(queries, source), axis=1, kind="stable")[:, :10]
        np.testing.assert_array_equal(labels[top], np.repeat((np.arange(32) % 8)[:, None], 10, axis=1))
        for block, bank in enumerate(banks):
            np.testing.assert_allclose((bank[40] + bank[41]) / 2, queries[block], rtol=0, atol=1e-15)
            distance = ((bank - bank.mean(axis=0)) ** 2).sum(axis=1)
            self.assertTrue(np.all(distance[:40] < distance.max()))


if __name__ == "__main__": unittest.main()
