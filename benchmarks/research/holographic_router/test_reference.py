# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np


class ReferenceTests(unittest.TestCase):
    def test_joint_reference_rotation_preserves_every_block_score(self):
        rng = np.random.default_rng(1231)
        query = np.exp(1j * rng.normal(size=(7, 64)))
        fields = np.exp(1j * rng.normal(size=(3, 11, 64))).sum(axis=1)
        reference = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(3, 64)))
        before = query @ fields.conj().T
        after = np.column_stack([
            (query * reference[block]) @ (fields[block] * reference[block]).conj()
            for block in range(3)])
        np.testing.assert_allclose(before, after, rtol=0, atol=2e-13)


if __name__ == "__main__":
    unittest.main()
