# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import longmemeval_retrieval_limits as L


class RetrievalLimitTests(unittest.TestCase):
    def test_weight_partition_covers_every_pair_order_including_adjacent_floats(self):
        samples = L.weight_intervals(3, 2)

        def signature(weight):
            return tuple(weight / (61 + assistant) > 1.0 / (61 + user)
                         for user in range(3) for assistant in range(2))

        covered = {signature(weight) for weight in samples}
        for weight in [index / 250 for index in range(1001)]:
            self.assertIn(signature(weight), covered)
        for weight in samples:
            if weight not in (0.0, math.nextafter(0.0, 1.0), 4.0):
                previous = math.nextafter(weight, 0.0)
                self.assertNotEqual(signature(weight), signature(previous))
                self.assertIn(signature(previous), covered)

    def test_replay_admits_whole_closures_at_round_source_and_byte_boundaries(self):
        candidates = [{"identity": "anchor", "role": "user", "rank": 0, "position": 0,
                       "closure": [("anchor", ("r2", "s"), 10), ("premise", ("r1", "s"), 10)]}]
        options = {"assistantWeight": 1.0, "maxRounds": 2, "maxSources": 2, "maxBytes": 23}
        for size in (22, 23, 24):
            result = L.simulate(candidates, {**options, "maxBytes": size}, "question")
            self.assertEqual(result, [] if size < 23 else [("r2", "s"), ("r1", "s")])
        for capacity in (1, 2, 3):
            result = L.simulate(candidates, {**options, "maxRounds": capacity}, "question")
            self.assertEqual(bool(result), capacity >= 2)
            result = L.simulate(candidates, {**options, "maxSources": capacity}, "question")
            self.assertEqual(bool(result), capacity >= 2)

    def test_assistant_query_override_matches_native_word_order_independence(self):
        self.assertTrue(L.assistant_intent("What was recommended by you?"))
        self.assertTrue(L.assistant_intent("What did you recommend?"))
        self.assertFalse(L.assistant_intent("What was my recommendation?"))
        self.assertTrue(L.aggregation_intent("How\nmany visits?"))
        self.assertFalse(L.aggregation_intent("The totality of my travels"))


if __name__ == "__main__":
    unittest.main()
