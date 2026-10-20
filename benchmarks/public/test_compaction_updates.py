# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Adversarial checks for the independent compaction oracle."""

import copy
import unittest

from check_compaction_updates import check_queries, expected, states
from compaction_updates import resources


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.documents = [{"id": f"d{i:08}", "text": f"abstract{i} hmsprobe",
                           "sourceUri": f"scifact:{i}", "metadata": {"pair": 0, "revision": 0}}
                          for i in range(2)]
        self.operations = [{"kind": "replace", "pair": 0, "revision": 1, "start_ns": 10, "end_ns": 20},
                           {"kind": "delete", "pair": 0, "revision": 1, "start_ns": 30, "end_ns": 40}]

    def query(self, start, end, versions):
        found = []
        for number, revision in versions.items():
            document = self.documents[number]
            row = expected(document, revision)
            row.update({"documentId": document["id"], "id": f"hms:chunk:9:{document['id']}:0",
                        "startByte": 0, "endByte": len(row["text"].encode()), "score": 1.0,
                        "lexicalScore": 1.0, "semanticScore": None})
            found.append(row)
        return {"reader": 0, "sequence": 0, "pair": 0, "start_ns": start, "end_ns": end, "results": found}

    def check(self, query):
        return check_queries([query], self.documents, self.operations, 0, 1)

    def test_atomic_and_acknowledged_states(self):
        for start, end, versions in [(1, 9, {0: 0, 1: 0}), (11, 19, {0: 0, 1: 0}),
                                     (11, 19, {0: 1, 1: 1}), (21, 29, {0: 1, 1: 1}),
                                     (31, 39, {0: 1, 1: 1}), (31, 39, {1: 1}), (41, 49, {1: 1})]:
            with self.subTest(start=start, versions=versions):
                self.check(self.query(start, end, versions))
        for start, end, versions in [(11, 19, {0: 0, 1: 1}), (21, 29, {0: 0, 1: 0}),
                                     (41, 49, {0: 1, 1: 1}), (1, 9, {1: 0})]:
            with self.subTest(start=start, versions=versions), self.assertRaises(ValueError):
                self.check(self.query(start, end, versions))

    def test_payload_and_sequence_tampering(self):
        original = self.query(21, 29, {0: 1, 1: 1})
        for field, value in [("text", "wrong"), ("sourceUri", "wrong"), ("metadata", {}),
                             ("endByte", 0), ("score", float("nan")), ("id", "wrong")]:
            bad = copy.deepcopy(original)
            bad["results"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(bad)
        for field, value in [("sequence", 1), ("reader", 1), ("pair", 1), ("end_ns", 21)]:
            bad = copy.deepcopy(original)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(bad)
        with self.assertRaises(ValueError):
            check_queries([], self.documents, self.operations, 0, 1)

    def test_unrelated_operations_do_not_change_snapshot(self):
        self.assertEqual(states(self.operations, 1, 41, 49), [{2: 0, 3: 0}])

    def test_resource_boundaries_and_load_gate(self):
        phases = ["static", "updates", "compacting", "updates-tail", "verification", "done"]
        receipt = {"events": [{"phase": phase, "epoch_ns": i * 10} for i, phase in enumerate(phases)],
                   "load_before": [1, 1, 1], "load_after": [1, 1, 1], "samples": [
                       {"begin_epoch_ns": 0, "end_epoch_ns": 9, "rss_bytes": 10, "load": [1, 1, 1]},
                       {"begin_epoch_ns": 9, "end_epoch_ns": 11, "rss_bytes": 100, "load": [1, 1, 1]},
                       {"begin_epoch_ns": 11, "end_epoch_ns": 19, "rss_bytes": 20, "load": [1, 1, 1]}]}
        measured = resources(receipt)
        self.assertTrue(measured["load_gate_met"])
        self.assertEqual(measured["phases"]["static"]["sampled_max_rss_bytes"], 10)
        self.assertEqual(measured["phases"]["updates"]["sampled_max_rss_bytes"], 20)
        self.assertIsNone(measured["phases"]["compacting"]["sampled_max_rss_bytes"])
        for load in (2.999, 3.0, 3.001):
            receipt["samples"][1]["load"][0] = load
            self.assertEqual(resources(receipt)["load_gate_met"], load < 3)


if __name__ == "__main__":
    unittest.main()
