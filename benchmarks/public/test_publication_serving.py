# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Adversarial oracle checks using preserved actual-core observations."""

import argparse
import copy
import json
from pathlib import Path
import unittest

from check_compaction_updates import expected
from check_publication_serving import check_data, load


class OracleTests(unittest.TestCase):
    data = None
    documents = None

    def reject(self, mutate):
        data = copy.deepcopy(self.data)
        mutate(data)
        with self.assertRaises(ValueError):
            check_data(data, self.documents, "baseline")

    def test_original_and_arrival_count_boundaries(self):
        self.assertTrue(check_data(self.data, self.documents, "baseline")["checked"])
        self.reject(lambda d: d["arrivals"].pop())
        self.reject(lambda d: d["arrivals"].append(d["arrivals"][-1].copy()))

    def test_queue_capacity_boundaries(self):
        for capacity in (63, 65):
            with self.subTest(capacity=capacity):
                self.reject(lambda d: d["report"].update(queue_capacity=capacity))

    def test_complete_request_accounting(self):
        self.reject(lambda d: d["readers"][0].pop())
        self.reject(lambda d: d["readers"][0].append(d["readers"][0][-1].copy()))
        self.reject(lambda d: d["arrivals"][d["readers"][0][0]["sequence"]].update(accepted=False))

    def test_scheduled_and_service_clocks(self):
        self.reject(lambda d: d["arrivals"][1].update(scheduled_ns=d["arrivals"][0]["scheduled_ns"]))
        self.reject(lambda d: d["readers"][0][0].update(start_ns=0))
        self.reject(lambda d: d["readers"][0][0].update(end_ns=float("nan")))

    def test_source_and_embedding_identity(self):
        changes = {
            "sourceUri": "forged", "text": "forged", "metadata": {"revision": -1},
            "startByte": 1, "score": float("nan"), "version": "01", "id": "forged",
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                self.reject(lambda d: d["readers"][0][0]["results"][0].update({key: value}))
        self.reject(lambda d: d["readers"][0][0]["results"][0].update(
            semanticScore=1.0 - d["readers"][0][0]["results"][0]["semanticScore"]))

    def test_torn_whole_bulk_with_internally_consistent_payload(self):
        def mutate(data):
            query = next(q for reader in data["readers"] for q in reader if len(q["results"]) == 64)
            row = query["results"][0]
            number = int(row["documentId"][1:])
            revision = (int(row["version"]) + 1) % 65
            row.update(expected(self.documents[number], revision))
            row["endByte"] = len(row["text"].encode())
            row["semanticScore"] = float((number // 2 + revision) % 128 == query["pair"] % 128)
        self.reject(mutate)

    def test_trace_accounting_and_phase_order(self):
        self.reject(lambda d: d["traces"].pop())
        self.reject(lambda d: d["traces"].append(d["traces"][0].copy()))
        def mutate(data):
            trace = next(row for row in data["traces"] if row["kind"] == "commit")
            trace["exclusion_start_ns"] = trace["append_start_ns"]
        self.reject(mutate)
        self.reject(lambda d: d["traces"][0].update(count=999))

    def test_restart_and_acknowledged_state(self):
        self.reject(lambda d: d["after"][0]["results"][0].update(text="forged"))
        def mutate(data):
            for phase in ("before", "after"):
                row = data[phase][0]["results"][0]
                row.update(expected(self.documents[0], 0))
                row["endByte"] = len(row["text"].encode())
                row["semanticScore"] = 1.0
        self.reject(mutate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    OracleTests.data = load(args.folder)
    OracleTests.documents = json.loads(args.input.read_text())
    unittest.main(argv=["test_publication_serving"], verbosity=2)
