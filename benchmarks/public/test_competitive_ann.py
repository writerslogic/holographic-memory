# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = ["numpy==2.2.6"]
# ///

import unittest
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from check_competitive_ann import choose, exact_truth, score_predictions
from competitive_ann import timing_lock
from restore_ann_checkpoint import restore


class CompetitiveOracleTests(unittest.TestCase):
    def test_checkpoint_restore_preserves_only_completed_and_rejects_changed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            attempt = campaign / "validation-interrupted-attempt-1"
            attempt.mkdir()
            files = []
            for name in ("complete-validation.json", "complete-validation-execution.log", "partial-validation-execution.log"):
                retained = attempt / name
                retained.write_bytes(name.encode())
                files.append({"retained_path": str(retained), "original_path": str(campaign / name),
                              "sha256": hashlib.sha256(retained.read_bytes()).hexdigest()})
            receipt_path = attempt / "interrupted-receipt.json"
            receipt_path.write_text(json.dumps({"schema_version": 1, "completed_result_files": [str(attempt / "complete-validation.json")], "files": files}))
            self.assertEqual(len(restore(receipt_path)), 2)
            self.assertEqual(len(restore(receipt_path)), 2)
            self.assertFalse((campaign / "partial-validation-execution.log").exists())
            (campaign / "complete-validation.json").write_bytes(b"concurrent change")
            with self.assertRaises(ValueError):
                restore(receipt_path)
            self.assertEqual((campaign / "complete-validation.json").read_bytes(), b"concurrent change")

    @staticmethod
    def _abort_campaign():
        raise RuntimeError("abort campaign")

    def test_campaign_lock_excludes_other_openers_and_releases_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.lock"
            with self.assertRaisesRegex(RuntimeError, "abort campaign"):
                with timing_lock(path) as held:
                    with timing_lock(path) as nested:
                        self.assertEqual(held.fileno(), nested.fileno())
                        with path.open("a+b") as competing:
                            with self.assertRaises(BlockingIOError):
                                fcntl.flock(competing.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._abort_campaign()
            with path.open("a+b") as next_campaign:
                fcntl.flock(next_campaign.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_oracle_is_block_independent_and_ties_use_ids(self):
        train = np.zeros((8193, 2), dtype=np.float64)
        train[0] = train[8192] = [1, 0]
        ids, values = exact_truth(train, np.array([[1.0, 0.0]]), k=2)
        np.testing.assert_array_equal(ids, [[0, 8192]])
        np.testing.assert_array_equal(values, [[1.0, 1.0]])

    def test_predictions_reject_duplicate_and_out_of_range_ids(self):
        train = np.eye(12, dtype=np.float64)
        query = train[:1]
        ids, scores = exact_truth(train, query)
        for bad in (list(range(9)) + [8], list(range(9)) + [12], list(range(9)) + [-2]):
            with self.assertRaises(ValueError):
                score_predictions([bad], train, query, ids, scores)
        valid = score_predictions([list(range(10))], train, query, ids, scores)
        self.assertEqual(valid["recall_at_10"], 1.0)
        absent = score_predictions([list(range(9)) + [-1]], train, query, ids, scores)
        self.assertEqual(absent["recall_at_10"], 0.9)
        self.assertEqual(absent["missing_result_count"], 1)

    def test_selection_keeps_target_boundary_and_missing_targets(self):
        rows = [{"index_bytes": 100, "configs": [
            {"config_id": 0, "recall_at_10": 0.949, "median_ns": 1},
            {"config_id": 1, "recall_at_10": 0.95, "median_ns": 2},
            {"config_id": 2, "recall_at_10": 0.951, "median_ns": 3},
        ]}]
        self.assertEqual(choose(rows, 0.95), {"row_id": 0, "config_id": 1})
        self.assertIsNone(choose(rows, 0.99))


if __name__ == "__main__":
    unittest.main()
