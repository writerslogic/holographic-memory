# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path
import tempfile
import unittest

from summarize import METRICS, assess, read_arm


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.arms = {f"{n}-{arm}": {"metrics": dict.fromkeys(METRICS, 1000),
                                  "files_sha256": {"CURRENT": "a" * 64}}
                     for n in (1, 2, 3) for arm in ("parent", "candidate")}
        for n in (1, 2, 3):
            self.arms[f"{n}-candidate"]["metrics"]["elapsed_ns"] = 970

    def test_time_and_memory_boundaries(self):
        for metric, cases in (
            ("elapsed_ns", [(979, True), (980, False), (981, False)]),
            ("isolated_peak_rss_kib", [(1049, True), (1050, True), (1051, False)]),
        ):
            for value, eligible in cases:
                with self.subTest(metric=metric, value=value):
                    arms = copy.deepcopy(self.arms)
                    for n in (1, 2, 3):
                        arms[f"{n}-candidate"]["metrics"][metric] = value
                    self.assertEqual(assess(arms)["decision"] == "eligible_for_scaled_validation", eligible)

    def test_loss_and_file_mismatch_are_retained(self):
        self.arms["3-candidate"]["metrics"]["elapsed_ns"] = 1100
        result = assess(self.arms)
        self.assertEqual(result["faster_pairs"], 2)
        self.assertEqual(result["pairs"][2]["ratios"]["elapsed_ns"], 1.1)
        self.assertEqual(result["decision"], "eligible_for_scaled_validation")
        self.arms["2-candidate"]["files_sha256"]["CURRENT"] = "b" * 64
        self.assertEqual(assess(self.arms)["decision"], "rejected")

    def test_incomplete_or_invalid_log_fails_closed(self):
        row = dict(self.arms["1-parent"]["metrics"], mode="Merged", live=211428, threads=8)
        hashes = {"files_sha256": {"CURRENT": "a" * 64}}
        good = json.dumps(row) + "\n" + json.dumps(hashes) + "\n\tExit status: 0\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.log"
            path.write_text(good)
            self.assertEqual(read_arm(path, 211428)["metrics"], row)
            for bad in (good.replace("Exit status: 0", "Exit status: 1"),
                        good.replace("Exit status: 0", ""),
                        good + json.dumps(row) + "\n",
                        good.replace('"threads": 8', '"threads": 1'),
                        good.replace('"live": 211428', '"live": 211427'),
                        good.replace('"isolated_peak_rss_kib": 1000', '"isolated_peak_rss_kib": 0')):
                path.write_text(bad)
                with self.assertRaises(ValueError):
                    read_arm(path, 211428)


if __name__ == "__main__":
    unittest.main()
