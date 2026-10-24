# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest

from paired import summarize


class SummaryTests(unittest.TestCase):
    def fixture(self, folder, load=0.5, pgo_factor=1.1):
        runs = []
        for dataset in ("glove", "sift"):
            for r in range(3):
                for arm, factor in (("v3", 1), ("pgo", pgo_factor), ("native", 1)):
                    stem = f"{dataset}-r{r}-{arm}"
                    runs.append(dict(stem=stem, load_before=load, load_after=load))
                    cells = [dict(params=dict(ef=ef, rerank=100), recall_at_10=0.9,
                                  mean_exact_evals_per_query=1000,
                                  qps_single_thread=(100 + r) * factor,
                                  load_1m_before_runs=[0.5], load_gate_met=True) for ef in (64, 128, 256)]
                    (folder / f"{stem}.json").write_text(json.dumps(dict(
                        sweep=cells, k=10, n_queries=10000, repeats=1, build=dict(threads=1),
                        index_file=dict(loaded="saved-index"))))
        (folder / "manifest.json").write_text(json.dumps(dict(runs=runs)))

    def test_pairs_and_load_boundary(self):
        for load in (2.99, 3, 3.01):
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                self.fixture(folder, load)
                result = summarize(folder)
                self.assertEqual(result["passes_development_bar"], load < 3)
                self.assertFalse(result["production_adoption"])
                self.assertEqual(len(result["rows"]), 12)
                for row in result["rows"]:
                    self.assertAlmostEqual(row["median_ratio"], 1.1 if row["arm"] == "pgo" else 1)

    def test_recall_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            path = folder / "sift-r1-pgo.json"
            doc = json.loads(path.read_text())
            doc["sweep"][1]["recall_at_10"] = 0.899
            path.write_text(json.dumps(doc))
            with self.assertRaises(AssertionError):
                summarize(folder)

    def test_evaluation_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            path = folder / "glove-r2-pgo.json"
            doc = json.loads(path.read_text())
            doc["sweep"][2]["mean_exact_evals_per_query"] += 1
            path.write_text(json.dumps(doc))
            with self.assertRaises(AssertionError):
                summarize(folder)

    def test_gain_boundary(self):
        for gain in (1.029, 1.03, 1.031):
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                self.fixture(folder, pgo_factor=gain)
                self.assertEqual(summarize(folder)["passes_development_bar"], gain >= 1.03)

    def test_inner_load_gate_rejected(self):
        for load, gate in ((3, True), (0.5, False)):
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                self.fixture(folder)
                path = folder / "sift-r0-pgo.json"
                doc = json.loads(path.read_text())
                doc["sweep"][0].update(load_1m_before_runs=[load], load_gate_met=gate)
                path.write_text(json.dumps(doc))
                self.assertFalse(summarize(folder)["passes_development_bar"])


if __name__ == "__main__":
    unittest.main()
