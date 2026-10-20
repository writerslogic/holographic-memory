# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.2.6", "lz4==4.4.5"]
# ///

import copy
import json
from pathlib import Path
import unittest

from check_encoder_version_diagnostic import encoding, result_scores, MAJORITY, UNION


class OracleTests(unittest.TestCase):
    def test_majority_tie_requires_an_eligible_prechange_witness(self):
        fixture = Path(__file__).resolve().parents[2] / "tests/fixtures/text-encoder-majority-tie.json"
        witness = json.loads(fixture.read_text())
        self.assertEqual(encoding(witness["text"], witness["dim"], MAJORITY, witness), witness["indices"])
        with self.assertRaises(AssertionError):
            encoding(witness["text"], witness["dim"], MAJORITY)
        tampered = copy.deepcopy(witness)
        tampered["majority"][4][0] = 1024
        with self.assertRaises(AssertionError):
            encoding(witness["text"], witness["dim"], MAJORITY, tampered)

    def test_union_matches_prechange_controls_at_density_boundaries(self):
        fixture = Path(__file__).resolve().parents[2] / "tests/fixtures/text-encoder-v1.json"
        rows = json.loads(fixture.read_text())
        for row in rows:
            with self.subTest(dim=row["dim"], text=row["text"]):
                self.assertEqual(encoding(row["text"], row["dim"], UNION), row["capped_union"])

    def test_rank_oracle_rejects_tampering_and_accepts_boundary_ties(self):
        vectors = {f"d{i}": [1, i+2] for i in range(11)}
        query = [1]
        valid = [{"id": f"d{i}", "similarity": 0.5} for i in range(10)]
        result_scores(valid, query, vectors)
        alternate = [{"id": f"d{i}", "similarity": 0.5} for i in range(1, 11)]
        result_scores(alternate, query, vectors)
        variants = [valid[:9], valid + [alternate[-1]], [valid[0]] * 10]
        incorrect_score = copy.deepcopy(valid)
        incorrect_score[0]["similarity"] = 0.51
        foreign = copy.deepcopy(valid)
        foreign[0]["id"] = "foreign"
        variants.extend([incorrect_score, foreign])
        for rows in variants:
            with self.assertRaises(AssertionError):
                result_scores(rows, query, vectors)

    def test_exact_scan_includes_zero_scores_and_empty_vector_identity(self):
        vectors = {"match": [1], "empty": [], "other": [2]}
        result_scores([{"id": "match", "similarity": 1.0},
                       {"id": "empty", "similarity": 0.0},
                       {"id": "other", "similarity": 0.0}], [1], vectors)
        result_scores([{"id": "empty", "similarity": 1.0},
                       {"id": "match", "similarity": 0.0},
                       {"id": "other", "similarity": 0.0}], [], vectors)


if __name__ == "__main__":
    unittest.main()
