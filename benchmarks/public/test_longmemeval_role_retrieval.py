# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6"]
# ///
"""Role-aware retrieval regressions without corpus access, labels, or model calls."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import longmemeval_role_retrieval as experiment


def group(index, user="Can you give me advice?", assistant="A generic response."):
    return {"group_id": f"opaque-group-{index}", "sources": [
        {"source_id": f"opaque-user-{index}", "role": "user", "text": user},
        {"source_id": f"opaque-assistant-{index}", "role": "assistant", "text": assistant}]}


class RoleRetrievalTests(unittest.TestCase):
    def test_assistant_only_rare_fact_enters_a_pool_behind_generic_user_candidates(self):
        groups = [group(index) for index in range(100)]
        groups[-1]["sources"][1]["text"] = "The heliotrope renderer increased frame rate by 20%."
        result = experiment.retrieve("What frame rate increase did the heliotrope renderer achieve?", groups, list(range(100)))
        self.assertEqual(result["selected"][:5], list(range(5)))
        self.assertEqual(result["selected"][5], 99)
        self.assertEqual(result["assistant_ranking"], [99])
        self.assertEqual(result["source_groups"][5], groups[99])
        self.assertEqual(len(result["selected"]), 64)

    def test_user_text_does_not_receive_an_assistant_vote(self):
        groups = [group(index) for index in range(8)]
        groups[7]["sources"][0]["text"] = "I own a heliotrope renderer."
        result = experiment.retrieve("heliotrope renderer", groups, list(range(8)))
        self.assertEqual(result["assistant_ranking"], [])
        self.assertEqual(result["selected"], list(range(8)))
        self.assertEqual(result["assistant_scores"], [0.0] * 8)

    def test_assistant_suggestions_remain_assistant_evidence_and_all_quotes_are_verbatim(self):
        groups = [group(index) for index in range(8)]
        groups[-1]["sources"][1]["text"] = "You could buy the heliotrope model; this is a suggestion, not a purchase. αβ"
        unchanged = copy.deepcopy(groups)
        result = experiment.retrieve("Which heliotrope model did I buy?", groups, list(range(8)))
        selected = next(item for item in result["source_groups"] if item["group_id"] == groups[-1]["group_id"])
        self.assertEqual(selected, groups[-1])
        self.assertEqual(selected["sources"][1]["role"], "assistant")
        self.assertNotIn("heliotrope", selected["sources"][0]["text"])
        result["source_groups"][0]["sources"][0]["text"] = "mutated"
        self.assertEqual(groups, unchanged)

    def test_original_first_five_remain_in_their_cached_order(self):
        groups = [group(index, assistant="heliotrope details") for index in range(100)]
        prior = [8, 5, 2, 13, 21, *[index for index in range(100) if index not in (8, 5, 2, 13, 21)]]
        result = experiment.retrieve("heliotrope", groups, prior)
        self.assertEqual(result["selected"][:5], prior[:5])
        self.assertEqual(result["reserved_user_prior"], prior[:5])
        self.assertEqual(len(set(result["selected"])), 64)

    def test_opaque_identity_renaming_does_not_affect_selection_or_scores(self):
        groups = [group(index, assistant="heliotrope" if index % 7 == 0 else "unrelated") for index in range(80)]
        renamed = copy.deepcopy(groups)
        for index, item in enumerate(renamed):
            item["group_id"] = f"renamed-{80 - index}"
            for source_index, source in enumerate(item["sources"]):
                source["source_id"] = f"renamed-source-{80 - index}-{source_index}"
        first = experiment.retrieve("heliotrope", groups, list(range(80)))
        second = experiment.retrieve("heliotrope", renamed, list(range(80)))
        for key in ("selected", "assistant_ranking", "assistant_scores", "fused_ranking", "fused_scores"):
            self.assertEqual(first[key], second[key])

    def test_only_positive_assistant_matches_vote_and_stop_words_do_not_match(self):
        groups = [group(0, assistant="the and or"), group(1, assistant="heliotrope"), group(2, assistant="unrelated")]
        result = experiment.retrieve("the heliotrope and", groups, [0, 1, 2])
        self.assertEqual(result["assistant_ranking"], [1])
        self.assertAlmostEqual(result["fused_scores"][0], 1 / 61)
        self.assertAlmostEqual(result["fused_scores"][1], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(result["fused_scores"][2], 1 / 63)
        empty = experiment.retrieve("the and or", groups, [0, 1, 2])
        self.assertEqual(empty["assistant_ranking"], [])

    def test_empty_assistant_documents_are_valid_without_spurious_scores(self):
        groups = [{"group_id": "one", "sources": [{"source_id": "source", "role": "user", "text": "A user fact."}]}]
        result = experiment.retrieve("heliotrope", groups, [0])
        self.assertEqual(result["selected"], [0])
        self.assertEqual(result["assistant_ranking"], [])

    def test_partial_user_prior_allows_unranked_assistant_sources(self):
        groups = [group(index) for index in range(8)]
        groups[-1]["sources"][1]["text"] = "heliotrope"
        result = experiment.retrieve("heliotrope", groups, [0, 1, 2, 3, 4], limit=6)
        self.assertEqual(result["selected"], [0, 1, 2, 3, 4, 7])

    def test_rare_assistant_only_block_is_selectable_without_a_fabricated_user(self):
        groups = [group(index) for index in range(8)]
        orphan = {"group_id": "orphan-assistant-block", "sources": [
            {"source_id": "orphan-source", "role": "assistant", "text": "The heliotrope renderer achieved 20%."}]}
        groups.append(orphan)
        result = experiment.retrieve("heliotrope renderer", groups, list(range(8)), limit=6)
        self.assertEqual(result["selected"], [0, 1, 2, 3, 4, 8])
        self.assertEqual(result["source_groups"][-1], orphan)
        self.assertEqual(result["source_groups"][-1]["sources"][0]["role"], "assistant")

    def test_candidate_limit_minus_one_at_and_plus_one(self):
        groups = [group(index) for index in range(80)]
        for limit in (63, 64, 65):
            with self.subTest(limit=limit):
                if limit > 64:
                    with self.assertRaisesRegex(ValueError, "1..=64"):
                        experiment.retrieve("heliotrope", groups, list(range(80)), limit=limit)
                else:
                    self.assertEqual(experiment.retrieve("heliotrope", groups, list(range(80)), limit=limit)["selected"], list(range(limit)))
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                experiment.retrieve("heliotrope", groups, list(range(80)), limit=limit)

    def test_duplicate_identities_sources_and_prior_indices_are_rejected(self):
        original = [group(0), group(1)]
        mutations = [
            lambda groups: groups[1].update(group_id=groups[0]["group_id"]),
            lambda groups: groups[1]["sources"][0].update(source_id=groups[0]["sources"][0]["source_id"]),
            lambda groups: groups[0]["sources"].append(copy.deepcopy(groups[0]["sources"][0])),
        ]
        for mutate in mutations:
            groups = copy.deepcopy(original)
            mutate(groups)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                experiment.retrieve("heliotrope", groups, [0, 1])
        for prior in ([0, 0], [-1], [2], [True], [], "0"):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                experiment.retrieve("heliotrope", original, prior)

    def test_complete_exchange_shape_and_text_are_required(self):
        for sources in ([],
                        [{"source_id": "assistant", "role": "assistant", "text": "Orphan prelude"},
                         {"source_id": "user", "role": "user", "text": "A user turn"}],
                        [{"source_id": "system", "role": "system", "text": "Wrong role"}],
                        [{"source_id": "user", "role": "user", "text": 42}]):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                experiment.retrieve("heliotrope", [{"group_id": "group", "sources": sources}], [0])

    def test_dev_proof_merges_repeated_alias_occurrences_without_duplicate_prior_indices(self):
        dev = [f"dev-{index}" for index in range(100)]
        turn = [{"role": "user", "content": "A user request."},
                {"role": "assistant", "content": "heliotrope details"}]
        questions = {qid: {"question_id": qid, "question": "heliotrope", "question_type": "synthetic",
                          "haystack_session_ids": ["repeat", "repeat", "other"],
                          "haystack_dates": ["2024/01/02 (Tue) 09:00"] * 3,
                          "haystack_sessions": [copy.deepcopy(turn) for _ in range(3)]} for qid in dev}
        historical = {"rows": [{"qid": qid, "ids": ["repeat_1", "repeat_1", "other_1"],
                                "baseline_ranking": ["repeat_1", "repeat_1", "other_1"], "scored": index < 84}
                               for index, qid in enumerate(dev)]}
        snapshots = {"rows": [{"qid": qid, "systems": {"matched_gpt54": {"arms": {}}}} for qid in dev]}
        root = experiment.ROOT
        fixtures = {root / "benchmarks/results/evidence_completeness_dev.json": historical,
                    root / "benchmarks/public/longmemeval_split.json": {"dev": dev},
                    **{root / f"benchmarks/results/longmemeval_dev_answers_{variant}.json": snapshots
                       for variant in ("v1", "v3")}}
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            return json.dumps(fixtures[path]) if path in fixtures else original_read(path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "read_text", read), \
                patch("longmemeval_dev_conversations.load_dev_conversations", return_value=questions), \
                redirect_stdout(io.StringIO()):
            report = experiment.prove_dev(Path("unused-source"), root / "benchmarks/public/longmemeval_split.json",
                                          Path(directory) / "proof.json")
        self.assertEqual(report["summary"]["n"], 100)
        self.assertEqual(report["summary"]["scored"], 84)
        self.assertTrue(all(row["input_groups"] == row["selected_groups"] == 2 for row in report["rows"]))
        self.assertTrue(all(row["selected_assistant_sources"] == 3 for row in report["rows"]))


if __name__ == "__main__":
    unittest.main()
