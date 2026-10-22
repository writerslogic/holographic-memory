# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import unittest

import longmemeval_retrieval as R


def source(identity, role="user"):
    return {"sourceId": identity, "role": role, "text": identity}


def group(identity, session, *sources):
    return {"id": identity, "session": session, "sources": list(sources)}


class RetrievalTests(unittest.TestCase):
    def test_round_cutoff_does_not_count_distinct_sessions_beyond_the_budget(self):
        groups = [group(str(i), "repeated", source(str(i))) for i in range(10)]
        groups.append(group("last", "evidence", source("gold", "assistant")))
        result = R.measure(groups, {"evidence"}, {"gold"})
        self.assertEqual(result["session_recall@10"], 0)
        self.assertEqual(result["session_recall@20"], 1)
        self.assertEqual(result["turn_recall@10"], 0)
        self.assertEqual(result["turn_recall@20"], 1)

    def test_turn_budget_counts_both_roles_and_deduplicates_sources(self):
        groups = [group("a", "s", source("a"), source("b", "assistant"))]
        groups.extend(group(str(i), "s", source("a"), source(str(i))) for i in range(1, 10))
        result = R.measure(groups, {"s"}, {"9"})
        self.assertEqual(result["turn_recall@10"], 0)
        self.assertEqual(result["round_turn_recall@10"], 1)
        self.assertIsNone(R.measure(groups, set(), set())["session_recall@10"])

    def test_aggregation_intent_is_text_only_and_word_bounded(self):
        for text in ("How many visits did I make?", "What is the total?", "List all my courses."):
            self.assertTrue(R.AGGREGATION.search(text))
        for text in ("Tell me about Totality", "My counter is broken", "What did I buy?"):
            self.assertFalse(R.AGGREGATION.search(text))

    def test_mmr_preserves_relevance_order_and_penalizes_repeated_sessions(self):
        groups = [group("a", "s"), group("b", "s"), group("c", "t")]
        scores = {0: 1.0, 1: 0.9, 2: 0.8}
        self.assertEqual(R.diverse_order([0, 1, 2], scores, groups, 0), [0, 1, 2])
        self.assertEqual(R.diverse_order([0, 1, 2], scores, groups, 0.25), [0, 2, 1])
        for strength in (-0.001, 1.001, float("nan")):
            with self.assertRaises(ValueError):
                R.diverse_order([0, 1, 2], scores, groups, strength)

    def test_expansion_uses_only_original_query_scores(self):
        groups = [group(str(i), str(i), source(str(i))) for i in range(3)]
        result = R.rank_rounds("a question", groups, [0, 1, 2], expansion=({0, 2}, {0: 0.1, 2: 0.9}))
        self.assertEqual(result, [2, 0, 1])
        with self.assertRaises(ValueError):
            R.rank_rounds("a question", groups, [0, 1, 2], expansion=({0, 2}, {0: 0.1}))

    def test_native_metrics_never_invent_the_unreturned_half_of_a_round(self):
        original = {"sourceId": "user", "sessionId": "s", "turnIndex": 0, "role": "user", "text": "hello"}
        returned = {**original, "recordId": "record", "premiseRecordIds": []}
        evidence = {"sources": [returned], "anchors": [{"recordId": "record"}]}
        raw = [group("round", "s", original, source("assistant", "assistant"))]
        selected = R.native_groups(evidence, {"user": original}, raw)
        self.assertEqual(R.measure(selected, {"s"}, {"assistant"})["turn_recall@10"], 0)
        evidence["sources"][0]["text"] = "changed"
        with self.assertRaises(ValueError):
            R.native_groups(evidence, {"user": original}, raw)

    def test_oracle_respects_premise_round_costs_and_missing_sessions(self):
        groups, sources, anchors = [], [], []
        for si in range(4):
            premises = []
            for ti in range(3):
                identity = f"{si}:{ti}"
                member = {**source(identity), "sessionId": str(si), "recordId": identity,
                          "premiseRecordIds": list(premises)}
                sources.append(member)
                groups.append(group(identity, str(si), member))
                premises.append(identity)
            anchors.append({"recordId": identity})
        result = R.budget_ceiling({"sources": sources, "anchors": anchors}, groups,
                                  {"0", "1", "2", "3", "missing"}, {"3:2"})
        self.assertEqual(result["candidate_session_recall"], 0.8)
        self.assertEqual(result["candidate_turn_recall"], 1)
        self.assertEqual(result["session_oracle@10"], 0.6)
        self.assertEqual(result["session_oracle@20"], 0.8)


if __name__ == "__main__":
    unittest.main()
