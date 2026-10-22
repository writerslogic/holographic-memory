# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Whole-turn prefix admission and user-preservation budget boundaries."""

import copy
import json
import unittest

from longmemeval_history_budget import select_prefix


def groups():
    return [{"group_id": "first", "sources": [{"source_id": "u0", "role": "user", "text": "Complete user fact. αβ"},
            {"source_id": "a0", "role": "assistant", "text": "Complete assistant fact."}]},
            {"group_id": "orphan", "sources": [{"source_id": "a1", "role": "assistant", "text": "An unpaired assistant fact."}]},
            {"group_id": "later", "sources": [{"source_id": "u1", "role": "user", "text": "A later complete user fact."},
            {"source_id": "a2", "role": "assistant", "text": "A later complete assistant fact."}]}]


def serialize(retained):
    return "Full reader framing: " + json.dumps(retained, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class HistoryBudgetTests(unittest.TestCase):
    def test_full_context_at_budget_minus_one_at_and_plus_one(self):
        original = groups()
        size = len(serialize(original))
        for cap in (size - 1, size, size + 1):
            with self.subTest(cap=cap):
                result = select_prefix(original, [0, 1, 2], serialize, len, cap)
                self.assertEqual(result["source_groups"], original if cap >= size else original[:2])
                self.assertEqual(result["input_tokens"], len(serialize(result["source_groups"])))
                self.assertLessEqual(result["input_tokens"], cap)

    def test_preserves_all_users_and_whole_selected_assistant_turns(self):
        original = groups()
        expected = copy.deepcopy(original)
        expected[2]["sources"] = expected[2]["sources"][:1]
        result = select_prefix(original, [1, 0, 2], serialize, len, len(serialize(expected)), preserve_users=True)
        self.assertEqual(result["source_groups"], expected)
        self.assertEqual(result["promoted_groups"], [1, 0])
        self.assertEqual({source["source_id"] for group in result["source_groups"] for source in group["sources"]
                          if source["role"] == "user"}, {"u0", "u1"})

    def test_user_only_overflow_keeps_the_complete_user_payload_and_marks_failure(self):
        original = groups()
        users = [dict(group, sources=[source for source in group["sources"] if source["role"] == "user"])
                 for group in original if any(source["role"] == "user" for source in group["sources"])]
        boundary = len(serialize(users))
        for cap in (boundary - 1, boundary, boundary + 1):
            with self.subTest(cap=cap):
                result = select_prefix(original, [0, 1, 2], serialize, len, cap, preserve_users=True)
                self.assertEqual(result["source_groups"], users)
                self.assertEqual(result["selection_failure"], cap < boundary)
                self.assertEqual(result["failure_reason"], "all_user_input_exceeds_cap" if cap < boundary else None)

    def test_whole_exchange_policy_never_splits_a_source_group_and_does_not_mutate_input(self):
        original = groups()
        before = copy.deepcopy(original)
        selected = [original[0], original[2]]
        result = select_prefix(original, [2, 0, 1], serialize, len, len(serialize(selected)))
        self.assertEqual(result["source_groups"], selected)
        self.assertEqual(result["promoted_groups"], [2, 0])
        result["source_groups"][0]["sources"][0]["text"] = "changed"
        self.assertEqual(original, before)

    def test_rejects_duplicate_or_incomplete_prior_and_invalid_token_counts(self):
        for prior in ([0, 0, 2], [0, 1], [0, 1, 3], [False, 1, 2]):
            with self.subTest(prior=prior), self.assertRaisesRegex(ValueError, "permutation"):
                select_prefix(groups(), prior, serialize, len, 1000)
        for count in (-1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "invalid count"):
                select_prefix(groups(), [0, 1, 2], serialize, lambda text: count, 1000)
        for cap in (0, True, 1.5):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "positive integer"):
                select_prefix(groups(), [0, 1, 2], serialize, len, cap)


if __name__ == "__main__":
    unittest.main()
