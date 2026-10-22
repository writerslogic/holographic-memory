# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Synthetic source-integrity and exact serialized-budget checks."""

import copy
import json
import unittest

from longmemeval_reader_budget import cap_sources_user_first


def source(source_id, role, text):
    return {"source_id": source_id, "session_id": "session", "role": role, "text": text}


def group(group_id, rank, *sources):
    return {"group_id": group_id, "selection_rank": rank, "sources": list(sources)}


def serialize(groups):
    return "reader framing: " + json.dumps({"question": "What changed?", "source_groups": groups},
                                          sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def user_groups(arm):
    return [dict(item, sources=[value for value in item["sources"] if value["role"] == "user"])
            for item in arm["source_groups"]]


class ReaderBudgetTests(unittest.TestCase):
    def setUp(self):
        self.arm = {"metadata": {"frozen": True}, "source_groups": [
            group("anchor", 0, source("u0", "user", "I used to live in Rome."),
                  source("a0", "assistant", "A very verbose recommendation. " * 100)),
            group("later", 3, source("u1", "user", "I moved to Paris."),
                  source("a1", "assistant", "The ticket cost was 35 euros.")),
        ]}

    def cap(self, cap, arm=None):
        return cap_sources_user_first(self.arm if arm is None else arm, serialize, len, cap)

    def test_user_budget_minus_one_at_and_plus_one(self):
        users = user_groups(self.arm)
        boundary = len(serialize(users))
        for cap in (boundary - 1, boundary, boundary + 1):
            with self.subTest(cap=cap):
                result = self.cap(cap)
                selected = result["source_groups"]
                self.assertEqual(selected, users if cap >= boundary else users[:1])
                self.assertEqual(result["reader_cap"]["dropped_group_ids"], [] if cap >= boundary else ["later"])
                self.assertFalse(result["reader_cap"]["selection_failure"])
                self.assertEqual(result["reader_cap"]["input_tokens"], len(serialize(selected)))
                self.assertLessEqual(result["reader_cap"]["input_tokens"], cap)

    def test_anchor_budget_minus_one_at_and_plus_one(self):
        users = user_groups(self.arm)
        boundary = len(serialize(users[:1]))
        for cap in (boundary - 1, boundary, boundary + 1):
            with self.subTest(cap=cap):
                result = self.cap(cap)
                self.assertEqual(result["source_groups"], users[:1])
                self.assertEqual(result["reader_cap"]["selection_failure"], cap < boundary)
                self.assertEqual(result["reader_cap"]["failure_reason"],
                                 "anchor_user_input_exceeds_cap" if cap < boundary else None)

    def test_verbose_assistant_does_not_displace_user_fact(self):
        users = user_groups(self.arm)
        result = self.cap(len(serialize(users)))
        self.assertEqual(result["source_groups"], users)
        self.assertEqual(result["reader_cap"]["preserved_user_source_ids"], ["u0", "u1"])
        self.assertEqual(result["reader_cap"]["omitted_assistant_source_ids"], ["a0", "a1"])
        self.assertEqual(result["reader_cap"]["source_policy"], "user_first")
        self.assertEqual(result["reader_cap"]["uncapped_input_tokens"], len(serialize(self.arm["source_groups"])))

    def test_assistant_origin_fact_retained_when_it_fits(self):
        wanted = user_groups(self.arm)
        wanted[1]["sources"].append(self.arm["source_groups"][1]["sources"][1])
        boundary = len(serialize(wanted))
        for cap in (boundary - 1, boundary, boundary + 1):
            with self.subTest(cap=cap):
                result = self.cap(cap)
                self.assertEqual(result["source_groups"], wanted if cap >= boundary else user_groups(self.arm))
                self.assertEqual(result["reader_cap"]["omitted_assistant_source_ids"],
                                 ["a0"] if cap >= boundary else ["a0", "a1"])

    def test_all_sources_remain_verbatim_and_input_is_immutable(self):
        before = copy.deepcopy(self.arm)
        result = self.cap(len(serialize(self.arm["source_groups"])))
        self.assertEqual(result["source_groups"], self.arm["source_groups"])
        self.assertEqual(result["metadata"], self.arm["metadata"])
        self.assertEqual(result["reader_cap"]["omitted_assistant_source_ids"], [])
        result["metadata"]["frozen"] = False
        result["source_groups"][0]["sources"][0]["text"] = "changed"
        self.assertEqual(self.arm, before)

    def test_assistants_are_considered_by_rank_and_source_order(self):
        first = source("a-first", "assistant", "First answer.")
        second = source("a-next-", "assistant", "Later answer.")
        arm = {"source_groups": [
            group("anchor", 0, source("u0", "user", "A")),
            group("low", 9, source("u9", "user", "B"), second),
            group("high", 2, source("u2", "user", "C"), first),
        ]}
        expected = user_groups(arm)
        expected[2]["sources"].append(first)
        result = self.cap(len(serialize(expected)), arm)
        self.assertEqual(result["source_groups"], expected)

    def test_multiple_user_occurrences_and_source_order_survive(self):
        arm = {"source_groups": [group("anchor", 0,
            source("u0", "user", "First complete turn. α"),
            source("a0", "assistant", "First complete answer."),
            source("u1", "user", "Second complete turn. β"),
            source("a1", "assistant", "Second complete answer."))]}
        result = self.cap(len(serialize(arm["source_groups"])), arm)
        self.assertEqual(result["source_groups"], arm["source_groups"])
        self.assertEqual(result["reader_cap"]["preserved_user_source_ids"], ["u0", "u1"])

    def test_dropped_groups_never_restore_their_assistants(self):
        arm = copy.deepcopy(self.arm)
        arm["source_groups"][1]["sources"][0]["text"] *= 100
        arm["source_groups"][1]["sources"][1]["text"] = "short"
        boundary = len(serialize(user_groups(arm)[:1]))
        result = self.cap(boundary + 200, arm)
        self.assertEqual(result["reader_cap"]["dropped_group_ids"], ["later"])
        self.assertEqual(result["source_groups"], user_groups(arm)[:1])
        self.assertIn("a1", result["reader_cap"]["omitted_assistant_source_ids"])

    def test_exact_full_serialized_count_controls_each_admission(self):
        calls = []
        def count(text):
            calls.append(text)
            return len(text) + (1000 if '"role":"assistant"' in text else 17)
        boundary = count(serialize(user_groups(self.arm)))
        result = cap_sources_user_first(self.arm, serialize, count, boundary)
        self.assertEqual(result["source_groups"], user_groups(self.arm))
        self.assertEqual(result["reader_cap"]["input_tokens"], count(serialize(result["source_groups"])))
        self.assertTrue(all(text.startswith("reader framing: ") for text in calls))

    def test_invalid_cap_count_and_serialization_are_rejected(self):
        for cap in (0, -1, True, 1.5, "100"):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "positive integer"):
                self.cap(cap)
        for count in (-1, True, 1.5, "100"):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "invalid count"):
                cap_sources_user_first(self.arm, serialize, lambda text: count, 10000)
        with self.assertRaisesRegex(ValueError, "serialization must be text"):
            cap_sources_user_first(self.arm, lambda groups: groups, len, 10000)

    def test_malformed_source_groups_are_rejected(self):
        invalid = [None, {}, {"source_groups": []}, {"source_groups": [None]}]
        mutations = [
            lambda arm: arm["source_groups"][0].update(selection_rank=1),
            lambda arm: arm["source_groups"][1].update(selection_rank=True),
            lambda arm: arm["source_groups"][1].update(selection_rank=-1),
            lambda arm: arm["source_groups"][1].update(group_id="anchor"),
            lambda arm: arm["source_groups"][0].update(sources=[]),
            lambda arm: arm["source_groups"][0].update(sources=[source("a", "assistant", "Only assistant")]),
            lambda arm: arm["source_groups"][0]["sources"][0].update(role="system"),
            lambda arm: arm["source_groups"][1]["sources"][0].update(source_id="u0"),
            lambda arm: arm["source_groups"][0]["sources"][0].update(text=42),
        ]
        for mutation in mutations:
            arm = copy.deepcopy(self.arm)
            mutation(arm)
            invalid.append(arm)
        for arm in invalid:
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                cap_sources_user_first(arm, serialize, len, 10000)

    def test_group_count_limit_minus_one_at_and_plus_one(self):
        for size in (63, 64, 65):
            arm = {"source_groups": [group(f"g{i}", i, source(f"u{i}", "user", "Whole turn."))
                                      for i in range(size)]}
            with self.subTest(size=size):
                if size > 64:
                    with self.assertRaisesRegex(ValueError, "1..=64"):
                        self.cap(100000, arm)
                else:
                    self.assertEqual(self.cap(100000, arm)["source_groups"], arm["source_groups"])


if __name__ == "__main__":
    unittest.main()
