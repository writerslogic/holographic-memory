# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Evidence integrity and cap boundaries using the existing pinned dev sources."""

import copy
import json
import unittest
from pathlib import Path

from longmemeval_answer_evidence import (
    ARMS, _groups_for_question, _observation_key, build_evidence, canonical, cap_sources, reader_payload,
)

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path.home() / ".cache/hms-bench/downloads/longmemeval_s_cleaned.json"


@unittest.skipUnless(SOURCE.exists(), "pinned LongMemEval S dev source is not cached")
class AnswerEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = build_evidence(
            SOURCE, REPO / "benchmarks/public/longmemeval_split.json",
            REPO / "benchmarks/results/evidence_budget_dev.json",
        )
        cls.dev_ids = {question["qid"] for question in cls.questions}
        cls.raw_dev = {question["question_id"]: question for question in json.loads(SOURCE.read_text()) if question["question_id"] in cls.dev_ids}

    def test_all_dev_exchanges_have_verbatim_roles_and_distinct_occurrences(self):
        self.assertEqual(len(self.questions), 100)
        for question in self.questions:
            raw = self.raw_dev[question["qid"]]
            for name in ARMS:
                with self.subTest(qid=question["qid"], arm=name):
                    arm = question["arms"][name]
                    payload = reader_payload(question, arm)
                    records = payload["evidence"]
                    self.assertEqual(set(payload), {"question", "question_date", "evidence"})
                    self.assertEqual(len({record["source_id"] for record in records}), len(records))
                    source_ids = set()
                    for group in arm["source_groups"]:
                        for source in group["sources"]:
                            session = raw["haystack_sessions"][source["session_index"]]
                            original = session[source["turn_index"]]
                            self.assertEqual((source["role"], source["text"]), (original["role"], original["content"]))
                            self.assertEqual(source["observation_time"], raw["haystack_dates"][source["session_index"]])
                            source_ids.add(source["source_id"])
                            if source["role"] == "user":
                                index = source["turn_index"] + 1
                                expected = []
                                while index < len(session) and session[index]["role"] == "assistant":
                                    expected.append(index)
                                    index += 1
                                actual = [item["turn_index"] for item in group["sources"] if item["session_index"] == source["session_index"] and source["turn_index"] < item["turn_index"] < index]
                                self.assertEqual(actual, expected)
                    self.assertEqual(source_ids, {record["source_id"] for record in records})
                    times = [_observation_key(record["observation_time"]) for record in records]
                    self.assertEqual(times, sorted(times))
                    for record in records:
                        self.assertEqual(set(record), {"source_id", "session_id", "observation_time", "role", "text"})

    def test_replay_is_independent_of_raw_answer_fields(self):
        question = self.questions[0]
        raw = copy.deepcopy(self.raw_dev[question["qid"]])
        original = json.loads((REPO / "benchmarks/results/evidence_completeness_dev.json").read_text())
        historical = next(row for row in original["rows"] if row["qid"] == question["qid"])
        expected = _groups_for_question(raw, set(historical["ids"]))
        for key in ("answer", "answer_session_ids", "question_type"):
            raw.pop(key, None)
        for session in raw["haystack_sessions"]:
            for turn in session:
                turn.pop("has_answer", None)
        self.assertEqual(_groups_for_question(raw, set(historical["ids"])), expected)

    def test_existing_repeated_session_occurrences_keep_distinct_identities(self):
        raw = next(question for question in self.raw_dev.values() if len(set(question["haystack_session_ids"])) < len(question["haystack_session_ids"]))
        original = json.loads((REPO / "benchmarks/results/evidence_completeness_dev.json").read_text())
        historical = next(row for row in original["rows"] if row["qid"] == raw["question_id"])
        _, groups = _groups_for_question(raw, set(historical["ids"]))
        repeated = [group for group in groups.values() if sum(source["role"] == "user" for source in group) > 1]
        self.assertTrue(repeated)
        for group in repeated:
            users = [source for source in group if source["role"] == "user"]
            self.assertEqual(len({source["source_id"] for source in group}), len(group))
            self.assertEqual(len({source["session_id"] for source in users}), len(users))
            self.assertEqual(len({source["session_index"] for source in users}), len(users))

    def test_complete_input_limit_minus_one_limit_plus_one(self):
        question = self.questions[0]
        arm = question["arms"]["first_five"]
        def serialize(groups):
            return canonical(reader_payload(question, {"source_groups": groups}))
        complete = len(serialize(arm["source_groups"]))
        unchanged = copy.deepcopy(arm)
        for cap in (complete - 1, complete, complete + 1):
            with self.subTest(cap=cap):
                capped = cap_sources(arm, serialize, len, cap)
                self.assertFalse(capped["reader_cap"]["selection_failure"])
                self.assertLessEqual(capped["reader_cap"]["input_tokens"], cap)
                self.assertEqual(capped["reader_cap"]["uncapped_input_tokens"], complete)
                expected = [] if cap >= complete else [max(arm["source_groups"], key=lambda group: group["selection_rank"])["group_id"]]
                self.assertEqual(capped["reader_cap"]["dropped_group_ids"], expected)
                for group in capped["source_groups"]:
                    self.assertEqual(group, next(item for item in arm["source_groups"] if item["group_id"] == group["group_id"]))
        self.assertEqual(arm, unchanged)

    def test_anchor_failure_preserves_complete_group_and_prevents_reader_input(self):
        question = self.questions[0]
        arm = question["arms"]["operand"]
        def serialize(groups):
            return canonical(reader_payload(question, {"source_groups": groups}))
        anchor = arm["source_groups"][0]
        anchor_count = len(serialize([anchor]))
        for cap in (anchor_count - 1, anchor_count, anchor_count + 1):
            with self.subTest(cap=cap):
                capped = cap_sources(arm, serialize, len, cap)
                self.assertEqual(capped["source_groups"][0], anchor)
                self.assertEqual(capped["reader_cap"]["selection_failure"], cap < anchor_count)
                if cap < anchor_count:
                    self.assertEqual(capped["source_groups"], [anchor])
                    with self.assertRaisesRegex(ValueError, "selection failed"):
                        reader_payload(question, capped)
                else:
                    self.assertLessEqual(capped["reader_cap"]["input_tokens"], cap)

    def test_invalid_tokenizer_counts_rejected(self):
        arm = self.questions[0]["arms"]["operand"]
        for invalid in (-1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "invalid count"):
                    cap_sources(arm, lambda groups: canonical(groups), lambda text: invalid, 32768)


if __name__ == "__main__":
    unittest.main()
