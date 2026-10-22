# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Complete-source input, stage-specific requests, frozen scope, and spend boundaries."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import longmemeval_complete_history as complete


CAPS = {"total": 250, "reader_judge_dev": 120, "qwen8b": 60, "x86": 60, "reserve": 10, "heldout400": 0}
QID = "synthetic-dev"
DATE = "2024/01/02 (Tue) 09:00"


def base_protocol():
    return {"reader_prompt": "Original reader instructions.\n", "judge_source": {"templates": {}}, "systems": {
        complete.SYSTEM: {"reader_model": "gpt-5.4-2026-03-05", "judge_model": "gpt-5.4-2026-03-05",
                          "reasoning_effort": "medium", "reader_max_output_tokens": 4096,
                          "judge_max_output_tokens": 2048}}}


def question():
    return {"qid": QID, "question": "What changed?", "question_date": DATE, "qtype": "SECRET_TYPE",
            "answer": "SECRET_REFERENCE", "has_answer": "SECRET_LABEL", "answer_session_ids": ["SECRET_SESSION"]}


def fixture():
    raw = {"question_id": QID, "question": "What changed?", "question_date": DATE,
           "question_type": "single-session-assistant", "answer": "SECRET_REFERENCE",
           "haystack_session_ids": ["answer_session", "orphan_session"], "haystack_dates": [DATE, DATE],
           "haystack_sessions": [
               [{"role": "assistant", "content": "An original leading assistant turn."},
                {"role": "user", "content": "An earlier user state. αβ", "has_answer": False},
                {"role": "assistant", "content": "The complete original reply."},
                {"role": "user", "content": "An explicit changed state.", "has_answer": True},
                {"role": "assistant", "content": "Another complete original reply."}],
               [{"role": "assistant", "content": "An original assistant-only conversation."}]]}
    return raw, {"noans_session_2", "answer_session_4"}


def ledger(cost, call_id="this-experiment"):
    return complete.experiment.merge_ledgers([{"caps_usd": CAPS, "entries": [
        {"cache_key": "previous", "call_id": call_id, "cost_usd": cost,
         "category": "reader_judge_dev", "status": "ok"}]}], CAPS)


class CompleteHistoryInputTests(unittest.TestCase):
    def setUp(self):
        self.protocol = complete.make_protocol(base_protocol())
        raw, aliases = fixture()
        self.raw = raw
        self.groups, _, self.orphans = complete.role.groups_for_question(raw, aliases)

    def test_every_original_source_including_leading_and_assistant_only_turns_survives(self):
        before = copy.deepcopy(self.groups)
        arm = complete.prepare_arm(question(), self.groups, self.protocol, len)
        payload = json.loads(arm["reader_input"][len(self.protocol["reader_prompt"]):])
        expected = [turn for session in self.raw["haystack_sessions"] for turn in session]
        self.assertEqual(len(payload["evidence"]), len(expected))
        self.assertEqual([source["text"] for source in payload["evidence"]], [turn["content"] for turn in expected])
        self.assertEqual([source["role"] for source in payload["evidence"]], [turn["role"] for turn in expected])
        self.assertEqual(len({source["source_id"] for source in payload["evidence"]}), len(expected))
        self.assertEqual(self.orphans, 2)
        self.assertEqual(arm["reader_cap"]["dropped_group_ids"], [])
        self.assertEqual(arm["source_groups"], before)
        arm["source_groups"][0]["sources"][0]["text"] = "changed"
        self.assertEqual(self.groups, before)

    def test_complete_input_limit_minus_one_at_and_plus_one_never_truncates(self):
        prompt = complete.serialize(question(), self.groups, self.protocol)
        boundary = len(prompt)
        for local_cap in (boundary - 1, boundary, boundary + 1):
            with self.subTest(local_cap=local_cap):
                protocol = {**self.protocol, "local_token_cap": local_cap, "token_cap": local_cap + 3,
                            "framing_reserve_tokens": 3}
                arm = complete.prepare_arm(question(), self.groups, protocol, len)
                self.assertEqual(arm["source_groups"], self.groups)
                self.assertEqual(arm["reader_input"], prompt)
                self.assertEqual(arm["reader_cap"]["input_tokens"], boundary)
                self.assertEqual(arm["reader_cap"]["selection_failure"], local_cap < boundary)
                self.assertEqual(arm["reader"]["status"], "selection_failure" if local_cap < boundary else "pending")

    def test_payload_has_only_question_date_and_original_evidence_without_labels(self):
        first = complete.prepare_arm(question(), self.groups, self.protocol, len)
        changed = {**question(), "qid": "SECRET_OTHER_ID", "qtype": "OTHER_TYPE", "answer": "OTHER_REFERENCE"}
        second = complete.prepare_arm(changed, self.groups, self.protocol, len)
        self.assertEqual(first["reader_input"], second["reader_input"])
        payload = json.loads(first["reader_input"][len(self.protocol["reader_prompt"]):])
        self.assertEqual(set(payload), {"question", "question_date", "evidence"})
        for marker in (QID, "SECRET_TYPE", "SECRET_REFERENCE", "SECRET_LABEL", "SECRET_SESSION", "OTHER_REFERENCE"):
            self.assertNotIn(marker, first["reader_input"])

    def test_group_count_is_not_limited_to_retrieval_top64(self):
        groups = []
        for index in range(65):
            source = copy.deepcopy(self.groups[0]["sources"][0])
            source["source_id"] = f"source-{index}"
            groups.append({"group_id": f"group-{index}", "sources": [source]})
        arm = complete.prepare_arm(question(), groups, self.protocol, len)
        self.assertEqual(len(arm["source_groups"]), 65)
        self.assertFalse(arm["reader_cap"]["selection_failure"])

    def test_stage_requests_pin_reader_high8192_and_judge_medium2048_without_mutating_base(self):
        base = base_protocol()
        unchanged = copy.deepcopy(base)
        protocol = complete.make_protocol(base)
        read = complete.request(protocol, "Reader prompt", "reader")
        judge = complete.request(protocol, "Official judge prompt", "judge")
        self.assertEqual(base, unchanged)
        self.assertEqual(read["reasoning"], {"effort": "high"})
        self.assertEqual(read["max_output_tokens"], 8192)
        self.assertEqual(judge["reasoning"], {"effort": "medium"})
        self.assertEqual(judge["max_output_tokens"], 2048)
        for req in (read, judge):
            self.assertEqual(req["model"], "gpt-5.4-2026-03-05")
            self.assertEqual(req["truncation"], "disabled")
            self.assertFalse(req["store"])
        self.assertEqual(protocol["token_cap"], 196608)
        self.assertEqual(protocol["local_token_cap"], 196096)
        self.assertEqual(protocol["new_spend_cap_usd"], 50)
        self.assertEqual(protocol["reader_prompt"], complete.experiment.make_protocol(base)["reader_prompt"]
                         + complete.COMPLETE_REASONING_RULES)
        with self.assertRaisesRegex(ValueError, "invalid experiment stage"):
            complete.request(protocol, "prompt", "other")

    def test_invalid_counts_roles_and_duplicate_source_occurrences_are_rejected(self):
        for invalid in (-1, True, 1.5):
            with self.subTest(count=invalid), self.assertRaisesRegex(ValueError, "invalid count"):
                complete.prepare_arm(question(), self.groups, self.protocol, lambda text: invalid)
        duplicate = copy.deepcopy(self.groups)
        duplicate[-1]["sources"][0]["source_id"] = duplicate[0]["sources"][0]["source_id"]
        with self.assertRaisesRegex(ValueError, "duplicate retained source"):
            complete.prepare_arm(question(), duplicate, self.protocol, len)
        malformed = copy.deepcopy(self.groups)
        malformed[0]["sources"][0]["role"] = "system"
        with self.assertRaisesRegex(ValueError, "invalid complete history source"):
            complete.prepare_arm(question(), malformed, self.protocol, len)

    def test_all_100_inputs_must_remain_frozen_including_the_last_question(self):
        arm = complete.prepare_arm(question(), self.groups, self.protocol, len)
        rows = [{**question(), "qid": f"dev-{index}", "systems": {complete.SYSTEM: {"arms": {complete.ARM: copy.deepcopy(arm)}}}}
                for index in range(100)]
        dev = {row["qid"] for row in rows}
        protocol = {**self.protocol, "prepared_inputs": complete.experiment.input_manifest(rows)}
        complete.validate_inputs(rows, dev, protocol)
        with self.assertRaisesRegex(ValueError, "100 dev"):
            complete.validate_inputs(rows[:-1], dev, protocol)
        rows[-1]["systems"][complete.SYSTEM]["arms"][complete.ARM]["source_groups"][0]["sources"][0]["text"] += " edited"
        with self.assertRaisesRegex(ValueError, "prepared complete history changed"):
            complete.validate_inputs(rows, dev, protocol)


class CompleteHistorySpendTests(unittest.TestCase):
    def setUp(self):
        self.protocol = complete.make_protocol(base_protocol())
        self.req = complete.request(self.protocol, "Frozen reader prompt.", "reader")
        self.reserve = complete.experiment.reservation_usd(len(self.req["input"]), self.req["max_output_tokens"])

    def guarded(self, work, current):
        with patch.object(complete.experiment, "refresh_ledger", return_value=current), \
                patch.object(complete.reader, "sha", return_value="protocol-fingerprint"), \
                patch.object(complete.reader, "api_call", return_value={"status": "paid-transport"}) as transport:
            result = complete.guarded_api_call(None, self.protocol, self.req, "new-call", work,
                                               {"this-experiment"}, len)
            return result, transport.call_count

    def test_new_spend_budget_minus_one_at_and_plus_one_uses_50_instead_of_40(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for extra in (-0.00001, 0, 0.00001):
                with self.subTest(extra=extra):
                    current = ledger(50 - self.reserve + extra)
                    result, calls = self.guarded(work, current)
                    self.assertEqual(calls, int(extra <= 0))
                    self.assertEqual(result["status"], "paid-transport" if extra <= 0 else "budget_exhausted")
                    if extra > 0:
                        self.assertEqual(result["budget_scope"], "new_variant_spend_cap")
                        self.assertEqual(result["cost_usd"], 0)

    def test_global120_and250_caps_are_enforced_under_the_larger_variant_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            current = ledger(120 - self.reserve + 0.00001, call_id="other-experiment")
            result, calls = self.guarded(work, current)
            self.assertEqual(calls, 0)
            self.assertEqual(result["budget_scope"], "reader_judge_dev_spend_cap")
            current["total_usd"] = 250 - self.reserve + 0.00001
            result, calls = self.guarded(work, current)
            self.assertEqual(calls, 0)
            self.assertEqual(result["budget_scope"], "global_total_spend_cap")

    def test_paid_cache_remains_resumable_after_spend_limit_without_new_charge(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            current = ledger(50)
            denied, calls = self.guarded(work, current)
            self.assertEqual(calls, 0)
            cache = work / "cache" / (denied["cache_key"] + ".json")
            cache.parent.mkdir()
            cache.write_text("{}")
            result, calls = self.guarded(work, current)
            self.assertEqual(calls, 1)
            self.assertEqual(result["status"], "paid-transport")


if __name__ == "__main__":
    unittest.main()
