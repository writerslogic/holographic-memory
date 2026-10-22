# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = ["tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Offline full-history input, source-integrity, acceptance, and spending checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import longmemeval_full_history as experiment


CAPS = {"total": 250, "reader_judge_dev": 120, "qwen8b": 60, "x86": 60, "reserve": 10, "heldout400": 0}
QID = "dev-fixture"
DATE = "2024/01/02 (Tue) 09:00"


def base_protocol():
    return {"reader_prompt": "Original reader instructions.\n", "judge_source": {"templates": {}}, "systems": {
        experiment.SYSTEM: {"reader_model": "gpt-5.4-2026-03-05", "judge_model": "gpt-5.4-2026-03-05",
                            "reasoning_effort": "medium", "reader_max_output_tokens": 4096,
                            "judge_max_output_tokens": 2048}}}


def fixture_groups(size=2, known_assistant=False):
    sessions = [{"sid": "archive-session", "date": DATE,
                 "turns": [(f"archive-session_{2 * i + 1}", f"Complete observation {i}. αβ") for i in range(size)]}]
    candidates = [tid for tid, _ in sessions[0]["turns"]]
    snapshots = []
    if known_assistant:
        user = {"source_id": experiment.reader.digest(experiment.reader.canonical(["lme-source-v1", QID, 5, 0])),
                "session_id": experiment.reader.digest(experiment.reader.canonical(["lme-session-v1", QID, 5])),
                "session_index": 5, "turn_index": 0, "observation_time": DATE, "role": "user",
                "text": sessions[0]["turns"][0][1]}
        assistant = {**user, "source_id": experiment.reader.digest("known-assistant"), "turn_index": 1,
                     "role": "assistant", "text": "The receipt recorded 35 euros."}
        snapshots = [{"systems": {experiment.SYSTEM: {"arms": {"operand": {"source_groups": [
            {"frozen_selected_id_sha256": experiment.reader.digest(candidates[0]), "sources": [user, assistant]}]}}}}}]
    groups = experiment.user_first.source_groups(QID, sessions, candidates, snapshots)
    return groups


def question():
    return {"qid": QID, "qtype": "synthetic-type", "question": "What changed?", "question_date": DATE,
            "answer": "SECRET_REFERENCE", "question_id": "SECRET_ID", "has_answer": "SECRET_LABEL"}


def ledger(entries=()):
    return experiment.merge_ledgers([{"caps_usd": CAPS, "entries": list(entries)}], CAPS)


def entry(key, cost, call_id="old", status="ok"):
    return {"cache_key": key, "cost_usd": cost, "call_id": call_id, "category": "reader_judge_dev", "status": status}


class FullHistoryInputTests(unittest.TestCase):
    def setUp(self):
        self.protocol = experiment.make_protocol(base_protocol())

    def test_more_than_64_groups_are_preserved_without_selection_or_clipping(self):
        groups = fixture_groups(65)
        before = copy.deepcopy(groups)
        arm = experiment.prepare_arm(question(), groups, self.protocol, len)
        payload = json.loads(arm["reader_input"][len(self.protocol["reader_prompt"]):])
        self.assertEqual(len(arm["source_groups"]), 65)
        self.assertEqual(len(payload["evidence"]), 65)
        self.assertEqual({source["text"] for source in payload["evidence"]},
                         {source["text"] for group in groups for source in group["sources"]})
        self.assertFalse(arm["reader_cap"]["selection_failure"])
        self.assertEqual(arm["reader_cap"]["dropped_group_ids"], [])
        arm["source_groups"][0]["sources"][0]["text"] = "mutated"
        self.assertEqual(groups, before)

    def test_exact_complete_prompt_boundary_minus_one_at_and_plus_one(self):
        groups = fixture_groups(3, known_assistant=True)
        full = experiment.serialize(question(), groups, self.protocol)
        boundary = len(full)
        for local_cap in (boundary - 1, boundary, boundary + 1):
            with self.subTest(local_cap=local_cap):
                protocol = {**self.protocol, "token_cap": local_cap + 3, "local_token_cap": local_cap,
                            "framing_reserve_tokens": 3}
                arm = experiment.prepare_arm(question(), groups, protocol, len)
                self.assertEqual(arm["reader_input"], full)
                self.assertEqual(arm["source_groups"], groups)
                self.assertEqual(arm["reader_cap"]["input_tokens"], boundary)
                self.assertEqual(arm["reader_cap"]["selection_failure"], local_cap < boundary)
                self.assertEqual(arm["reader"]["status"], "selection_failure" if local_cap < boundary else "pending")
                self.assertEqual(arm["reader_cap"]["failure_reason"],
                                 "complete_history_input_exceeds_cap" if local_cap < boundary else None)

    def test_reader_payload_excludes_gold_and_question_metadata(self):
        groups = fixture_groups(2)
        first = experiment.prepare_arm(question(), groups, self.protocol, len)
        changed = {**question(), "qid": "A_DIFFERENT_ID", "qtype": "SECRET_TYPE", "answer": "OTHER_REFERENCE"}
        second = experiment.prepare_arm(changed, groups, self.protocol, len)
        self.assertEqual(first["reader_input"], second["reader_input"])
        payload = json.loads(first["reader_input"][len(self.protocol["reader_prompt"]):])
        self.assertEqual(set(payload), {"question", "question_date", "evidence"})
        for secret in ("SECRET_REFERENCE", "SECRET_ID", "SECRET_LABEL", "SECRET_TYPE", "OTHER_REFERENCE", QID):
            self.assertNotIn(secret, first["reader_input"])

    def test_only_known_verbatim_assistant_occurrences_are_added(self):
        groups = fixture_groups(3, known_assistant=True)
        arm = experiment.prepare_arm(question(), groups, self.protocol, len)
        assistants = [source for source in arm["retained_sources"] if source["role"] == "assistant"]
        self.assertEqual(len(assistants), 1)
        self.assertEqual(assistants[0], groups[0]["sources"][1])
        self.assertEqual(assistants[0]["text"], "The receipt recorded 35 euros.")
        self.assertEqual([source["turn_index"] for source in arm["retained_sources"]], [0, 1, 2, 4])

    def test_pinned_model_settings_context_and_prompt_are_frozen_separately(self):
        base = base_protocol()
        before = copy.deepcopy(base)
        protocol = experiment.make_protocol(base)
        self.assertEqual(base, before)
        self.assertTrue(protocol["reader_prompt"].startswith(base["reader_prompt"]))
        self.assertEqual(protocol["token_cap"], 131072)
        self.assertEqual(protocol["local_token_cap"], 130560)
        self.assertEqual(protocol["reader_judge_concurrency"], 1)
        self.assertEqual(protocol["new_spend_cap_usd"], 40)
        changed = copy.deepcopy(base)
        changed["systems"][experiment.SYSTEM]["reasoning_effort"] = "high"
        with self.assertRaisesRegex(ValueError, "pinned reasoning_effort"):
            experiment.make_protocol(changed)

    def test_duplicate_occurrences_and_invalid_counts_fail_before_calls(self):
        groups = fixture_groups(2)
        groups[1]["sources"][0]["source_id"] = groups[0]["sources"][0]["source_id"]
        with self.assertRaisesRegex(ValueError, "duplicate retained source"):
            experiment.prepare_arm(question(), groups, self.protocol, len)
        for invalid in (-1, True, 1.5):
            with self.subTest(count=invalid), self.assertRaisesRegex(ValueError, "invalid count"):
                experiment.prepare_arm(question(), fixture_groups(), self.protocol, lambda text: invalid)

    def test_acceptance_requires_all_100_correct_with_finished_readers_and_judges(self):
        arm = experiment.prepare_arm(question(), fixture_groups(), self.protocol, len)
        arm["reader"]["status"] = "ok"
        arm["judge"]["status"] = "ok"
        arm["scores"]["correct"] = True
        rows = [{"qid": f"dev-{i}", "qtype": "synthetic", "scored_evidence": True, "actual_complete": True,
                 "systems": {experiment.SYSTEM: {"arms": {experiment.ARM: copy.deepcopy(arm)}}}} for i in range(100)]
        report = {"complete": True, "rows": rows}
        self.assertTrue(experiment.summarize(report)["acceptance_met"])
        rows[-1]["systems"][experiment.SYSTEM]["arms"][experiment.ARM]["scores"]["correct"] = False
        summary = experiment.summarize(report)
        self.assertEqual(summary["correct"], 99)
        self.assertFalse(summary["acceptance_met"])
        rows[-1]["systems"][experiment.SYSTEM]["arms"][experiment.ARM]["scores"]["correct"] = True
        rows[-1]["systems"][experiment.SYSTEM]["arms"][experiment.ARM]["judge"]["status"] = "pending"
        self.assertFalse(experiment.summarize(report)["acceptance_met"])
        with self.assertRaisesRegex(ValueError, "all 100"):
            experiment.summarize({"complete": True, "rows": rows[:-1]})


class FullHistorySpendTests(unittest.TestCase):
    def test_reservation_matches_existing_transport_prices_and_framing(self):
        self.assertAlmostEqual(experiment.reservation_usd(0, 4096), 0.06272)
        self.assertAlmostEqual(experiment.reservation_usd(130560, 4096), 0.38912)
        for counts in ((-1, 4096), (True, 4096), (0, 0), (0, True)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                experiment.reservation_usd(*counts)

    def test_merge_preserves_conservative_unique_costs_and_running_reservations(self):
        primary = {"caps_usd": CAPS, "entries": [entry("same", 1.0)]}
        running = {"caps_usd": CAPS, "entries": [entry("same", 1.5), entry("inflight", 2.0, status="reserved")]}
        merged = experiment.merge_ledgers([primary, running], CAPS)
        self.assertEqual(len(merged["entries"]), 2)
        self.assertEqual(merged["total_usd"], 3.5)
        self.assertEqual(next(value for value in merged["entries"] if value["cache_key"] == "inflight")["status"], "reserved")
        merged["entries"][0]["cost_usd"] = 99
        self.assertEqual(primary["entries"][0]["cost_usd"], 1.0)
        for cost in (-1, True, float("inf"), float("nan")):
            with self.subTest(cost=cost), self.assertRaisesRegex(ValueError, "invalid ledger entry"):
                experiment.merge_ledgers([{"caps_usd": CAPS, "entries": [entry("bad", cost)]}], CAPS)

    def test_variant_budget_minus_one_at_and_plus_one(self):
        current = ledger([entry("own", 39.0, call_id="this-experiment"), entry("other", 10.0)])
        for reserve in (0.99999, 1.0, 1.00001):
            with self.subTest(reserve=reserve):
                reason = experiment.budget_rejection(current, {"this-experiment"}, reserve)
                self.assertEqual(reason, "new_variant_spend_cap" if reserve > 1 else None)
        self.assertEqual(experiment.variant_spend(current, {"this-experiment"}), 39.0)

    def test_global_reader_and_total_caps_remain_enforced(self):
        current = ledger([entry("existing", 119.0)])
        self.assertIsNone(experiment.budget_rejection(current, set(), 1.0))
        self.assertEqual(experiment.budget_rejection(current, set(), 1.00001), "reader_judge_dev_spend_cap")
        current["total_usd"] = 249.0
        current["by_category_usd"]["reader_judge_dev"] = 0
        self.assertIsNone(experiment.budget_rejection(current, set(), 1.0))
        self.assertEqual(experiment.budget_rejection(current, set(), 1.00001), "global_total_spend_cap")

    def test_refresh_includes_new_experiment_ledgers_and_deduplicates_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "benchmarks/results"
            work = results / "longmemeval_full_history_v1"
            sibling = results / "longmemeval_user_first_v1"
            for path, value in ((results / "spend_2026-10.json", ledger([entry("seed", 1)])),
                                (sibling / "spend.json", ledger([entry("seed", 1), entry("running", 3, status="reserved")])),
                                (work / "spend.json", ledger([entry("own", 2)]))):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            with patch.object(experiment, "ROOT", root), patch.object(experiment, "RESULTS", results):
                refreshed = experiment.refresh_ledger(work)
            self.assertEqual(refreshed["total_usd"], 6)
            self.assertEqual(len(refreshed["entries"]), 3)
            self.assertIn(str((sibling / "spend.json").relative_to(root)), refreshed["observed_ledger_paths"])

    def test_variant_budget_refuses_paid_transport_but_allows_already_paid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            req = {"input": "Frozen question.", "max_output_tokens": 4096}
            own = ledger([entry("paid", 40, call_id="own")])
            with patch.object(experiment, "refresh_ledger", return_value=own), \
                    patch.object(experiment.reader, "sha", return_value="protocol-fingerprint"), \
                    patch.object(experiment.reader, "api_call", return_value={"status": "cached"}) as transport:
                denied = experiment.guarded_api_call(None, {}, req, "new", work, {"own"}, len)
                self.assertEqual(denied["status"], "budget_exhausted")
                self.assertEqual(denied["cost_usd"], 0)
                transport.assert_not_called()
                cache = work / "cache" / (denied["cache_key"] + ".json")
                cache.parent.mkdir()
                cache.write_text("{}")
                result = experiment.guarded_api_call(None, {}, req, "new", work, {"own"}, len)
                self.assertEqual(result, {"status": "cached"})
                transport.assert_called_once()


if __name__ == "__main__":
    unittest.main()
