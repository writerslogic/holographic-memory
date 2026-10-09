# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Offline reader integrity checks using the frozen real dev requests and sources."""

import copy
import io
import json
import random
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import longmemeval_reader_api as reader

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "target/superiority-validation/prepared_answers.json"


@unittest.skipUnless(PREPARED.exists(), "frozen real dev requests are not prepared")
class ReaderIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = json.loads(PREPARED.read_text())
        cls.protocol = json.loads((ROOT / "benchmarks/public/reader_protocol_v1.json").read_text())
        cls.ledger_template = json.loads((ROOT / "benchmarks/results/spend_2026-10.json").read_text())
        cls.real_arm = cls.prepared["rows"][0]["systems"]["matched_gpt54"]["arms"]["first_five"]
        cls.real_request = reader.request(cls.protocol, cls.real_arm["reader_input"], "reader")
        cls.source = next(source for source in cls.real_arm["retained_sources"] if source["role"] == "user")
        cls.quotation = cls.source["text"][:80]

    def source_answer(self):
        return {
            "answer": self.quotation,
            "abstain": False,
            "claims": [{"text": self.quotation, "citations": [
                {"source_id": self.source["source_id"], "quote": self.quotation},
            ]}],
        }

    def scored_arm(self, answer):
        arm = copy.deepcopy(self.real_arm)
        arm["reader"] = {"status": "ok", "raw_text": reader.canonical(answer), "parsed": answer}
        arm["judge"] = {"status": "pending", "score": False, "raw_text": ""}
        return arm

    def test_corrupted_or_missing_citations_never_inflate_support(self):
        answer = self.source_answer()
        scores = reader.scores(self.scored_arm(answer))
        self.assertEqual((scores["citation_valid_claims"], scores["citation_total_claims"]), (1, 1))
        self.assertTrue(scores["all_claims_supported"])
        for corruption in ("unobserved_quote", "unobserved_source", "empty_quote", "missing_citation", "additional_invalid_citation"):
            with self.subTest(corruption=corruption):
                changed = copy.deepcopy(answer)
                citations = changed["claims"][0]["citations"]
                if corruption == "unobserved_quote":
                    citations[0]["quote"] += "\nCORRUPTED CITATION"
                elif corruption == "unobserved_source":
                    citations[0]["source_id"] = "0" * 64
                elif corruption == "empty_quote":
                    citations[0]["quote"] = ""
                elif corruption == "missing_citation":
                    changed["claims"][0]["citations"] = []
                else:
                    citations.append({"source_id": "0" * 64, "quote": self.quotation})
                scores = reader.scores(self.scored_arm(changed))
                self.assertEqual((scores["citation_valid_claims"], scores["citation_total_claims"]), (0, 1))
                self.assertFalse(scores["all_claims_supported"])
                self.assertFalse(scores["correct"])

    def test_abstention_has_no_asserted_claims_or_citation_support(self):
        answer = {"answer": "I don't know: the retained evidence is incomplete.", "abstain": True, "claims": []}
        self.assertEqual(reader.parse_answer(reader.canonical(answer)), answer)
        scores = reader.scores(self.scored_arm(answer))
        self.assertTrue(scores["abstain"])
        self.assertEqual((scores["citation_valid_claims"], scores["citation_total_claims"]), (0, 0))
        self.assertFalse(scores["all_claims_supported"])
        answer["claims"] = self.source_answer()["claims"]
        self.assertIsNone(reader.parse_answer(reader.canonical(answer)))

    def test_malformed_reader_json_and_types_fail_closed(self):
        answer = self.source_answer()
        valid = reader.canonical(answer)
        cases = [valid[:-1], f"```json\n{valid}\n```", reader.canonical([answer])]
        for key, value in (("answer", None), ("abstain", 1), ("claims", {})):
            changed = copy.deepcopy(answer)
            changed[key] = value
            cases.append(reader.canonical(changed))
        changed = copy.deepcopy(answer)
        changed["claims"][0]["citations"][0]["quote"] = 1
        cases.append(reader.canonical(changed))
        for raw_text in cases:
            with self.subTest(raw_text=raw_text[:30]):
                self.assertIsNone(reader.parse_answer(raw_text))

    def test_bad_json_completion_stays_in_the_100_question_denominator(self):
        report = copy.deepcopy(self.prepared)
        report["spend_usd"] = 0
        for row in report["rows"]:
            for arm in row["systems"]["matched_gpt54"]["arms"].values():
                arm["reader"] = {"status": "error", "raw_text": "", "parsed": None}
                arm["scores"] = reader.scores(arm)
        selected = report["rows"][0]["systems"]["matched_gpt54"]["arms"]["first_five"]
        selected["reader"]["status"] = "pending"
        call = {"status": "ok", "raw_text": reader.canonical(self.source_answer())[:-1]}
        with patch.object(reader, "client"), patch.object(reader, "api_call", return_value=call) as intercepted, \
                patch.object(reader, "save_report"), redirect_stdout(io.StringIO()):
            reader.run_readers(report, self.protocol)
        intercepted.assert_called_once()
        self.assertEqual(selected["reader"]["status"], "error")
        self.assertIsNone(selected["reader"]["parsed"])
        self.assertEqual(selected["scores"]["citation_total_claims"], 0)
        summary = reader.summary(report)["matched_gpt54"]["first_five"]
        self.assertEqual((summary["n"], summary["correct_count"], summary["answer_accuracy"]), (100, 0, 0))

    def ledger(self, total=0.0, category=0.0):
        value = copy.deepcopy(self.ledger_template)
        value["entries"] = []
        value["total_usd"] = total
        value["by_category_usd"] = {name: 0.0 for name in value["by_category_usd"]}
        value["by_category_usd"]["reader_judge_dev"] = category
        return value

    def offline_call(self, directory, ledger, call_id):
        ledger_path = directory / "ledger.json"
        cache_path = directory / "cache"
        reader.write(ledger_path, ledger)
        api = Mock()
        api.responses.create.side_effect = RuntimeError("offline reservation test")
        with patch.object(reader, "LEDGER", ledger_path), patch.object(reader, "CACHE", cache_path):
            call = reader.api_call(api, self.protocol, self.real_request, call_id)
        return call, json.loads(ledger_path.read_text()), api

    def test_subcap_and_total_cap_at_minus_one_at_limit_plus_one_microdollar(self):
        with tempfile.TemporaryDirectory(prefix="hms-reader-integrity-") as temporary:
            work = Path(temporary)
            _, calibration, _ = self.offline_call(work / "calibration", self.ledger(), "calibration")
            reserve = calibration["entries"][0]["reserved_usd"]
            self.assertGreater(reserve, 0)
            for category, cap in (("reader_judge_dev", 120), ("total", 250)):
                for adjustment in (-0.000001, 0, 0.000001):
                    with self.subTest(category=category, adjustment=adjustment):
                        prospective = cap - reserve + adjustment
                        ledger = self.ledger(total=prospective, category=prospective if category == "reader_judge_dev" else 0)
                        directory = work / f"{category}_{adjustment}"
                        call, final, api = self.offline_call(directory, ledger, f"{category}:{adjustment}")
                        if adjustment > 0:
                            self.assertEqual(call["status"], "budget_exhausted")
                            api.responses.create.assert_not_called()
                            self.assertEqual(final, ledger)
                        else:
                            self.assertEqual(call["status"], "error")
                            api.responses.create.assert_called_once_with(**self.real_request)
                            self.assertEqual(len(final["entries"]), 1)
                            self.assertEqual(final["entries"][0]["status"], "error")
                            self.assertLessEqual(final["total_usd"], 250)
                            self.assertLessEqual(final["by_category_usd"]["reader_judge_dev"], 120)
                            self.assertAlmostEqual(final["total_usd"], prospective + reserve)

    def test_reserved_or_cached_calls_are_never_charged_twice(self):
        with tempfile.TemporaryDirectory(prefix="hms-reader-integrity-") as temporary:
            work = Path(temporary)
            original, final, _ = self.offline_call(work, self.ledger(), "same_real_dev_call")
            ledger_path = work / "ledger.json"
            cache_path = work / "cache"
            api = Mock()
            with patch.object(reader, "LEDGER", ledger_path), patch.object(reader, "CACHE", cache_path):
                cached = reader.api_call(api, self.protocol, self.real_request, "same_real_dev_call")
                self.assertEqual(cached, original)
                self.assertEqual(json.loads(ledger_path.read_text()), final)
                api.responses.create.assert_not_called()
                (cache_path / (original["cache_key"] + ".json")).unlink()
                final["entries"][0]["status"] = "reserved"
                reader.write(ledger_path, final)
                with self.assertRaisesRegex(RuntimeError, "unsettled call already reserved"):
                    reader.api_call(api, self.protocol, self.real_request, "same_real_dev_call")
                api.responses.create.assert_not_called()
                self.assertEqual(json.loads(ledger_path.read_text()), final)

    def test_instability_resumes_out_of_order_real_dev_repeats_without_duplicates(self):
        report = copy.deepcopy(self.prepared)
        report["spend_usd"] = 0
        rows = {row["qid"]: row for row in report["rows"]}
        entries = []
        existing = [19, 1, 13]
        random.Random(20261009).shuffle(existing)
        for index, qid in enumerate(self.protocol["instability"]["sample_qids"]):
            name = self.protocol["arms"][index % len(self.protocol["arms"])]
            arm = rows[qid]["systems"]["matched_gpt54"]["arms"][name]
            arm["judge"] = {"status": "error", "score": False,
                            "blind_id": reader.digest(f"judge-blind-v1:{qid}:matched_gpt54:{name}"),
                            "call": {"request": reader.request(self.protocol, arm["reader_input"], "reader")}}
            entries.append({"qid": qid, "arm": name, "system": "matched_gpt54", "outcomes": [
                {"repeat_index": repeat, "status": "ok", "raw_text": "no", "score": False} for repeat in existing
            ]})
        report["instability"]["entries"] = entries
        requested = []
        capture_lock = threading.Lock()

        def intercept(api, protocol, request, call_id):
            with capture_lock:
                requested.append(call_id)
            return {"status": "ok", "raw_text": "no"}

        with patch.object(reader, "client"), patch.object(reader, "api_call", side_effect=intercept), \
                patch.object(reader, "save_report"), redirect_stdout(io.StringIO()):
            reader.instability(report, self.protocol)
        self.assertEqual(len(requested), 340)
        self.assertEqual(len(set(requested)), 340)
        for entry in entries:
            self.assertEqual([outcome["repeat_index"] for outcome in entry["outcomes"]], list(range(20)))
            prefix = rows[entry["qid"]]["systems"]["matched_gpt54"]["arms"][entry["arm"]]["judge"]["blind_id"]
            for repeat in existing:
                self.assertNotIn(prefix + ":repeat:" + str(repeat), requested)
        with patch.object(reader, "client"), patch.object(reader, "api_call") as intercepted, \
                patch.object(reader, "save_report"), redirect_stdout(io.StringIO()):
            reader.instability(report, self.protocol)
        intercepted.assert_not_called()


if __name__ == "__main__":
    unittest.main()
