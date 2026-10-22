# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Independent-reader inputs, exact framing, cache accounting, and resume boundaries."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import longmemeval_verifier as verifier

CAPS = {"total": 250, "reader_judge_dev": 120, "qwen8b": 60, "x86": 60, "reserve": 10, "heldout400": 0}
DATE = "2024/01/02 (Tue) 09:00"


def base():
    return {"reader_prompt": "Answer in strict JSON.\n", "judge_source": {"templates": {}}, "systems": {
        "matched_gpt54": {"reader_model": verifier.JUDGE_MODEL, "judge_model": verifier.JUDGE_MODEL,
                          "reasoning_effort": "medium", "reader_max_output_tokens": 4096,
                          "judge_max_output_tokens": 2048}}}


def groups():
    return [{"group_id": "group-1", "frozen_selected_id_sha256": "alias-1", "sources": [
        {"source_id": "user-1", "session_id": "session-1", "session_index": 0, "turn_index": 0,
         "observation_time": DATE, "role": "user", "text": "My complete original observation. αβ"},
        {"source_id": "assistant-1", "session_id": "session-1", "session_index": 0, "turn_index": 1,
         "observation_time": DATE, "role": "assistant", "text": "An entire original reply, with its qualifiers."}]}]


def count_record(prompt, tokens=100):
    req = {"model": verifier.READER_MODEL, "input": prompt}
    return {"status": "ok", "request": req, "request_sha256": verifier.reader.digest(verifier.reader.canonical(req)),
            "response": {"object": "response.input_tokens", "input_tokens": tokens},
            "cache_key": verifier.reader.digest(verifier.reader.canonical({"namespace": "sol61-input-count-v1", "request": req}))}


def usage(incoming=100, outgoing=20, cached=10, writes=0, classes=None):
    details = {"cached_tokens": cached, "cache_write_tokens": writes}
    if classes is not None:
        details["cache_write_tokens_details"] = classes
    return {"input_tokens": incoming, "output_tokens": outgoing, "total_tokens": incoming + outgoing,
            "input_tokens_details": details, "output_tokens_details": {"reasoning_tokens": outgoing // 2}}


def entry(status="reserved", cost=0.2, identity="own", reserved=0.2):
    return {"cache_key": "key", "call_id": identity, "category": "reader_judge_dev", "status": status,
            "cost_usd": cost, "reserved_usd": reserved, "usage": None}


def ledger(entries):
    return verifier.ledger_totals(entries, CAPS)


class VerifierInputTests(unittest.TestCase):
    def setUp(self):
        self.protocol = verifier.make_protocol(base())
        self.question = {"question": "What did the assistant report?", "question_date": DATE,
                         "qid": "SECRET_ID", "qtype": "SECRET_TYPE", "answer": "SECRET_GOLD"}
        self.groups = groups()
        self.seeds = [{"source_groups": self.groups, "cited_source_ids": ["assistant-1"]}]

    def prepare(self, tokens=100):
        return verifier.prepare_arm(self.question, self.groups, self.seeds, ["group-1"], self.protocol,
                                    lambda text: 100, lambda prompt: count_record(prompt, tokens))

    def test_source_packet_keeps_complete_turns_roles_and_no_prior_answers_or_labels(self):
        before = copy.deepcopy(self.groups)
        arm = self.prepare()
        payload = json.loads(arm["reader_input"][len(self.protocol["reader_prompt"]):])
        self.assertEqual(set(payload), {"question", "question_date", "evidence"})
        self.assertEqual(payload["evidence"], [{key: source[key] for key in
                         ("source_id", "session_id", "observation_time", "role", "text")}
                         for source in before[0]["sources"]])
        for secret in ("SECRET_ID", "SECRET_TYPE", "SECRET_GOLD"):
            self.assertNotIn(secret, arm["reader_input"])
        self.assertEqual(self.groups, before)
        self.assertNotIn("diagnostics", arm["reader_input"])
        self.assertEqual(arm["source_selection"]["diagnostics"]["dropped_cited_source_ids"], [])

    def test_exact_endpoint_limit_minus_one_at_and_plus_one(self):
        limit = self.protocol["exact_preflight_cap"]
        for tokens in (limit - 1, limit, limit + 1):
            with self.subTest(tokens=tokens):
                arm = self.prepare(tokens)
                self.assertEqual(arm["input_tokens"], tokens)
                self.assertEqual(arm["reader_cap"]["selection_failure"], tokens > limit)
                self.assertEqual(arm["reader"]["status"], "selection_failure" if tokens > limit else "pending")
                self.assertEqual(arm["retained_sources"], self.groups[0]["sources"])

    def test_protocol_uses_separate_model_effort_limits_and_explicit_flat_cache_request(self):
        original = base()
        before = copy.deepcopy(original)
        protocol = verifier.make_protocol(original)
        read = verifier.request(protocol, "frozen prompt", "reader")
        judge = verifier.request(protocol, "official judge", "judge")
        self.assertEqual(original, before)
        self.assertEqual((read["model"], read["reasoning"], read["max_output_tokens"]),
                         ("gpt-6.1-sol", {"effort": "high"}, 4096))
        self.assertEqual((judge["model"], judge["reasoning"], judge["max_output_tokens"]),
                         (verifier.JUDGE_MODEL, {"effort": "medium"}, 2048))
        self.assertEqual(read["prompt_cache_options"], {"mode": "explicit"})
        transported = verifier.transport_request(read)
        self.assertEqual(transported["extra_body"], {"prompt_cache_options": {"mode": "explicit"}})
        self.assertNotIn("prompt_cache_options", transported)
        self.assertEqual(read["prompt_cache_options"], {"mode": "explicit"})
        for req in (read, judge):
            self.assertEqual(req["service_tier"], "default")
            self.assertEqual(req["truncation"], "disabled")
            self.assertFalse(req["store"])
        self.assertEqual(protocol["local_proxy_cap"], 24000)
        self.assertEqual(protocol["exact_preflight_cap"], 32256)
        self.assertEqual(protocol["new_spend_cap_usd"], 14)

    def test_stripped_priors_wait_for_last_reader_but_ignore_unfinished_judges(self):
        rows = []
        for index in range(100):
            arm = {"source_groups": self.groups, "reader": {"status": "ok", "raw_text": "SECRET_OLD_ANSWER",
                   "parsed": {"answer": "SECRET_OLD_ANSWER", "claims": [{"text": "SECRET_CLAIM_TEXT",
                              "citations": [{"source_id": "assistant-1", "quote": "SECRET_QUOTE"}]}]}},
                   "judge": {"status": "pending", "score": "SECRET_JUDGE"}, "scores": {"correct": "SECRET_SCORE"}}
            rows.append({"qid": f"dev-{index}", "question": "question", "question_date": DATE,
                         "qtype": "SECRET_TYPE", "systems": {"old": {"arms": {"oldarm": arm}}}})
        report = {"dev_only": True, "heldout_run": False, "rows": rows}
        dev = {row["qid"] for row in rows}
        stripped = verifier.stripped_prior(report, "old", "oldarm", dev)
        encoded = json.dumps(stripped)
        for secret in ("SECRET_OLD_ANSWER", "SECRET_CLAIM_TEXT", "SECRET_QUOTE", "SECRET_JUDGE", "SECRET_SCORE", "SECRET_TYPE"):
            self.assertNotIn(secret, encoded)
        self.assertEqual(stripped[0]["cited_source_ids"], ["assistant-1"])
        proof = verifier.prior_reader_proof(report, "old", "oldarm", dev)
        self.assertEqual(proof[0]["reader"]["raw_text"], "SECRET_OLD_ANSWER")
        self.assertNotIn("call", proof[0]["reader"])
        for secret in ("SECRET_JUDGE", "SECRET_SCORE", "SECRET_TYPE"):
            self.assertNotIn(secret, json.dumps(proof))
        rows[-1]["systems"]["old"]["arms"]["oldarm"]["reader"]["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "finish all 100"):
            verifier.stripped_prior(report, "old", "oldarm", dev)

    def test_preflight_count_is_cached_without_generation_or_semantic_retry(self):
        response = SimpleNamespace(model_dump=lambda **kwargs: {"input_tokens": 42, "object": "response.input_tokens"})
        count = Mock(return_value=response)
        api = SimpleNamespace(responses=SimpleNamespace(input_tokens=SimpleNamespace(count=count), create=Mock()))
        with tempfile.TemporaryDirectory() as directory:
            first = verifier.exact_input_count(api, "harmless synthetic input", Path(directory))
            second = verifier.exact_input_count(api, "harmless synthetic input", Path(directory))
        self.assertEqual(first, second)
        count.assert_called_once_with(model="gpt-6.1-sol", input="harmless synthetic input")
        api.responses.create.assert_not_called()

    def test_100_input_freeze_and_failure_inclusive_acceptance(self):
        arm = self.prepare()
        rows = [{"qid": f"dev-{index}", **{key: self.question[key] for key in ("question", "question_date")},
                 "systems": {verifier.SYSTEM: {"arms": {verifier.ARM: copy.deepcopy(arm)}}}} for index in range(100)]
        protocol = {**self.protocol, "prepared_inputs": verifier.input_manifest(rows)}
        verifier.validate_inputs(rows, {row["qid"] for row in rows}, protocol)
        report = {"rows": rows, "complete": True}
        for row in rows:
            target = row["systems"][verifier.SYSTEM]["arms"][verifier.ARM]
            target["reader"]["status"] = "error"
            target["judge"]["status"] = "reader_failure"
            target["scores"]["correct"] = True
        rows[-1]["systems"][verifier.SYSTEM]["arms"][verifier.ARM]["scores"]["correct"] = False
        summary = verifier.summarize(report)
        self.assertEqual(summary["n"], 100)
        self.assertEqual(summary["answer_accuracy"], 0.99)
        self.assertFalse(summary["acceptance_met"])
        with self.assertRaisesRegex(ValueError, "100 dev"):
            verifier.validate_inputs(rows[:-1], {row["qid"] for row in rows}, protocol)
        rows[-1]["systems"][verifier.SYSTEM]["arms"][verifier.ARM]["source_groups"][0]["sources"][0]["text"] += "changed"
        with self.assertRaisesRegex(ValueError, "prompt changed"):
            verifier.validate_inputs(rows, {row["qid"] for row in rows}, protocol)


class VerifierCostTests(unittest.TestCase):
    def test_nonoverlapping_cache_read_write_rates_and_reasoning_not_charged_twice(self):
        record = usage(writes=30, classes={"ttl_30m_tokens": 10, "ttl_12h_tokens": 20})
        priced = verifier.usage_cost(record, "reader")
        self.assertAlmostEqual(priced["cost_usd"], (60 * 2 + 10 * 0.1 + 10 * 2.5 + 20 * 5 + 20 * 10) / 1e6)
        self.assertTrue(priced["unexpected_cache_writes"])
        self.assertFalse(priced["cost_is_conservative"])
        record["output_tokens_details"]["reasoning_tokens"] = 0
        self.assertEqual(verifier.usage_cost(record, "reader")["cost_usd"], priced["cost_usd"])
        conservative = verifier.usage_cost(usage(writes=30), "reader")
        self.assertAlmostEqual(conservative["cost_usd"], (60 * 2 + 10 * 0.1 + 30 * 5 + 20 * 10) / 1e6)
        self.assertTrue(conservative["cost_is_conservative"])
        self.assertAlmostEqual(verifier.usage_cost(usage(), "judge")["cost_usd"], (90 * 2.5 + 10 * .25 + 20 * 15) / 1e6)

    def test_usage_rejects_negative_bool_overlap_unknown_lifetime_and_bad_sum(self):
        bad = [usage(incoming=-1), usage(cached=True), usage(cached=80, writes=30),
               usage(writes=30, classes={"ttl_30m_tokens": 29}),
               usage(writes=30, classes={"ttl_30m_tokens": 30, "unknown_lifetime": 10})]
        for record in bad:
            with self.subTest(record=record), self.assertRaises(ValueError):
                verifier.usage_cost(record, "reader")
        with self.assertRaisesRegex(ValueError, "judge cache writes"):
            verifier.usage_cost(usage(writes=1), "judge")

    def test_reservations_include_512_and_max_write_rate(self):
        self.assertAlmostEqual(verifier.reservation_usd(32256, 4096, "reader"), 32768 * 5e-6 + 4096 * 10e-6)
        self.assertAlmostEqual(verifier.reservation_usd(100, 2048, "judge"), 612 * 2.5e-6 + 2048 * 15e-6)

    def test_own14_and_global120_250_budget_boundaries(self):
        reserve = 0.1
        for delta in (-0.000001, 0, 0.000001):
            with self.subTest(delta=delta):
                current = ledger([entry(cost=14 - reserve + delta)])
                self.assertEqual(verifier.budget_rejection(current, {"own"}, reserve),
                                 "new_variant_spend_cap" if delta > 0 else None)
        current = ledger([entry(cost=120 - reserve + .000001, identity="other")])
        self.assertEqual(verifier.budget_rejection(current, {"own"}, reserve), "reader_judge_dev_spend_cap")
        current["total_usd"] = 250 - reserve + .000001
        self.assertEqual(verifier.budget_rejection(current, {"own"}, reserve), "global_total_spend_cap")

    def test_proven_terminal_replaces_stale_reservation_but_conflicts_fail(self):
        reserved = entry()
        terminal = entry(status="ok", cost=.03)
        proof = {**terminal}
        merged = verifier.merge_ledgers([ledger([reserved]), ledger([terminal]), ledger([reserved])], CAPS, {"own"}, {"key": proof})
        self.assertAlmostEqual(merged["total_usd"], .03)
        self.assertEqual(merged["entries"][0]["status"], "ok")
        conflict = entry(status="ok", cost=.04)
        with self.assertRaisesRegex(ValueError, "conflicting terminal"):
            verifier.merge_ledgers([ledger([terminal]), ledger([conflict])], CAPS, {"own"}, {"key": proof})
        with self.assertRaisesRegex(ValueError, "lacks durable cache"):
            verifier.merge_ledgers([ledger([terminal])], CAPS, {"own"}, {})
        foreign = verifier.merge_ledgers([ledger([entry(identity="other")]), ledger([entry(status="ok", cost=.03, identity="other")])],
                                         CAPS, {"own"}, {})
        self.assertAlmostEqual(foreign["total_usd"], .2)


class VerifierPaidTransportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.results = self.root / "results"
        self.work = self.results / "longmemeval_synthetic"
        self.work.mkdir(parents=True)
        self.protocol = verifier.make_protocol(base())
        verifier.reader.write(self.work / "protocol.json", self.protocol)
        verifier.reader.write(self.results / "spend_2026-10.json", ledger([]))
        self.patch_root = patch.object(verifier, "ROOT", self.root)
        self.patch_results = patch.object(verifier, "RESULTS", self.results)
        self.patch_root.start()
        self.patch_results.start()
        self.addCleanup(self.patch_root.stop)
        self.addCleanup(self.patch_results.stop)
        self.req = verifier.request(self.protocol, "Synthetic source-only prompt", "reader")

    def api(self, record=None, model=None, status="completed"):
        record = usage() if record is None else record
        response = SimpleNamespace(output_text='{"answer":"A","abstain":false,"claims":[]}',
                                   status=status, model=model or verifier.READER_MODEL,
                                   usage=SimpleNamespace(model_dump=lambda **kwargs: record))
        response.model_dump = lambda **kwargs: {"model": response.model, "status": status, "usage": record}
        return SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))

    def call(self, api):
        return verifier.api_call(api, self.protocol, self.req, "own", "reader", self.work, {"own"}, 100)

    def retry(self, api, sleep):
        identities = {verifier.retry_call_id("own", attempt) for attempt in range(3)}
        return verifier.call_with_transport_retries(api, self.protocol, self.req, "own", "reader", self.work,
                                                   identities, 100, sleep_callback=sleep)

    def test_paid_cache_resume_settles_once_and_does_not_resample_incomplete(self):
        api = self.api(status="incomplete")
        first = self.call(api)
        second = self.call(api)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "incomplete")
        self.assertEqual(api.responses.create.call_count, 1)
        current = json.loads((self.work / "spend.json").read_text())
        self.assertAlmostEqual(current["total_usd"], verifier.usage_cost(usage(), "reader")["cost_usd"])
        self.assertEqual(current["entries"][0]["status"], "incomplete")
        self.assertEqual(api.responses.create.call_args.kwargs["extra_body"], {"prompt_cache_options": {"mode": "explicit"}})

    def test_unexpected_cache_writes_and_model_alias_drift_are_charged_failed_calls(self):
        api = self.api(usage(writes=5), model="unfrozen-other-model")
        result = self.call(api)
        self.assertEqual(result["status"], "validation_failure")
        self.assertTrue(result["cost_is_conservative"])
        self.assertIn("unexpected_explicit_mode_cache_writes", result["validation_failures"])
        self.assertIn("response_model_differs_from_requested_alias", result["validation_failures"])
        self.assertIsNotNone(result["response"])
        self.assertAlmostEqual(result["cost_usd"], verifier.usage_cost(usage(writes=5), "reader")["cost_usd"])

    def test_invalid_usage_retains_reservation_and_is_not_retried(self):
        api = self.api(usage(incoming=True))
        result = self.call(api)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["cost_is_conservative_reservation"])
        self.assertEqual(result["cost_usd"], result["reserved_usd"])
        self.assertEqual(self.call(api), result)
        self.assertEqual(api.responses.create.call_count, 1)

    def test_unsettled_reservation_without_terminal_cache_blocks_duplicate_generation(self):
        key = verifier.cache_key(self.req, "own", verifier.reader.sha(self.work / "protocol.json"))
        pending = entry(reserved=verifier.reservation_usd(100, 4096, "reader"))
        pending.update(cache_key=key, cost_usd=pending["reserved_usd"])
        verifier.reader.write(self.work / "spend.json", ledger([pending]))
        api = self.api()
        with self.assertRaisesRegex(ValueError, "unsettled call"):
            self.call(api)
        api.responses.create.assert_not_called()

    def test_transport_520_then_success_preserves_both_charges_and_cache_first_resume(self):
        class TransportFailure(Exception):
            status_code = 520

        api = self.api()
        success = api.responses.create.return_value
        api.responses.create.side_effect = [TransportFailure(), success]
        sleep = Mock()
        result, previous = self.retry(api, sleep)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["call_id"], "own:transport-retry:1")
        self.assertEqual(len(previous), 1)
        self.assertEqual(previous[0]["call_id"], "own")
        self.assertTrue(previous[0]["cost_is_conservative_reservation"])
        self.assertEqual(api.responses.create.call_count, 2)
        self.assertEqual(api.responses.create.call_args_list[0], api.responses.create.call_args_list[1])
        sleep.assert_called_once_with(2)
        current = json.loads((self.work / "spend.json").read_text())
        self.assertEqual(len(current["entries"]), 2)
        self.assertAlmostEqual(current["total_usd"], previous[0]["reserved_usd"] + result["cost_usd"])
        self.assertEqual(len(list((self.work / "cache").glob("*.json"))), 2)
        resumed, old_attempts = self.retry(api, Mock())
        self.assertEqual(resumed, result)
        self.assertEqual(old_attempts, previous)
        self.assertEqual(api.responses.create.call_count, 2)

    def test_transport_errors_exhaust_after_three_attempts_with_2_and_8_backoffs(self):
        class TransportFailure(Exception):
            status_code = 503

        api = self.api()
        api.responses.create.side_effect = TransportFailure()
        sleep = Mock()
        result, previous = self.retry(api, sleep)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["call_id"], "own:transport-retry:2")
        self.assertEqual([item["call_id"] for item in previous], ["own", "own:transport-retry:1"])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 8])
        self.assertEqual(api.responses.create.call_count, 3)
        current = json.loads((self.work / "spend.json").read_text())
        self.assertEqual(len(current["entries"]), 3)
        self.assertAlmostEqual(current["total_usd"], 3 * result["reserved_usd"])

    def test_retry_reservation_respects_own_budget_and_keeps_first_failed_charge(self):
        class TransportFailure(Exception):
            status_code = 429

        reserve = verifier.reservation_usd(100, 4096, "reader")
        api = self.api()
        api.responses.create.side_effect = TransportFailure()
        with patch.object(verifier, "NEW_SPEND_CAP_USD", reserve * 1.5):
            result, previous = self.retry(api, Mock())
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["budget_scope"], "new_variant_spend_cap")
        self.assertEqual(len(previous), 1)
        self.assertEqual(api.responses.create.call_count, 1)
        current = json.loads((self.work / "spend.json").read_text())
        self.assertAlmostEqual(current["total_usd"], reserve)


class VerifierRetryPolicyTests(unittest.TestCase):
    def test_only_response_less_transient_transport_errors_are_retryable(self):
        for status in (429, 500, 520, 599):
            self.assertTrue(verifier.retryable_transport_failure(
                {"status": "error", "response": None, "usage": None, "http_status": status}))
        for error_type in ("APIConnectionError", "APITimeoutError"):
            self.assertTrue(verifier.retryable_transport_failure(
                {"status": "error", "response": None, "usage": None, "error_type": error_type}))
        for status in (400, 401, 404, 499, 600, True):
            self.assertFalse(verifier.retryable_transport_failure(
                {"status": "error", "response": None, "usage": None, "http_status": status}))
        for response, usage_record in (({}, None), (None, {})):
            self.assertFalse(verifier.retryable_transport_failure(
                {"status": "error", "response": response, "usage": usage_record, "http_status": 520}))

    def test_completed_wrong_invalid_json_no_incomplete_and_validation_failure_never_retry(self):
        protocol = verifier.make_protocol(base())
        req = verifier.request(protocol, "identical source packet", "reader")
        cases = [("ok", "no"), ("ok", "invalid JSON"), ("ok", '{"answer":"wrong"}'),
                 ("incomplete", "partial"), ("validation_failure", "invalid model identity"),
                 ("error", "", 400)]
        for case in cases:
            status, raw = case[:2]
            output = {"status": status, "raw_text": raw, "response": {} if len(case) == 2 else None,
                      "usage": {} if len(case) == 2 else None, "http_status": case[2] if len(case) > 2 else None}
            with self.subTest(case=case), patch.object(verifier, "api_call", return_value=output) as call:
                sleep = Mock()
                final, previous = verifier.call_with_transport_retries(None, protocol, req, "base", "reader", Path("."),
                                                                        {"base"}, 100, sleep_callback=sleep)
                self.assertIs(final, output)
                self.assertEqual(previous, [])
                call.assert_called_once()
                sleep.assert_not_called()

    def test_all_potential_retry_ids_are_in_own_budget_scope(self):
        work = Path("longmemeval_synthetic")
        rows = [{"qid": "q1"}, {"qid": "q2"}]
        identities = verifier.all_call_ids(work, rows)
        self.assertEqual(len(identities), 12)
        for row in rows:
            for stage in ("reader", "judge"):
                base_id = verifier.call_id(work, stage, row["qid"])
                for attempt in range(3):
                    self.assertIn(verifier.retry_call_id(base_id, attempt), identities)


if __name__ == "__main__":
    unittest.main()
