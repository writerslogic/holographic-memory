# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.13"
# dependencies = ["tiktoken==0.14.0"]
# ///
"""Checker regression tests using retained turns from the real pinned dev release."""
import copy
import json
import unittest
from pathlib import Path

from check_longmemeval_answers import (
    ROOT,
    canonical,
    check_call,
    check_source_groups,
    judge_score,
    paired_interval,
    reader_attempts,
    recompute_scores,
    reconstruct_groups,
    same,
    sorted_sources,
    wilson,
)


class AnswerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path.home() / '.cache/hms-bench/downloads/longmemeval_s_cleaned.json'
        cls.question = next(question for question in json.loads(source.read_text())
                            if question['question_id'] == '18bc8abd')
        frozen = next(row for row in json.loads((ROOT / 'benchmarks/results/evidence_budget_dev.json').read_text())['rows']
                      if row['qid'] == cls.question['question_id'])
        historical = next(row for row in json.loads((ROOT / 'benchmarks/results/evidence_completeness_dev.json').read_text())['rows']
                          if row['qid'] == cls.question['question_id'])
        cls.groups = reconstruct_groups(cls.question, frozen, set(historical['ids']))['first_five']

    def arm(self):
        source = self.groups[0]['sources'][0]
        parsed = {'answer': source['text'][:80], 'abstain': False,
                  'claims': [{'text': source['text'][:80], 'citations': [
                      {'source_id': source['source_id'], 'quote': source['text'][:80]}]}]}
        return {'source_groups': copy.deepcopy(self.groups), 'retained_sources': sorted_sources(self.groups),
                'reader': {'status': 'ok', 'raw_text': canonical(parsed), 'parsed': parsed},
                'judge': {'status': 'ok', 'raw_text': 'yes', 'score': True}}

    def test_exact_citation_and_failed_denominators(self):
        arm = self.arm()
        scores = recompute_scores(arm)
        self.assertTrue(scores['correct'])
        self.assertTrue(scores['all_claims_supported'])
        self.assertEqual(scores['citation_valid_claims'], 1)
        arm['judge'] = {'status': 'error', 'raw_text': '', 'score': False}
        self.assertFalse(recompute_scores(arm)['correct'])
        arm['reader'] = {'status': 'error', 'raw_text': '', 'parsed': None}
        failed = recompute_scores(arm)
        self.assertEqual(failed['citation_total_claims'], 0)
        self.assertFalse(failed['all_claims_supported'])

    def test_corrupted_citation_rejected_after_raw_response_agrees(self):
        arm = self.arm()
        original = recompute_scores(arm)
        arm['reader']['parsed']['claims'][0]['citations'][0]['quote'] += '\nCORRUPTED CITATION'
        arm['reader']['raw_text'] = canonical(arm['reader']['parsed'])
        actual = recompute_scores(arm)
        self.assertEqual(actual['citation_valid_claims'], 0)
        with self.assertRaisesRegex(ValueError, 'citation_valid_claims'):
            same(original, actual, 'citation score mismatch')

    def test_occurrences_keep_assistant_turns_and_full_bytes(self):
        arm = self.arm()
        check_source_groups(arm, self.groups)
        self.assertTrue(any(source['role'] == 'assistant' for source in arm['retained_sources']))
        source_ids = [source['source_id'] for source in arm['retained_sources']]
        self.assertEqual(len(source_ids), len(set(source_ids)))
        arm['source_groups'][0]['sources'][0]['text'] += 'corrupted'
        with self.assertRaisesRegex(ValueError, 'retained source changed or incomplete'):
            check_source_groups(arm, self.groups)

    def test_abstention_does_not_get_vacuous_citation_support(self):
        arm = self.arm()
        arm['reader']['parsed'] = {'answer': "I don't know", 'abstain': True, 'claims': []}
        arm['reader']['raw_text'] = canonical(arm['reader']['parsed'])
        score = recompute_scores(arm)
        self.assertTrue(score['correct'])
        self.assertFalse(score['all_claims_supported'])
        self.assertEqual(score['citation_valid_claims'], 0)

    def test_scores_predictions_and_interval_boundaries(self):
        arm = self.arm()
        arm['judge']['score'] = False
        with self.assertRaisesRegex(ValueError, 'judge prediction mismatch'):
            recompute_scores(arm)
        for successes in (99, 100):
            interval = wilson(successes, 100)
            self.assertLessEqual(interval[0], successes / 100)
            self.assertGreaterEqual(interval[1], successes / 100)
        with self.assertRaisesRegex(ValueError, 'invalid accuracy denominator'):
            wilson(101, 100)
        for output in ('yes indeed', 'yesterday', 'no yes'):
            with self.assertRaisesRegex(ValueError, 'official yes/no'):
                judge_score(output)
        self.assertEqual(paired_interval([0] * 100), {'mean': 0.0, 'ci95': [0.0, 0.0]})

    def test_rejected_real_credit_preflight_did_not_queue_other_readers(self):
        ledger = json.loads((ROOT / 'benchmarks/results/spend_2026-10.json').read_text())
        calls = [entry for entry in ledger['entries'] if entry['call_id'].endswith(':restored-credit-attempt:1')]
        if not calls:
            self.skipTest('the real restored-credit preflight has not run')
        self.assertEqual(len(calls), 1)
        entry = calls[0]
        self.assertEqual(entry['status'], 'error')
        self.assertIsNone(entry['usage'])
        report = json.loads((ROOT / 'benchmarks/results/longmemeval_dev_answers_v1.json').read_text())
        protocol = json.loads((ROOT / report['protocol_path']).read_text())
        recorded = [(row, name, arm) for row in report['rows']
                    for name, arm in row['systems']['matched_gpt54']['arms'].items()
                    if any(call['cache_key'] == entry['cache_key'] for call in
                           arm['reader'].get('prior_attempts', []) + [arm['reader'].get('call') or {}]
                           if 'cache_key' in call)]
        self.assertEqual(len(recorded), 1)
        row, name, arm = recorded[0]
        attempts = reader_attempts(arm, row['qid'], name, report, protocol)
        preflight = next(call for call, _, call_id in attempts if call_id == entry['call_id'])
        self.assertEqual(preflight['http_status'], 429)
        self.assertIs(preflight['cost_is_conservative_reservation'], True)
        self.assertTrue(attempts[0][2].endswith(f"{row['qid']}:{name}"))
        self.assertGreaterEqual(len(attempts), 2)
        changed = copy.deepcopy(arm)
        changed['reader']['prior_attempts'][0]['request']['input'] += '\nALTERED PROMPT'
        with self.assertRaisesRegex(ValueError, 'authorized original or restoration call'):
            reader_attempts(changed, row['qid'], name, report, protocol)

    def test_api_response_snapshot_matches_the_frozen_request(self):
        protocol = json.loads((ROOT / 'benchmarks/public/reader_protocol_v1.json').read_text())
        model = protocol['systems']['matched_gpt54']['reader_model']
        cache = Path.home() / '.cache/hms-bench/answers-v1'
        call = None
        for path in sorted(cache.glob('*.json')):
            recorded = json.loads(path.read_text())
            if recorded['status'] == 'ok' and recorded['request']['model'] == model and recorded['request']['max_output_tokens'] == 4096:
                call = recorded
                break
        if call is None:
            self.skipTest('an immutable completed matched-reader API call is not cached')
        arguments = (call['raw_text'], 'ok', protocol, call['request']['input'], model, 'reader')
        self.assertAlmostEqual(check_call(call, *arguments), call['cost_usd'], places=12)
        changed = copy.deepcopy(call)
        changed['response']['model'] = 'gpt-5.4'
        with self.assertRaisesRegex(ValueError, 'API response model differs from requested frozen snapshot'):
            check_call(changed, *arguments)


if __name__ == '__main__':
    unittest.main()
