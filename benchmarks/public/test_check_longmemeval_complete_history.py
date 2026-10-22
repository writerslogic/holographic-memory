# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Complete original histories and distinct reader/judge policies from synthetic records."""

import copy
import statistics
import tempfile
import unittest
from pathlib import Path

import tiktoken

from check_longmemeval_answers import canonical, digest, recompute_scores, sorted_sources
from check_longmemeval_complete_history import (
    ARM, SYSTEM, SOURCE_POLICY, check_stage_call, complete_reasoning_rules, validate,
)
from check_longmemeval_full_history import independent_summary
from check_longmemeval_user_first import check_historical_policy
from longmemeval_answer_evidence import reader_payload
from test_check_longmemeval_full_history import update_spend
from test_check_longmemeval_role_reader import make_role_fixture


COMPLETE_RULES = '\nUse the complete original conversation and keep entity scopes distinct.\n'


def make_complete_fixture(root):
    fixture = make_role_fixture(root)
    protocol, report = fixture['protocol'], fixture['report']
    full_rules = fixture.pop('rules')
    protocol.pop('retrieval')
    protocol.update(reader_prompt=protocol['reader_prompt'] + COMPLETE_RULES, additional_reasoning_rules=COMPLETE_RULES,
                    source_policy=SOURCE_POLICY, token_cap=196608, local_token_cap=196096, new_spend_cap_usd=50.0)
    settings = protocol['systems'][SYSTEM]
    settings.pop('reasoning_effort')
    settings.update(reader_reasoning_effort='high', judge_reasoning_effort='medium', reader_max_output_tokens=8192,
                    framing_verification='complete local prompt <=196096; actual API input <=196608')
    tokenizer = tiktoken.get_encoding('o200k_base')
    counts = []
    for row in report['rows']:
        row.pop('retrieval')
        arm = row['systems'][SYSTEM]['arms'][ARM]
        arm['source_groups'] = [{key: value for key, value in arm['source_groups'][index].items() if key != 'selection_rank'}
                                for index in (1, 0, 2)]
        arm['retained_sources'] = sorted_sources(arm['source_groups'])
        prompt = protocol['reader_prompt'] + canonical(reader_payload(row, arm))
        count = len(tokenizer.encode(prompt))
        counts.append(count)
        arm.update(reader_input=prompt, reader_input_sha256=digest(prompt), input_tokens=count,
                   reader_cap={'cap': 196608, 'local_cap': 196096, 'input_tokens': count, 'uncapped_input_tokens': count,
                               'dropped_group_ids': [], 'selection_failure': False, 'failure_reason': None,
                               'source_policy': 'complete_original_history'})
        for stage in ('reader', 'judge'):
            call = arm[stage]['call']
            request = call['request']
            if stage == 'reader':
                request.update(input=prompt, reasoning={'effort': 'high'}, max_output_tokens=8192)
            usage = {'input_tokens': len(tokenizer.encode(request['input'])) + 10, 'output_tokens': 20,
                     'input_tokens_details': {'cached_tokens': 0}}
            cost = (usage['input_tokens'] * 2.5 + usage['output_tokens'] * 15) / 1_000_000
            call_id = digest(f"full-history:{fixture['work_name']}:{stage}:{row['qid']}")
            key = digest(canonical({'request': request, 'call_id': call_id, 'protocol_sha256': fixture['official_sha256']}))
            call.update(request_sha256=digest(canonical(request)), cache_key=key, usage=usage, cost_usd=cost)
            call['response']['usage'] = copy.deepcopy(usage)
            entry = next(entry for entry in fixture['ledger']['entries'] if entry.get('call_id') == call_id)
            entry.update(cache_key=key, usage=copy.deepcopy(usage), cost_usd=cost,
                         reserved_usd=((len(tokenizer.encode(request['input'])) + 512) * 2.5
                                       + request['max_output_tokens'] * 15) / 1_000_000)
    protocol['prepared_inputs'] = [{'qid': row['qid'],
                                    'reader_input_sha256': row['systems'][SYSTEM]['arms'][ARM]['reader_input_sha256'],
                                    'input_tokens': row['systems'][SYSTEM]['arms'][ARM]['input_tokens']} for row in report['rows']]
    report['summary']['input_tokens'] = {'median': statistics.median(counts), 'max': max(counts)}
    update_spend(fixture)
    fixture.update(full_rules=full_rules, complete_rules=COMPLETE_RULES)
    return fixture


class CompleteHistoryChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = make_complete_fixture(self.root)

    def arm(self, fixture=None):
        return (fixture or self.fixture)['report']['rows'][0]['systems'][SYSTEM]['arms'][ARM]

    def test_all_original_roles_and_distinct_stage_requests_pass_without_mutation(self):
        before = copy.deepcopy(self.fixture)
        result = validate(**self.fixture)
        self.assertEqual(result['summary']['user_occurrences'], 200)
        self.assertEqual(result['summary']['assistant_occurrences'], 400)
        self.assertEqual(result['paid_calls_verified'], 200)
        self.assertEqual(result['summary']['n'], 100)
        self.assertTrue(result['summary']['acceptance_met'])
        self.assertEqual(result['source_gate']['heldout_records_decoded'], 0)
        self.assertEqual(self.fixture, before)

    def test_truncated_exchanges_missing_orphans_and_original_indices_are_rejected(self):
        for corruption in ('assistant', 'orphan', 'user', 'index'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                groups = self.arm(fixture)['source_groups']
                if corruption == 'assistant':
                    groups[0]['sources'].pop()
                elif corruption == 'orphan':
                    groups.pop()
                elif corruption == 'user':
                    groups[0]['sources'][0]['text'] = 'I prefer'
                else:
                    groups[0]['sources'][0]['session_index'] = 0
                with self.assertRaisesRegex(ValueError, 'complete original exchange or orphan history changed'):
                    validate(**fixture)

    def test_reader_high_and_judge_medium_are_verified_in_original_calls(self):
        for stage, field, wrong in (('reader', 'reasoning', {'effort': 'medium'}),
                                     ('judge', 'reasoning', {'effort': 'high'}),
                                     ('reader', 'max_output_tokens', 4096), ('judge', 'max_output_tokens', 8192)):
            with self.subTest(stage=stage, field=field):
                fixture = copy.deepcopy(self.fixture)
                call = self.arm(fixture)[stage]['call']
                call['request'][field] = wrong
                call['request_sha256'] = digest(canonical(call['request']))
                with self.assertRaisesRegex(ValueError, 'request differs from frozen stage policy'):
                    validate(**fixture)

    def test_protocol_cannot_hide_ambiguous_effort_retrieval_or_weaken_context_limits(self):
        for corruption in ('ambiguous_effort', 'retrieval', 'local_cap', 'own_cap', 'source_gate', 'gold_rules'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                protocol = fixture['protocol']
                if corruption == 'ambiguous_effort':
                    protocol['systems'][SYSTEM]['reasoning_effort'] = 'medium'
                elif corruption == 'retrieval':
                    protocol['retrieval'] = {'candidate_limit': 64}
                elif corruption == 'local_cap':
                    protocol['local_token_cap'] = 130560
                elif corruption == 'own_cap':
                    protocol['new_spend_cap_usd'] = 51.0
                elif corruption == 'source_gate':
                    protocol['source_gate']['heldout_records_decoded'] = 1
                else:
                    protocol['reader_prompt'] += 'The gold answer is blue.'
                with self.assertRaises(ValueError):
                    validate(**fixture)

    def test_raw_response_model_usage_status_and_cost_cannot_be_rewritten(self):
        for corruption in ('model', 'raw_text', 'usage', 'status', 'cost'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                call = self.arm(fixture)['reader']['call']
                if corruption == 'model':
                    call['response']['model'] = 'different-model'
                elif corruption == 'raw_text':
                    call['response']['text'] += ' extra'
                elif corruption == 'usage':
                    call['usage']['input_tokens'] += 1
                elif corruption == 'status':
                    call['response']['status'] = 'incomplete'
                else:
                    call['cost_usd'] += 0.01
                with self.assertRaises(ValueError):
                    validate(**fixture)

    def test_transport_failure_keeps_exact_high_request_and_conservative_reservation(self):
        arm = self.arm()
        output = copy.deepcopy(arm['reader'])
        call = output['call']
        call.update(status='error', response=None, usage=None, raw_text='', cost_is_conservative_reservation=True)
        output.update(status='error', raw_text='', parsed=None)
        tokenizer = tiktoken.get_encoding('o200k_base')
        reserve = ((len(tokenizer.encode(arm['reader_input'])) + 512) * 2.5 + 8192 * 15) / 1_000_000
        call['cost_usd'] = reserve
        actual = check_stage_call(call, output, self.fixture['protocol'], self.fixture['base_protocol'],
                                  arm['reader_input'], 'reader', tokenizer)
        self.assertEqual(actual, reserve)
        call['cost_usd'] = 0.0
        with self.assertRaisesRegex(ValueError, 'API cost mismatch'):
            check_stage_call(call, output, self.fixture['protocol'], self.fixture['base_protocol'],
                             arm['reader_input'], 'reader', tokenizer)

    def test_actual_reader_context_limit_is_196608(self):
        call = self.arm()['reader']['call']
        call['usage']['input_tokens'] = 196609
        call['response']['usage'] = copy.deepcopy(call['usage'])
        call['cost_usd'] = (196609 * 2.5 + 20 * 15) / 1_000_000
        with self.assertRaisesRegex(ValueError, 'actual complete-history reader context cap exceeded'):
            validate(**self.fixture)

    def test_forged_100_correct_is_rejected_without_changing_official_judge_text(self):
        arm = self.arm()
        arm['judge'].update(raw_text='no', score=False)
        arm['judge']['call']['raw_text'] = 'no'
        arm['judge']['call']['response']['text'] = 'no'
        with self.assertRaisesRegex(ValueError, 'score mismatch: correct'):
            validate(**self.fixture)
        arm['scores'] = recompute_scores(arm)
        with self.assertRaisesRegex(ValueError, 'summary mismatch: correct'):
            validate(**self.fixture)

    def test_failures_remain_in_100_denominator_and_pending_stages_prevent_acceptance(self):
        rows = self.fixture['report']['rows']
        arm = self.arm()
        arm['reader'] = {'status': 'error', 'raw_text': '', 'parsed': None}
        arm['judge'] = {'status': 'reader_failure', 'raw_text': '', 'score': False}
        scores = [recompute_scores(row['systems'][SYSTEM]['arms'][ARM]) for row in rows]
        summary = independent_summary(rows, scores, True)
        self.assertEqual(summary['n'], 100)
        self.assertEqual(summary['correct'], 99)
        self.assertEqual(summary['answer_accuracy'], 0.99)
        self.assertFalse(summary['acceptance_met'])
        scores[0]['correct'] = True
        self.assertFalse(independent_summary(rows, scores, False)['acceptance_met'])
        arm['judge']['status'] = 'pending'
        self.assertFalse(independent_summary(rows, scores, True)['acceptance_met'])
        with self.assertRaisesRegex(ValueError, 'pending stage'):
            validate(**self.fixture)

    def test_own_50_dollar_cap_and_duplicate_call_identity_are_verified(self):
        for corruption in ('own_cap', 'duplicate', 'charge'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                if corruption == 'own_cap':
                    fixture['ledger']['entries'][2]['cost_usd'] = 50.1
                elif corruption == 'duplicate':
                    extra = copy.deepcopy(fixture['ledger']['entries'][2])
                    extra['cache_key'] = 'new-key-same-experiment-call'
                    fixture['ledger']['entries'].append(extra)
                else:
                    fixture['ledger']['entries'][2]['cost_usd'] += 0.01
                update_spend(fixture)
                with self.assertRaises(ValueError):
                    validate(**fixture)

    def test_all_original_labels_and_prepared_input_hashes_are_independently_checked(self):
        self.fixture['report']['rows'][0]['actual_complete'] = False
        with self.assertRaisesRegex(ValueError, 'labeled coverage mismatch'):
            validate(**self.fixture)
        fixture = make_complete_fixture(self.root)
        fixture['protocol']['prepared_inputs'][0]['reader_input_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'prepared input manifest'):
            validate(**fixture)

    def test_extra_rule_constant_is_read_without_executing_frozen_producer(self):
        path = self.root / 'producer.py'
        path.write_text("raise RuntimeError('must not execute')\nCOMPLETE_REASONING_RULES = ('generic ' 'rules')\n")
        self.assertEqual(complete_reasoning_rules(path), 'generic rules')

    def test_old_protocol_linkage_uses_historical_medium_requests(self):
        baseline = {'protocol_sha256': '0' * 64, 'rows': self.fixture['baseline_rows']}
        result = check_historical_policy(self.fixture['base_protocol'], self.fixture['preregistration'],
                                         baseline, self.fixture['references'])
        self.assertEqual(result['reader_requests_verified'], 100)
        self.assertNotEqual(self.fixture['official_sha256'], result['preregistered_protocol_sha256'])


if __name__ == '__main__':
    unittest.main()
