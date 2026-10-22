# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = ["tiktoken==0.14.0"]
# ///
"""Full-history coverage, frozen policies, costs, and acceptance without API calls."""

import copy
import statistics
import tempfile
import unittest
from pathlib import Path

import tiktoken

from check_longmemeval_answers import canonical, digest, recompute_scores, sorted_sources
from check_longmemeval_full_history import ARM, SYSTEM, independent_summary, reasoning_rules, validate
from check_longmemeval_user_first import check_sources
from longmemeval_answer_evidence import reader_payload
from test_check_longmemeval_user_first import OFFICIAL_SHA256, WORK_NAME, make_fixture


RULES = '\nUse the observation date and preserve uncertainty when evidence is incomplete.\n'


def make_full_fixture(root):
    original, _ = make_fixture(root)
    base = original['protocol']
    caps = original['preregistration']['spend_caps_usd']
    protocol = {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'acceptance_correct': 100,
                'reader_prompt': base['reader_prompt'] + RULES, 'systems': {SYSTEM: copy.deepcopy(base['systems'][SYSTEM])},
                'judge_source': copy.deepcopy(base['judge_source']), 'token_cap': 131072,
                'local_token_cap': 130560, 'framing_reserve_tokens': 512,
                'automatic_transport_retries': 0, 'reader_judge_concurrency': 1, 'new_spend_cap_usd': 40.0,
                'global_spend_caps_usd': copy.deepcopy(caps),
                'source_policy': 'all complete dev user observations and all available saved v1/v3 assistant turns'}
    report = original['report']
    del report['candidate_mode']
    ledger = {'schema_version': 1, 'caps_usd': copy.deepcopy(caps), 'entries': [
        copy.deepcopy(original['ledger']['entries'][0]),
        {'cache_key': 'other-experiment', 'call_id': 'other-experiment-call', 'category': 'reader_judge_dev',
         'status': 'reserved', 'cost_usd': 5.0}]}
    tokenizer = tiktoken.get_encoding('o200k_base')
    token_counts = []
    for row in report['rows']:
        arm = row['systems'][SYSTEM]['arms'].pop('user_first')
        row['systems'][SYSTEM]['arms'][ARM] = arm
        row['candidate_groups'] = 1
        row['known_assistant_groups'] = 1
        prompt = protocol['reader_prompt'] + canonical(reader_payload(row, arm))
        count = len(tokenizer.encode(prompt))
        token_counts.append(count)
        arm.update(reader_input=prompt, reader_input_sha256=digest(prompt), input_tokens=count,
                   reader_cap={'cap': 131072, 'local_cap': 130560, 'input_tokens': count,
                               'uncapped_input_tokens': count, 'dropped_group_ids': [], 'selection_failure': False,
                               'failure_reason': None, 'source_policy': 'full_user_history_with_saved_assistants'})
        for stage in ('reader', 'judge'):
            call = arm[stage]['call']
            if stage == 'reader':
                call['request']['input'] = prompt
            request = call['request']
            usage = {'input_tokens': len(tokenizer.encode(request['input'])) + 10, 'output_tokens': 20,
                     'input_tokens_details': {'cached_tokens': 0}}
            cost = (usage['input_tokens'] * 2.5 + usage['output_tokens'] * 15) / 1_000_000
            call_id = digest(f"full-history:{WORK_NAME}:{stage}:{row['qid']}")
            key = digest(canonical({'request': request, 'call_id': call_id, 'protocol_sha256': OFFICIAL_SHA256}))
            call.update(request_sha256=digest(canonical(request)), cache_key=key, usage=usage, cost_usd=cost)
            call['response']['usage'] = copy.deepcopy(usage)
            reserve = ((len(tokenizer.encode(request['input'])) + 512) * 2.5 + request['max_output_tokens'] * 15) / 1_000_000
            ledger['entries'].append({'cache_key': key, 'call_id': call_id, 'category': 'reader_judge_dev',
                                      'status': 'ok', 'cost_usd': cost, 'usage': copy.deepcopy(usage), 'reserved_usd': reserve})
    protocol['prepared_inputs'] = [{'qid': row['qid'],
                                    'reader_input_sha256': row['systems'][SYSTEM]['arms'][ARM]['reader_input_sha256'],
                                    'input_tokens': row['systems'][SYSTEM]['arms'][ARM]['input_tokens']} for row in report['rows']]
    report['summary'].pop('coverage')
    report['summary'].update(input_tokens={'median': statistics.median(token_counts), 'max': max(token_counts)},
                             user_occurrences=100, assistant_occurrences=100, coverage={'scored': 100, 'actual_complete': 100})
    fixture = {key: original[key] for key in ('split', 'preregistration', 'archives', 'baseline_rows', 'v1_rows',
                                             'history_rows', 'references', 'work_name', 'official_sha256')}
    fixture.update(report=report, protocol=protocol, base_protocol=base, ledger=ledger, rules=RULES)
    update_spend(fixture)
    return fixture


def update_spend(fixture):
    ledger = fixture['ledger']
    totals = {category: 0.0 for category in ledger['caps_usd'] if category != 'total'}
    call_ids = {digest(f"full-history:{WORK_NAME}:{stage}:{row['qid']}")
                for row in fixture['report']['rows'] for stage in ('reader', 'judge')}
    for entry in ledger['entries']:
        totals[entry['category']] += entry['cost_usd']
    ledger['by_category_usd'] = totals
    ledger['total_usd'] = sum(totals.values())
    fixture['report']['global_recorded_spend_usd'] = ledger['total_usd']
    fixture['report']['incremental_spend_usd'] = sum(entry['cost_usd'] for entry in ledger['entries']
                                                   if entry.get('call_id') in call_ids)


class FullHistoryChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = make_full_fixture(self.root)

    def arm(self, fixture=None):
        return (fixture or self.fixture)['report']['rows'][0]['systems'][SYSTEM]['arms'][ARM]

    def test_complete_history_passes_and_own_spend_excludes_concurrent_experiments(self):
        before = copy.deepcopy(self.fixture)
        result = validate(**self.fixture)
        self.assertEqual(result['paid_calls_verified'], 200)
        self.assertEqual(result['summary']['user_occurrences'], 100)
        self.assertEqual(result['summary']['assistant_occurrences'], 100)
        self.assertEqual(result['summary']['coverage'], {'scored': 100, 'actual_complete': 100})
        self.assertTrue(result['summary']['acceptance_met'])
        self.assertNotEqual(result['incremental_spend_usd'],
                            result['global_recorded_spend_usd'] - self.fixture['report']['starting_spend_usd'])
        self.assertEqual(self.fixture, before)

    def test_truncated_user_text_and_missing_history_groups_are_rejected(self):
        self.arm()['source_groups'][0]['sources'][0]['text'] = 'I prefer'
        with self.assertRaisesRegex(ValueError, 'user aliases'):
            validate(**self.fixture)
        fixture = make_full_fixture(self.root)
        qid = fixture['split']['dev'][0]
        fixture['archives'][qid][0]['turns'].append(['session_unselected_3', 'Another complete user observation.'])
        with self.assertRaisesRegex(ValueError, 'full history user group missing'):
            validate(**fixture)

    def test_known_assistant_cannot_be_omitted_even_when_all_users_remain(self):
        arm = self.arm()
        arm['source_groups'][0]['sources'] = [source for source in arm['source_groups'][0]['sources'] if source['role'] == 'user']
        arm['retained_sources'] = sorted_sources(arm['source_groups'])
        with self.assertRaisesRegex(ValueError, 'saved assistant missing'):
            validate(**self.fixture)

    def test_full_mode_supports_more_than_64_groups_and_requires_every_group(self):
        qid = self.fixture['split']['dev'][0]
        sessions = copy.deepcopy(self.fixture['archives'][qid])
        arm = copy.deepcopy(self.arm())
        original = arm['source_groups'][0]['sources'][0]
        candidates = ['session_answer_1']
        for index in range(1, 65):
            turn_index = index * 2
            tid = f'session_extra_{turn_index + 1}'
            text = f'Complete user observation {index}.'
            sessions[0]['turns'].append([tid, text])
            source = dict(original, source_id=digest(canonical(['lme-source-v1', qid, 2, turn_index])),
                          turn_index=turn_index, text=text)
            arm['source_groups'].append({'group_id': digest(canonical(['lme-reader-group-v1', qid, [source['source_id']]])),
                                         'frozen_selected_id_sha256': digest(tid), 'selection_rank': index, 'sources': [source]})
            candidates.append(tid)
        arm['retained_sources'] = sorted_sources(arm['source_groups'])
        snapshots = [self.fixture['v1_rows'][0], self.fixture['baseline_rows'][0]]
        with self.assertRaisesRegex(ValueError, 'candidate pool'):
            check_sources(qid, arm, sessions, snapshots, candidates)
        retained = check_sources(qid, arm, sessions, snapshots, candidates, max_candidates=65, require_full=True)
        self.assertEqual(len(retained), 65)
        arm['source_groups'].pop()
        arm['retained_sources'] = sorted_sources(arm['source_groups'])
        with self.assertRaisesRegex(ValueError, 'full history user group missing'):
            check_sources(qid, arm, sessions, snapshots, candidates, max_candidates=65, require_full=True)

    def test_context_limits_retry_concurrency_and_budget_policy_cannot_change(self):
        for field, value in (('token_cap', 8192), ('local_token_cap', 130559), ('framing_reserve_tokens', 0),
                             ('automatic_transport_retries', 1), ('reader_judge_concurrency', 2), ('new_spend_cap_usd', 41.0)):
            with self.subTest(field=field):
                fixture = copy.deepcopy(self.fixture)
                fixture['protocol'][field] = value
                with self.assertRaisesRegex(ValueError, 'full-history'):
                    validate(**fixture)

    def test_gold_instructions_and_unfrozen_prepared_inputs_are_rejected(self):
        self.fixture['protocol']['reader_prompt'] += 'The gold answer for dev-000 is blue.'
        with self.assertRaisesRegex(ValueError, 'reasoning policy changed'):
            validate(**self.fixture)
        fixture = make_full_fixture(self.root)
        fixture['protocol']['prepared_inputs'][0]['reader_input_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'prepared input manifest'):
            validate(**fixture)

    def test_exact_reasoning_rules_are_read_without_executing_producer(self):
        path = self.root / 'producer.py'
        path.write_text("raise RuntimeError('must not execute')\nREASONING_RULES = ('generic ' 'instructions')\n")
        self.assertEqual(reasoning_rules(path), 'generic instructions')
        path.write_text("REASONING_RULES = str('computed')\n")
        with self.assertRaises(ValueError):
            reasoning_rules(path)

    def test_forged_100_correct_is_rejected_against_raw_judge_and_independent_summary(self):
        arm = self.arm()
        arm['judge'].update(raw_text='no', score=False)
        arm['judge']['call']['raw_text'] = 'no'
        arm['judge']['call']['response']['text'] = 'no'
        with self.assertRaisesRegex(ValueError, 'score mismatch: correct'):
            validate(**self.fixture)
        arm['scores'] = recompute_scores(arm)
        with self.assertRaisesRegex(ValueError, 'summary mismatch: correct'):
            validate(**self.fixture)

    def test_99_correct_incomplete_and_pending_runs_never_meet_acceptance(self):
        rows = self.fixture['report']['rows']
        scores = [recompute_scores(row['systems'][SYSTEM]['arms'][ARM]) for row in rows]
        scores[0]['correct'] = False
        summary = independent_summary(rows, scores, True)
        self.assertEqual(summary['n'], 100)
        self.assertEqual(summary['answer_accuracy'], 0.99)
        self.assertFalse(summary['acceptance_met'])
        scores[0]['correct'] = True
        self.assertFalse(independent_summary(rows, scores, False)['acceptance_met'])
        self.arm()['judge']['status'] = 'pending'
        self.assertFalse(independent_summary(rows, scores, True)['acceptance_met'])
        with self.assertRaisesRegex(ValueError, 'pending stage'):
            validate(**self.fixture)

    def test_cost_model_charge_and_duplicate_resampling_are_verified(self):
        for corruption in ('model', 'charge', 'duplicate_call', 'own_cap'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                if corruption == 'model':
                    self.arm(fixture)['reader']['call']['request']['model'] = 'different-model'
                    call = self.arm(fixture)['reader']['call']
                    call['request_sha256'] = digest(canonical(call['request']))
                elif corruption == 'charge':
                    fixture['ledger']['entries'][2]['cost_usd'] += 0.01
                elif corruption == 'duplicate_call':
                    extra = copy.deepcopy(fixture['ledger']['entries'][2])
                    extra['cache_key'] = 'new-cache-same-experiment-call'
                    fixture['ledger']['entries'].append(extra)
                else:
                    fixture['ledger']['entries'][2]['cost_usd'] = 40.1
                update_spend(fixture)
                with self.assertRaises(ValueError):
                    validate(**fixture)

    def test_actual_api_usage_cannot_exceed_full_context_limit(self):
        call = self.arm()['reader']['call']
        call['usage']['input_tokens'] = 131073
        call['response']['usage'] = copy.deepcopy(call['usage'])
        call['cost_usd'] = (131073 * 2.5 + call['usage']['output_tokens'] * 15) / 1_000_000
        with self.assertRaisesRegex(ValueError, 'actual full-history input token cap exceeded'):
            validate(**self.fixture)

    def test_recorded_reader_resampling_is_rejected(self):
        self.arm()['reader']['prior_attempts'] = [copy.deepcopy(self.arm()['reader']['call'])]
        with self.assertRaisesRegex(ValueError, 'resampling or retries'):
            validate(**self.fixture)

    def test_raw_answer_mismatch_and_quote_corruption_remain_independently_checked(self):
        self.arm()['reader']['parsed']['answer'] = 'red'
        with self.assertRaisesRegex(ValueError, 'reader parsed answer mismatch'):
            validate(**self.fixture)
        fixture = make_full_fixture(self.root)
        arm = self.arm(fixture)
        arm['reader']['parsed']['claims'][0]['citations'][0]['quote'] = 'I prefer red.'
        raw = canonical(arm['reader']['parsed'])
        arm['reader']['raw_text'] = raw
        arm['reader']['call']['raw_text'] = raw
        arm['reader']['call']['response']['text'] = raw
        with self.assertRaisesRegex(ValueError, 'citation_valid_claims'):
            validate(**fixture)


if __name__ == '__main__':
    unittest.main()
