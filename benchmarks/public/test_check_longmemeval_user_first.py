# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = ["tiktoken==0.14.0"]
# ///
"""Synthetic tampering checks without model calls or corpus access."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import tiktoken

from check_longmemeval_answers import canonical, digest, file_digest, recompute_scores, sorted_sources, wilson
from check_longmemeval_user_first import (
    ARM, SYSTEM, check_historical_policy, check_references, independent_summary, validate, verify_artifacts,
)
from longmemeval_answer_evidence import reader_payload


MODEL = 'gpt-5.4-2026-03-05'
WORK_NAME = 'synthetic-experiment'
OFFICIAL_SHA256 = '1' * 64
DATE = '2024/01/02 (Tue) 09:00'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_fixture(root):
    judge_source = root / 'official_judge.py'
    judge_source.write_text('def get_anscheck_prompt():\n    return "I will give you {}|{}|{}"\n')
    settings = {'reader_model': MODEL, 'judge_model': MODEL, 'reader_max_output_tokens': 4096,
                'judge_max_output_tokens': 2048, 'reasoning_effort': 'medium', 'tokenizer': 'o200k_base'}
    template = 'I will give you {question}|{answer}|{response}'
    protocol = {'schema_version': 1, 'created_before_first_reader_call': True,
                'complete_input_counting': True, 'token_cap': 8192, 'reader_prompt': 'Use the evidence.\n',
                'arms': ['first_five', 'knapsack', 'operand'],
                'systems': {SYSTEM: settings, 'weak_qwen4b': {'judge_model': MODEL}},
                'judge_source': {'path': str(judge_source), 'sha256': file_digest(judge_source),
                                 'templates': {'knowledge-update': template}, 'abstention_template': template},
                'cost': {'subcap_usd': 120, 'automatic_retries': 0, 'input_usd_per_million': 2.5,
                         'cached_input_usd_per_million': 0.25, 'output_usd_per_million': 15}}
    caps = {'total': 250, 'reader_judge_dev': 120, 'heldout400': 0, 'preprocessing': 130}
    preregistration = {'session': 1, 'heldout_authorized': False, 'spend_caps_usd': caps,
                      'reader_protocol_sha256': '0' * 64}
    frozen = {'reader': copy.deepcopy(settings), 'dev_only': True, 'heldout_run': False,
              'acceptance_correct': 100, 'token_cap': 8192, 'local_token_cap': 7680,
              'automatic_transport_retries': 0, 'reader_judge_concurrency': 2,
              'reader_prompt': protocol['reader_prompt'], 'candidate_mode': 'pool'}
    split = {'dev': [f'dev-{index:03d}' for index in range(100)],
             'heldout': [f'heldout-{index:03d}' for index in range(400)]}
    tokenizer = tiktoken.get_encoding('o200k_base')
    ledger = {'schema_version': 1, 'caps_usd': caps, 'entries': [
        {'cache_key': 'seeded-prior-call', 'call_id': 'prior', 'category': 'preprocessing',
         'status': 'ok', 'cost_usd': 1.0}]}
    rows, baseline, history, evidence, exported, archives, references = [], [], [], [], [], {}, {}
    for qid in split['dev']:
        tid = 'session_answer_1'
        source = {'source_id': digest(canonical(['lme-source-v1', qid, 2, 0])),
                  'session_id': digest(canonical(['lme-session-v1', qid, 2])), 'session_index': 2,
                  'turn_index': 0, 'observation_time': DATE, 'role': 'user', 'text': 'I prefer blue.'}
        assistant = {'source_id': digest(canonical(['lme-source-v1', qid, 2, 1])),
                     'session_id': source['session_id'], 'session_index': 2, 'turn_index': 1,
                     'observation_time': DATE, 'role': 'assistant', 'text': 'Your preference is blue.'}
        group = {'group_id': digest(canonical(['lme-reader-group-v1', qid, [source['source_id']]])),
                 'frozen_selected_id_sha256': digest(tid), 'selection_rank': 0, 'sources': [source, assistant]}
        row = {'qid': qid, 'qtype': 'knowledge-update', 'question': 'What color do I prefer?',
               'question_date': '2024/02/03 (Sat) 10:00', 'scored_evidence': True,
               'actual_complete': True, 'baseline_actual_complete': True}
        arm = {'source_groups': [group], 'retained_sources': sorted_sources([group])}
        prompt = protocol['reader_prompt'] + canonical(reader_payload(row, arm))
        count = len(tokenizer.encode(prompt))
        arm.update(reader_input=prompt, reader_input_sha256=digest(prompt), input_tokens=count,
                   reader_cap={'cap': 7680, 'input_tokens': count, 'selection_failure': False})
        parsed = {'answer': 'blue', 'abstain': False,
                  'claims': [{'text': 'The preferred color is blue.', 'citations': [
                      {'source_id': source['source_id'], 'quote': 'I prefer blue.'}]}]}
        arm['reader'] = {'status': 'ok', 'raw_text': canonical(parsed), 'parsed': parsed}
        arm['judge'] = {'status': 'ok', 'raw_text': 'yes', 'score': True}
        for stage in ('reader', 'judge'):
            request_input = prompt if stage == 'reader' else template.format(
                question=row['question'], answer='blue', response='blue')
            request = {'model': MODEL, 'input': request_input, 'reasoning': {'effort': 'medium'},
                       'max_output_tokens': settings[stage + '_max_output_tokens'],
                       'store': False, 'truncation': 'disabled'}
            usage = {'input_tokens': len(tokenizer.encode(request_input)) + 10, 'output_tokens': 20,
                     'input_tokens_details': {'cached_tokens': 0}}
            cost = (usage['input_tokens'] * 2.5 + usage['output_tokens'] * 15) / 1_000_000
            call_id = digest(f'{WORK_NAME}:{stage}:{qid}')
            key = digest(canonical({'request': request, 'call_id': call_id, 'protocol_sha256': OFFICIAL_SHA256}))
            raw = arm[stage]['raw_text']
            arm[stage]['call'] = {'request': request, 'request_sha256': digest(canonical(request)),
                                  'cache_key': key, 'status': 'ok', 'raw_text': raw,
                                  'response': {'model': MODEL, 'status': 'completed', 'text': raw,
                                               'usage': copy.deepcopy(usage)}, 'usage': usage, 'cost_usd': cost}
            reserve = ((len(tokenizer.encode(request_input)) + 512) * 2.5 + request['max_output_tokens'] * 15) / 1_000_000
            ledger['entries'].append({'cache_key': key, 'call_id': call_id, 'category': 'reader_judge_dev',
                                      'status': 'ok', 'cost_usd': cost, 'usage': copy.deepcopy(usage),
                                      'reserved_usd': reserve})
        arm['scores'] = recompute_scores(arm)
        row['systems'] = {SYSTEM: {'arms': {ARM: arm}}}
        rows.append(row)
        old_arm = {'source_groups': copy.deepcopy([group]), 'reader_input': prompt,
                   'reader': copy.deepcopy(arm['reader']), 'judge': copy.deepcopy(arm['judge'])}
        baseline.append({key: row[key] for key in ('qid', 'qtype', 'question', 'question_date')} | {
            'systems': {SYSTEM: {'arms': {'operand': old_arm}}}})
        history.append({'qid': qid, 'ids': [tid], 'scored': True})
        evidence.append({'qid': qid, 'candidates': [tid], 'arms': {'operand': {'selected': [tid]}}})
        archives[qid] = [{'date': DATE, 'turns': [[tid, source['text']]]}]
        references[qid] = 'blue'
        exported.append({'qid': qid, 'question': row['question'], 'date': row['question_date'], 'answer': 'blue'})
    summary = {'n': 100, 'correct': 100, 'answer_accuracy': 1.0, 'answer_accuracy_ci95': wilson(100, 100),
               'acceptance_correct': 100, 'acceptance_met': True, 'by_type': {'knowledge-update': {'n': 100, 'correct': 100}},
               'reader_status': {'ok': 100}, 'judge_status': {'ok': 100},
               'coverage': {'scored': 100, 'baseline_actual_complete': 100, 'actual_complete': 100},
               'abstentions': 0, 'exact_quote_claims': 100, 'total_claims': 100}
    report = {'schema_version': 1, 'dev_only': True, 'heldout_run': False, 'complete': True,
              'reader_systems': [SYSTEM], 'candidate_mode': 'pool', 'rows': rows,
              'summary': summary, 'starting_spend_usd': 1.0}
    fixture = {'report': report, 'split': split, 'frozen': frozen, 'protocol': protocol,
               'preregistration': preregistration, 'archives': archives, 'baseline_rows': baseline,
               'v1_rows': copy.deepcopy(baseline), 'history_rows': history, 'evidence_rows': evidence,
               'references': references, 'ledger': ledger, 'work_name': WORK_NAME,
               'official_sha256': OFFICIAL_SHA256}
    update_spend(fixture)
    return fixture, {'questions': exported}


def update_spend(fixture):
    ledger = fixture['ledger']
    categories = {category: 0.0 for category in ledger['caps_usd'] if category != 'total'}
    for entry in ledger['entries']:
        categories[entry['category']] += entry['cost_usd']
    ledger['by_category_usd'] = categories
    ledger['total_usd'] = sum(categories.values())
    fixture['report']['incremental_spend_usd'] = ledger['total_usd'] - fixture['report']['starting_spend_usd']


class CompletedUserFirstChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture, self.exported = make_fixture(self.root)

    def arm(self):
        return self.fixture['report']['rows'][0]['systems'][SYSTEM]['arms'][ARM]

    def test_complete_synthetic_run_passes_and_inputs_remain_unchanged(self):
        before = copy.deepcopy(self.fixture)
        result = validate(**self.fixture)
        self.assertEqual(result['dev_questions_verified'], 100)
        self.assertEqual(result['paid_calls_verified'], 200)
        self.assertEqual(result['summary']['correct'], 100)
        self.assertTrue(result['summary']['acceptance_met'])
        self.assertEqual(self.fixture, before)

    def test_decoded_answer_cannot_disagree_with_recorded_raw_response(self):
        self.arm()['reader']['parsed']['answer'] = 'red'
        with self.assertRaisesRegex(ValueError, 'reader parsed answer mismatch'):
            validate(**self.fixture)

    def test_quote_corruption_is_detected_when_raw_and_decoded_answers_agree(self):
        arm = self.arm()
        arm['reader']['parsed']['claims'][0]['citations'][0]['quote'] = 'I prefer red.'
        raw = canonical(arm['reader']['parsed'])
        arm['reader']['raw_text'] = raw
        arm['reader']['call']['raw_text'] = raw
        arm['reader']['call']['response']['text'] = raw
        with self.assertRaisesRegex(ValueError, 'citation_valid_claims'):
            validate(**self.fixture)

    def test_source_text_dates_ordinals_and_saved_assistant_bytes_are_verified(self):
        for role, field, changed in (('user', 'text', 'Invented user text.'),
                                     ('user', 'observation_time', '2024/01/03 (Wed) 09:00'),
                                     ('user', 'turn_index', 2), ('assistant', 'text', 'Truncated assistant.')):
            with self.subTest(role=role, field=field):
                fixture = copy.deepcopy(self.fixture)
                arm = fixture['report']['rows'][0]['systems'][SYSTEM]['arms'][ARM]
                source = next(source for source in arm['source_groups'][0]['sources'] if source['role'] == role)
                source[field] = changed
                with self.assertRaises(ValueError):
                    validate(**fixture)

    def test_all_archive_user_aliases_are_required(self):
        fixture = copy.deepcopy(self.fixture)
        qid = fixture['split']['dev'][0]
        fixture['archives'][qid].append(copy.deepcopy(fixture['archives'][qid][0]))
        with self.assertRaisesRegex(ValueError, 'alias count mismatch'):
            validate(**fixture)

    def test_reader_prompt_cannot_include_unrecorded_evidence(self):
        arm = self.arm()
        arm['reader_input'] += 'Additional unrecorded evidence.'
        arm['reader_input_sha256'] = digest(arm['reader_input'])
        with self.assertRaisesRegex(ValueError, 'reader input differs'):
            validate(**self.fixture)

    def test_judge_prompt_and_exported_reference_must_match_frozen_reference(self):
        baseline = {row['qid']: row for row in self.fixture['baseline_rows']}
        references = check_references(self.exported, baseline, set(self.fixture['split']['dev']), self.fixture['protocol'])
        self.assertEqual(references, self.fixture['references'])
        altered = copy.deepcopy(self.exported)
        altered['questions'][0]['answer'] = 'red'
        with self.assertRaisesRegex(ValueError, 'reference differs from frozen judge input'):
            check_references(altered, baseline, set(self.fixture['split']['dev']), self.fixture['protocol'])
        call = self.arm()['judge']['call']
        call['request']['input'] = call['request']['input'].replace('|blue|', '|red|')
        call['request_sha256'] = digest(canonical(call['request']))
        with self.assertRaisesRegex(ValueError, 'API input differs'):
            validate(**self.fixture)

    def test_sanitized_protocol_preserves_historical_hash_and_exact_reader_judge_policy(self):
        baseline = {'protocol_sha256': '0' * 64, 'rows': self.fixture['baseline_rows']}
        old_reader = baseline['rows'][0]['systems'][SYSTEM]['arms']['operand']['reader']
        old_reader['prior_attempts'] = [copy.deepcopy(old_reader['call'])]
        result = check_historical_policy(self.fixture['protocol'], self.fixture['preregistration'],
                                         baseline, self.fixture['references'])
        self.assertNotEqual(OFFICIAL_SHA256, baseline['protocol_sha256'])
        self.assertEqual(result, {'preregistered_protocol_sha256': '0' * 64, 'reader_inputs_verified': 100,
                                  'reader_requests_verified': 101, 'judge_requests_verified': 100})
        baseline['protocol_sha256'] = '2' * 64
        with self.assertRaisesRegex(ValueError, 'historical baseline protocol'):
            check_historical_policy(self.fixture['protocol'], self.fixture['preregistration'],
                                    baseline, self.fixture['references'])

    def test_historical_prompt_request_settings_and_prior_attempts_cannot_change(self):
        baseline = {'protocol_sha256': '0' * 64, 'rows': self.fixture['baseline_rows']}
        for corruption in ('prompt', 'reader_model', 'reader_output_limit', 'judge_model', 'judge_reference', 'prior_attempt'):
            with self.subTest(corruption=corruption):
                changed = copy.deepcopy(baseline)
                arm = changed['rows'][0]['systems'][SYSTEM]['arms']['operand']
                if corruption == 'prompt':
                    arm['reader_input'] = 'Changed instruction\n' + arm['reader_input']
                elif corruption == 'prior_attempt':
                    prior = copy.deepcopy(arm['reader']['call'])
                    prior['request']['truncation'] = 'auto'
                    arm['reader']['prior_attempts'] = [prior]
                else:
                    stage = 'reader' if corruption.startswith('reader') else 'judge'
                    request = arm[stage]['call']['request']
                    if corruption.endswith('model'):
                        request['model'] = 'different-model'
                    elif corruption == 'reader_output_limit':
                        request['max_output_tokens'] = 8192
                    else:
                        request['input'] = request['input'].replace('|blue|', '|red|')
                with self.assertRaisesRegex(ValueError, 'historical'):
                    check_historical_policy(self.fixture['protocol'], self.fixture['preregistration'],
                                            changed, self.fixture['references'])

    def test_labels_coverage_and_denominator_cannot_be_rewritten(self):
        for field in ('scored_evidence', 'actual_complete', 'baseline_actual_complete'):
            with self.subTest(field=field):
                fixture = copy.deepcopy(self.fixture)
                fixture['report']['rows'][0][field] = False
                with self.assertRaisesRegex(ValueError, 'label|support'):
                    validate(**fixture)
        self.fixture['report']['summary']['n'] = 99
        with self.assertRaisesRegex(ValueError, 'summary mismatch: n'):
            validate(**self.fixture)
        self.fixture['report']['rows'].pop()
        with self.assertRaisesRegex(ValueError, '100 dev rows'):
            validate(**self.fixture)

    def test_failures_remain_in_denominator_and_99_correct_never_meets_acceptance(self):
        arm = self.arm()
        arm['reader'] = {'status': 'error', 'raw_text': '', 'parsed': None}
        arm['judge'] = {'status': 'reader_failure', 'raw_text': '', 'score': False}
        scores = [recompute_scores(row['systems'][SYSTEM]['arms'][ARM]) for row in self.fixture['report']['rows']]
        summary = independent_summary(self.fixture['report']['rows'], scores, True)
        self.assertEqual(summary['n'], 100)
        self.assertEqual(summary['correct'], 99)
        self.assertEqual(summary['answer_accuracy'], 0.99)
        self.assertEqual(summary['reader_status'], {'error': 1, 'ok': 99})
        self.assertFalse(summary['acceptance_met'])
        self.assertFalse(independent_summary(self.fixture['report']['rows'], [dict(score, correct=True) for score in scores], False)['acceptance_met'])

    def test_spend_charge_duplicates_and_caps_are_verified(self):
        for corruption in ('charge', 'duplicate', 'cap'):
            with self.subTest(corruption=corruption):
                fixture = copy.deepcopy(self.fixture)
                if corruption == 'charge':
                    fixture['ledger']['entries'][1]['cost_usd'] += 0.01
                elif corruption == 'duplicate':
                    fixture['ledger']['entries'].append(copy.deepcopy(fixture['ledger']['entries'][1]))
                else:
                    fixture['ledger']['entries'][0]['cost_usd'] = 251.0
                update_spend(fixture)
                with self.assertRaisesRegex(ValueError, 'spend'):
                    validate(**fixture)

    def test_api_usage_cannot_exceed_8192_reader_tokens(self):
        call = self.arm()['reader']['call']
        call['usage']['input_tokens'] = 8193
        call['response']['usage'] = copy.deepcopy(call['usage'])
        call['cost_usd'] = (8193 * 2.5 + call['usage']['output_tokens'] * 15) / 1_000_000
        with self.assertRaisesRegex(ValueError, 'actual reader input token cap exceeded'):
            validate(**self.fixture)

    def test_complete_report_cannot_contain_pending_stages(self):
        self.arm()['judge']['status'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'pending stage'):
            validate(**self.fixture)

    def test_artifact_and_experiment_protocol_hashes_are_checked(self):
        work = self.root / 'work'
        artifact = self.root / 'frozen.json'
        write_json(work / 'protocol.json', self.fixture['frozen'])
        write_json(artifact, {'frozen': True})
        report = {'protocol_path': str(work / 'protocol.json'), 'protocol_sha256': file_digest(work / 'protocol.json'),
                  'artifacts': [{'path': str(artifact), 'sha256': file_digest(artifact)}]}
        verify_artifacts(report, work, [artifact])
        artifact.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'artifact fingerprint mismatch'):
            verify_artifacts(report, work)
        report['artifacts'][0]['sha256'] = file_digest(artifact)
        (work / 'protocol.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'experiment protocol fingerprint mismatch'):
            verify_artifacts(report, work)


if __name__ == '__main__':
    unittest.main()
