# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = ["tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Offline experiment integrity, spend accounting, and 100-question acceptance."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import longmemeval_user_first as experiment


QID = 'synthetic-dev'
SESSION_IDS = (
    '340c948e714f4202cf2ac18ebe84eeb463211ee145e6e3bca65f5a1a4537d09b',
    '036a98faccbcb794d57b19081f8a236dd0a4aab7ded4163a8468a967b1af74c0',
)
SOURCE_IDS = (
    '93f1d7a1c1674d5169880896744efc1e46e0f319cf2c4abcb338fb4d4d29213b',
    '3a26fe2e99dc91792da32178a64e28553c4b0401434c35b64b8a7bdc4a74117f',
    '11e4ba92772da3efece36714c8d35ccabd36a1d32e92c3a816197810f66ed45d',
)
DATES = ('2024/01/02 (Tue) 09:00', '2024/02/03 (Sat) 10:00')


def source_fixture():
    sessions = [
        {'date': DATES[0], 'turns': [('repeated_session_1', 'An earlier preference.'),
                                   ('repeated_session_3', 'A changed preference.')],
         'answer': 'GOLD ANSWER', 'has_answer': True},
        {'date': DATES[1], 'turns': [('repeated_session_1', 'An earlier preference.')],
         'answer_session_ids': ['GOLD SESSION']},
    ]
    sources = [
        {'source_id': SOURCE_IDS[0], 'session_id': SESSION_IDS[0], 'session_index': 0,
         'turn_index': 0, 'observation_time': DATES[0], 'role': 'user', 'text': 'An earlier preference.'},
        {'source_id': SOURCE_IDS[1], 'session_id': SESSION_IDS[0], 'session_index': 0,
         'turn_index': 2, 'observation_time': DATES[0], 'role': 'user', 'text': 'A changed preference.'},
        {'source_id': SOURCE_IDS[2], 'session_id': SESSION_IDS[1], 'session_index': 1,
         'turn_index': 0, 'observation_time': DATES[1], 'role': 'user', 'text': 'An earlier preference.'},
    ]
    return sessions, sources


def snapshot(tid, sources):
    return {'answer': 'GOLD ANSWER', 'systems': {experiment.SYSTEM: {'arms': {'operand': {
        'source_groups': [{'frozen_selected_id_sha256': hashlib.sha256(tid.encode()).hexdigest(),
                           'sources': copy.deepcopy(sources)}]}}}}}


def make_report(correct=100, complete=True):
    rows = []
    for index in range(100):
        accepted = index < correct
        arm = {'reader': {'status': 'ok'}, 'judge': {'status': 'ok'},
               'scores': {'correct': accepted, 'abstain': False, 'citation_valid_claims': 1,
                          'citation_total_claims': 1}}
        rows.append({'qid': f'dev-{index:03d}', 'qtype': 'temporal-reasoning' if index % 2 else 'knowledge-update',
                     'systems': {experiment.SYSTEM: {'arms': {experiment.ARM: arm}}},
                     'scored_evidence': index < 90, 'baseline_actual_complete': index < 80,
                     'actual_complete': index < 85})
    return {'rows': rows, 'complete': complete}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class UserFirstSources(unittest.TestCase):
    def test_candidate_order_and_repeated_occurrences_keep_exact_ids_and_dates(self):
        sessions, sources = source_fixture()
        archive_sessions = (
            'b9b4d5c249f4da6fac4e09ff7b0dd226a1eb002abdfdb52a4504150382156e12',
            'bd0e48d1c52777164e4ca1cc18df132d9d70c0b0e6cb585c35aa324168cbf862',
        )
        archive_sources = (
            'f95723ee4b8df12780bcf3e9a9c991ac1391086a579ea8d52dacb42eb14cc4c1',
            '4562f45780adc1b8f125a2360f4c984ccbbc76d6a50bd895b3047d4f2a6162bc',
            '7e611b06cbdab3e8673f54e82ecbf7c47f7869a778e98ee1746b85b872948107',
        )
        for source, source_id in zip(sources, archive_sources):
            source['source_id'] = source_id
            source['session_id'] = archive_sessions[source['session_index']]
        candidates = ['repeated_session_3', 'repeated_session_1']
        groups = experiment.source_groups(QID, sessions, candidates, [])
        self.assertEqual([group['selection_rank'] for group in groups], [0, 1])
        self.assertEqual([group['group_id'] for group in groups], [
            'bf245c2b9670ed9815197b256d8a44d1819db847de79b9b80794876571c73a0d',
            'bd8939fb261edcd2064692e7cee379dbedf3fd86757e6a0c28a60f82fddc706f'])
        self.assertEqual(groups[0]['sources'], [sources[1]])
        self.assertEqual(groups[1]['sources'], [sources[0], sources[2]])
        self.assertNotEqual(groups[1]['sources'][0]['source_id'], groups[1]['sources'][1]['source_id'])
        self.assertEqual([group['frozen_selected_id_sha256'] for group in groups],
                         [hashlib.sha256(tid.encode()).hexdigest() for tid in candidates])
        for group in groups:
            for source in group['sources']:
                self.assertEqual(set(source), {'source_id', 'session_id', 'session_index', 'turn_index',
                                               'observation_time', 'role', 'text'})

    def test_snapshot_preserves_complete_assistant_text_and_does_not_alias_inputs(self):
        sessions, sources = source_fixture()
        assistant = {'source_id': '8f35c4dc1b45d443ac0d7d1dfc2418d0f943a64c607e16c9e55272405e7fb6e4',
                     'session_id': SESSION_IDS[0], 'session_index': 0, 'turn_index': 1,
                     'observation_time': DATES[0], 'role': 'assistant',
                     'text': 'A complete assistant continuation.\nIncluding its final line.'}
        known = [sources[0], assistant, sources[2]]
        saved = snapshot('repeated_session_1', known)
        original = copy.deepcopy(saved)
        groups = experiment.source_groups(QID, sessions, ['repeated_session_1'], [saved])
        self.assertEqual(groups[0]['sources'], known)
        groups[0]['sources'][0]['text'] = 'mutated result'
        self.assertEqual(saved, original)

    def test_snapshot_session_offset_maps_unknown_turns_without_identity_collisions(self):
        sessions = [
            {'date': DATES[0], 'turns': [('mapped_session_1', 'Known user.'),
                                       ('mapped_session_3', 'Later user.')]},
            {'date': DATES[1], 'turns': [('unmapped_session_1', 'Other user.')]},
        ]
        known_user = {'source_id': SOURCE_IDS[2], 'session_id': SESSION_IDS[1], 'session_index': 1,
                      'turn_index': 0, 'observation_time': DATES[0], 'role': 'user', 'text': 'Known user.'}
        known_assistant = {'source_id': 'a94ce5a2b026e6a8988108e3c5f4054c48e1617e776357a90fcfb0791c13bc06',
                           'session_id': SESSION_IDS[1], 'session_index': 1, 'turn_index': 1,
                           'observation_time': DATES[0], 'role': 'assistant', 'text': 'Complete saved reply.'}
        known = [known_user, known_assistant]
        saved = snapshot('mapped_session_1', known)
        original = copy.deepcopy(saved)
        groups = experiment.source_groups(
            QID, sessions, ['mapped_session_1', 'mapped_session_3', 'unmapped_session_1'], [saved])
        self.assertEqual(groups[0]['sources'], known)
        self.assertEqual(groups[1]['sources'], [{
            'source_id': 'b69408ba752e02497ae3edf90ffc658cdeff7c0dac2d6ec8c1b13ebb81a1035c',
            'session_id': SESSION_IDS[1], 'session_index': 1, 'turn_index': 2,
            'observation_time': DATES[0], 'role': 'user', 'text': 'Later user.'}])
        unmapped = groups[2]['sources'][0]
        self.assertEqual(unmapped['session_index'], 1)
        self.assertEqual(unmapped['source_id'], '7e611b06cbdab3e8673f54e82ecbf7c47f7869a778e98ee1746b85b872948107')
        self.assertEqual(unmapped['session_id'], 'bd0e48d1c52777164e4ca1cc18df132d9d70c0b0e6cb585c35aa324168cbf862')
        all_sources = [source for group in groups for source in group['sources']]
        self.assertEqual(len({source['source_id'] for source in all_sources}), 4)
        self.assertEqual(len({group['group_id'] for group in groups}), 3)
        self.assertEqual(len({source['session_id'] for source in all_sources}), 2)
        self.assertEqual(saved, original)

    def test_snapshot_user_text_or_cross_snapshot_disagreement_is_rejected(self):
        sessions, sources = source_fixture()
        valid = snapshot('repeated_session_1', [sources[0], sources[2]])
        changed = copy.deepcopy(valid)
        changed['systems'][experiment.SYSTEM]['arms']['operand']['source_groups'][0]['sources'][0]['text'] += ' altered'
        with self.assertRaisesRegex(ValueError, 'snapshot user source'):
            experiment.source_groups(QID, sessions, ['repeated_session_1'], [changed])
        with self.assertRaisesRegex(ValueError, 'snapshot source disagreement'):
            experiment.source_groups(QID, sessions, ['repeated_session_1'], [valid, changed])

    def test_serialized_evidence_excludes_gold_labels_and_internal_selection_fields(self):
        sessions, _ = source_fixture()
        groups = experiment.source_groups(QID, sessions, ['repeated_session_1'], [])
        groups[0]['gold_answer'] = 'GOLD ANSWER'
        groups[0]['sources'][0]['has_answer'] = True
        groups[0]['sources'][0]['gold_answer'] = 'GOLD ANSWER'
        question = {'question': 'What changed?', 'question_date': '2024/03/01 (Fri) 12:00',
                    'answer': 'GOLD ANSWER', 'answer_session_ids': ['GOLD SESSION'], 'qtype': 'GOLD TYPE'}
        prefix = 'Frozen reader instruction\n'
        serialized = experiment.serialize(question, groups, {'reader_prompt': prefix})
        self.assertTrue(serialized.startswith(prefix))
        payload = json.loads(serialized[len(prefix):])
        self.assertEqual(set(payload), {'question', 'question_date', 'evidence'})
        self.assertEqual(len(payload['evidence']), 2)
        for source in payload['evidence']:
            self.assertEqual(set(source), {'source_id', 'session_id', 'observation_time', 'role', 'text'})
        self.assertNotIn('GOLD', serialized)
        self.assertEqual([source['observation_time'] for source in payload['evidence']], list(DATES))


class UserFirstSpend(unittest.TestCase):
    def test_seed_ledger_deduplicates_by_conservative_max_and_keeps_category_totals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / 'benchmarks/results'
            primary = results / 'spend_2026-10.json'
            supplemental = results / 'longmemeval_dev_answers_v1_spend.json'
            reservation = results / 'longmemeval_dev_answers_v3_spend.json'
            caps = {'total': 100, 'reader': 30, 'judge': 20, 'preprocessing': 10}
            write_json(primary, {'caps_usd': caps, 'entries': [
                {'cache_key': 'shared', 'cost_usd': 4.5, 'category': 'reader', 'tag': 'original-reservation'},
                {'cache_key': 'shared', 'cost_usd': 1.0, 'category': 'reader', 'tag': 'duplicate'},
                {'cache_key': 'base-judge', 'cost_usd': 0.5, 'category': 'judge'}]})
            write_json(supplemental, {'entries': [
                {'cache_key': 'shared', 'cost_usd': 2.0, 'category': 'reader', 'tag': 'lower-actual'},
                {'cache_key': 'new-reader', 'cost_usd': 0.25, 'category': 'reader'}]})
            write_json(reservation, {'entries': [
                {'cache_key': 'shared', 'cost_usd': 6.0, 'category': 'reader', 'tag': 'larger-reservation'},
                {'cache_key': 'new-judge', 'cost_usd': 0.75, 'category': 'judge'}]})
            work = results / 'synthetic-experiment'
            with patch.object(experiment, 'ROOT', root):
                experiment.seed_ledger(work)
                destination = work / 'spend.json'
                seeded = json.loads(destination.read_text())
                self.assertEqual(len(seeded['entries']), 4)
                entries = {entry['cache_key']: entry for entry in seeded['entries']}
                self.assertEqual(entries['shared']['cost_usd'], 6.0)
                self.assertEqual(entries['shared']['tag'], 'larger-reservation')
                self.assertEqual(seeded['total_usd'], 7.5)
                self.assertEqual(seeded['by_category_usd'], {'reader': 6.25, 'judge': 1.25, 'preprocessing': 0})
                self.assertEqual(seeded['caps_usd'], caps)
                self.assertEqual({item['path'] for item in seeded['seed_paths']},
                                 {str(path.relative_to(root)) for path in (primary, supplemental, reservation)})
                for item in seeded['seed_paths']:
                    self.assertEqual(item['sha256'], hashlib.sha256((root / item['path']).read_bytes()).hexdigest())
                previous = destination.read_bytes()
                write_json(primary, {'caps_usd': caps, 'entries': []})
                experiment.seed_ledger(work)
                self.assertEqual(destination.read_bytes(), previous)


class UserFirstAcceptance(unittest.TestCase):
    def test_target_is_exactly_100_correct_and_requires_completed_judging(self):
        for correct, complete, expected in ((99, True, False), (100, False, False), (100, True, True)):
            with self.subTest(correct=correct, complete=complete):
                summary = experiment.summarize(make_report(correct, complete))
                self.assertEqual(summary['n'], 100)
                self.assertEqual(summary['acceptance_correct'], 100)
                self.assertEqual(summary['correct'], correct)
                self.assertEqual(summary['answer_accuracy'], correct / 100)
                self.assertIs(summary['acceptance_met'], expected)

    def test_reader_and_selection_failures_stay_in_100_question_denominator(self):
        report = make_report()
        for index, status in ((0, 'error'), (1, 'selection_failure')):
            arm = report['rows'][index]['systems'][experiment.SYSTEM]['arms'][experiment.ARM]
            arm['reader']['status'] = status
            arm['judge']['status'] = 'reader_failure'
            arm['scores'].update(correct=False, citation_valid_claims=0, citation_total_claims=0)
        summary = experiment.summarize(report)
        self.assertEqual(summary['n'], 100)
        self.assertEqual(summary['correct'], 98)
        self.assertEqual(summary['answer_accuracy'], 0.98)
        self.assertFalse(summary['acceptance_met'])
        self.assertEqual(summary['reader_status'], {'ok': 98, 'error': 1, 'selection_failure': 1})
        self.assertEqual(summary['judge_status'], {'ok': 98, 'reader_failure': 2})
        self.assertEqual(summary['by_type'], {'knowledge-update': {'n': 50, 'correct': 49},
                                             'temporal-reasoning': {'n': 50, 'correct': 49}})
        self.assertEqual(summary['coverage'], {'scored': 90, 'baseline_actual_complete': 80, 'actual_complete': 85})

    def test_judging_pending_readers_fails_before_reference_or_client_access(self):
        report = make_report(complete=False)
        report['artifacts'] = []
        arm = report['rows'][0]['systems'][experiment.SYSTEM]['arms'][experiment.ARM]
        arm['reader']['status'] = 'pending'
        arm['judge']['status'] = 'pending'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / 'benchmarks/results/synthetic-experiment'
            split = root / 'benchmarks/public/longmemeval_split.json'
            baseline = root / 'benchmarks/results/baseline.json'
            write_json(split, {'dev': [row['qid'] for row in report['rows']]})
            write_json(baseline, {})
            write_json(work / 'protocol.json', {})
            report['protocol_sha256'] = hashlib.sha256((work / 'protocol.json').read_bytes()).hexdigest()
            with (patch.object(experiment, 'ROOT', root),
                  patch.object(experiment, 'SPLIT', split),
                  patch.object(experiment, 'BASELINE', baseline),
                  patch.object(experiment, 'load_protocol', return_value={}),
                  patch.object(experiment, 'reference_answers', side_effect=AssertionError('reference accessed')) as references,
                  patch.object(experiment.reader, 'client', side_effect=AssertionError('client accessed')) as client,
                  patch.object(experiment.reader, 'api_call', side_effect=AssertionError('model called')) as api_call):
                with self.assertRaisesRegex(ValueError, 'reader|pending'):
                    experiment.run_stage(report, work, 'judge')
                references.assert_not_called()
                client.assert_not_called()
                api_call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
