# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Dev quality diagnostics from synthetic, source-grounded reader records."""

import copy
import json
import unittest

from check_longmemeval_answers import ARMS, recompute_scores
from longmemeval_quality import diagnose


SYSTEM = 'matched_gpt54'
OUTCOMES = ('correct', 'selection_failure', 'reader_failure', 'judge_failure',
            'judge_rejected_abstention', 'judge_rejected_answer')
QUESTION_TYPES = ('temporal-reasoning', 'knowledge-update', 'multi-session', 'single-session-user')


def make_arm():
    source = {'source_id': 'session:0:turn:0', 'text': 'My preferred color is blue.'}
    parsed = {'answer': 'blue', 'abstain': False,
              'claims': [{'text': 'The preferred color is blue.', 'citations': [
                  {'source_id': source['source_id'], 'quote': source['text']}]}]}
    arm = {'reader': {'status': 'ok', 'raw_text': json.dumps(parsed), 'parsed': parsed},
           'judge': {'status': 'ok', 'raw_text': 'yes', 'score': True},
           'retained_sources': [source],
           'reader_cap': {'dropped_group_ids': [], 'selection_failure': False}}
    arm['scores'] = recompute_scores(arm)
    return arm


def make_inputs(systems=(SYSTEM,)):
    dev = [f'dev-{index:03d}' + ('_abs' if index >= 90 else '') for index in range(100)]
    split = {'dev': dev, 'heldout': [f'heldout-{index:03d}' for index in range(400)]}
    report = {'schema_version': 1, 'complete': True, 'dev_only': True, 'heldout_run': False,
              'reader_systems': list(systems), 'rows': []}
    evidence = {'rows': []}
    for index, qid in enumerate(dev):
        qtype = QUESTION_TYPES[index % len(QUESTION_TYPES)]
        report['rows'].append({'qid': qid, 'qtype': qtype, 'systems': {
            system: {'arms': {name: make_arm() for name in ARMS}} for system in systems}})
        evidence['rows'].append({'qid': qid, 'qtype': qtype, 'scored': True,
                                 'arms': {name: {'complete': True} for name in ARMS}})
    return report, split, evidence


def set_answer(arm, *, abstain=False, accepted=True):
    if abstain:
        parsed = {'answer': "I don't know", 'abstain': True, 'claims': []}
        arm['reader'] = {'status': 'ok', 'raw_text': json.dumps(parsed), 'parsed': parsed}
    arm['judge'] = {'status': 'ok', 'raw_text': 'yes' if accepted else 'no', 'score': accepted}
    arm['scores'] = recompute_scores(arm)


class QualityDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.report, self.split, self.evidence = make_inputs()

    def arm(self, index, name='first_five', system=SYSTEM):
        return self.report['rows'][index]['systems'][system]['arms'][name]

    def diagnose(self):
        return diagnose(self.report, self.split, self.evidence)

    def test_complete_dev_denominators_question_types_and_input_purity(self):
        original = copy.deepcopy((self.report, self.split, self.evidence))
        result = self.diagnose()
        self.assertEqual(result['failures'], [])
        self.assertEqual(set(result['systems']), {SYSTEM})
        for name in ARMS:
            aggregate = result['systems'][SYSTEM][name]
            self.assertEqual(aggregate['n'], 100)
            self.assertEqual(aggregate['correct_count'], 100)
            self.assertEqual(aggregate['answer_accuracy'], 1.0)
            self.assertEqual(aggregate['outcomes'], {key: 100 if key == 'correct' else 0 for key in OUTCOMES})
            self.assertEqual(set(aggregate['by_question_type']), set(QUESTION_TYPES))
            for qtype in QUESTION_TYPES:
                subgroup = aggregate['by_question_type'][qtype]
                self.assertEqual(subgroup['n'], 25)
                self.assertEqual(subgroup['correct_count'], 25)
                self.assertEqual(subgroup['answer_accuracy'], 1.0)
                self.assertEqual(subgroup['outcomes']['correct'], 25)
            self.assertEqual(aggregate['abstention'], {
                'expected_answerable': 90, 'expected_unanswerable': 10,
                'on_answerable': 0, 'on_unanswerable': 0, 'judge_accepted': 0, 'judge_rejected': 0})
        self.assertEqual((self.report, self.split, self.evidence), original)

    def test_failures_are_mutually_exclusive_and_remain_in_accuracy_denominator(self):
        for index, reader_status, judge_status in (
                (0, 'selection_failure', 'selection_failure'), (1, 'error', 'reader_failure')):
            arm = self.arm(index)
            arm['reader'] = {'status': reader_status, 'raw_text': '', 'parsed': None}
            arm['judge'] = {'status': judge_status, 'raw_text': '', 'score': False}
            arm['reader_cap']['selection_failure'] = reader_status == 'selection_failure'
            arm['scores'] = recompute_scores(arm)
        self.arm(2)['judge'] = {'status': 'error', 'raw_text': '', 'score': False}
        self.arm(2)['scores'] = recompute_scores(self.arm(2))
        set_answer(self.arm(3), abstain=True, accepted=False)
        set_answer(self.arm(4), accepted=False)
        self.assertTrue(self.arm(4)['scores']['all_claims_supported'])
        result = self.diagnose()
        aggregate = result['systems'][SYSTEM]['first_five']
        self.assertEqual(aggregate['n'], 100)
        self.assertEqual(aggregate['correct_count'], 95)
        self.assertEqual(aggregate['answer_accuracy'], 0.95)
        self.assertEqual(aggregate['outcomes'], {key: 95 if key == 'correct' else 1 for key in OUTCOMES})
        self.assertEqual(sum(aggregate['outcomes'].values()), 100)
        expected = {(self.report['rows'][index]['qid'], self.report['rows'][index]['qtype'],
                     SYSTEM, 'first_five', outcome) for index, outcome in enumerate(OUTCOMES[1:])}
        actual = {(failure['qid'], failure['qtype'], failure['system'], failure['arm'], failure['outcome'])
                  for failure in result['failures']}
        self.assertEqual(actual, expected)
        self.assertEqual(len(result['failures']), 5)
        for name in ('knapsack', 'operand'):
            self.assertEqual(result['systems'][SYSTEM][name]['correct_count'], 100)

    def test_abstention_labels_and_official_judge_acceptance_are_independent(self):
        for index, accepted in ((0, True), (1, False), (90, True), (91, False)):
            set_answer(self.arm(index), abstain=True, accepted=accepted)
        aggregate = self.diagnose()['systems'][SYSTEM]['first_five']
        self.assertEqual(aggregate['abstention'], {
            'expected_answerable': 90, 'expected_unanswerable': 10,
            'on_answerable': 2, 'on_unanswerable': 2, 'judge_accepted': 2, 'judge_rejected': 2})
        self.assertEqual(aggregate['correct_count'], 98)
        self.assertEqual(aggregate['outcomes']['judge_rejected_abstention'], 2)

    def test_unscored_evidence_is_excluded_from_complete_and_incomplete_counts(self):
        for index in (0, 1, 2):
            set_answer(self.arm(index), accepted=False)
        for index in (1, 3, 4):
            self.evidence['rows'][index]['arms']['first_five']['complete'] = False
        for index in (2, 4):
            self.evidence['rows'][index]['scored'] = False
        result = self.diagnose()['systems'][SYSTEM]
        self.assertEqual(result['first_five']['selected_evidence'], {
            'scored': 98, 'complete': 96, 'incomplete': 2,
            'incorrect_with_complete': 1, 'incorrect_with_incomplete': 1})
        self.assertEqual(result['first_five']['correct_count'], 97)
        self.assertEqual(result['operand']['selected_evidence'], {
            'scored': 98, 'complete': 98, 'incomplete': 0,
            'incorrect_with_complete': 0, 'incorrect_with_incomplete': 0})

    def test_declared_reader_systems_are_reported_separately(self):
        self.report, self.split, self.evidence = make_inputs((SYSTEM, 'local_qwen30b'))
        set_answer(self.arm(0, system='local_qwen30b'), accepted=False)
        result = self.diagnose()
        self.assertEqual(set(result['systems']), {SYSTEM, 'local_qwen30b'})
        self.assertEqual(result['systems'][SYSTEM]['first_five']['correct_count'], 100)
        self.assertEqual(result['systems']['local_qwen30b']['first_five']['correct_count'], 99)
        self.report['reader_systems'] = ['local_qwen30b']
        for row in self.report['rows']:
            del row['systems'][SYSTEM]
        self.assertEqual(set(self.diagnose()['systems']), {'local_qwen30b'})

    def test_corrupted_stored_scores_are_rejected(self):
        for field, value in (('correct', False), ('citation_valid_claims', 0),
                             ('citation_total_claims', 2), ('all_claims_supported', False), ('abstain', True)):
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                report['rows'][0]['systems'][SYSTEM]['arms']['first_five']['scores'][field] = value
                with self.assertRaises(ValueError):
                    diagnose(report, self.split, self.evidence)

    def test_incomplete_nondev_and_heldout_report_flags_are_rejected(self):
        for field, value in (('schema_version', 2), ('schema_version', True), ('complete', False),
                             ('dev_only', False), ('heldout_run', True)):
            with self.subTest(field=field, value=value):
                report = copy.deepcopy(self.report)
                report[field] = value
                with self.assertRaises(ValueError):
                    diagnose(report, self.split, self.evidence)

    def test_report_and_evidence_row_count_boundaries(self):
        for target in ('report', 'evidence'):
            for size in (99, 101):
                with self.subTest(target=target, size=size):
                    report, split, evidence = make_inputs()
                    rows = (report if target == 'report' else evidence)['rows']
                    if size == 99:
                        rows.pop()
                    else:
                        rows.append(copy.deepcopy(rows[0]) | {'qid': 'extra-dev'})
                    with self.assertRaises(ValueError):
                        diagnose(report, split, evidence)

    def test_split_count_boundaries_duplicates_and_overlap_are_rejected(self):
        for partition, expected in (('dev', 100), ('heldout', 400)):
            for size in (expected - 1, expected + 1):
                with self.subTest(partition=partition, size=size):
                    split = copy.deepcopy(self.split)
                    split[partition] = split[partition][:size] if size < expected else split[partition] + ['extra-id']
                    with self.assertRaises(ValueError):
                        diagnose(self.report, split, self.evidence)
            split = copy.deepcopy(self.split)
            split[partition][1] = split[partition][0]
            with self.subTest(partition=partition, issue='duplicate'):
                with self.assertRaises(ValueError):
                    diagnose(self.report, split, self.evidence)
        self.split['heldout'][0] = self.split['dev'][0]
        with self.assertRaises(ValueError):
            self.diagnose()

    def test_duplicate_and_heldout_rows_are_rejected_in_both_inputs(self):
        for target in ('report', 'evidence'):
            for replacement in (self.split['dev'][1], self.split['heldout'][0]):
                with self.subTest(target=target, replacement=replacement):
                    report, split, evidence = make_inputs()
                    (report if target == 'report' else evidence)['rows'][0]['qid'] = replacement
                    with self.assertRaises(ValueError):
                        diagnose(report, split, evidence)

    def test_evidence_labels_and_complete_arms_are_validated(self):
        for mutation in ('qtype', 'scored', 'complete', 'missing_evidence_arm', 'missing_report_arm'):
            with self.subTest(mutation=mutation):
                report, split, evidence = make_inputs()
                row = evidence['rows'][0]
                if mutation == 'qtype':
                    row['qtype'] = 'different-type'
                elif mutation == 'scored':
                    row['scored'] = 1
                elif mutation == 'complete':
                    row['arms']['operand']['complete'] = 1
                elif mutation == 'missing_evidence_arm':
                    del row['arms']['operand']
                else:
                    del report['rows'][0]['systems'][SYSTEM]['arms']['operand']
                with self.assertRaises(ValueError):
                    diagnose(report, split, evidence)

    def test_pending_reader_or_judge_in_a_complete_report_is_rejected(self):
        for stage in ('reader', 'judge'):
            with self.subTest(stage=stage):
                report = copy.deepcopy(self.report)
                report['rows'][0]['systems'][SYSTEM]['arms']['first_five'][stage]['status'] = 'pending'
                with self.assertRaises(ValueError):
                    diagnose(report, self.split, self.evidence)


if __name__ == '__main__':
    unittest.main()
