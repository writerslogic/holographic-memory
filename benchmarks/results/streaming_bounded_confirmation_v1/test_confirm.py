# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import unittest

from confirm import FIELDS, PHASES, assess, expected_runs, validate_receipt, validate_records


def records():
    queries = [{'event': 'query', 'phase': phase, 'ef': 128, 'queries': 128,
                'batch_ns': [1, 2, 3, 4, 5]} for phase in PHASES]
    return [
        {'event': 'start', 'action': 'run', 'rows': 1_000_000,
         'capacity': 250_000, 'threads': 8},
        queries[0], {'event': 'mutated', 'elapsed_ns': 1}, queries[1],
        {'event': 'consolidated', 'cgroup_peak_reset': True, 'dirty': [[0, 241000], [3, 160000]],
         'how': ['Merged', 'Merged'], **dict.fromkeys(FIELDS, 1)},
        queries[2], {'event': 'complete', 'reopen_results_equal': True},
    ]


def scored():
    runs = []
    pairs = []
    for name, (rows, _, arm) in expected_runs().items():
        runs.append({'name': name, 'rows': rows, 'capacity': rows if arm == 'global' else 250_000,
                     'start': 640, 'mutation': {'elapsed_ns': 10},
                     'queries': [{'phase': phase, 'fingerprint': f'{rows}-{arm}-{phase}',
                                  'hits': 100 if arm == 'global' else 99} for phase in PHASES]})
        if arm == 'segments':
            pairs.append({'candidate': name, 'base': name.replace('-segments-', '-global-'),
                          'consolidation_ratios': dict.fromkeys(FIELDS, 0.5),
                          'query_ratios': [{'phase': phase, 'candidate_ef': 128, 'base_ef': 256,
                                            'time_ratio': 1.5, 'recall_delta': -1 / 1280}
                                           for phase in PHASES]})
    return {'runs': runs, 'pairs': pairs}


class ConfirmationTests(unittest.TestCase):
    def test_valid_records(self):
        validate_records(records(), 1_000_000, 'segments')

    def test_successful_receipt_required(self):
        success = ('test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; '
                   '570 filtered out; finished in 6.40s\n\tExit status: 0\n')
        validate_receipt(success)
        for log in (success.replace('ok.', 'FAILED.'), success.replace('status: 0', 'status: 1'),
                    success.splitlines()[0], '\tExit status: 0\n', success + success):
            with self.subTest(log=log), self.assertRaises(ValueError):
                validate_receipt(log)

    def test_unexpected_consolidation_outcome_rejected(self):
        rows = records()
        rows[4]['how'][0] = 'Rebuilt'
        with self.assertRaises(ValueError):
            validate_records(rows, 1_000_000, 'segments')

    def test_query_count_boundaries(self):
        for count in (127, 128, 129):
            with self.subTest(count=count):
                rows = records()
                rows[1]['queries'] = count
                if count == 128:
                    validate_records(rows, 1_000_000, 'segments')
                else:
                    with self.assertRaises(ValueError):
                        validate_records(rows, 1_000_000, 'segments')

    def test_reopen_and_isolation_fail_closed(self):
        for position, field in ((-1, 'reopen_results_equal'), (4, 'cgroup_peak_reset')):
            rows = records()
            rows[position][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_records(rows, 1_000_000, 'segments')

    def test_extra_or_missing_event_rejected(self):
        for rows in (records()[:-1], records() + [records()[-1]]):
            with self.assertRaises(ValueError):
                validate_records(rows, 1_000_000, 'segments')

    def test_nonfinite_or_missing_memory_rejected(self):
        for value in (None, 0, float('nan'), float('inf')):
            rows = records()
            rows[4]['isolated_cgroup_peak'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_records(rows, 1_000_000, 'segments')

    def test_losses_are_retained(self):
        result = assess(scored())
        for size in result['sizes']:
            self.assertFalse(size['every_phase_meets_control_recall'])
            self.assertFalse(size['every_phase_query_time_improves'])
            self.assertEqual(size['consolidation_median_ratios']['elapsed_ns'], 0.5)
            for query in size['queries']:
                self.assertEqual(query['candidate_hits'], 99)
                self.assertEqual(query['median_time_ratio'], 1.5)
                self.assertEqual(query['recall_deltas'], [-1 / 1280] * 3)

    def test_incomplete_or_duplicate_arm_rejected(self):
        for missing in (True, False):
            result = scored()
            if missing:
                result['runs'].pop()
            else:
                result['runs'][-1] = copy.deepcopy(result['runs'][0])
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                assess(result)

    def test_nondeterministic_results_rejected(self):
        result = scored()
        result['runs'][0]['queries'][0]['fingerprint'] = 'changed'
        with self.assertRaises(ValueError):
            assess(result)


if __name__ == '__main__':
    unittest.main()
