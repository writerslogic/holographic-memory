# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


PHASES = ('before', 'churn', 'after')
FIELDS = ('elapsed_ns', 'peak_rss_kib', 'written_bytes', 'cgroup_peak',
          'isolated_cgroup_peak')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_runs():
    return {
        f'{rows}-250000-{round_number}-{arm}-640': (rows, round_number, arm)
        for rows in (1_000_000, 3_000_000)
        for round_number in (1, 2, 3)
        for arm in ('global', 'segments')
    }


def validate_records(records, rows, arm):
    events = ['start', 'query', 'mutated', 'query', 'consolidated', 'query', 'complete']
    require([record['event'] for record in records] == events, 'event sequence differs')
    start = records[0]
    capacity = rows if arm == 'global' else 250_000
    ef = 256 if arm == 'global' else 128
    require(start['action'] == 'run' and start['rows'] == rows
            and start['capacity'] == capacity and start['threads'] == 8,
            'frozen layout or thread count differs')
    require(records[-1].get('reopen_results_equal') is True, 'reopen check missing')
    for phase, record in zip(PHASES, (records[1], records[3], records[5])):
        require(record['phase'] == phase and record['ef'] == ef
                and record['queries'] == 128, 'frozen query setting differs')
        require(len(record['batch_ns']) == 5
                and all(type(value) is int and value > 0 for value in record['batch_ns']),
                'five positive integer timing samples required')
    require(records[2]['elapsed_ns'] > 0, 'mutation timing missing')
    consolidation = records[4]
    require(consolidation.get('cgroup_peak_reset') is True, 'isolated cgroup reset missing')
    for field in FIELDS:
        value = consolidation.get(field)
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                f'positive finite {field} required')
    require(len(consolidation['dirty']) == (1 if arm == 'global' else 2),
            'changed segment count differs')
    require(consolidation['how'] == ['Merged'] * len(consolidation['dirty']),
            'expected merged consolidation outcomes')


def validate_receipt(log):
    tests = [line for line in log.splitlines() if line.startswith('test result:')]
    require(len(tests) == 1 and re.fullmatch(
        r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; '
        r'\d+ filtered out; finished in .+', tests[0]) is not None,
        'successful Rust test receipt missing')
    exits = re.findall(r'^\s*Exit status: (\d+)\s*$', log, re.MULTILINE)
    require(exits == ['0'], 'successful time receipt missing')


def validate_arm(evidence, name):
    require(name in expected_runs(), 'unknown confirmation arm')
    rows, _, arm = expected_runs()[name]
    log = evidence / f'{name}.log'
    text = log.read_text()
    validate_receipt(text)
    records = [json.loads(line.split('BOUNDED ', 1)[1])
               for line in text.splitlines() if 'BOUNDED ' in line]
    validate_records(records, rows, arm)
    preflight_path = evidence / f'{name}.preflight.json'
    preflight = json.loads(preflight_path.read_text())
    require(preflight['available_memory_kib'] >= 10 * 1024 * 1024,
            '10 GiB available-memory guard missing')
    ef = 256 if arm == 'global' else 128
    paths = sorted((evidence / name).iterdir())
    require({path.name for path in paths} == {f'{phase}-ef{ef}.u32' for phase in PHASES},
            'raw answer file set differs')
    require(all(path.is_file() and path.stat().st_size == 128 * 10 * 4 for path in paths),
            'raw answer file sizes differ')
    return {str(path.relative_to(evidence)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [log, preflight_path, *paths]}


def assess(result):
    expected = expected_runs()
    require(len(result['runs']) == len(expected)
            and {run['name'] for run in result['runs']} == set(expected),
            'exactly twelve frozen confirmation arms required')
    require(len(result['pairs']) == 6, 'exactly six confirmation pairs required')
    fingerprints = {}
    by_name = {run['name']: run for run in result['runs']}
    for run in result['runs']:
        rows, _, arm = expected[run['name']]
        require(run['rows'] == rows and run['start'] == 640
                and run['capacity'] == (rows if arm == 'global' else 250_000),
                'scored run identity differs')
        require(len(run['queries']) == 3
                and {query['phase'] for query in run['queries']} == set(PHASES),
                'three query phases required')
        for query in run['queries']:
            key = (rows, arm, query['phase'])
            fingerprints.setdefault(key, set()).add((query['fingerprint'], query['hits']))
    require(all(len(values) == 1 for values in fingerprints.values()),
            'repeat fingerprints or recall counts differ')
    sizes = []
    for rows in (1_000_000, 3_000_000):
        pairs = [pair for pair in result['pairs'] if by_name[pair['base']]['rows'] == rows]
        require(len(pairs) == 3, 'three pairs per size required')
        queries = []
        for phase in PHASES:
            comparisons = [query for pair in pairs for query in pair['query_ratios']
                           if query['phase'] == phase]
            require(len(comparisons) == 3
                    and all(query['candidate_ef'] == 128 and query['base_ef'] == 256
                            for query in comparisons), 'paired query setting differs')
            control = fingerprints[(rows, 'global', phase)]
            candidate = fingerprints[(rows, 'segments', phase)]
            ratios = [query['time_ratio'] for query in comparisons]
            queries.append({'phase': phase, 'control_hits': next(iter(control))[1],
                            'candidate_hits': next(iter(candidate))[1], 'denominator': 1280,
                            'recall_deltas': [query['recall_delta'] for query in comparisons],
                            'time_ratios': ratios, 'median_time_ratio': statistics.median(ratios)})
        mutation_ratios = [by_name[pair['candidate']]['mutation']['elapsed_ns']
                           / by_name[pair['base']]['mutation']['elapsed_ns'] for pair in pairs]
        sizes.append({
            'rows': rows, 'queries': queries,
            'consolidation_median_ratios': {
                field: statistics.median(pair['consolidation_ratios'][field] for pair in pairs)
                for field in FIELDS},
            'mutation_time_ratios': mutation_ratios,
            'median_mutation_time_ratio': statistics.median(mutation_ratios),
            'every_phase_meets_control_recall': all(query['candidate_hits'] >= query['control_hits']
                                                   for query in queries),
            'every_phase_query_time_improves': all(query['median_time_ratio'] < 1
                                                  for query in queries),
        })
    return {'frozen_capacity': 250_000, 'candidate_ef': 128, 'control_ef': 256,
            'query_ids_half_open': [640, 768], 'runs': 12, 'pairs': 6,
            'checks': {'liveness': True, 'duplicates': True, 'reopen': True,
                       'repeat_fingerprints': True, 'complete_frozen_matrix': True},
            'production_coordinator_implemented': False, 'sizes': sizes}


def confirmation(evidence, truth):
    from score import score

    expected = expected_runs()
    actual = {path.name for path in evidence.iterdir() if path.is_dir()}
    require(actual == set(expected), 'confirmation directory contains missing or extra arms')
    hashes = {}
    for name in expected:
        hashes.update(validate_arm(evidence, name))
    result = score(evidence, truth)
    result['confirmation'] = assess(result)
    result['evidence_sha256'] = hashes
    result['truth_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(truth.glob('*-640-*.npz'))}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--truth', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm')
    args = parser.parse_args()
    if args.arm:
        result = {'arm': args.arm, 'evidence_sha256': validate_arm(args.evidence, args.arm)}
    else:
        if args.truth is None:
            parser.error('--truth is required without --arm')
        result = confirmation(args.evidence, args.truth)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result.get('confirmation', result), indent=2))
