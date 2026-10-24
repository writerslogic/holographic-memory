# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def score(root, truth):
    runs = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not directory.name[0].isdigit():
            continue
        log = root / (directory.name + '.log')
        records = [json.loads(line.split('BOUNDED ', 1)[1]) for line in log.read_text().splitlines() if 'BOUNDED ' in line]
        start = next(row for row in records if row['event'] == 'start')
        assert records[-1]['event'] == 'complete', directory
        n = start['rows']
        split = int(directory.name.split('-')[-1])
        result = {'name': directory.name, 'rows': n, 'capacity': start['capacity'], 'start': split,
                  'consolidation': next(row for row in records if row['event'] == 'consolidated'),
                  'mutation': next(row for row in records if row['event'] == 'mutated'), 'queries': []}
        for row in records:
            if row['event'] != 'query':
                continue
            phase = row['phase']
            expected_phase = 'before' if phase == 'before' else 'churn'
            with np.load(truth / f'{n}-{split}-{expected_phase}.npz') as saved:
                ids = saved['ids']
                source = saved['source']
            found = np.fromfile(directory / f'{phase}-ef{row["ef"]}.u32', dtype='<u4').reshape(128, 10)
            assert (found < len(source)).all() and (source[found] >= 0).all()
            assert all(len(set(query)) == 10 for query in found)
            hits = sum(len(set(a) & set(b)) for a, b in zip(found, ids))
            result['queries'].append({**row, 'hits': hits, 'recall': hits / 1280,
                                      'median_batch_ns': statistics.median(row['batch_ns']),
                                      'fingerprint': hashlib.sha256(found.tobytes()).hexdigest()})
        runs.append(result)
    groups = {}
    for run in runs:
        for query in run['queries']:
            key = (run['rows'], run['capacity'], run['start'], query['phase'], query['ef'])
            groups.setdefault(key, []).append(query)
    summary = [{'rows': k[0], 'capacity': k[1], 'start': k[2], 'phase': k[3], 'ef': k[4],
                'rounds': len(v), 'recalls': [r['recall'] for r in v],
                'median_ms_128_queries': statistics.median(r['median_batch_ns'] for r in v) / 1e6}
               for k, v in sorted(groups.items())]
    by_name = {run['name']: run for run in runs}
    pairs = []
    for candidate in runs:
        if '-segments-' not in candidate['name']:
            continue
        base = by_name[candidate['name'].replace('-segments-', '-global-')]
        before = base['consolidation']
        after = candidate['consolidation']
        consolidation = {field: after[field] / before[field] for field in ('elapsed_ns', 'peak_rss_kib', 'written_bytes', 'cgroup_peak')}
        if before['isolated_cgroup_peak'] and after['isolated_cgroup_peak']:
            consolidation['isolated_cgroup_peak'] = after['isolated_cgroup_peak'] / before['isolated_cgroup_peak']
        query_ratios = []
        for q in candidate['queries']:
            for parent in base['queries']:
                if parent['phase'] == q['phase']:
                    query_ratios.append({'phase': q['phase'], 'candidate_ef': q['ef'], 'base_ef': parent['ef'],
                                         'time_ratio': q['median_batch_ns'] / parent['median_batch_ns'],
                                         'recall_delta': q['recall'] - parent['recall']})
        pairs.append({'candidate': candidate['name'], 'base': base['name'], 'consolidation_ratios': consolidation, 'query_ratios': query_ratios})
    return {'measurement': 'paired development only', 'runs': runs, 'summary': summary, 'pairs': pairs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = score(args.evidence, args.truth)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))
