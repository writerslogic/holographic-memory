# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Dev-only exact source-budget diagnosis; research record section 16."""

import argparse
import hashlib
import json
import math
import random
import statistics
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

from check_evidence_completeness import source_block

PREFIX = b'{"v":1,"evidence":['
SUFFIX = b']}'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def encode(records):
    return json.dumps(records, ensure_ascii=False, separators=(',', ':')).encode()


def packet(groups, chosen):
    return PREFIX + b','.join(encode(groups[i]) for i in chosen) + SUFFIX


def knapsack(costs, ranks, limit):
    if not costs or costs[0] > limit or len(costs) != len(ranks) or len(costs) > 20:
        raise ValueError('invalid anchored knapsack')
    states = {costs[0]: (Fraction(1, ranks[0] + 1), (0,))}
    for i in range(1, len(costs)):
        expanded = dict(states)
        for used, (value, chosen) in states.items():
            used += costs[i]
            if used > limit:
                continue
            candidate = (value + Fraction(1, ranks[i] + 1), chosen + (i,))
            current = expanded.get(used)
            if current is None or candidate[0] > current[0] or (candidate[0] == current[0] and candidate[1] < current[1]):
                expanded[used] = candidate
        states = {}
        best = Fraction(-1)
        for used, result in sorted(expanded.items()):
            if result[0] > best:
                states[used] = result
                best = result[0]
    return min(states.items(), key=lambda pair: (-pair[1][0], pair[0], pair[1][1]))[1][1]


def pack(costs, limit, order):
    if costs[0] > limit:
        raise ValueError('anchor exceeds source budget')
    chosen, used = [0], costs[0]
    for i in order:
        if i != 0 and used + costs[i] <= limit:
            chosen.append(i)
            used += costs[i]
    return tuple(sorted(chosen))


def paired(values):
    rng = random.Random(20261008)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    def quantile(p):
        index = p * 1999
        low = int(index)
        return samples[low] + (index - low) * (samples[min(low + 1, 1999)] - samples[low])
    return {'mean': statistics.mean(values), 'ci95': [quantile(.025), quantile(.975)]}


def aggregate(rows):
    scored = [r for r in rows if r['scored']]
    result = {}
    for kind in ['overall'] + sorted({r['qtype'] for r in scored}):
        rr = [r for r in scored if kind == 'overall' or r['qtype'] == kind]
        table = {'n': len(rr), 'gold_count_distribution': dict(sorted(Counter(r['gold_count'] for r in rr).items())),
                 'top_k_complete': {str(k): statistics.mean(r['top_k_complete'][str(k)] for r in rr) for k in (5, 10, 20, 64)},
                 'oracle': {key: statistics.mean(r['oracle'][key] for r in rr) for key in rr[0]['oracle']}, 'arms': {}}
        for name in rr[0]['arms']:
            table['arms'][name] = {
                'complete': statistics.mean(r['arms'][name]['complete'] for r in rr),
                'paired_vs_first_five': paired([int(r['arms'][name]['complete']) - int(r['arms']['first_five']['complete']) for r in rr]),
                'packet_bytes_mean': statistics.mean(r['arms'][name]['packet_bytes'] for r in rr),
                'selected_ids_mean': statistics.mean(len(r['arms'][name]['selected']) for r in rr)}
        result[kind] = table
    return result


def self_test():
    groups = [[['turn', 'session', 'date', 'user', 'literal α']],
              [['alias', 'session', 'date1', 'user', 'same'], ['alias', 'session', 'date2', 'user', 'same']]]
    assert json.loads(packet(groups, [0, 1]))['evidence'] == groups
    costs = [len(encode(g)) + 1 for g in groups]
    for limit in (sum(costs) - 1, sum(costs), sum(costs) + 1):
        expected = (0,) if limit < sum(costs) else (0, 1)
        assert pack(costs, limit, [0, 1]) == expected
        assert knapsack(costs, [0, 1], limit) == expected
    assert knapsack([1, 6, 3, 3], [0, 1, 2, 3], 7) == (0, 2, 3)
    for costs, limit in (([2], 1), ([], 1), ([1] * 21, 21)):
        try:
            knapsack(costs, list(range(len(costs))), limit)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid resource bound accepted')
    print('budget-1/budget/budget+1, exact packets, alias dates and anchored packing passed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('benchmarks/results/evidence_completeness_dev.json'))
    p.add_argument('--validated-input', type=Path, default=Path('benchmarks/results/evidence_completeness_summary.json'))
    p.add_argument('--work', type=Path, default=Path('target/conversation-geometry-validation/models'))
    p.add_argument('--output', type=Path, default=Path('benchmarks/results/evidence_budget_dev.json'))
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    original = json.loads(args.input.read_text())
    manifest = json.loads((args.work / 'manifest.json').read_text())
    assert digest(args.input) == json.loads(args.validated_input.read_text())['report_sha256']
    assert digest(args.work / 'manifest.json') == original['source_manifest_sha256']
    rows = []
    for qi, row in enumerate(original['rows']):
        groups_by_id = {}
        entry = manifest['questions'][qi]
        assert row['qid'] == entry['qid'] and Counter(row['ids']) == Counter(row['baseline_ranking'])
        with zipfile.ZipFile(args.work / f'q{qi:03}.npz') as archive:
            for j in range(len(entry['sids'])):
                source, checksum = source_block(archive, j)
                assert checksum == entry['source_hashes'][j]
                for tid, text in source['turns']:
                    groups_by_id.setdefault(tid, []).append([
                        hashlib.sha256(tid.encode()).hexdigest(), hashlib.sha256(source['sid'].encode()).hexdigest(),
                        source['date'], 'user', text])
        base = row['baseline_ranking']
        if set(groups_by_id) != set(base):
            raise ValueError('source membership mismatch')
        candidates = list(dict.fromkeys(base))[:20]
        groups = [groups_by_id[t] for t in candidates]
        costs = [len(encode(g)) + 1 for g in groups]
        positions = [base.index(t) for t in candidates]
        first = tuple(candidates.index(t) for t in dict.fromkeys(base[:5]))
        limit = sum(costs[i] for i in first)
        budget = len(packet(groups, first))
        selections = {'first_five': first, 'rank_skip': pack(costs, limit, range(len(costs))),
                      'shortest': pack(costs, limit, sorted(range(len(costs)), key=lambda i: (costs[i], i))),
                      'knapsack': knapsack(costs, positions, limit)}
        gold = {t for t in row['ids'] if 'answer' in t}
        arms = {}
        for name, selected in selections.items():
            encoded = packet(groups, selected)
            assert len(encoded) <= budget and json.loads(encoded)['evidence'] == [groups[i] for i in selected]
            selected_ids = [candidates[i] for i in selected]
            arms[name] = {'selected': selected_ids, 'packet_bytes': len(encoded), 'complete': gold <= set(selected_ids),
                          'packet_sha256': hashlib.sha256(encoded).hexdigest()}
        all_gold = gold <= set(candidates)
        gold_cost = sum(costs[i] for i, tid in enumerate(candidates) if tid in gold)
        anchor_cost = gold_cost + (0 if candidates[0] in gold else costs[0])
        rows.append({'qid': row['qid'], 'qtype': row['qtype'], 'scored': row['scored'], 'candidates': candidates,
                     'costs': costs, 'ranks': positions, 'budget': budget, 'gold_count': len(gold),
                     'top_k_complete': {str(k): gold <= set(base[:k]) for k in (5, 10, 20, 64)},
                     'oracle': {'five_item_ceiling': len(gold) <= 5,
                                'unanchored_source_budget': all_gold and gold_cost <= limit,
                                'anchored_source_budget': all_gold and anchor_cost <= limit}, 'arms': arms})
        if (qi + 1) % 20 == 0:
            print(f'source-budget diagnosis {qi + 1}/100', flush=True)
    report = {'protocol': 'IDEAS-2026-10 section 16; exploratory dev source bytes, not answer accuracy',
              'code_sha256': digest(__file__), 'lock_sha256': digest(str(__file__) + '.lock'),
              'input_sha256': digest(args.input), 'validated_input_sha256': digest(args.validated_input),
              'source_manifest_sha256': digest(args.work / 'manifest.json'),
              'source_sha256': manifest['source_sha256'], 'split_sha256': original['split_sha256'],
              'rows': rows, 'summary': aggregate(rows), 'overall_superiority_established': False}
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    print(json.dumps(report['summary']['overall']), flush=True)


if __name__ == '__main__':
    main()
