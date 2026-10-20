# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Independent raw-source and meet-in-the-middle check of section 16."""

import argparse
import bisect
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def encoded(groups, selected):
    return json.dumps({'v': 1, 'evidence': [groups[i] for i in selected]}, ensure_ascii=False,
                      separators=(',', ':')).encode()


def optimize(costs, ranks, limit):
    denominator = math.lcm(*(r + 1 for r in ranks))
    values = [denominator // (r + 1) for r in ranks]
    available = limit - costs[0]
    middle = 1 + (len(costs) - 1) // 2
    def enumerate_half(indices):
        states = [(0, 0, ())]
        for i in indices:
            states += [(used + costs[i], value + values[i], chosen + (i,))
                       for used, value, chosen in states.copy() if used + costs[i] <= available]
        return states
    left = enumerate_half(range(1, middle))
    right = sorted(enumerate_half(range(middle, len(costs))))
    best, prefix = None, []
    for state in right:
        if best is None or (-state[1], state[0], state[2]) < (-best[1], best[0], best[2]):
            best = state
        prefix.append(best)
    right_costs = [state[0] for state in right]
    choices = []
    for cost, value, chosen in left:
        other = prefix[bisect.bisect_right(right_costs, available - cost) - 1]
        choices.append((-(value + other[1]), cost + other[0], (0,) + chosen + other[2]))
    return min(choices)[2]


def ci(values):
    rng = random.Random(20261008)
    draws = []
    for _ in range(2000):
        draws.append(sum(rng.choices(values, k=len(values))) / len(values))
    draws.sort()
    bounds = []
    for fraction in (.025, .975):
        at = fraction * 1999
        low = int(at)
        bounds.append(draws[low] * (1 - (at - low)) + draws[min(low + 1, 1999)] * (at - low))
    return {'mean': sum(values) / len(values), 'ci95': bounds}


def same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for k in expected:
            same(actual[k], expected[k])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            same(a, b)
    elif isinstance(expected, (int, float)):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('--input', type=Path, default=Path('benchmarks/results/evidence_completeness_dev.json'))
    p.add_argument('--validated-input', type=Path, default=Path('benchmarks/results/evidence_completeness_summary.json'))
    p.add_argument('--source', type=Path, default=Path.home() / '.cache/hms-bench/downloads/longmemeval_s_cleaned.json')
    p.add_argument('--output', type=Path, default=Path('benchmarks/results/evidence_budget_summary.json'))
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    assert report['code_sha256'] == sha(Path(__file__).with_name('evidence_budget_diagnostic.py'))
    assert report['lock_sha256'] == sha(Path(__file__).with_name('evidence_budget_diagnostic.py.lock'))
    original = json.loads(args.input.read_text())
    assert report['input_sha256'] == sha(args.input) == json.loads(args.validated_input.read_text())['report_sha256']
    assert report['validated_input_sha256'] == sha(args.validated_input)
    assert report['source_manifest_sha256'] == original['source_manifest_sha256']
    assert report['source_sha256'] == sha(args.source) == 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
    frozen_file = Path(__file__).with_name('longmemeval_split.json')
    frozen = json.loads(frozen_file.read_text())
    assert report['split_sha256'] == sha(frozen_file)
    assert {r['qid'] for r in report['rows']} == set(frozen['dev'])
    assert not {r['qid'] for r in report['rows']} & set(frozen['heldout'])
    raw = {r['question_id']: r for r in json.loads(args.source.read_text()) if r['question_id'] in frozen['dev']}
    source_instances = 0
    for row, previous in zip(report['rows'], original['rows']):
        assert row['qid'] == previous['qid']
        source = raw[row['qid']]
        groups_by_id = {}
        for sid, date, turns in zip(source['haystack_session_ids'], source['haystack_dates'], source['haystack_sessions']):
            observations = []
            for i, turn in enumerate(turns):
                if turn['role'] != 'user':
                    continue
                tid = f'{sid}_{i + 1}'
                if 'answer' in sid and not turn.get('has_answer'):
                    tid = tid.replace('answer', 'noans')
                observations.append((tid, turn['content']))
            observed_sid = sid.replace('answer', 'noans') if 'answer' in sid and not any('answer' in t for t, _ in observations) else sid
            for tid, text in observations:
                groups_by_id.setdefault(tid, []).append([
                    hashlib.sha256(tid.encode()).hexdigest(), hashlib.sha256(observed_sid.encode()).hexdigest(), date, 'user', text])
                source_instances += 1
        assert set(groups_by_id) == set(previous['ids'])
        assert row['qtype'] == source['question_type'] and row['scored'] == previous['scored']
        base = previous['baseline_ranking']
        candidates = list(dict.fromkeys(base))[:20]
        assert row['candidates'] == candidates
        groups = [groups_by_id[t] for t in candidates]
        costs = [len(json.dumps(g, ensure_ascii=False, separators=(',', ':')).encode()) + 1 for g in groups]
        ranks = [base.index(t) for t in candidates]
        assert row['costs'] == costs and row['ranks'] == ranks
        first = tuple(candidates.index(t) for t in dict.fromkeys(base[:5]))
        budget = len(encoded(groups, first))
        assert budget == row['budget']
        limit = budget - len(b'{"v":1,"evidence":[]}') + 1
        assert limit == sum(costs[i] for i in first)
        selections = {'first_five': first}
        for name, order in (('rank_skip', range(len(costs))), ('shortest', sorted(range(len(costs)), key=lambda i: (costs[i], i)))):
            used, chosen = costs[0], [0]
            for i in order:
                if i != 0 and used + costs[i] <= limit:
                    used += costs[i]
                    chosen.append(i)
            selections[name] = tuple(sorted(chosen))
        selections['knapsack'] = optimize(costs, ranks, limit)
        gold = {t for t in previous['ids'] if 'answer' in t}
        assert row['gold_count'] == len(gold)
        same(row['top_k_complete'], {str(k): gold <= set(base[:k]) for k in (5, 10, 20, 64)})
        needed = sum(costs[i] for i, t in enumerate(candidates) if t in gold)
        same(row['oracle'], {'five_item_ceiling': len(gold) <= 5,
                             'unanchored_source_budget': gold <= set(candidates) and needed <= limit,
                             'anchored_source_budget': gold <= set(candidates) and needed + (0 if candidates[0] in gold else costs[0]) <= limit})
        for name, chosen in selections.items():
            blob = encoded(groups, chosen)
            assert len(blob) <= budget
            ids = [candidates[i] for i in chosen]
            same(row['arms'][name], {'selected': ids, 'packet_bytes': len(blob), 'complete': gold <= set(ids),
                                    'packet_sha256': hashlib.sha256(blob).hexdigest()})
    scored = [r for r in report['rows'] if r['scored']]
    expected = {}
    for kind in ['overall'] + sorted({r['qtype'] for r in scored}):
        rr = [r for r in scored if kind == 'overall' or r['qtype'] == kind]
        table = {'n': len(rr), 'gold_count_distribution': {str(k): v for k, v in sorted(Counter(r['gold_count'] for r in rr).items())},
                 'top_k_complete': {str(k): statistics.mean(r['top_k_complete'][str(k)] for r in rr) for k in (5, 10, 20, 64)},
                 'oracle': {k: statistics.mean(r['oracle'][k] for r in rr) for k in rr[0]['oracle']}, 'arms': {}}
        for name in rr[0]['arms']:
            table['arms'][name] = {
                'complete': statistics.mean(r['arms'][name]['complete'] for r in rr),
                'paired_vs_first_five': ci([int(r['arms'][name]['complete']) - int(r['arms']['first_five']['complete']) for r in rr]),
                'packet_bytes_mean': statistics.mean(r['arms'][name]['packet_bytes'] for r in rr),
                'selected_ids_mean': statistics.mean(len(r['arms'][name]['selected']) for r in rr)}
        expected[kind] = table
    same(report['summary'], expected)
    assert not report['overall_superiority_established']
    result = {'report_sha256': sha(args.report), 'validator_sha256': sha(__file__),
              'source_instances_verified': source_instances, 'scored_dev_questions': len(scored),
              'summary': expected, 'overall_superiority_established': False}
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'summary'}))
    print(json.dumps(expected['overall']))


if __name__ == '__main__':
    main()
