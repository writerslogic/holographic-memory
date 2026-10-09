# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Independent raw-source verification of frozen operand packets and controls."""
import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from check_evidence_budget import ci, encoded, optimize, same, sha


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def percentile95(values):
    values = sorted(values)
    index = (len(values) - 1) * 95 / 100
    lo = int(index)
    return values[lo] * (1 - (index - lo)) + values[min(lo + 1, len(values) - 1)] * (index - lo)


def stable(report):
    return [(r['qid'], r['arms']['operand'], r['operand']['packet'], r['operand']['required'], r['operand']['witnessed'], r['operand']['request_sha256']) for r in report['rows']]


def check_witness(key, sources, required):
    if key.startswith('date:'):
        assert key == 'date:' + sources[0][1] + ':' + sources[0][2]
    elif ':session:' in key:
        field, session = key.rsplit(':session:', 1)
        if field.startswith('observation-date:'):
            date = field.split(':', 1)[1]
            assert date in required['date_literals']
            assert any(source[1] == session and source[2][:10].replace('/', '-') == date for source in sources), 'unobserved requested date'
            return
        assert session == sources[0][1]
        if field.startswith('identity:'):
            identity = field[len('identity:'):]
            assert identity and re.search(r'(?<![^\W_])' + re.escape(identity) + r'(?![^\W_])', sources[0][4]), 'unobserved literal identity'
        elif field.startswith('quotation:'):
            literal = field[len('quotation:'):]
            assert literal and literal in required['literal_quotes']
            assert literal in sources[0][4], 'unobserved literal quotation'
        elif field.startswith('literal-date:'):
            date = field.split(':', 1)[1]
            assert date in required['date_literals']
            dates = {d.replace('/', '-') for d in re.findall(r'(?<!\w)\d{4}[/-]\d{2}[/-]\d{2}(?!\w)', sources[0][4])}
            assert date in dates, 'unobserved requested date'
        elif field.startswith('quantity:') or field.startswith('time:'):
            for value in field.split(':', 1)[1].split('/'):
                assert value in sources[0][4].lower(), 'unobserved source value'
        else:
            assert field in {'premise', 'boundary', 'confirmation', 'companion', 'transport', 'anxiety'}, 'unknown operand witness'
    elif ':value:' in key:
        assert key.rsplit(':value:', 1)[1] in sources[0][4].lower()
    else:
        raise AssertionError('unknown operand witness')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('--reproduction', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('benchmarks/results/evidence_budget_summary.json'))
    p.add_argument('--binary', type=Path)
    p.add_argument('--kernel', type=Path, default=Path('src/core/operand_retriever.rs'))
    p.add_argument('--harness', type=Path, default=Path('src/bin/operand-retriever.rs'))
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    repeat = json.loads(args.reproduction.read_text())
    work = Path('target/conversation-geometry-validation')
    source_file = Path.home() / '.cache/hms-bench/downloads/longmemeval_s_cleaned.json'
    input_file = Path('benchmarks/results/evidence_completeness_dev.json')
    validated = Path('benchmarks/results/evidence_completeness_summary.json')
    split_file = Path('benchmarks/public/longmemeval_split.json')
    protocol = Path('docs/research/PROTOCOL-2026-10-08-OPERAND.md')
    baseline_file = Path('benchmarks/results/evidence_budget_baseline_dev.json')
    assert report['baseline_report_sha256'] == sha(baseline_file) == json.loads(Path('benchmarks/results/evidence_budget_baseline_summary.json').read_text())['report_sha256']
    assert report['code_sha256'] == sha(Path(__file__).with_name('operand_experiment.py'))
    if args.binary is not None:
        assert report['binary_sha256'] == sha(args.binary), 'binary fingerprint mismatch'
    assert report['kernel_sha256'] == sha(args.kernel) and report['harness_sha256'] == sha(args.harness)
    assert report['protocol_sha256'] == sha(protocol)
    assert report['input_sha256'] == sha(input_file) == json.loads(validated.read_text())['report_sha256']
    assert report['validated_input_sha256'] == sha(validated)
    assert report['source_sha256'] == sha(source_file) == 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
    assert report['split_sha256'] == sha(split_file)
    frozen = json.loads(split_file.read_text())
    assert len(frozen['dev']) == 100 and len(frozen['heldout']) == 400
    assert len(report['rows']) == 100 and len({r['qid'] for r in report['rows']}) == 100
    assert {r['qid'] for r in report['rows']} == set(frozen['dev'])
    assert not {r['qid'] for r in report['rows']} & set(frozen['heldout'])
    gates = ['input fingerprints and untouched held-out split']
    original = json.loads(input_file.read_text())
    baseline = json.loads(baseline_file.read_text())
    raw = {q['question_id']: q for q in json.loads(source_file.read_text()) if q['question_id'] in set(frozen['dev'])}
    source_instances = 0
    for row, previous, old in zip(report['rows'], original['rows'], baseline['rows']):
        assert row['qid'] == previous['qid'] == old['qid']
        source = raw[row['qid']]
        groups_by_id, ordinals = {}, {}
        for sid, date, turns in zip(source['haystack_session_ids'], source['haystack_dates'], source['haystack_sessions']):
            observations = []
            for i, turn in enumerate(turns):
                if turn['role'] != 'user':
                    continue
                tid = f'{sid}_{i + 1}'
                if 'answer' in sid and not turn.get('has_answer'):
                    tid = tid.replace('answer', 'noans')
                observations.append((tid, i + 1, turn['content']))
            observed_sid = sid.replace('answer', 'noans') if 'answer' in sid and not any('answer' in t for t, _, _ in observations) else sid
            for tid, ordinal, text in observations:
                groups_by_id.setdefault(tid, []).append([hashlib.sha256(tid.encode()).hexdigest(), hashlib.sha256(observed_sid.encode()).hexdigest(), date, 'user', text])
                ordinals[tid] = ordinal
                source_instances += 1
        assert set(groups_by_id) == set(previous['ids'])
        assert row['qtype'] == source['question_type']
        gold = {tid for tid in previous['ids'] if 'answer' in tid}
        assert row['scored'] == ('_abs' not in row['qid'] and bool(gold)) == previous['scored']
        base = previous['baseline_ranking']
        candidates = list(dict.fromkeys(base))[:20]
        assert row['candidates'] == candidates
        groups = [groups_by_id[t] for t in candidates]
        costs = [len(canonical(g).encode()) + 1 for g in groups]
        ranks = [base.index(t) for t in candidates]
        assert row['costs'] == costs and row['ranks'] == ranks
        first = tuple(candidates.index(t) for t in dict.fromkeys(base[:5]))
        budget = len(encoded(groups, first))
        assert row['budget'] == old['budget'] == budget
        limit = budget - len(b'{"v":1,"evidence":[]}') + 1
        assert row['gold_count'] == len(gold)
        same(row['top_k_complete'], {str(k): gold <= set(base[:k]) for k in (5, 10, 20, 64)})
        needed = sum(costs[i] for i, t in enumerate(candidates) if t in gold)
        same(row['oracle'], {'five_item_ceiling': len(gold) <= 5, 'unanchored_source_budget': gold <= set(candidates) and needed <= limit,
                             'anchored_source_budget': gold <= set(candidates) and needed + (0 if candidates[0] in gold else costs[0]) <= limit})
        selections = {'first_five': first, 'knapsack': optimize(costs, ranks, limit)}
        for name, order in (('rank_skip', range(len(costs))), ('shortest', sorted(range(len(costs)), key=lambda i: (costs[i], i)))):
            used, chosen = costs[0], [0]
            for i in order:
                if i != 0 and used + costs[i] <= limit:
                    used += costs[i]
                    chosen.append(i)
            selections[name] = tuple(sorted(chosen))
        for name, chosen in selections.items():
            blob = encoded(groups, chosen)
            ids = [candidates[i] for i in chosen]
            same(row['arms'][name], {'selected': ids, 'packet_bytes': len(blob), 'complete': gold <= set(ids), 'packet_sha256': hashlib.sha256(blob).hexdigest()})
        arm = row['arms']['operand']
        ids = arm['selected']
        assert len(ids) == len(set(ids)) and set(ids) <= set(candidates), 'unobserved source membership'
        chosen = [candidates.index(t) for t in ids]
        assert chosen == sorted(chosen)
        expected_packet = canonical({'v': 2, 'q': [groups[i] for i in chosen], 'i': []})
        blob = row['operand']['packet'].encode()
        assert blob == expected_packet.encode(), 'partial or altered source quotation'
        assert len(blob) == arm['packet_bytes'] <= budget, 'invalid packet byte budget'
        assert hashlib.sha256(blob).hexdigest() == arm['packet_sha256'], 'corrupted packet hash'
        assert arm['complete'] == (gold <= set(ids)), 'full semantic source support mismatch'
        request = {'question': source['question'], 'budget': budget, 'candidates': [
            {'observations': groups[i], 'ordinal': ordinals[t], 'rank': ranks[i]} for i, t in enumerate(candidates)]}
        assert hashlib.sha256(canonical(request).encode()).hexdigest() == row['operand']['request_sha256'], 'label-free source request mismatch'
        for key, indices in row['operand']['witnessed'].items():
            assert indices and len(indices) == len(set(indices)) and set(indices) <= set(chosen)
            for i in indices:
                check_witness(key, groups[i], row['operand']['required'])
        for key in ['operand_ns', 'control_ns']:
            samples = row['operand'][key]
            assert len(samples) == 11 and all(isinstance(n, int) and n > 0 for n in samples)
    gates += ['observed full source turns, dates and alias membership', 'exact packet hashes and original byte budgets']
    scored = [r for r in report['rows'] if r['scored']]
    assert len(scored) == 84
    expected = {}
    for kind in ['overall'] + sorted({r['qtype'] for r in scored}):
        rr = [r for r in scored if kind == 'overall' or r['qtype'] == kind]
        table = {'n': len(rr), 'gold_count_distribution': {str(k): v for k, v in sorted(Counter(r['gold_count'] for r in rr).items())},
                 'top_k_complete': {str(k): statistics.mean(r['top_k_complete'][str(k)] for r in rr) for k in (5, 10, 20, 64)},
                 'oracle': {k: statistics.mean(r['oracle'][k] for r in rr) for k in rr[0]['oracle']}, 'arms': {}}
        for name in rr[0]['arms']:
            table['arms'][name] = {'complete': statistics.mean(r['arms'][name]['complete'] for r in rr),
                                  'paired_vs_first_five': ci([int(r['arms'][name]['complete']) - int(r['arms']['first_five']['complete']) for r in rr]),
                                  'packet_bytes_mean': statistics.mean(r['arms'][name]['packet_bytes'] for r in rr),
                                  'selected_ids_mean': statistics.mean(len(r['arms'][name]['selected']) for r in rr)}
        expected[kind] = table
    same(report['summary'], expected)
    assert sum(r['arms']['first_five']['complete'] for r in scored) == 68
    assert sum(r['arms']['knapsack']['complete'] for r in scored) == 68
    assert sum(r['oracle']['unanchored_source_budget'] for r in scored) == 79
    gates += ['controls, gold oracles and complete semantic support']
    assert stable(report) == stable(repeat), 'reproduction mismatch'
    for row, other in zip(report['rows'], repeat['rows']):
        assert {k: v for k, v in row.items() if k != 'operand'} == {k: v for k, v in other.items() if k != 'operand'}, 'reproduction source accounting mismatch'
        for key in ['operand_ns', 'control_ns']:
            assert len(other['operand'][key]) == 11 and all(isinstance(n, int) and n > 0 for n in other['operand'][key])
    for key in ['attempt', 'kernel_sha256', 'harness_sha256', 'binary_sha256', 'protocol_sha256', 'input_sha256', 'source_sha256', 'split_sha256', 'summary']:
        assert report[key] == repeat[key]
    count = sum(r['arms']['operand']['complete'] for r in scored)
    operand_p95 = percentile95([statistics.median(r['operand']['operand_ns']) for r in scored])
    control_p95 = percentile95([statistics.median(r['operand']['control_ns']) for r in scored])
    decision = {'complete': count, 'n': 84, 'quality_75': count >= 75, 'quality_79': count >= 79, 'equal_budget': True,
                'operand_p95_ns': operand_p95, 'control_p95_ns': control_p95, 'no_worse_p95': operand_p95 <= control_p95,
                'success_75': count >= 75 and operand_p95 <= control_p95, 'success_79': count >= 79 and operand_p95 <= control_p95}
    same(report['decision'], decision)
    other_scored = [r for r in repeat['rows'] if r['scored']]
    repeated_operand = percentile95([statistics.median(r['operand']['operand_ns']) for r in other_scored])
    repeated_control = percentile95([statistics.median(r['operand']['control_ns']) for r in other_scored])
    repeat_decision = dict(decision, operand_p95_ns=repeated_operand, control_p95_ns=repeated_control, no_worse_p95=repeated_operand <= repeated_control,
                           success_75=count >= 75 and repeated_operand <= repeated_control, success_79=count >= 79 and repeated_operand <= repeated_control)
    same(repeat['decision'], repeat_decision)
    assert not report['overall_superiority_established']
    gates += ['reproduced packets, aggregates and recorded success decisions']
    assert len(gates) == 5
    summary = {'report_sha256': sha(args.report), 'reproduction_sha256': sha(args.reproduction), 'validator_sha256': sha(__file__),
               'protocol_sha256': sha(protocol), 'source_instances_verified': source_instances, 'scored_dev_questions': 84,
               'heldout_questions_untouched': 400, 'validation_gates_passed': gates, 'decision': decision, 'reproduction_decision': repeat_decision, 'summary': expected,
               'overall_superiority_established': False}
    args.output.write_text(json.dumps(summary, indent=1, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'summary'}))
    print('5 validation gates passed')


if __name__ == '__main__':
    main()
