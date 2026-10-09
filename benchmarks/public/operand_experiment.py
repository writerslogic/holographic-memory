# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Frozen dev-only source operand retrieval; no models or network calls."""
import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from evidence_budget_diagnostic import aggregate

WORK = Path('target/conversation-geometry-validation')
PROTOCOL = Path('docs/research/PROTOCOL-2026-10-08-OPERAND.md')
INPUT = Path('benchmarks/results/evidence_completeness_dev.json')
VALIDATED = Path('benchmarks/results/evidence_completeness_summary.json')
SOURCE = Path.home() / '.cache/hms-bench/downloads/longmemeval_s_cleaned.json'
SPLIT = Path('benchmarks/public/longmemeval_split.json')
KERNEL = Path('src/core/operand_retriever.rs')
HARNESS = Path('src/bin/operand-retriever.rs')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def p95(values):
    ordered = sorted(values)
    at = .95 * (len(ordered) - 1)
    low = int(at)
    return ordered[low] + (at - low) * (ordered[min(low + 1, len(ordered) - 1)] - ordered[low])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/evidence_budget_dev.json'))
    parser.add_argument('--attempt', choices=['predicate', 'premise'], default='predicate')
    args = parser.parse_args()
    assert PROTOCOL.exists(), 'protocol must exist before measurement'
    baseline_path = Path('benchmarks/results/evidence_budget_baseline_dev.json')
    original = json.loads(INPUT.read_text())
    baseline = json.loads(baseline_path.read_text())
    assert sha(INPUT) == json.loads(VALIDATED.read_text())['report_sha256']
    assert sha(baseline_path) == json.loads(Path('benchmarks/results/evidence_budget_baseline_summary.json').read_text())['report_sha256']
    frozen = json.loads(SPLIT.read_text())
    dev = set(frozen['dev'])
    assert len(dev) == 100 and len(frozen['heldout']) == 400
    assert sha(SOURCE) == 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442'
    raw = {q['question_id']: q for q in json.loads(SOURCE.read_text()) if q['question_id'] in dev}
    requests = []
    for row, previous in zip(baseline['rows'], original['rows']):
        q = raw[row['qid']]
        groups, ordinals = {}, {}
        for sid, date, turns in zip(q['haystack_session_ids'], q['haystack_dates'], q['haystack_sessions']):
            users = [(i + 1, t) for i, t in enumerate(turns) if t['role'] == 'user']
            observed_sid = sid.replace('answer', 'noans') if 'answer' in sid and not any(t.get('has_answer') for _, t in users) else sid
            for ordinal, t in users:
                tid = f'{sid}_{ordinal}'
                if 'answer' in sid and not t.get('has_answer'):
                    tid = tid.replace('answer', 'noans')
                groups.setdefault(tid, []).append([hashlib.sha256(tid.encode()).hexdigest(), hashlib.sha256(observed_sid.encode()).hexdigest(), date, 'user', t['content']])
                ordinals[tid] = ordinal
        candidates = list(dict.fromkeys(previous['baseline_ranking']))[:20]
        assert candidates == row['candidates']
        requests.append({'question': q['question'], 'budget': row['budget'], 'candidates': [
            {'observations': groups[t], 'ordinal': ordinals[t], 'rank': previous['baseline_ranking'].index(t)} for t in candidates]})
    metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--offline', '--no-deps', '--format-version=1']))
    executable = Path(metadata['target_directory']) / 'release/operand-retriever'
    proc = subprocess.run([str(executable)], input='\n'.join(canonical(q) for q in requests) + '\n', text=True, capture_output=True, check=True)
    outputs = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(outputs) == 100
    report = json.loads(canonical(baseline))
    for row, previous, request, output in zip(report['rows'], original['rows'], requests, outputs):
        result = output['retrieval']
        selected = result['selected']
        selected_ids = [row['candidates'][i] for i in selected]
        packet = result['packet'].encode()
        gold = {tid for tid in previous['ids'] if 'answer' in tid}
        assert [row['candidates'][i] for i in output['control']] == row['arms']['knapsack']['selected']
        assert len(packet) <= row['budget']
        assert json.loads(packet)['q'] == [request['candidates'][i]['observations'] for i in selected]
        row['arms']['operand'] = {'selected': selected_ids, 'packet_bytes': len(packet), 'packet_sha256': hashlib.sha256(packet).hexdigest(), 'complete': gold <= set(selected_ids)}
        row['operand'] = {'packet': result['packet'], 'required': result['required'], 'witnessed': result['witnessed'],
                          'operand_ns': output['operand_ns'], 'control_ns': output['control_ns'],
                          'index_ns': output['index_ns'], 'request_sha256': hashlib.sha256(canonical(request).encode()).hexdigest()}
    scored = [r for r in report['rows'] if r['scored']]
    count = sum(r['arms']['operand']['complete'] for r in scored)
    operand_p95 = p95([statistics.median(r['operand']['operand_ns']) for r in scored])
    control_p95 = p95([statistics.median(r['operand']['control_ns']) for r in scored])
    report.update({'protocol': 'PROTOCOL-2026-10-08-OPERAND.md', 'attempt': args.attempt, 'code_sha256': sha(__file__),
                   'kernel_sha256': sha(KERNEL), 'harness_sha256': sha(HARNESS), 'binary_sha256': sha(executable),
                   'protocol_sha256': sha(PROTOCOL), 'baseline_report_sha256': sha(baseline_path),
                   'input_sha256': sha(INPUT), 'validated_input_sha256': sha(VALIDATED), 'source_sha256': sha(SOURCE),
                   'split_sha256': sha(SPLIT), 'summary': aggregate(report['rows']),
                   'decision': {'complete': count, 'n': len(scored), 'quality_75': count >= 75, 'quality_79': count >= 79,
                                'equal_budget': True, 'operand_p95_ns': operand_p95, 'control_p95_ns': control_p95,
                                'no_worse_p95': operand_p95 <= control_p95,
                                'success_75': count >= 75 and operand_p95 <= control_p95,
                                'success_79': count >= 79 and operand_p95 <= control_p95},
                   'overall_superiority_established': False})
    source_archive = WORK / 'operand-sources'
    source_archive.mkdir(exist_ok=True)
    for source_path in [Path(__file__), KERNEL, HARNESS, PROTOCOL]:
        (source_archive / (sha(source_path) + source_path.suffix)).write_bytes(source_path.read_bytes())
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    print(json.dumps(report['decision']))
    for row in scored:
        if row['arms']['operand']['complete'] != row['arms']['knapsack']['complete']:
            print(row['qid'], 'gain' if row['arms']['operand']['complete'] else 'loss', row['arms']['operand']['selected'])


if __name__ == '__main__':
    main()
