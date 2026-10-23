# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect locomo_bench.py predictions into benchmarks/results/locomo_v1.json.

  uv run --no-project python benchmarks/public/locomo/assemble_results.py OUT.json --meta RUN/summary.json \
    --spend USD --arm primary:3 RUN1/predictions.jsonl RUN2/predictions.jsonl ...
"""

import argparse
import json
import statistics
from pathlib import Path

# Mem0 paper (arXiv 2504.19413) Tables 1-2, columns re-keyed to LoCoMo category ids: weighting the paper's
# J columns by the id counts (1: 282, 2: 321, 3: 96, 4: 841) reproduces its overall J only with
# "single-hop"=1, "temporal"=2, "multi-hop"=3, "open-domain"=4.
PUBLISHED = {
    'source': 'arXiv 2504.19413 Tables 1-2 (10 runs, mean +- sd); Zep own: getzep.com blog 2025-05-06 and '
              'getzep/zep-papers kg_architecture_agent_memory/locomo_eval',
    'category_names': {'1': 'multi-hop', '2': 'temporal', '3': 'open-domain', '4': 'single-hop'},
    'mem0': {'overall_j': 66.88, 'overall_j_sd': 0.15, 'context_tokens': 1764,
             'f1_b1_j': {'1': [38.72, 27.13, 67.13], '2': [48.93, 40.51, 55.51], '3': [28.64, 21.58, 51.15],
                         '4': [47.65, 38.72, 72.93]}},
    'mem0_graph': {'overall_j': 68.44, 'overall_j_sd': 0.17, 'context_tokens': 3616,
                   'f1_b1_j': {'1': [38.09, 26.03, 65.71], '2': [51.55, 40.28, 58.13], '3': [24.32, 18.82, 47.19],
                               '4': [49.27, 40.30, 75.71]}},
    'zep_in_mem0_harness': {'overall_j': 65.99, 'overall_j_sd': 0.16, 'context_tokens': 3911,
                            'f1_b1_j': {'1': [35.74, 23.30, 61.70], '2': [42.00, 34.53, 49.31],
                                        '3': [19.37, 14.82, 41.35], '4': [49.56, 38.92, 76.60]}},
    'full_context': {'overall_j': 72.90, 'overall_j_sd': 0.19, 'context_tokens': 26031},
    'zep_own_harness_not_protocol_matched': {
        'overall_j': 75.14, 'overall_j_sd': 0.17,
        'j_by_category_single_run': {'1': 74.11, '2': 66.04, '3': 67.71, '4': 79.79},
        'why_unmatched': "Zep's own reader prompt and retrieval template; Mem0 reran it with Mem0's prompt: 58.44 +- 0.20"},
}


def block(rows, runs):
    if not rows:
        return None
    def runwise(key):
        return [statistics.fmean(r[key][i] for r in rows) for i in range(runs)]
    out = {'n': len(rows), 'evidence_recall': round(statistics.fmean(r['evidence_recall'] for r in rows), 4),
           'context_tokens_mean': round(statistics.fmean(r['context_tokens'] for r in rows), 1),
           'f1': round(statistics.fmean(runwise('f1')), 4), 'bleu1': round(statistics.fmean(runwise('bleu1')), 4),
           'f1_runs': [round(x, 4) for x in runwise('f1')]}
    if all('judge' in r and len(r['judge']) >= runs for r in rows):
        j = runwise('judge')
        out.update({'j': round(statistics.fmean(j), 4), 'j_runs': [round(x, 4) for x in j],
                    'j_std': round(statistics.stdev(j), 4) if runs > 1 else None})
    return out


def groups(rows, runs, split):
    out = {}
    for name, ids in {'all_scored': sorted({r['sample_id'] for r in rows}), **split}.items():
        rs = [r for r in rows if r['sample_id'] in ids]
        if rs:
            out[name] = {'conversations': sorted({r['sample_id'] for r in rs}), 'overall': block(rs, runs),
                         **{f'category_{c}': block([r for r in rs if r['category'] == c], runs) for c in (1, 2, 3, 4)}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--meta', required=True, help='a locomo_bench summary.json for models, split and sha')
    ap.add_argument('--arm', action='append', nargs='+', metavar=('NAME:RUNS', 'PREDICTIONS'), required=True,
                    help='arm name, passes to score, then predictions.jsonl files')
    ap.add_argument('--spend', type=float, required=True, help='OpenAI spend over all runs, USD (estimate when a run was interrupted)')
    ap.add_argument('--note', default='')
    ap.add_argument('--date', default='')
    ap.add_argument('--machine', default='', help='CPU, cores, server threads and load average')
    ap.add_argument('--arm-spend', action='append', default=[], metavar='NAME=USD',
                    help='per-arm spend from that run\'s summary.json new_api_spend_usd')
    args = ap.parse_args()
    meta = json.loads(Path(args.meta).read_text())
    arms = {}
    for name_runs, *files in args.arm:
        name, runs = name_runs.split(':')
        rows = []
        for f in files:
            rows += [json.loads(line) for line in Path(f).read_text().splitlines()]
        arms[name] = groups(rows, int(runs), meta['split'])
        arms[name]['passes'] = int(runs)
    result = {
        'benchmark': 'LoCoMo locomo10.json (snap-research/locomo 3eb6f2c), categories 1-4, category 5 excluded',
        'dataset_sha256': '79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4',
        'protocol': "mem0ai/mem0 evaluation/ at 393a4fd5: ANSWER_PROMPT reader as system message, top_k=30 per "
                    "speaker, ACCURACY_PROMPT judge (json_object), set-token F1 and nltk BLEU-1 on the raw response",
        'reader_model': meta['reader_model'], 'judge_model': meta['judge_model'], 'temperature': 0.0,
        'embedding_model': meta['embedding_model'], 'k_per_speaker': meta['k_per_speaker'],
        'token_budget': meta['token_budget'], 'recall_options': meta['recall_options'],
        'hms_sha': meta['hms_sha'], 'hms_surface': 'hms-server memory API (POST /api/v1/memory/remember, recall)',
        'split': meta['split'],
        'split_rule': 'first three conversations in file order are the tuning set, fixed before any run',
        'scores': 'fractions; f1/bleu1/j are means over independent reader passes',
        'note': args.note, 'date': args.date, 'machine': args.machine,
        'arms': arms,
        'published': PUBLISHED,
        'openai_spend_usd': args.spend,
        'openai_spend_usd_by_arm': {k: float(v) for k, v in (a.split('=') for a in args.arm_spend)},
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    for name, arm in arms.items():
        for g, v in arm.items():
            if isinstance(v, dict):
                o = v['overall']
                print(name, g, o['n'], o.get('j'), o['f1'], o['bleu1'], o['evidence_recall'], o['context_tokens_mean'])
                for c in (1, 2, 3, 4):
                    b = v[f'category_{c}']
                    print(f"   cat{c} n={b['n']} J={b.get('j')} F1={b['f1']} B1={b['bleu1']} ev={b['evidence_recall']}")


if __name__ == '__main__':
    main()
