# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Collect locomo_bench.py summaries into benchmarks/results/locomo_v1.json.

  uv run --no-project python benchmarks/public/locomo/assemble_results.py OUT.json \
    primary=DIR uncapped=DIR lexical=DIR full_context=DIR --tuning DIR [DIR...]
"""

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('arms', nargs='+', help='name=run_dir')
    ap.add_argument('--tuning', nargs='*', default=[], help='tuning-set run dirs (spend only)')
    args = ap.parse_args()
    arms, spend = {}, 0.0
    for spec in args.arms:
        name, run_dir = spec.split('=', 1)
        summary = json.loads((Path(run_dir) / 'summary.json').read_text())
        spend += summary['new_api_spend_usd']
        arms[name] = {k: summary[k] for k in ('arm', 'embedding_model', 'k_per_speaker', 'token_budget',
                                              'recall_options', 'runs', 'new_api_tokens', 'new_api_spend_usd')}
        arms[name]['results'] = {g: summary['results'][g] for g in ('heldout', 'run', 'tuning')}
        arms[name]['results']['all'] = arms[name]['results'].pop('run')
        meta = summary
    for run_dir in args.tuning:
        spend += json.loads((Path(run_dir) / 'summary.json').read_text())['new_api_spend_usd']
    result = {
        'benchmark': 'LoCoMo locomo10.json (snap-research/locomo 3eb6f2c), categories 1-4 (n=1540), category 5 excluded',
        'dataset_sha256': '79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4',
        'protocol': "mem0ai/mem0 evaluation/ at 393a4fd5: ANSWER_PROMPT reader as system message, top_k=30 per "
                    "speaker, ACCURACY_PROMPT judge (json_object), set-token F1 and nltk BLEU-1 on the raw response",
        'reader_model': meta['reader_model'], 'judge_model': meta['judge_model'], 'temperature': 0.0,
        'hms_sha': meta['hms_sha'], 'hms_surface': 'hms-server memory API (POST /api/v1/memory/remember, recall)',
        'split': meta['split'],
        'split_rule': 'first three conversations in file order are the tuning set, fixed before any run',
        'scores': 'j/f1/bleu1 are fractions; j is the mean over independent reader+judge passes, j_std their sample sd',
        'arms': arms,
        'published': PUBLISHED,
        'openai_spend_usd_total': round(spend, 2),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    for name, arm in arms.items():
        for g in ('heldout', 'all'):
            o = arm['results'][g]['overall']
            print(name, g, o['n'], o['j'], o['j_std'], o['f1'], o['bleu1'], o['context_tokens_mean'])
    print('spend', round(spend, 2))


if __name__ == '__main__':
    main()
