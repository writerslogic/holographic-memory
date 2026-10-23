# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""uv run --no-project --with pytest --with openai==1.109.1 --with nltk==3.9.1 --with tiktoken==0.12.0 \
--with jinja2==3.1.6 pytest benchmarks/public/locomo/test_locomo_bench.py"""

import json

import tiktoken

import locomo_bench as lb


def test_token_cap_keeps_rank_prefixes_within_budget():
    enc = tiktoken.get_encoding('o200k_base')
    lists = {'a': [f'8 May, 2023: Alice: fact number {i} ' * (1 + i % 3) for i in range(30)],
             'b': [f'9 May, 2023: Bob: other fact {i}' for i in range(30)]}
    for budget in (0, 40, 41, 500, 10_000):
        kept = lb.cap_tokens(lists, enc, budget)
        for side, items in kept.items():
            assert items == lists[side][:len(items)]
        assert sum(len(enc.encode(json.dumps(x))) + 1 for v in kept.values() for x in v) <= budget
    assert kept == lists


def test_scores_match_mem0_definitions():
    # Set-based F1 over lowercase tokens with . , ! ? stripped, as in Mem0's metrics/utils.py.
    assert lb.f1_score('Sweden, last year!', 'sweden') == 0.5  # precision 1/3, recall 1
    assert lb.f1_score('', 'x') == 0.0
    assert lb.f1_score('Paris paris', 'paris') == 1.0


def test_spend_cap_counts_prior_runs_and_cached_input():
    usage = lb.Usage(prior_usd=0.9, max_usd=1.0)
    # 1M prompt tokens of which 0.5M cached: 0.5 * 0.15 + 0.5 * 0.075 = 0.1125 USD.
    usage.add(lb.READER_MODEL, 1_000_000, 0, cached=500_000)
    assert usage.usd() == 0.1125
    try:
        usage.check()
    except lb.SpendCap:
        pass
    else:
        raise AssertionError('cap not enforced')
    lb.Usage(prior_usd=0.9, max_usd=0.0).check()  # 0 disables the cap
