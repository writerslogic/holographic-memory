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


class OnePerItem:
    @staticmethod
    def encode(_text):
        return []


def test_similarity_packing_fits_budget_and_merges_ranked_lists():
    lists = {'a': ['a0', 'a1', 'a2', 'a3'], 'b': ['b0', 'b1', 'b2']}
    sims = {'a': [0.9, 0.2, 0.8, 0.7], 'b': [0.5, 0.6, 0.1]}
    for budget in range(9):
        zipper = lb.pack_by_similarity(lists, sims, OnePerItem, budget, True)
        cosine = lb.pack_by_similarity(lists, sims, OnePerItem, budget, False)
        assert sum(map(len, zipper.values())) == sum(map(len, cosine.values())) == min(budget, 7)
        for side in lists:  # the merge keeps each speaker's recall order
            assert zipper[side] == list(range(len(zipper[side])))
    # Merge order compares only the two list heads: a0 (.9) > b0 (.5), b0 > a1 (.2), b1 (.6) > a1, a1 > b2 (.1).
    assert lb.pack_by_similarity(lists, sims, OnePerItem, 4, True) == {'a': [0, 1], 'b': [0, 1]}
    assert lb.pack_by_similarity(lists, sims, OnePerItem, 4, False) == {'a': [0, 2, 3], 'b': [1]}
    # One empty side: the other side's prefix up to the budget.
    assert lb.pack_by_similarity({'a': ['x', 'y'], 'b': []}, {'a': [0.1, 0.2], 'b': []}, OnePerItem, 1, True) == \
        {'a': [0], 'b': []}


def test_similarity_packing_token_boundary():
    enc = tiktoken.get_encoding('o200k_base')
    lists = {'a': ['8 May, 2023: Alice: one fact'], 'b': ['9 May, 2023: Bob: another fact']}
    sims = {'a': [0.9], 'b': [0.1]}
    first = len(enc.encode(json.dumps(lists['a'][0]))) + 1
    second = len(enc.encode(json.dumps(lists['b'][0]))) + 1
    limit = first + second
    assert lb.pack_by_similarity(lists, sims, enc, limit - 1, True) == {'a': [0], 'b': []}
    assert lb.pack_by_similarity(lists, sims, enc, limit, True) == {'a': [0], 'b': [0]}
    assert lb.pack_by_similarity(lists, sims, enc, limit + 1, True) == {'a': [0], 'b': [0]}
