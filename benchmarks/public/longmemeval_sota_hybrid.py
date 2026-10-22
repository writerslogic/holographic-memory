# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Matched cached-model interventions; annotation access is confined to evaluation."""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import tiktoken

import longmemeval_retrieval as R
import longmemeval_sota as S


def rerank_order(prior, main_scores, depth):
    pool = prior[:depth]
    scored = [i for i in pool if i in main_scores]
    position = {i: rank for rank, i in enumerate(prior)}
    return sorted(scored, key=lambda i: (-main_scores[i], position[i])) + [i for i in prior if i not in set(scored)]


def study(entries):
    raw = R.P.load_scores(R.RUN / "scores.json")
    targets = json.loads((R.RUN / "targets.json").read_text())
    rerank = json.loads((R.RUN / "rerank.json").read_text())
    config = json.loads((R.ROOT / "benchmarks/public/longmemeval_config.json").read_text())["turn"]
    enc = tiktoken.get_encoding("o200k_base")
    R.P._ranks = lambda values: {t: r for r, t in enumerate(sorted(values, key=lambda t: (-values[t], R.hashlib.sha256(t.encode()).digest())))}
    arms = defaultdict(list)
    for qid, entry in sorted(entries.items()):
        groups, _, gs, gt = R.rounds_for(entry)
        S.inputs(groups)
        q, aliases = R.reference_question(entry)
        # Query IDs and types never enter the ranker.
        q.qid = q.qtype = ""
        variants = {}
        for name, changes in (("hybrid_base", {}), ("hybrid_pair", {"kinds": {"turn": 1., "turn_exp": 2., "round": 1.}}),
                              ("hybrid_wide", {"rerank_n": 50}),
                              ("hybrid_pair_wide", {"kinds": {"turn": 1., "turn_exp": 2., "round": 1.}, "rerank_n": 50})):
            cfg = {**copy.deepcopy(config), **changes}
            ranked = R.P.rank_question(q, raw[qid], targets[qid], cfg, "turn", None, rerank["turn"][qid])
            variants[name] = list(dict.fromkeys(aliases[tid] for tid in ranked))
        main = {aliases[tid]: value for tid, value in rerank["turn"][qid].items()}
        prior = variants["hybrid_base"]
        for depth in (20, 50):
            variants[f"main_relevance_{depth}"] = rerank_order(prior, main, depth)
        variants["all_cached_relevance"] = rerank_order(prior, main, len(prior))
        for name, ranking in variants.items():
            selected = [groups[i] for i in ranking[:10]]
            arms[name].append({"qid": qid, "qtype": entry["question_type"],
                               "cached_main_scored_in_first_50": len(set(ranking[:50]) & set(main)),
                               "metrics": S.metrics(selected, gs, gt, enc),
                               "round_ids": [groups[i]["id"] for i in ranking[:100]]})
    return {name: {"summary": R.summarize(rows), "rows": rows} for name, rows in arms.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = R.load_dev_conversations(R.SOURCE, R.SPLIT)
    result = study(entries)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for name, arm in result.items():
        print(name, {k: arm["summary"]["overall"][k] for k in ("session_recall@10", "evidence_all@10", "round_turn_recall@10", "context_tokens")})


if __name__ == "__main__":
    main()
