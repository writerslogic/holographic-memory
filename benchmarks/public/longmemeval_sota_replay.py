# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Candidate and selection interventions on pinned native dev receipts."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tiktoken

import longmemeval_retrieval as R
import longmemeval_retrieval_limits as L
import longmemeval_sota as S


def replay(entries, receipts, canonical):
    enc = tiktoken.get_encoding("o200k_base")
    arms = defaultdict(list)
    for qid, entry in sorted(entries.items()):
        groups, originals, gs, gt = R.rounds_for(entry)
        sources, closures, contexts, membership = S.inputs(groups)
        positions = {source["sourceId"]: i for i, source in enumerate(sources)}
        receipt = receipts[qid]["replies"]["candidate_ceiling"]["evidence"]
        by_role = defaultdict(list)
        for anchor in receipt["anchors"]:
            i = positions[anchor["sourceId"]]
            by_role[sources[i]["role"]].append((i, anchor["score"]))
        raw = np.zeros(len(sources))
        candidate = set()
        for role, items in by_role.items():
            for rank, (i, score) in enumerate(sorted(items, key=lambda v: -v[1])):
                candidate.add(i)
                raw[i] = (1 if role == "user" or L.assistant_intent(entry["question"]) else .96) / (61 + rank)
        consensus = np.zeros(len(sources))
        round_scores = defaultdict(float)
        for i in candidate:
            round_scores[membership[i]] += raw[i]
        for i in candidate:
            consensus[i] = round_scores[membership[i]]
        lexical = S.BM25([contexts[i] for i in range(len(sources))], S.focus_tokens).score(entry["question"])
        combined = raw.copy()
        for rank, i in enumerate(S.order(lexical)):
            if i in candidate:
                combined[i] += 1 / (61 + rank)
        for name, scores, cost in (("native_wide_replay", raw, False), ("native_cost", raw, True),
                                   ("round_consensus", consensus, False), ("round_consensus_cost", consensus, True),
                                   ("contextual_union", combined, False), ("contextual_union_cost", combined, True)):
            selected = S.select(S.order(scores), sources, closures, membership, scores=scores, cost_aware=cost)
            arms[name].append({"qid": qid, "qtype": entry["question_type"], "metrics": S.metrics(selected, gs, gt, enc)})
        lookup = {group["id"]: group for group in groups}
        for key in ("baseline", "selected"):
            ids = next(row["round_ids"] for row in canonical["reference"][key]["rows"] if row["qid"] == qid)
            selected = [lookup[identity] for identity in ids[:10]]
            arms[f"reference_{key}"].append({"qid": qid, "qtype": entry["question_type"], "metrics": S.metrics(selected, gs, gt, enc)})
    return {name: {"summary": R.summarize(rows), "rows": rows} for name, rows in arms.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = R.load_dev_conversations(R.SOURCE, R.SPLIT)
    receipts = json.loads(gzip.decompress((S.CACHE / "native_receipts.json.gz").read_bytes()))
    canonical = json.loads((R.ROOT / "benchmarks/results/longmemeval_retrieval_dev_v1.json").read_text())
    R.require(R.digest(entries) == canonical["dev_sha256"] and set(entries) == set(receipts), "dev identity differs")
    result = replay(entries, receipts, canonical)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for name, arm in result.items():
        print(name, {k: arm["summary"]["overall"][k] for k in ("session_recall@10", "evidence_all@10", "round_turn_recall@10", "context_tokens")})


if __name__ == "__main__":
    main()
