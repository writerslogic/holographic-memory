# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Monotone enumeration candidate expansion within a lossless context byte budget."""

import argparse
import json
from pathlib import Path

import tiktoken

import longmemeval_retrieval as R
import longmemeval_retrieval_limits as L
import longmemeval_sota as S
import longmemeval_sota_context as C
import longmemeval_sota_receipts as N


def expand(query, seed, candidates, max_bytes, source_round, max_rounds=50, max_sources=64, aggregation_only=True):
    selected = {source["recordId"]: source for source in seed["sources"]}
    available = {source["recordId"]: source for source in candidates["sources"]}
    current = C.pack(list(selected.values()), max_bytes)
    if aggregation_only and not L.aggregation_intent(query):
        return current
    positions = {source["recordId"]: i for i, source in enumerate(candidates["sources"])}
    ranked = sorted(candidates["anchors"], key=lambda anchor: (
        -anchor["score"] * (.96 if available[anchor["recordId"]]["role"] == "assistant" and not L.assistant_intent(query) else 1.),
        available[anchor["recordId"]]["role"] == "assistant", positions[anchor["recordId"]]))
    for anchor in ranked:
        if anchor["score"] < 1 / (60 + 128) - 1e-15:
            continue
        source = available[anchor["recordId"]]
        closure = {identity: available[identity] for identity in [source["recordId"], *source["premiseRecordIds"]]}
        pending = selected | closure
        if len(pending) > max_sources or len({source_round[s["sourceId"]] for s in pending.values()}) > max_rounds:
            continue
        try:
            packed = C.pack(list(pending.values()), max_bytes)
        except ValueError as error:
            if str(error) != "context exceeds byte budget":
                raise
            continue
        selected, current = pending, packed
    return current


def study(entries, receipts, aggregation_only=True):
    enc = tiktoken.get_encoding("o200k_base")
    rows = []
    for qid, entry in sorted(entries.items()):
        groups, originals, gs, gt = R.rounds_for(entry)
        membership = {s["sourceId"]: group["id"] for group in groups for s in group["sources"]}
        seed = receipts[qid]["replies"]["selected"]["evidence"]
        broad = receipts[qid]["replies"]["candidate_ceiling"]["evidence"]
        budget = max(seed["serializedBytes"], len(C.wire(C.pack(seed["sources"]))))
        context = expand(entry["question"], seed, broad, budget, membership, aggregation_only=aggregation_only)
        selected = C.unpack(context)
        R.require({s["recordId"] for s in seed["sources"]} <= {s["recordId"] for s in selected}, "expansion removes a seed source")
        identities = {s["sourceId"] for s in selected}
        chosen_groups = [{**g, "sources": [s for s in selected if membership[s["sourceId"]] == g["id"]]}
                         for g in groups if any(s["sourceId"] in identities for s in g["sources"])]
        metrics = S.metrics(chosen_groups, gs, gt, enc)
        metrics.update({"evidence_all_selected": float(gt <= identities) if gt else None,
                        "evidence_coverage_selected": len(gt & identities) / len(gt) if gt else None,
                        "session_coverage_selected": len(gs & {s["sessionId"] for s in selected}) / len(gs) if gs else None,
                        "context_bytes": len(C.wire(context)),
                        "context_tokens": len(enc.encode(C.wire(context).decode(), disallowed_special=())),
                        "context_byte_budget": budget})
        # A variable-size candidate census has no top-ten ranking claim.
        metrics = {k: v for k, v in metrics.items() if "@" not in k and k != "evidence_mrr"}
        seed_ids = {s["sourceId"] for s in seed["sources"]}
        rows.append({"qid": qid, "qtype": entry["question_type"], "aggregation_intent": L.aggregation_intent(entry["question"]),
                     "sources_sha256": context["sourcesSha256"],
                     "added_source_ids": sorted(identities - seed_ids),
                     "recovered_annotated_source_ids": sorted((identities - seed_ids) & gt),
                     "unresolved_premise_references": sum(type(p) is str for turn in context["turns"] for p in turn[6]),
                     "exhaustive_count_verified": False, "metrics": metrics})
    return {"summary": R.summarize(rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-queries", action="store_true")
    args = parser.parse_args()
    entries = R.load_dev_conversations(R.SOURCE, R.SPLIT)
    result = study(entries, N.load(entries), aggregation_only=not args.all_queries)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
