# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Complete source dependencies inside the existing raw rounds without replacing evidence."""

import argparse
import json
from collections import Counter
from pathlib import Path

import tiktoken

import longmemeval_retrieval as R
import longmemeval_sota as S
import longmemeval_sota_context as C
import longmemeval_sota_receipts as N


def complete(seed, available, source_round, max_bytes, max_tokens, encoder, dependencies_only=False):
    R.require(type(max_tokens) is int and max_tokens >= 0 and len(seed) <= 64, "invalid completion bounds")
    chosen = {source["recordId"]: source for source in seed}
    allowed_rounds = {source_round[source["sourceId"]] for source in seed}
    pending = {source["recordId"]: source for source in available
               if source_round[source["sourceId"]] in allowed_rounds and source["recordId"] not in chosen}
    context = C.pack(list(chosen.values()), max_bytes)
    R.require(len(encoder.encode(C.wire(context).decode(), disallowed_special=())) <= max_tokens,
              "seed exceeds token budget")
    while pending:
        references = Counter(identity for source in chosen.values() for identity in source["premiseRecordIds"] if identity not in chosen)
        trials = []
        for identity, source in pending.items():
            if dependencies_only and not references[identity]:
                continue
            proposal = chosen | {identity: source}
            if len(proposal) > 64:
                continue
            packed = C.pack(list(proposal.values()))
            size = len(C.wire(packed))
            if size > max_bytes or len(encoder.encode(C.wire(packed).decode(), disallowed_special=())) > max_tokens:
                continue
            gain = references[identity]
            trials.append((gain, -size, -source["turnIndex"], identity, packed))
        if not trials:
            break
        _, _, _, identity, context = max(trials, key=lambda row: row[:4])
        chosen[identity] = pending.pop(identity)
    return context


def study(entries, receipts):
    encoder = tiktoken.get_encoding("o200k_base")
    rows = []
    for qid, entry in sorted(entries.items()):
        groups, _, gs, gt = R.rounds_for(entry)
        available, _, _, _ = S.inputs(groups)
        membership = {s["sourceId"]: g["id"] for g in groups for s in g["sources"]}
        seed = receipts[qid]["replies"]["selected"]["evidence"]["sources"]
        max_bytes = len(C.wire(C.canonical(seed)))
        max_tokens = len(encoder.encode(C.wire(C.canonical(seed)).decode(), disallowed_special=()))
        context = complete(seed, available, membership, max_bytes, max_tokens, encoder)
        sources = C.unpack(context)
        seed_ids = {s["sourceId"] for s in seed}
        ids = {s["sourceId"] for s in sources}
        selected_rounds = {membership[identity] for identity in ids}
        R.require(seed_ids <= ids and selected_rounds == {membership[i] for i in seed_ids}, "same-round completion changes seed round scope")
        selected = [{**g, "sources": [s for s in sources if membership[s["sourceId"]] == g["id"]]}
                    for g in groups if g["id"] in selected_rounds]
        m = S.metrics(selected, gs, gt, encoder)
        m["context_bytes"] = len(C.wire(context))
        m["context_tokens"] = len(encoder.encode(C.wire(context).decode(), disallowed_special=()))
        m.update({"context_byte_budget": max_bytes, "context_token_budget": max_tokens})
        R.require(m["context_bytes"] <= max_bytes and m["context_tokens"] <= max_tokens,
                  "same-round completion exceeds a context budget")
        m = {key: value for key, value in m.items() if "@" not in key and key != "evidence_mrr"}
        rows.append({"qid": qid, "qtype": entry["question_type"], "metrics": m,
                     "sources_sha256": context["sourcesSha256"], "added_source_ids": sorted(ids - seed_ids),
                     "recovered_annotated_source_ids": sorted((ids - seed_ids) & gt),
                     "recoverable_annotated_source_ids_in_seed_rounds": sorted({
                         s["sourceId"] for s in available
                         if membership[s["sourceId"]] in selected_rounds} & (gt - seed_ids)),
                     "missing_annotated_source_ids_outside_seed_rounds": sorted({
                         s["sourceId"] for s in available
                         if membership[s["sourceId"]] not in selected_rounds} & (gt - seed_ids)),
                     "unresolved_premise_references": sum(type(p) is str for turn in context["turns"] for p in turn[6])})
    return {"summary": R.summarize(rows), "rows": rows,
            "fixed_round_ceiling": {
                "evaluation_only": True,
                "questions_with_recoverable_missing_annotations": sum(bool(row["recoverable_annotated_source_ids_in_seed_rounds"]) for row in rows),
                "missing_annotated_sources_inside_seed_rounds": sum(len(row["recoverable_annotated_source_ids_in_seed_rounds"]) for row in rows),
                "missing_annotated_sources_outside_seed_rounds": sum(len(row["missing_annotated_source_ids_outside_seed_rounds"]) for row in rows),
                "scope": "All original sources in the fixed seed rounds, with no byte, token or source cap; annotations are inspected only after completion. This ceiling does not constrain selection of a different ten rounds."
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = R.load_dev_conversations(R.SOURCE, R.SPLIT)
    result = study(entries, N.load(entries))
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["summary"]["overall"]))


if __name__ == "__main__":
    main()
