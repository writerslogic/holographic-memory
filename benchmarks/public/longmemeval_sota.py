# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0"]
# ///
"""Offline, frozen-dev evidence assembly experiments. No reader or network clients."""

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tiktoken

import longmemeval_retrieval as R
from proxies.beir_proxies import BM25

CACHE = Path.home() / ".cache/hms-bench/longmemeval_sota_dev_v1"
DEPTHS = (1, 5, 10, 20, 50, 100)
FOCUS_STOP = set("many much times total number count list all every ever have had been any about tell me recall remember previously earlier before after during last first latest current currently month months week weeks year years day days conversation conversations chat chats discuss discussed mention mentioned ask asked".split())


def focus_tokens(text):
    return [word for word in R.tokens(text) if word not in FOCUS_STOP]


def inputs(groups):
    sources = [source for group in groups for source in group["sources"]]
    positions = {source["sourceId"]: i for i, source in enumerate(sources)}
    sessions = defaultdict(list)
    for source in sources:
        sessions[source["sessionId"]].append(source)
    closures, context = {}, {}
    for members in sessions.values():
        users = []
        for index, source in enumerate(members):
            sid = source["sourceId"]
            if source["role"] == "user":
                users.append(index)
            user = users[-1] if users else None
            premises = set(users[:2]) - {index}
            if user is not None:
                if user != index:
                    premises.add(user)
                if user > 0 and members[user - 1]["role"] == "assistant":
                    premises.add(user - 1)
            closures[positions[sid]] = {positions[members[i]["sourceId"]] for i in premises | {index}}
            context[positions[sid]] = (members[user]["text"] + "\n" + source["text"]
                                       if user is not None and user != index else source["text"])
    membership = {positions[source["sourceId"]]: i for i, group in enumerate(groups) for source in group["sources"]}
    for source in sources:
        parts = [1, "source", [source["sessionId"], "1", source["turnIndex"], source["sourceId"]]]
        source["recordId"] = "conversation:source:" + hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        source["version"] = "1"
    for i, source in enumerate(sources):
        source["premiseRecordIds"] = [sources[j]["recordId"] for j in sorted(closures[i] - {i})]
    return sources, closures, context, membership


def order(scores):
    return sorted((i for i, score in enumerate(scores) if score > 0), key=lambda i: (-float(scores[i]), i))


def select(ranking, sources, closures, membership, max_rounds=10, max_bytes=65536,
           scores=None, cost_aware=False, coverage=False, query_terms=None):
    chosen, rounds, anchors, size = set(), set(), [], 2
    pending = list(dict.fromkeys(ranking))
    observed = Counter()
    position = {index: rank for rank, index in enumerate(pending)}
    sizes = {i: len(json.dumps(s, ensure_ascii=False, separators=(",", ":")).encode()) for i, s in enumerate(sources)}
    terms = {i: set(focus_tokens(sources[i]["text"])) & query_terms for i in pending} if coverage else {}
    while pending:
        if cost_aware or coverage:
            def value(i):
                new = closures[i] - chosen
                extra = {membership[j] for j in new} - rounds
                score = float(scores[i])
                if coverage:
                    score *= 1 + sum(1 / (1 + observed[t]) for t in terms[i]) / max(1, len(query_terms))
                return score / max(1, len(extra)), -position[i]
            index = max(pending, key=value)
            pending.remove(index)
        else:
            index = pending.pop(0)
        new = closures[index] - chosen
        extra = {membership[j] for j in new} - rounds
        added = sum(sizes[j] + 1 for j in new) - int(not chosen and bool(new))
        if len(rounds | extra) > max_rounds or len(chosen | new) > 64 or size + added > max_bytes:
            continue
        anchors.append(index)
        chosen |= new
        rounds |= extra
        size += added
        if coverage:
            observed.update(terms[index])
    # Round order follows anchors then their canonical dependencies, as in the native evaluator.
    round_order = list(dict.fromkeys(membership[j] for i in anchors for j in [i, *sorted(closures[i] - {i})]))
    return [{"id": str(group), "session": sources[next(i for i in chosen if membership[i] == group)]["sessionId"],
             "sources": [sources[i] for i in sorted(chosen) if membership[i] == group]} for group in round_order]


def metrics(selected, gold_sessions, gold_turns, enc):
    result = R.measure(selected, gold_sessions, gold_turns)
    for k in DEPTHS:
        groups = selected[:k]
        sessions = {group["session"] for group in groups}
        turns = {source["sourceId"] for group in groups for source in group["sources"]}
        for label, gold, got in (("session", gold_sessions, sessions), ("evidence", gold_turns, turns)):
            result[f"{label}_any@{k}"] = float(bool(gold & got)) if gold else None
            result[f"{label}_all@{k}"] = float(gold <= got) if gold else None
            result[f"{label}_coverage@{k}"] = len(gold & got) / len(gold) if gold else None
    source_array = [source for group in selected for source in group["sources"]]
    text = json.dumps(source_array, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    result["context_bytes"] = len(text.encode())
    result["context_tokens"] = len(enc.encode(text, disallowed_special=()))
    result["source_text_tokens"] = sum(len(enc.encode(source["text"], disallowed_special=())) for source in source_array)
    result["source_text_bytes"] = sum(len(source["text"].encode()) for source in source_array)
    selected_sources = {source["sourceId"] for source in source_array}
    selected_sessions = {group["session"] for group in selected}
    result["evidence_all_selected"] = float(gold_turns <= selected_sources) if gold_turns else None
    result["evidence_coverage_selected"] = len(gold_turns & selected_sources) / len(gold_turns) if gold_turns else None
    result["session_coverage_selected"] = len(gold_sessions & selected_sessions) / len(gold_sessions) if gold_sessions else None
    hit = [bool({s["sourceId"] for s in group["sources"]} & gold_turns) for group in selected]
    result["evidence_mrr"] = next((1 / (i + 1) for i, relevant in enumerate(hit) if relevant), 0) if gold_turns else None
    gains = [int(source["sourceId"] in gold_turns) for source in source_array[:20]]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(20, len(gold_turns))))
    result["source_ndcg@20"] = dcg / ideal if ideal else None
    return result


def experiment(entries, receipts, enc):
    arms = defaultdict(list)
    for n, (qid, entry) in enumerate(sorted(entries.items()), 1):
        started = time.perf_counter()
        groups, originals, gold_sessions, gold_turns = R.rounds_for(entry)
        sources, closures, contexts, membership = inputs(groups)
        texts = [source["text"] for source in sources]
        raw = BM25(texts, R.tokens).score(entry["question"])
        contextual = BM25([contexts[i] for i in range(len(sources))], R.tokens).score(entry["question"])
        focused = BM25([contexts[i] for i in range(len(sources))], focus_tokens).score(entry["question"])
        # Structured postings retain literal source text; sentences are retrieval keys only.
        sentences, owners = [], []
        import re
        for i, text in enumerate(texts):
            for sentence in re.split(r"(?<=[.!?\n])\s+", text):
                if sentence.strip():
                    sentences.append(sentence)
                    owners.append(i)
        sentence_scores = BM25(sentences, focus_tokens).score(entry["question"])
        event = np.zeros(len(sources))
        for i, score in zip(owners, sentence_scores, strict=True):
            event[i] = max(event[i], score)
        variants = {"raw_full_turn": (raw, False, False),
                    "contextual_pair": (contextual, False, False),
                    "contextual_focus": (focused, False, False),
                    "dependency_utility": (focused, True, False),
                    "aspect_coverage": (focused, True, True),
                    "literal_event_postings": (event, False, False)}
        for name, (scores, cost_aware, coverage) in variants.items():
            ranking = order(scores)
            selected = select(ranking, sources, closures, membership, scores=scores,
                              cost_aware=cost_aware, coverage=coverage, query_terms=set(focus_tokens(entry["question"])))
            arms[name].append({"qid": qid, "qtype": entry["question_type"],
                               "metrics": metrics(selected, gold_sessions, gold_turns, enc)})
        round_closure = {i: {j for j in range(len(sources)) if membership[j] == membership[i]} for i in range(len(sources))}
        round_user = ["\n".join(s["text"] for s in group["sources"] if s["role"] == "user") for group in groups]
        round_text = ["\n".join(s["text"] for s in group["sources"]) for group in groups]
        for label, docs, tokenizer in (("round_user", round_user, R.tokens),
                                       ("round_pair", round_text, R.tokens),
                                       ("round_focus", round_text, focus_tokens)):
            scores = BM25(docs, tokenizer).score(entry["question"])
            ranks = order(scores)
            representatives = {membership[i]: i for i in range(len(sources))}
            ranking = [representatives[i] for i in ranks]
            selected = select(ranking, sources, round_closure, membership)
            arms[label].append({"qid": qid, "qtype": entry["question_type"],
                                "metrics": metrics(selected, gold_sessions, gold_turns, enc)})
        required = {membership[i] for i, source in enumerate(sources) if source["sourceId"] in gold_turns
                    for i in closures[i]}
        arms["gold_dependency_cost"].append({"qid": qid, "qtype": entry["question_type"], "metrics": {
            "required_rounds": len(required) if gold_turns else None,
            "complete_possible@10": float(len(required) <= 10) if gold_turns else None}})
        for name in ("selected", "candidate_ceiling"):
            evidence = receipts[qid]["replies"][name]["evidence"]
            selected = R.native_groups(evidence, originals, groups)
            arms[f"native_{name}"].append({"qid": qid, "qtype": entry["question_type"],
                                           "metrics": metrics(selected, gold_sessions, gold_turns, enc)})
        if n % 10 == 0:
            print(json.dumps({"completed": n, "last_question_seconds": round(time.perf_counter() - started, 3)}), flush=True)
    return {name: {"summary": R.summarize(rows), "rows": rows} for name, rows in arms.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, default=CACHE / "native_receipts.json.gz")
    args = parser.parse_args()
    entries = R.load_dev_conversations(R.SOURCE, R.SPLIT)
    receipts = json.loads(gzip.decompress(args.receipts.read_bytes()))
    R.require(set(entries) == set(receipts), "receipt membership must equal frozen dev")
    result = {"schema_version": 1, "dev_only": True, "n_questions": len(entries),
              "heldout_records_decoded": 0, "llm_api_calls": 0, "dev_sha256": R.digest(entries),
              "receipt_sha256": R.file_digest(args.receipts),
              "arms": experiment(entries, receipts, tiktoken.get_encoding("o200k_base"))}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for name, arm in result["arms"].items():
        m = arm["summary"]["overall"]
        print(name, {k: m[k] for k in ("session_recall@10", "evidence_all@10", "round_turn_recall@10", "context_tokens") if k in m})


if __name__ == "__main__":
    main()
