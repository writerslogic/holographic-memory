# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6"]
# ///
"""Retrieve whole exchanges by cached user rank and independently indexed assistant text."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from proxies.beir_proxies import BM25, STOP, tok_warm

RRF_K0 = 60
MAX_GROUPS = 4096
MAX_SOURCES = 131072
MAX_TEXT_BYTES = 16 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tokens(text):
    return [token for token in tok_warm(text) if token not in STOP]


def retrieve(question, source_groups, user_prior, limit=64):
    """Keep complete exchanges or unpaired assistant blocks, with their original roles."""
    _require(isinstance(question, str) and 0 < len(question) <= 8192
             and len(question.encode("utf-8")) <= 8192, "invalid question length")
    _require(type(limit) is int and 1 <= limit <= 64, "candidate limit must be in 1..=64")
    _require(isinstance(source_groups, list) and 1 <= len(source_groups) <= MAX_GROUPS,
             "invalid source group count")
    _require(isinstance(user_prior, list) and user_prior and all(type(index) is int
             and 0 <= index < len(source_groups) for index in user_prior), "invalid user prior index")
    _require(len(set(user_prior)) == len(user_prior), "duplicate user prior index")
    group_ids, source_ids = set(), set()
    text_bytes, source_count = 0, 0
    assistant_documents = []
    for group in source_groups:
        _require(isinstance(group, dict), "source group must be an object")
        group_id = group.get("group_id")
        _require(isinstance(group_id, str) and 0 < len(group_id) <= 256, "invalid group identity")
        _require(group_id not in group_ids, "duplicate group identity")
        group_ids.add(group_id)
        sources = group.get("sources")
        _require(isinstance(sources, list) and sources, "source group must contain sources")
        has_user, assistants = False, []
        for source in sources:
            _require(isinstance(source, dict), "source must be an object")
            source_id = source.get("source_id")
            _require(isinstance(source_id, str) and 0 < len(source_id) <= 256, "invalid source identity")
            _require(source_id not in source_ids, "duplicate source occurrence")
            source_ids.add(source_id)
            source_count += 1
            _require(source_count <= MAX_SOURCES, "source allocation exceeds limit")
            role, text = source.get("role"), source.get("text")
            _require(role in ("user", "assistant"), "unsupported source role")
            _require(isinstance(text, str) and len(text) <= MAX_TEXT_BYTES, "invalid source text")
            text_bytes += len(text.encode("utf-8"))
            _require(text_bytes <= MAX_TEXT_BYTES, "source text allocation exceeds limit")
            if role == "user":
                has_user = True
            else:
                assistants.append(text)
        _require(not has_user or sources[0]["role"] == "user", "mixed exchange must begin with a user turn")
        assistant_documents.append("\n".join(assistants))

    scores = (BM25(assistant_documents, _tokens).score(question)
              if any(_tokens(document) for document in assistant_documents) else [0.0] * len(source_groups))
    scores = [float(score) for score in scores]
    _require(all(math.isfinite(score) and score >= 0 for score in scores), "invalid assistant retrieval score")
    prior_position = {index: rank for rank, index in enumerate(user_prior)}
    tie = lambda index: prior_position.get(index, len(user_prior) + index)
    assistant_ranking = sorted((index for index, score in enumerate(scores) if score > 0),
                               key=lambda index: (-scores[index], tie(index)))
    fused = [0.0] * len(source_groups)
    for rank, index in enumerate(user_prior):
        fused[index] = 1.0 / (RRF_K0 + rank + 1)
    for rank, index in enumerate(assistant_ranking):
        fused[index] += 1.0 / (RRF_K0 + rank + 1)
    ranking = sorted(range(len(source_groups)), key=lambda index: (-fused[index], tie(index)))
    reserved = user_prior[:min(5, limit)]
    selected = list(reserved)
    selected.extend(index for index in ranking if index not in reserved)
    selected = selected[:limit]
    return {"version": 1, "selected": selected,
            "source_groups": [copy.deepcopy(source_groups[index]) for index in selected],
            "reserved_user_prior": list(reserved), "assistant_ranking": assistant_ranking,
            "assistant_scores": scores, "fused_ranking": ranking, "fused_scores": fused,
            "rrf_k0": RRF_K0,
            "source_policy": "whole exchanges; original first five user candidates reserved; equal-weight user/assistant RRF"}


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prove_dev(source, split, output):
    from longmemeval_answer_evidence import SOURCE_SHA256, _groups_for_question, canonical, sha256_file
    from longmemeval_dev_conversations import load_dev_conversations

    historical_path = ROOT / "benchmarks/results/evidence_completeness_dev.json"
    historical = json.loads(historical_path.read_text())
    dev = set(json.loads(Path(split).read_text())["dev"])
    history = {row["qid"]: row for row in historical["rows"]}
    _require(len(history) == 100 and set(history) == dev, "historical prior must cover frozen dev only")
    questions = load_dev_conversations(Path(source), Path(split))
    old_assistants = {qid: set() for qid in dev}
    snapshot_paths = [ROOT / f"benchmarks/results/longmemeval_dev_answers_{variant}.json" for variant in ("v1", "v3")]
    for path in snapshot_paths:
        snapshot = json.loads(path.read_text())
        _require(len(snapshot["rows"]) == 100 and {row["qid"] for row in snapshot["rows"]} == dev,
                 "snapshot must cover frozen dev only")
        for row in snapshot["rows"]:
            for arm in row["systems"]["matched_gpt54"]["arms"].values():
                old_assistants[row["qid"]].update(source["source_id"] for group in arm["source_groups"]
                                                  for source in group["sources"] if source["role"] == "assistant")
    rows = []
    for qid in sorted(dev):
        question, prior = questions[qid], history[qid]
        prior = {**prior, "baseline_ranking": list(dict.fromkeys(prior["baseline_ranking"]))}
        _, exchanges = _groups_for_question(question, set(prior["ids"]))
        aliases = list(exchanges)
        positions = {alias: index for index, alias in enumerate(aliases)}
        _require(len(prior["baseline_ranking"]) == len(aliases)
                 and set(prior["baseline_ranking"]) == set(aliases), "cached user prior membership differs")
        groups = []
        for alias in aliases:
            occurrences = [[source["session_index"], source["turn_index"]] for source in exchanges[alias]
                           if source["role"] == "user"]
            groups.append({"group_id": _digest(canonical(["lme-group-v1", qid, occurrences])),
                           "frozen_selected_id_sha256": _digest(alias), "sources": exchanges[alias]})
        covered = {source["source_id"] for group in groups for source in group["sources"]}
        for session_index, turns in enumerate(question["haystack_sessions"]):
            orphan = []
            for turn_index, turn in enumerate(turns):
                source_id = _digest(canonical(["lme-source-v1", qid, session_index, turn_index]))
                if turn["role"] == "assistant" and source_id not in covered:
                    orphan.append({"source_id": source_id,
                                   "session_id": _digest(canonical(["lme-session-v1", qid, session_index])),
                                   "session_index": session_index, "turn_index": turn_index,
                                   "observation_time": question["haystack_dates"][session_index],
                                   "role": "assistant", "text": turn["content"]})
                elif orphan:
                    groups.append({"group_id": _digest(canonical(["lme-orphan-group-v1", qid,
                                    [source["source_id"] for source in orphan]])), "sources": orphan})
                    aliases.append(None)
                    orphan = []
            if orphan:
                groups.append({"group_id": _digest(canonical(["lme-orphan-group-v1", qid,
                                [source["source_id"] for source in orphan]])), "sources": orphan})
                aliases.append(None)
        user_prior = [positions[alias] for alias in prior["baseline_ranking"]]
        result = retrieve(question["question"], groups, user_prior)
        selected_aliases = {aliases[index] for index in result["selected"]}
        baseline_aliases = {aliases[index] for index in user_prior[:64]}
        gold = {alias for alias in prior["ids"] if "answer" in alias}
        selected_assistants = {source["source_id"] for group in result["source_groups"] for source in group["sources"]
                               if source["role"] == "assistant"}
        rows.append({"qid": qid, "qtype": question["question_type"], "scored": prior["scored"],
                     "user_prior_complete": gold <= baseline_aliases if prior["scored"] else None,
                     "role_retrieval_complete": gold <= selected_aliases if prior["scored"] else None,
                     "input_groups": len(groups), "selected_groups": len(result["selected"]),
                     "selected_group_sha256": [groups[index]["group_id"] for index in result["selected"]],
                     "assistant_positive_matches": len(result["assistant_ranking"]),
                     "selected_assistant_sources": len(selected_assistants),
                     "old_saved_assistant_sources": len(old_assistants[qid]),
                     "newly_available_assistant_sources": len(selected_assistants - old_assistants[qid]),
                     "old_saved_assistant_sources_selected": len(selected_assistants & old_assistants[qid])})
    scored = [row for row in rows if row["scored"]]
    artifacts = [Path(__file__), Path(__file__).with_name("longmemeval_dev_conversations.py"),
                 Path(__file__).with_name("longmemeval_answer_evidence.py"),
                 Path(__file__).with_name("proxies") / "beir_proxies.py", Path(split), historical_path, *snapshot_paths]
    report = {"schema_version": 1, "dev_only": True, "heldout_run": False, "model_calls": 0,
              "source_sha256": SOURCE_SHA256, "source_policy": "fixed top64 whole exchanges after equal-weight RRF60",
              "selection_inputs": ["question text", "opaque whole-exchange source groups", "cached user prior indices"],
              "limitations": ["User support is labeled coverage before any later reader token cap, not answer accuracy.",
                              "Assistant availability counts are unlabeled; they do not establish answer support.",
                              "Unpaired assistant blocks retain assistant roles; retrieval does not establish personal facts."],
              "summary": {"n": 100, "scored": len(scored),
                          "user_prior_complete": sum(row["user_prior_complete"] for row in scored),
                          "role_retrieval_complete": sum(row["role_retrieval_complete"] for row in scored),
                          "selected_assistant_sources": sum(row["selected_assistant_sources"] for row in rows),
                          "old_saved_assistant_sources": sum(row["old_saved_assistant_sources"] for row in rows),
                          "newly_available_assistant_sources": sum(row["newly_available_assistant_sources"] for row in rows)},
              "artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for path in artifacts],
              "rows": rows}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical(report) + "\n")
    print(json.dumps(report["summary"], sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / ".cache/warm-bench/downloads/longmemeval_s_cleaned.json")
    parser.add_argument("--split", type=Path, default=Path(__file__).with_name("longmemeval_split.json"))
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/longmemeval_role_retrieval_dev_v1.json")
    args = parser.parse_args()
    prove_dev(args.source, args.split, args.output)


if __name__ == "__main__":
    main()
