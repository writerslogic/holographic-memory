# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "tokenizers==0.23.3", "openai==3.27.0", "python-dotenv==1.2.4"]
# ///
"""Measure complete dev history and generic ranked prefixes under the exact reader input cap."""

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path

import tiktoken

from longmemeval_answer_evidence import SOURCE_SHA256, canonical, reader_payload, sha256_file
from longmemeval_dev_conversations import load_dev_conversations
from longmemeval_role_reader import groups_for_question
from longmemeval_role_retrieval import retrieve

ROOT = Path(__file__).resolve().parents[2]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def select_prefix(source_groups, priority, serialize_callback, count_callback, cap, preserve_users=False):
    """Fit a ranked prefix of complete groups, optionally keeping every unpromoted user turn."""
    _require(type(cap) is int and cap > 0, "input cap must be a positive integer")
    _require(type(preserve_users) is bool, "invalid user preservation policy")
    _require(isinstance(source_groups, list) and source_groups, "history requires source groups")
    _require(isinstance(priority, list) and len(priority) == len(source_groups)
             and all(type(index) is int for index in priority)
             and set(priority) == set(range(len(source_groups))), "priority must be a complete index permutation")
    groups = copy.deepcopy(source_groups)

    def materialize(prefix):
        promoted = set(priority[:prefix])
        retained = []
        for index, group in enumerate(groups):
            sources = group["sources"] if index in promoted else (
                [source for source in group["sources"] if source["role"] == "user"] if preserve_users else [])
            if sources:
                retained.append(dict(group, sources=list(sources)))
        return retained

    measured = {}

    def count(prefix):
        if prefix not in measured:
            text = serialize_callback(materialize(prefix))
            _require(isinstance(text, str), "complete reader serialization must be text")
            tokens = count_callback(text)
            _require(type(tokens) is int and tokens >= 0, "tokenizer returned invalid count")
            measured[prefix] = tokens
        return measured[prefix]

    uncapped = count(len(groups))
    failed = False
    if uncapped <= cap:
        prefix = len(groups)
    elif count(0) > cap:
        prefix, failed = 0, True
    else:
        low, high = 0, len(groups) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if count(middle) <= cap:
                low = middle
            else:
                high = middle - 1
        prefix = low
    selected = materialize(prefix)
    return {"source_groups": selected, "input_tokens": count(prefix), "uncapped_input_tokens": uncapped,
            "cap": cap, "selection_failure": failed,
            "failure_reason": ("all_user_input_exceeds_cap" if preserve_users else "reader_framing_exceeds_cap") if failed else None,
            "preserve_users": preserve_users, "promoted_groups": priority[:prefix],
            "token_counts": [{"prefix_groups": index, "input_tokens": tokens} for index, tokens in sorted(measured.items())],
            "policy": "all history if it fits; otherwise binary search a ranked full-group prefix using complete serialized counts"}


def measure(source, split, protocol_path, output):
    protocol = json.loads(Path(protocol_path).read_text())
    _require(protocol["dev_only"] is True and protocol["heldout_run"] is False
             and protocol["token_cap"] == 131072 and protocol["local_token_cap"] == 130560,
             "expected frozen dev full-history input budget")
    questions = load_dev_conversations(Path(source), Path(split))
    historical_path = ROOT / "benchmarks/results/evidence_completeness_dev.json"
    history = {row["qid"]: row for row in json.loads(historical_path.read_text())["rows"]}
    _require(len(history) == len(questions) == 100 and set(history) == set(questions), "dev history membership differs")
    tokenizer = tiktoken.get_encoding("o200k_base")
    count = lambda text: len(tokenizer.encode(text))
    cap = protocol["local_token_cap"]
    rows = []
    for qid in sorted(questions):
        raw, prior = questions[qid], history[qid]
        groups, by_alias, orphan_count = groups_for_question(raw, set(prior["ids"]))
        indices = [by_alias[alias] for alias in dict.fromkeys(prior["baseline_ranking"])]
        ranked = retrieve(raw["question"], groups, indices)
        priority = list(dict.fromkeys(ranked["reserved_user_prior"] + ranked["fused_ranking"]))
        question = {"question": raw["question"], "question_date": raw["question_date"]}
        serialize = lambda retained: protocol["reader_prompt"] + canonical(reader_payload(question, {"source_groups": retained}))
        selected_policies = {name: select_prefix(groups, priority, serialize, count, cap, preserve_users)
                             for name, preserve_users in (("whole_exchange_prefix", False),
                                                          ("all_users_ranked_assistants", True))}
        gold = {hashlib.sha256(alias.encode()).hexdigest() for alias in prior["ids"] if "answer" in alias}
        policies = {}
        for name, selected in selected_policies.items():
            kept = {group["frozen_selected_id_sha256"] for group in selected["source_groups"]}
            sources = [source for group in selected["source_groups"] for source in group["sources"]]
            policies[name] = {key: value for key, value in selected.items() if key != "source_groups"}
            policies[name].update(user_occurrences=sum(source["role"] == "user" for source in sources),
                                  assistant_occurrences=sum(source["role"] == "assistant" for source in sources),
                                  actual_complete=(not selected["selection_failure"] and gold <= kept) if prior["scored"] else None)
        all_sources = [source for group in groups for source in group["sources"]]
        full_tokens = policies["whole_exchange_prefix"]["uncapped_input_tokens"]
        rows.append({"qid": qid, "scored": prior["scored"], "groups": len(groups), "orphan_assistant_groups": orphan_count,
                     "full_input_tokens": full_tokens, "full_history_fits": full_tokens <= cap,
                     "user_occurrences": sum(source["role"] == "user" for source in all_sources),
                     "assistant_occurrences": sum(source["role"] == "assistant" for source in all_sources), "policies": policies})
        if len(rows) % 10 == 0:
            print(json.dumps({"measured_dev_questions": len(rows)}), flush=True)
    counts = [row["full_input_tokens"] for row in rows]
    summary = {"n": 100, "full_history_fits": sum(row["full_history_fits"] for row in rows),
               "full_input_tokens": {"min": min(counts), "median": statistics.median(counts), "max": max(counts)},
               "raw_user_occurrences": sum(row["user_occurrences"] for row in rows),
               "raw_assistant_occurrences": sum(row["assistant_occurrences"] for row in rows), "policies": {}}
    for name in ("whole_exchange_prefix", "all_users_ranked_assistants"):
        summary["policies"][name] = {
            "selection_failures": sum(row["policies"][name]["selection_failure"] for row in rows),
            "labeled_user_complete": sum(row["policies"][name]["actual_complete"] for row in rows if row["scored"]),
            "labeled_user_scored": sum(row["scored"] for row in rows),
            "user_occurrences": sum(row["policies"][name]["user_occurrences"] for row in rows),
            "assistant_occurrences": sum(row["policies"][name]["assistant_occurrences"] for row in rows)}
    artifacts = [Path(__file__), Path(__file__).with_name("longmemeval_role_reader.py"),
                 Path(__file__).with_name("longmemeval_role_retrieval.py"),
                 Path(__file__).with_name("longmemeval_dev_conversations.py"),
                 Path(__file__).with_name("longmemeval_answer_evidence.py"),
                 Path(__file__).with_name("proxies") / "beir_proxies.py", historical_path, Path(split), Path(protocol_path)]
    report = {"schema_version": 1, "dev_only": True, "heldout_run": False, "model_calls": 0,
              "source_sha256": SOURCE_SHA256, "cap": cap, "summary": summary,
              "selection_inputs": ["question text", "opaque original source groups", "cached user prior", "full serialized token counts"],
              "limitations": ["Labeled user coverage and assistant availability are not answer accuracy.",
                              "The ranked prefix is a feasible policy, not a globally optimal evidence subset.",
                              "The user-preserving policy may omit complete assistant turns while retaining their preceding users.",
                              "Binary search assumes increasing prefix token cost; every retained prefix is counted in full."],
              "artifacts": [{"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for path in artifacts], "rows": rows}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical(report) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / ".cache/hms-bench/downloads/longmemeval_s_cleaned.json")
    parser.add_argument("--split", type=Path, default=ROOT / "benchmarks/public/longmemeval_split.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks/results/longmemeval_full_history_v1/protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/longmemeval_history_budget_dev_v1.json")
    args = parser.parse_args()
    measure(args.source, args.split, args.protocol, args.output)


if __name__ == "__main__":
    main()
