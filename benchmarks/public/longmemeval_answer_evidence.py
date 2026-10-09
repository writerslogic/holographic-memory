# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Reconstruct the frozen dev selections as complete, source-identifiable exchanges."""

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ARMS = ("first_five", "knapsack", "operand")
SOURCE_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _handle(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _observation_key(value):
    match = re.fullmatch(r"(\d{4})/(\d{2})/(\d{2}) \([A-Za-z]{3}\) (\d{2}):(\d{2})", value)
    _require(match is not None, "unrecognized pinned observation time")
    return datetime(*(int(part) for part in match.groups()))


def _groups_for_question(question, historical_ids):
    sessions = question["haystack_sessions"]
    dates = question["haystack_dates"]
    session_ids = question["haystack_session_ids"]
    _require(len(sessions) == len(dates) == len(session_ids), "source session lengths differ")
    legacy_groups, exchange_groups = {}, {}
    for session_index, (session_id, date, turns) in enumerate(zip(session_ids, dates, sessions)):
        _observation_key(date)
        observations = []
        for turn_index, turn in enumerate(turns):
            _require(turn["role"] in ("user", "assistant"), "unsupported pinned source role")
            _require(isinstance(turn["content"], str), "source content must be text")
            if turn["role"] != "user":
                continue
            legacy_id = f"{session_id}_{turn_index + 1}"
            aliases = {legacy_id, legacy_id.replace("answer", "noans")}
            matches = aliases & historical_ids
            _require(len(matches) == 1, "ambiguous or missing frozen source occurrence")
            observations.append((turn_index, next(iter(matches))))
        # Historical identity aliases are replayed; source labels are never consulted.
        observed_session_id = session_id if any(
            legacy_id == f"{session_id}_{turn_index + 1}" for turn_index, legacy_id in observations
        ) else session_id.replace("answer", "noans")
        for turn_index, legacy_id in observations:
            turn = turns[turn_index]
            legacy_groups.setdefault(legacy_id, []).append([
                hashlib.sha256(legacy_id.encode()).hexdigest(),
                hashlib.sha256(observed_session_id.encode()).hexdigest(), date, "user", turn["content"],
            ])
            records = exchange_groups.setdefault(legacy_id, [])
            end = turn_index + 1
            while end < len(turns) and turns[end]["role"] == "assistant":
                end += 1
            for index in range(turn_index, end):
                records.append({
                    "source_id": _handle(["lme-source-v1", question["question_id"], session_index, index]),
                    "session_id": _handle(["lme-session-v1", question["question_id"], session_index]),
                    "session_index": session_index,
                    "turn_index": index,
                    "observation_time": date,
                    "role": turns[index]["role"],
                    "text": turns[index]["content"],
                })
    _require(set(legacy_groups) == historical_ids, "frozen source membership differs")
    return legacy_groups, exchange_groups


def _legacy_packet(groups, selected, arm):
    packet = {"v": 2, "q": [groups[item] for item in selected], "i": []} if arm == "operand" else {
        "v": 1, "evidence": [groups[item] for item in selected],
    }
    return json.dumps(packet, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def build_evidence(source_path, split_path, report_path):
    source_path, split_path, report_path = map(Path, (source_path, split_path, report_path))
    report = json.loads(report_path.read_text())
    validated_path = report_path.with_name("evidence_budget_summary.json")
    validated = json.loads(validated_path.read_text())
    _require(sha256_file(report_path) == validated["report_sha256"], "frozen report fingerprint mismatch")
    _require(sha256_file(source_path) == report["source_sha256"] == SOURCE_SHA256, "pinned source fingerprint mismatch")
    _require(sha256_file(split_path) == report["split_sha256"], "frozen split fingerprint mismatch")
    historical_path = report_path.with_name("evidence_completeness_dev.json")
    _require(sha256_file(historical_path) == report["input_sha256"], "historical identity fingerprint mismatch")
    historical = {row["qid"]: row for row in json.loads(historical_path.read_text())["rows"]}
    dev = json.loads(split_path.read_text())["dev"]
    _require(len(dev) == 100 and len(set(dev)) == 100 and all(isinstance(qid, str) for qid in dev), "invalid dev split")
    dev = set(dev)
    _require(len(report["rows"]) == 100 and {row["qid"] for row in report["rows"]} == dev, "frozen selections must cover all dev questions")
    _require(set(historical) == dev, "historical identities must cover dev only")
    raw_dev = {}
    for question in json.loads(source_path.read_text()):
        if question["question_id"] in dev:
            _require(question["question_id"] not in raw_dev, "duplicate dev question")
            raw_dev[question["question_id"]] = question
    _require(set(raw_dev) == dev, "pinned source is missing dev questions")
    result = []
    for row in report["rows"]:
        question = raw_dev[row["qid"]]
        legacy_groups, exchanges = _groups_for_question(question, set(historical[row["qid"]]["ids"]))
        _require(isinstance(question["question"], str), "question must be text")
        _observation_key(question["question_date"])
        arms = {}
        for name in ARMS:
            frozen = row["arms"][name]
            selected = frozen["selected"]
            _require(selected and len(selected) == len(set(selected)), "invalid frozen selection")
            _require(selected[0] == row["candidates"][0], "frozen anchor differs")
            _require(set(selected) <= set(row["candidates"]), "selected source outside frozen candidates")
            packet = _legacy_packet(legacy_groups, selected, name)
            _require(len(packet) == frozen["packet_bytes"] <= row["budget"], "frozen packet byte mismatch")
            _require(hashlib.sha256(packet).hexdigest() == frozen["packet_sha256"], "frozen packet fingerprint mismatch")
            if name == "operand":
                _require(packet.decode() == row["operand"]["packet"], "frozen operand packet differs")
            groups = []
            for item in selected:
                sources = copy.deepcopy(exchanges[item])
                occurrences = [[source["session_index"], source["turn_index"]] for source in sources if source["role"] == "user"]
                groups.append({
                    "group_id": _handle(["lme-group-v1", row["qid"], occurrences]),
                    "selection_rank": row["candidates"].index(item),
                    "frozen_selected_id_sha256": hashlib.sha256(item.encode()).hexdigest(),
                    "sources": sources,
                })
            arms[name] = {
                "source_groups": groups,
                "frozen_packet_sha256": frozen["packet_sha256"],
                "frozen_packet_bytes": frozen["packet_bytes"],
                "frozen_budget_bytes": row["budget"],
                "selection_sha256": _handle([group["group_id"] for group in groups]),
            }
        result.append({
            "qid": row["qid"], "qtype": question["question_type"], "question": question["question"],
            "question_date": question["question_date"], "arms": arms,
        })
    return result


def reader_payload(question, arm):
    _require(not arm.get("reader_cap", {}).get("selection_failure", False), "reader evidence selection failed its token cap")
    records = [source for group in arm["source_groups"] for source in group["sources"]]
    records.sort(key=lambda source: (_observation_key(source["observation_time"]), source["session_index"], source["turn_index"]))
    _require(len({source["source_id"] for source in records}) == len(records), "duplicate retained source occurrence")
    return {
        "question": question["question"],
        "question_date": question["question_date"],
        "evidence": [{key: source[key] for key in ("source_id", "session_id", "observation_time", "role", "text")} for source in records],
    }


def cap_sources(arm, serialize_callback, count_callback, cap):
    _require(isinstance(cap, int) and not isinstance(cap, bool) and cap > 0, "reader token cap must be a positive integer")
    result = copy.deepcopy(arm)
    groups = result["source_groups"]
    _require(groups and groups[0]["selection_rank"] == 0, "reader evidence requires frozen anchor")
    _require(len({group["group_id"] for group in groups}) == len(groups), "duplicate selected group")
    _require(all(isinstance(group["selection_rank"], int) and group["selection_rank"] >= 0 for group in groups), "invalid selection rank")

    def count(retained):
        serialized = serialize_callback(retained)
        _require(isinstance(serialized, str), "complete reader serialization must be text")
        tokens = count_callback(serialized)
        _require(isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0, "tokenizer returned invalid count")
        return tokens

    uncapped = count(groups)
    retained = list(groups)
    dropped = []
    current = uncapped
    for group in sorted(groups[1:], key=lambda group: group["selection_rank"], reverse=True):
        if current <= cap:
            break
        retained.remove(group)
        dropped.append(group["group_id"])
        current = count(retained)
    failed = current > cap
    if failed:
        _require(len(retained) == 1, "invalid cap failure state")
    result["source_groups"] = retained
    result["reader_cap"] = {
        "cap": cap,
        "input_tokens": current,
        "uncapped_input_tokens": uncapped,
        "dropped_group_ids": dropped,
        "selection_failure": failed,
        "failure_reason": "anchor_complete_input_exceeds_cap" if failed else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.home() / ".cache/hms-bench/downloads/longmemeval_s_cleaned.json")
    parser.add_argument("--split", type=Path, default=Path(__file__).with_name("longmemeval_split.json"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/results/evidence_budget_dev.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    questions = build_evidence(args.source, args.split, args.report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical({"version": 1, "questions": questions}) + "\n")
    print(canonical({
        "dev_questions": len(questions),
        "arms": {name: {
            "source_groups": sum(len(question["arms"][name]["source_groups"]) for question in questions),
            "user_occurrences": sum(source["role"] == "user" for question in questions for group in question["arms"][name]["source_groups"] for source in group["sources"]),
            "assistant_occurrences": sum(source["role"] == "assistant" for question in questions for group in question["arms"][name]["source_groups"] for source in group["sources"]),
        } for name in ARMS},
    }))


if __name__ == "__main__":
    main()
