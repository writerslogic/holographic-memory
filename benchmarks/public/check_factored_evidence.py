# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Independent raw-source verification of bounded factored evidence packets."""

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def handle(value):
    return base64.urlsafe_b64encode(bytes.fromhex(value)).decode().rstrip("=")


def reconstruct(question, identities):
    groups = {}
    for sid, observed, turns in zip(question["haystack_session_ids"], question["haystack_dates"], question["haystack_sessions"]):
        user_ids = []
        for ordinal, source in enumerate(turns, 1):
            if source["role"] == "user":
                original = f"{sid}_{ordinal}"
                matching = {original, original.replace("answer", "noans")} & identities
                require(len(matching) == 1, "ambiguous observed source")
                user_ids.append((original, next(iter(matching)), source["content"]))
        observed_sid = sid if any(original == legacy for original, legacy, _ in user_ids) else sid.replace("answer", "noans")
        for _, legacy, text in user_ids:
            groups.setdefault(legacy, []).append([
                hashlib.sha256(legacy.encode()).hexdigest(),
                hashlib.sha256(observed_sid.encode()).hexdigest(), observed, "user", text])
    require(set(groups) == identities, "observed source membership mismatch")
    return groups


def independent_packet(groups):
    texts, sessions, occurrences = [], [], []
    seen_texts, seen_sessions = {}, {}
    for sources in groups:
        references = []
        for turn, session, observed, role, text in sources:
            if turn not in seen_texts:
                seen_texts[turn] = len(texts)
                texts.append([handle(turn), role, text])
            if (session, observed) not in seen_sessions:
                seen_sessions[(session, observed)] = len(sessions)
                sessions.append([handle(session), observed])
            references.append([seen_texts[turn], seen_sessions[(session, observed)]])
        occurrences.append(references)
    return compact({"v": 3, "t": texts, "s": sessions, "g": occurrences, "i": []})


def check(report, source, split, baseline, historical, fingerprints):
    for key, expected in fingerprints.items():
        require(report[key] == expected, "fingerprint mismatch: " + key)
    dev = set(split["dev"])
    require(len(dev) == 100 and len(split["heldout"]) == 400
            and not dev & set(split["heldout"]), "invalid protected split")
    require(report["dev_only"] is True and report["heldout_run"] is False
            and report["answer_accuracy_measured"] is False
            and report["reader_token_budget_matched"] is False
            and report["competitor_superiority_established"] is False,
            "unsupported evaluation claim")
    require(report["candidate_limit"] == 64, "changed candidate limit")
    require(len(report["rows"]) == 100 and {row["qid"] for row in report["rows"]} == dev,
            "report membership mismatch")
    frozen = {row["qid"]: row for row in baseline["rows"]}
    old = {row["qid"]: row for row in historical["rows"]}
    require(set(frozen) == set(old) == set(source) == dev, "input membership mismatch")
    instances = 0
    for row in report["rows"]:
        qid = row["qid"]
        base, ranked_row = frozen[qid], old[qid]
        groups = reconstruct(source[qid], set(ranked_row["ids"]))
        require(row["qtype"] == base["qtype"] and row["scored"] == base["scored"], "scoring membership mismatch")
        budget = base["budget"]
        require(row["budget_bytes"] == budget, "budget mismatch")
        mandatory = base["arms"]["operand"]["selected"]
        require(row["baseline_selected"] == mandatory, "changed baseline selection")
        original = compact({"v": 2, "q": [groups[item] for item in mandatory], "i": []})
        require(original == base["operand"]["packet"], "baseline raw-source mismatch")
        require(row["baseline_packet_bytes"] == len(original.encode()), "baseline byte mismatch")
        compressed = independent_packet([groups[item] for item in mandatory])
        require(row["factored_baseline_bytes"] == len(compressed.encode()), "factored baseline byte mismatch")
        retained = list(mandatory)
        for candidate in list(dict.fromkeys(ranked_row["baseline_ranking"]))[:64]:
            if candidate not in retained:
                trial = independent_packet([groups[item] for item in retained + [candidate]])
                if len(trial.encode()) <= budget:
                    retained.append(candidate)
        require(row["selected"] == retained, "ranked complete-evidence selection mismatch")
        expected = independent_packet([groups[item] for item in retained])
        require(row["packet"] == expected, "complete observed source packet mismatch")
        require(row["packet_bytes"] == len(expected.encode()) <= budget, "packet byte cap mismatch")
        require(row["packet_sha256"] == hashlib.sha256(expected.encode()).hexdigest(), "packet hash mismatch")
        gold = {item for item in ranked_row["ids"] if "answer" in item}
        require(row["baseline_complete"] == (gold <= set(mandatory))
                and row["complete"] == (gold <= set(retained)), "full-turn support score mismatch")
        instances += sum(len(groups[item]) for item in retained)
    scored = [row for row in report["rows"] if row["scored"]]
    summary = {"scored_questions": len(scored),
               "baseline_complete": sum(row["baseline_complete"] for row in scored),
               "complete": sum(row["complete"] for row in scored),
               "gains": [row["qid"] for row in scored if row["complete"] and not row["baseline_complete"]],
               "losses": [row["qid"] for row in scored if row["baseline_complete"] and not row["complete"]],
               "same_selection_bytes_ratio": sum(row["factored_baseline_bytes"] for row in scored) /
                                             sum(row["baseline_packet_bytes"] for row in scored),
               "additional_groups": sum(len(row["selected"]) - len(row["baseline_selected"]) for row in scored)}
    require(report["summary"] == summary, "summary mismatch")
    return {"questions_verified": 100, "source_occurrences_verified": instances,
            "heldout_questions_untouched": 400, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--source", type=Path, default=Path.home() / ".cache/hms-bench/downloads/longmemeval_s_cleaned.json")
    parser.add_argument("--reproduction", type=Path)
    parser.add_argument("--tamper", action="store_true")
    args = parser.parse_args()
    baseline_path = ROOT / "benchmarks/results/evidence_budget_dev.json"
    historical_path = ROOT / "benchmarks/results/evidence_completeness_dev.json"
    split_path = ROOT / "benchmarks/public/longmemeval_split.json"
    require(sha(args.source) == SOURCE_SHA256, "pinned raw source changed")
    baseline = json.loads(baseline_path.read_text())
    require(sha(baseline_path) == json.loads((ROOT / "benchmarks/results/evidence_budget_summary.json").read_text())["report_sha256"],
            "baseline validation fingerprint mismatch")
    require(sha(historical_path) == baseline["input_sha256"], "historical input changed")
    split = json.loads(split_path.read_text())
    historical = json.loads(historical_path.read_text())
    source = {question["question_id"]: question for question in json.loads(args.source.read_text())
              if question["question_id"] in set(split["dev"])}
    fingerprints = {"source_sha256": SOURCE_SHA256, "split_sha256": sha(split_path),
                    "baseline_sha256": sha(baseline_path), "historical_sha256": sha(historical_path),
                    "protocol_sha256": sha(args.report.parent / "protocol.json"),
                    "codec_sha256": sha(ROOT / "bindings/python/holographic-sdk/holographic_sdk/evidence.py"),
                    "producer_sha256": sha(ROOT / "benchmarks/public/factored_evidence_diagnostic.py")}
    report = json.loads(args.report.read_text())
    summary = check(report, source, split, baseline, historical, fingerprints)
    if args.reproduction:
        repeated = json.loads(args.reproduction.read_text())
        check(repeated, source, split, baseline, historical, fingerprints)
        require(repeated == report, "independent reproduction differs")
        summary["reproduction_identical"] = True
    if args.tamper:
        tests = {}
        for label in ("source_text", "observation_time", "source_handle", "membership", "byte_cap", "score", "hash", "fingerprint", "mandatory_evidence"):
            changed = copy.deepcopy(report)
            row = changed["rows"][0]
            if label in ("source_text", "observation_time", "source_handle"):
                packet = json.loads(row["packet"])
                if label == "source_text":
                    packet["t"][0][2] += " fabricated"
                elif label == "observation_time":
                    packet["s"][0][1] = "1900-01-01"
                else:
                    packet["t"][0][0] = handle("0" * 64)
                row["packet"] = compact(packet)
            elif label == "membership":
                row["qid"] = split["heldout"][0]
            elif label == "byte_cap":
                row["budget_bytes"] += 1
            elif label == "score":
                row["complete"] = not row["complete"]
            elif label == "hash":
                row["packet_sha256"] = "0" * 64
            elif label == "fingerprint":
                changed["codec_sha256"] = "0" * 64
            else:
                row["selected"] = row["selected"][1:]
            try:
                check(changed, source, split, baseline, historical, fingerprints)
            except ValueError as error:
                tests[label] = str(error)
            else:
                raise ValueError("corruption was accepted: " + label)
        summary["tamper_rejections"] = tests
    summary["report_sha256"] = sha(args.report)
    summary["checker_sha256"] = sha(__file__)
    output = args.report.with_name("checked_summary.json")
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
