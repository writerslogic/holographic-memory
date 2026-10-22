# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6"]
# ///
"""Bound native role-rank fusion on dev using verified retrieval receipts."""

import argparse
import json
import math
import re
import struct
from functools import lru_cache
from pathlib import Path

import longmemeval_retrieval as R


def words(query):
    return re.findall(r"[^\W_]+", query.lower())


def assistant_intent(query):
    parts = words(query)
    return any(part in ("you", "your", "assistant") for part in parts) and any(
        part in ("say", "said", "give", "gave") or part.startswith(("suggest", "recommend", "mention", "provid"))
        for part in parts)


def aggregation_intent(query):
    parts = words(query)
    return "total" in parts or any(pair in (("how", "many"), ("list", "all"), ("list", "every"),
                                           ("count", "of"), ("number", "of")) for pair in zip(parts, parts[1:]))


def from_bits(bits):
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


@lru_cache(maxsize=64)
def weight_intervals(users, assistants):
    """First representable weight for each possible cross-role ordering, including zero.

    User candidates precede assistant candidates on exact ties. An ordering changes only
    when w/(61+a) first exceeds 1/(61+u); binary search IEEE754 bit patterns exactly.
    """
    maximum = struct.unpack(">Q", struct.pack(">d", 4.0))[0]
    samples = {0.0, math.nextafter(0.0, 1.0), 4.0}
    for user in range(users):
        score = 1.0 / (61 + user)
        for assistant in range(assistants):
            low, high = 0, maximum
            while low < high:
                middle = (low + high) // 2
                if from_bits(middle) / (61 + assistant) > score:
                    high = middle
                else:
                    low = middle + 1
            samples.add(from_bits(low))
    return sorted(samples)


def prepare(replies, groups):
    originals = {source["recordId"]: source for reply in replies.values() for source in reply["evidence"]["sources"]}
    baseline = replies["baseline"]["evidence"]
    pool = {anchor["recordId"] for anchor in baseline["anchors"]} | {drop["recordId"] for drop in baseline["dropped"]}
    R.require(pool <= set(originals), "native candidate source missing from receipts")
    full = {anchor["recordId"]: anchor["score"] for anchor in replies["candidate_ceiling"]["evidence"]["anchors"]}
    R.require(pool <= set(full), "native candidate rank missing from receipts")
    by_role = {role: sorted((identity for identity in pool if originals[identity]["role"] == role),
                            key=lambda identity: -full[identity]) for role in ("user", "assistant")}
    source_round = {source["sourceId"]: (group["id"], group["session"]) for group in groups for source in group["sources"]}
    candidates = []
    for role, identities in by_role.items():
        for rank, identity in enumerate(identities):
            R.require(full[identity] == 1.0 / (61 + rank), "broader candidate pool changes the baseline role ranks")
            source = originals[identity]
            closure = [source, *(originals[record] for record in source["premiseRecordIds"])]
            candidates.append({"identity": identity, "role": role, "rank": rank, "position": len(candidates),
                               "closure": [(member["recordId"], source_round[member["sourceId"]],
                                            len(json.dumps(member, ensure_ascii=False, separators=(",", ":")).encode()))
                                           for member in closure]})
    return candidates, by_role


def simulate(candidates, options, query):
    weight = 1.0 if assistant_intent(query) else options["assistantWeight"]
    order = sorted((candidate for candidate in candidates if candidate["role"] != "assistant" or weight > 0),
                   key=lambda candidate: (-(weight if candidate["role"] == "assistant" else 1.0) / (61 + candidate["rank"]),
                                          candidate["position"]))
    selected, seen_rounds, rounds, size = set(), set(), [], 2
    for candidate in order:
        group = [member for member in candidate["closure"] if member[0] not in selected]
        extra = {member[1][0] for member in group} - seen_rounds
        if options["maxRounds"] is not None and len(seen_rounds) + len(extra) > options["maxRounds"]:
            continue
        if len(selected) + len(group) > options["maxSources"]:
            continue
        new_size = size + sum(member[2] for member in group) + len(group) - int(not selected and bool(group))
        if new_size > options["maxBytes"]:
            continue
        size = new_size
        for identity, (round_id, session), _ in candidate["closure"]:
            selected.add(identity)
            if round_id not in seen_rounds:
                seen_rounds.add(round_id)
                rounds.append((round_id, session))
    return rounds


def analyze(entries, report, work):
    R.require(report["dev_sha256"] == R.digest(entries), "native receipt dev source differs")
    identity = report["native"]["identity"]
    R.require(identity["profile"]["mode"] == "lexical", "role-fusion bound requires native lexical receipts")
    rows, validated, partition_count, tuning = [], 0, 0, []
    for qid, entry in sorted(entries.items()):
        receipt = json.loads((work / f"{qid}.json").read_text())
        R.require(receipt["key"] == R.digest([identity, entry]), "native receipt identity mismatch")
        replies = receipt["replies"]
        groups, originals, gold_sessions, _ = R.rounds_for(entry)
        selected = {name: R.native_groups(reply["evidence"], originals, groups) for name, reply in replies.items()}
        candidates, by_role = prepare(replies, groups)
        if gold_sessions:
            tuning.append((entry["question"], entry["question_type"], candidates, gold_sessions))
        if not aggregation_intent(entry["question"]):
            for name, options in identity["variants"].items():
                if name == "candidate_ceiling":
                    continue
                actual = [group["id"] for group in selected[name]]
                replay = [round_id for round_id, _ in simulate(candidates, options, entry["question"])]
                R.require(replay == actual, f"native rank/closure replay differs: {qid} {name}")
                validated += 1
        if not gold_sessions:
            best, best_weight = None, None
        elif aggregation_intent(entry["question"]):
            # An optimistic bound for every possible session MMR strength, including untested ones.
            best, best_weight = 1.0, None
        else:
            samples = [1.0] if assistant_intent(entry["question"]) else weight_intervals(len(by_role["user"]), len(by_role["assistant"]))
            best, best_weight = -1.0, None
            for weight in samples:
                options = {**identity["variants"]["baseline"], "assistantWeight": weight, "maxRounds": 10}
                rounds = simulate(candidates, options, entry["question"])
                recall = len({session for _, session in rounds[:10]} & gold_sessions) / len(gold_sessions)
                if recall > best:
                    best, best_weight = recall, weight
            partition_count += len(samples)
        rows.append({"qid": qid, "qtype": entry["question_type"], "aggregation_intent": aggregation_intent(entry["question"]),
                     "metrics": {"role_fusion_upper_bound@10": best}, "oracle_weight": best_weight})
    sweep = []
    base = report["native"]["arms"]["baseline"]["summary"]
    for tick in range(90, 131):
        weight, measured = tick / 100, []
        options = {**identity["variants"]["baseline"], "assistantWeight": weight, "maxRounds": 10}
        for query, qtype, candidates, gold in tuning:
            selected = simulate(candidates, options, query)
            measured.append({"qtype": qtype, "metrics": {"session_recall@10":
                len({session for _, session in selected[:10]} & gold) / len(gold)}})
        summary = R.summarize(measured)
        sweep.append({"weight": weight, "summary": summary, "no_type_regression": all(
            summary[kind]["session_recall@10"] >= base[kind]["session_recall@10"] for kind in summary)})
    best = max((row for row in sweep if row["no_type_regression"]), key=lambda row: (
        row["summary"]["overall"]["session_recall@10"], -abs(row["weight"] - 1.0)))
    confirmed = False
    if identity["variants"].get("selected", {}).get("assistantWeight") == best["weight"]:
        actual = report["native"]["arms"]["selected"]["summary"]
        R.require(all(abs(values["session_recall@10"] - actual[kind]["session_recall@10"]) < 1e-12
                      for kind, values in best["summary"].items()), "selected native confirmation differs from the weight sweep")
        confirmed = True
    return {"scope": "Native lexical path, k=32/role, candidateLimit=128, 64 sources, 65536 bytes, maxRounds=10; every finite assistantWeight in [0,4] and every aggregation-only MMR strength in [0,1].",
            "method": "Exact IEEE754 ordering intervals for each non-aggregation question, allowing a separate oracle weight per question. Aggregation questions receive perfect recall as an optimistic bound. Replayed order and closure budgets must match actual native receipts.",
            "limitations": ["This bounds the measured lexical role-rank fusion class, not retrieval with different relevance scores, model stages, candidate limits, premise policies, or additional query variants."],
            "native_identity_sha256": R.digest(identity), "dev_sha256": R.digest(entries),
            "scored_questions": sum(row["metrics"]["role_fusion_upper_bound@10"] is not None for row in rows),
            "validated_native_replays": validated, "weight_intervals_evaluated": partition_count,
            "global_weight_sweep": {"method": "0.90..1.30 at 0.01 steps, ten rounds, MMR off; reject any per-type session regression, then maximize macro recall and prefer the weight closest to 1 on ties.",
                                    "best": best, "selected_confirmed_through_native_api": confirmed, "rows": sweep},
            "summary": R.summarize(rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-export", type=Path)
    parser.add_argument("--native-report", required=True, type=Path)
    parser.add_argument("--native-work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    entries = (json.loads(args.dev_export.read_text()) if args.dev_export else R.load_dev_conversations(R.SOURCE, R.SPLIT))
    R.require(set(entries) == set(json.loads(R.SPLIT.read_text())["dev"]), "limits require exactly frozen dev")
    result = analyze(entries, json.loads(args.native_report.read_text()), args.native_work)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
