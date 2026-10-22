# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.2.6"]
# ///
"""Dev-only evidence retrieval measurements; no reader, judge, or model API calls."""

import argparse
import copy
import hashlib
import json
import math
import re
import selectors
import subprocess
from collections import defaultdict
from datetime import timezone
from pathlib import Path

import longmemeval_pipeline as P
from longmemeval_answer_evidence import _observation_key
from longmemeval_dev_conversations import load_dev_conversations
from proxies.beir_proxies import BM25, STOP, tok_hms

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "benchmarks/public/longmemeval_split.json"
SOURCE = Path.home() / ".cache/hms-bench/downloads/longmemeval_s_cleaned.json"
RUN = Path.home() / ".cache/hms-bench/longmemeval_s/runs/s-dev-small-all2"
AGGREGATION = re.compile(r"\b(?:how many|total|list all|list every|count of|number of)\b", re.I)
ASSISTANT = re.compile(r"\b(?:you|your|assistant)\b.*\b(?:suggest\w*|recommend\w*|mention\w*|say|said|give|gave|provid\w*)\b|\b(?:suggest\w*|recommend\w*|provid\w*)\b.*\byou\b", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(text):
    return [token for token in tok_hms(text) if token not in STOP]


def rounds_for(entry):
    """Ranking inputs contain opaque identities and original sources, without annotations."""
    groups, originals, gold_turns, gold_sessions = [], {}, set(), set()
    abstention = entry["question_id"].endswith("_abs")
    for si, (sid, date, session) in enumerate(zip(entry["haystack_session_ids"], entry["haystack_dates"],
                                                entry["haystack_sessions"], strict=True)):
        session_id = digest(["session", si])
        if not abstention and sid in entry["answer_session_ids"]:
            gold_sessions.add(session_id)
        group = None
        for ti, turn in enumerate(session):
            source_id = digest(["source", si, ti])
            source = {"sourceId": source_id, "sessionId": session_id, "turnIndex": ti,
                      "role": turn["role"], "text": turn["content"], "observationTime":
                      int(_observation_key(date).replace(tzinfo=timezone.utc).timestamp() * 1000)}
            originals[source_id] = source
            if not abstention and turn.get("has_answer"):
                gold_turns.add(source_id)
            if group is None or turn["role"] == "user":
                group = {"id": digest(["round", si, ti]), "session": session_id, "sources": []}
                groups.append(group)
            group["sources"].append(source)
    return groups, originals, gold_sessions, gold_turns


def measure(groups, gold_sessions, gold_turns):
    result = {}
    sources = list(dict.fromkeys(source["sourceId"] for group in groups for source in group["sources"]))
    for k in (5, 10, 20):
        sessions = {group["session"] for group in groups[:k]}
        result[f"session_recall@{k}"] = len(sessions & gold_sessions) / len(gold_sessions) if gold_sessions else None
        result[f"session_all@{k}"] = float(gold_sessions <= sessions) if gold_sessions else None
    for k in (10, 20):
        result[f"turn_recall@{k}"] = len(set(sources[:k]) & gold_turns) / len(gold_turns) if gold_turns else None
        round_sources = {source["sourceId"] for group in groups[:k] for source in group["sources"]}
        result[f"round_turn_recall@{k}"] = len(round_sources & gold_turns) / len(gold_turns) if gold_turns else None
    result["returned_rounds"] = len(groups)
    result["returned_turns"] = len(sources)
    return result


def summarize(rows):
    out = {}
    for kind in ["overall", *sorted({row["qtype"] for row in rows})]:
        members = [row for row in rows if kind == "overall" or row["qtype"] == kind]
        summary = {"n_questions": len(members)}
        for metric in members[0]["metrics"]:
            values = [row["metrics"][metric] for row in members if row["metrics"][metric] is not None]
            summary[metric] = sum(values) / len(values) if values else None
            if "recall" in metric or "all@" in metric:
                summary[f"{metric}_n"] = len(values)
        out[kind] = summary
    return out


def diverse_order(ranking, scores, groups, strength):
    require(math.isfinite(strength) and 0 <= strength <= 1, "invalid diversity strength")
    if not strength or not ranking:
        return ranking
    maximum = max(scores.values(), default=0)
    selected, sessions, remaining = [], set(), list(ranking)
    position = {index: rank for rank, index in enumerate(ranking)}
    while remaining:
        index = max(remaining, key=lambda i: (scores[i] / maximum - strength * (groups[i]["session"] in sessions)
                                             if maximum else 0, -position[i]))
        remaining.remove(index)
        selected.append(index)
        sessions.add(groups[index]["session"])
    return selected


def rank_rounds(question, groups, prior, assistant_weight=0.0, diversity=0.0, expansion=None):
    require(sorted(prior) == list(range(len(groups))), "reference round membership differs")
    require(math.isfinite(assistant_weight) and 0 <= assistant_weight <= 4, "invalid assistant weight")
    scores = {index: 1.0 / (61 + rank) for rank, index in enumerate(prior)}
    if assistant_weight:
        documents = ["\n".join(source["text"] for source in group["sources"] if source["role"] == "assistant")
                     for group in groups]
        if any(tokens(document) for document in documents):
            lexical = BM25(documents, tokens).score(question)
            order = sorted((i for i in range(len(groups)) if lexical[i] > 0), key=lambda i: (-lexical[i], i))
            weight = assistant_weight if ASSISTANT.search(question) else assistant_weight * 0.1
            for rank, index in enumerate(order):
                scores[index] += weight / (61 + rank)
    if expansion:
        # Expansion changes membership only. The original question scores every candidate.
        pool, main = expansion
        require(set(pool) <= set(scores), "invalid expansion pool")
        require(set(pool) <= set(main) and all(math.isfinite(main[i]) for i in pool),
                "missing original-question expansion scores")
        head = sorted(pool, key=lambda i: (-main[i], prior.index(i)))
        order = head + [i for i in prior if i not in pool]
        scores = {index: 1.0 / (61 + rank) for rank, index in enumerate(order)}
    ranking = sorted(scores, key=lambda i: (-scores[i], prior.index(i)))
    if AGGREGATION.search(question):
        ranking = diverse_order(ranking, scores, groups, diversity)
    return ranking


def reference_question(entry):
    q = P.Question(entry["question_id"], "", entry["question"], entry["question_date"], P.parse_date(entry["question_date"]))
    aliases = {}
    groups, _, _, _ = rounds_for(entry)
    by_source = {source["sourceId"]: i for i, group in enumerate(groups) for source in group["sources"]}
    for si, (sid, date, session) in enumerate(zip(entry["haystack_session_ids"], entry["haystack_dates"],
                                                entry["haystack_sessions"], strict=True)):
        turns = []
        for ti, turn in enumerate(session):
            if turn["role"] != "user":
                continue
            tid = f"{sid}_{ti + 1}"
            if "answer" in sid and not turn.get("has_answer"):
                tid = tid.replace("answer", "noans")
            reply = session[ti + 1]["content"] if ti + 1 < len(session) and session[ti + 1]["role"] == "assistant" else ""
            turns.append((tid, turn["content"], reply))
            aliases[tid] = by_source[digest(["source", si, ti])]
        if turns:
            osid = sid.replace("answer", "noans") if "answer" in sid and not any("answer" in tid for tid, _, _ in turns) else sid
            q.sessions.append(P.Session(osid, P.parse_date(date), date, turns))
    return q, aliases


def reference(entries, run):
    scores = P.load_scores(run / "scores.json")
    targets = json.loads((run / "targets.json").read_text())
    query_file = json.loads((run / "queries.json").read_text())
    rerank = json.loads((run / "rerank.json").read_text())
    require(set(scores) == set(targets) == set(entries), "score cache must contain exactly frozen dev")
    cfg = json.loads((ROOT / "benchmarks/public/longmemeval_config.json").read_text())
    P._ranks = lambda values: {t: r for r, t in enumerate(sorted(values, key=lambda t: (-values[t], hashlib.sha256(t.encode()).digest())))}
    arms = defaultdict(list)
    for qid, entry in sorted(entries.items()):
        groups, _, gold_sessions, gold_turns = rounds_for(entry)
        q, aliases = reference_question(entry)
        tr = P.time_range(q, query_file["by_key"], query_file["llm"])
        session = P.rank_question(q, scores[qid], targets[qid], cfg["session"], "session", tr, rerank["session"][qid])

        def ranked(config):
            return P.rank_question(q, scores[qid], targets[qid], config, "turn", tr, rerank["turn"][qid], session)

        baseline = ranked(cfg["turn"])
        prior = list(dict.fromkeys(aliases[tid] for tid in baseline))
        prior += [i for i in range(len(groups)) if i not in set(prior)]
        variants = {"baseline": prior}
        for weight in (0.25, 0.5, 1.0, 2.0, 4.0):
            variants[f"assistant_{weight:g}"] = rank_rounds(entry["question"], groups, prior, weight)
        for strength in (0.1, 0.25, 0.5, 1.0):
            variants[f"diversity_{strength:g}"] = rank_rounds(entry["question"], groups, prior, diversity=strength)
        sub_cfg = copy.deepcopy(cfg["turn"])
        sub_cfg["variants"]["sub"] = 1.0
        sub_prior = list(dict.fromkeys(aliases[tid] for tid in ranked(sub_cfg)))
        variants["subquery_fusion"] = sub_prior + [i for i in prior if i not in set(sub_prior)]
        # Cache coverage is explicit: unscored expansion candidates cannot be silently promoted.
        pool = set(prior[:20]) | set(variants["subquery_fusion"][:20])
        main = {aliases[tid]: value for tid, value in rerank["turn"][qid].items() if tid in aliases}
        eligible = pool & set(main)
        variants["subquery_expansion"] = rank_rounds(entry["question"], groups, prior, expansion=(eligible, main))
        for name, ranking in variants.items():
            arms[name].append({"qid": qid, "qtype": entry["question_type"],
                               "aggregation_intent": bool(AGGREGATION.search(entry["question"])),
                               "metrics": measure([groups[i] for i in ranking], gold_sessions, gold_turns),
                               "round_ids": [groups[i]["id"] for i in ranking[:20]],
                               "expansion_pool": len(pool), "expansion_scored": len(eligible)})
    return {name: {"summary": summarize(rows), "rows": rows} for name, rows in arms.items()}


class NativeClient:
    def __init__(self, binary, log):
        self.log = Path(log).open("ab")
        self.process = subprocess.Popen([str(binary), "--lexical"], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def call(self, command):
        data = json.dumps(command).encode()
        require(len(data) <= 8 * 1024 * 1024, "native request exceeds frame limit")
        self.process.stdin.write(data + b"\n")
        self.process.stdin.flush()
        require(self.selector.select(1200), "native request timed out")
        line = self.process.stdout.readline(16 * 1024 * 1024 + 1)
        require(line.endswith(b"\n") and len(line) <= 16 * 1024 * 1024, "invalid native response frame")
        response = json.loads(line)
        require(response.get("status") == "ok", f"native command failed: {response.get('error')}")
        return response

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.selector.close()
        self.process.stdout.close()
        self.log.close()


def native_groups(evidence, originals, source_groups):
    source_round = {source["sourceId"]: group for group in source_groups for source in group["sources"]}
    returned = {}
    for source in evidence["sources"]:
        original = originals.get(source["sourceId"])
        require(original is not None and all(source[key] == original[key] for key in original),
                "native source differs from original")
        returned[source["recordId"]] = source
    groups, seen = [], set()
    for anchor in evidence["anchors"]:
        require(anchor["recordId"] in returned, "native anchor source missing")
        source = returned[anchor["recordId"]]
        premises = [returned[record] for record in source["premiseRecordIds"]]
        for member in [source, *premises]:
            group = source_round[member["sourceId"]]
            if group["id"] not in seen:
                seen.add(group["id"])
                groups.append({"id": group["id"], "session": group["session"], "sources": []})
        # Only actually returned sources count, including required premises.
    for group in groups:
        group["sources"] = sorted((source for source in returned.values()
                                   if source_round[source["sourceId"]]["id"] == group["id"]),
                                  key=lambda source: source["turnIndex"])
    require({source["recordId"] for group in groups for source in group["sources"]} == set(returned),
            "unranked native evidence source")
    return groups


def native(entries, binary, work):
    work.mkdir(parents=True, exist_ok=True)
    arms = defaultdict(list)
    baseline = {"roles": ["user", "assistant"], "k": 32, "candidateLimit": 128,
                "maxSources": 64, "maxBytes": 65536,
                "maxRounds": None, "assistantWeight": 1.0, "sessionDiversity": 0.0}
    variants = {"baseline": baseline}
    for weight in (0.0, 0.25, 0.5, 1.0, 2.0):
        variants[f"assistant_{weight:g}"] = {**baseline, "assistantWeight": weight}
    for strength in (0.1, 0.25, 0.5, 1.0):
        variants[f"diversity_{strength:g}"] = {**baseline, "sessionDiversity": strength}
    for budget in (5, 10, 20):
        variants[f"budget_{budget}"] = {**baseline, "maxRounds": budget}
    variants["candidate_ceiling"] = {**baseline, "k": 128, "candidateLimit": 512,
                                      "maxSources": 4096, "maxBytes": 2 * 1024 * 1024}
    client = NativeClient(binary, work / "native.stderr.log")
    identity = {"binary_sha256": file_digest(binary), "script_sha256": file_digest(__file__),
                "backend": "native lexical BM25; no local model stages", "variants": variants}
    try:
        initialized = client.call({"command": "initialize"})
        identity["profile"] = initialized["profile"]
        for qid, entry in sorted(entries.items()):
            path = work / f"{qid}.json"
            key = digest([identity, entry])
            groups, originals, gold_sessions, gold_turns = rounds_for(entry)
            if path.exists():
                cached = json.loads(path.read_text())
                require(cached["key"] == key, "native cache identity changed; use a new work directory")
                replies = cached["replies"]
            else:
                client.call({"command": "reset"})
                sessions = defaultdict(list)
                for source in originals.values():
                    sessions[source["sessionId"]].append({k: source[k] for k in ("sourceId", "role", "text", "observationTime")})
                for sid, turns in sessions.items():
                    client.call({"command": "ingest", "conversation": {"sessionId": sid, "version": "1", "turns": turns}})
                replies = {name: client.call({"command": "recall", "query": entry["question"], "options": options})
                           for name, options in variants.items()}
                path.write_text(json.dumps({"key": key, "replies": replies}))
            for name, reply in replies.items():
                selected = native_groups(reply["evidence"], originals, groups)
                arms[name].append({"qid": qid, "qtype": entry["question_type"],
                                   "metrics": measure(selected, gold_sessions, gold_turns),
                                   "round_ids": [group["id"] for group in selected],
                                   "dropped": len(reply["evidence"]["dropped"]),
                                   "anchors": len(reply["evidence"]["anchors"])})
            print(json.dumps({"completed": len(arms["baseline"]), "qid": qid}), flush=True)
    finally:
        client.close()
    return {"identity": identity, "arms": {name: {"summary": summarize(rows), "rows": rows} for name, rows in arms.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--dev-export", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--native-binary", type=Path)
    parser.add_argument("--native-work", type=Path, default=ROOT / "benchmarks/results/longmemeval_retrieval_native_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/longmemeval_retrieval_dev_v1.json")
    args = parser.parse_args()
    dev = set(json.loads(SPLIT.read_text())["dev"])
    if args.dev_export:
        entries = json.loads(args.dev_export.read_text())
        require(set(entries) == dev and all(entry["question_id"] == qid for qid, entry in entries.items()),
                "export must contain exactly frozen dev")
    else:
        entries = load_dev_conversations(args.source, SPLIT)
    if args.export:
        args.export.write_text(json.dumps(entries))
        return
    report = {"schema_version": 1, "split": "dev", "dev_only": True, "n_questions": 100,
              "heldout_records_decoded": 0, "llm_api_calls": 0, "dev_sha256": digest(entries),
              "metric_definition": {"session_recall@k": "macro fraction of annotated evidence sessions in the first k returned rounds",
                                    "turn_recall@k": "macro fraction of annotated evidence turns in the first k returned original turns",
                                    "round_turn_recall@k": "macro fraction of annotated evidence turns in the first k returned rounds",
                                    "unscored": "empty gold sets and abstention questions excluded separately per metric",
                                    "native_rounds": "rank anchors by fused score; include only returned original turns and their required premises"},
              "source_sha256": file_digest(args.source) if not args.dev_export else None,
              "split_sha256": file_digest(SPLIT), "artifacts": {},
              "limitations": ["Dev tuning only; retrieval coverage does not establish reader accuracy.",
                              "Native lexical measurements do not estimate native Qwen model-stage recall.",
                              "Assistant reference retrieval uses lexical scores; existing cached dense keys index user turns."]}
    if args.native_binary:
        report["native"] = native(entries, args.native_binary.resolve(), args.native_work)
    else:
        report["reference"] = reference(entries, args.run)
        report["artifacts"] = {name: file_digest(args.run / name) for name in ("scores.json", "targets.json", "queries.json", "rerank.json")}
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
