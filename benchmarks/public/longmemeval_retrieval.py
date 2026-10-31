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
from proxies.beir_proxies import BM25, STOP, tok_warm

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "benchmarks/public/longmemeval_split.json"
SOURCE = Path.home() / ".cache/warm-bench/downloads/longmemeval_s_cleaned.json"
RUN = Path.home() / ".cache/warm-bench/longmemeval_s/runs/s-dev-small-all2"
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


def rank_rounds(question, groups, prior, assistant_weight=0.0, diversity=0.0, expansion=None, reserve_main=0):
    require(sorted(prior) == list(range(len(groups))), "reference round membership differs")
    require(math.isfinite(assistant_weight) and 0 <= assistant_weight <= 4, "invalid assistant weight")
    require(type(reserve_main) is int and 0 <= reserve_main <= 10, "invalid main-query reservation")
    scores = {index: 1.0 / (61 + rank) for rank, index in enumerate(prior)}
    if assistant_weight:
        documents = ["\n".join(source["text"] for source in group["sources"] if source["role"] == "assistant")
                     for group in groups]
        if any(tokens(document) for document in documents):
            lexical = BM25(documents, tokens).score(question)
            order = sorted((i for i in range(len(groups)) if lexical[i] > 0), key=lambda i: (-lexical[i], i))
            weight = assistant_weight if reserve_main or ASSISTANT.search(question) else assistant_weight * 0.1
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
    if reserve_main:
        reserved = prior[:reserve_main]
        ranking = reserved + [i for i in ranking if i not in reserved]
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


def reference(entries, run, assistant_weight=0.5, diversity=0.0, max_rounds=10, subquery_mode="off"):
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
        for weight in (0.5, 1.0, 2.0, 4.0):
            variants[f"assistant_reserved_{weight:g}"] = rank_rounds(entry["question"], groups, prior, weight, reserve_main=5)
        for strength in (0.1, 0.25, 0.5, 1.0):
            variants[f"diversity_{strength:g}"] = rank_rounds(entry["question"], groups, prior, diversity=strength)
        variants["combined"] = rank_rounds(entry["question"], groups, prior, assistant_weight=0.5, diversity=0.25)
        for budget in (5, 10, 20):
            variants[f"budget_{budget}"] = prior[:budget]
        sub_cfg = copy.deepcopy(cfg["turn"])
        sub_cfg["variants"]["sub"] = 1.0
        sub_prior = list(dict.fromkeys(aliases[tid] for tid in ranked(sub_cfg)))
        variants["subquery_fusion"] = sub_prior + [i for i in prior if i not in set(sub_prior)]
        # Cache coverage is explicit: unscored expansion candidates cannot be silently promoted.
        pool = set(prior[:20]) | set(variants["subquery_fusion"][:20])
        main = {aliases[tid]: value for tid, value in rerank["turn"][qid].items() if tid in aliases}
        eligible = pool & set(main)
        variants["subquery_expansion"] = rank_rounds(entry["question"], groups, prior, expansion=(eligible, main))
        selected_prior = {"off": prior, "fused": variants["subquery_fusion"],
                          "expansion": variants["subquery_expansion"]}[subquery_mode]
        variants["selected"] = rank_rounds(entry["question"], groups, selected_prior,
                                           assistant_weight=assistant_weight, diversity=diversity)[:max_rounds]
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


def budget_ceiling(evidence, groups, gold_sessions, gold_turns):
    """Oracle bound for this candidate pool, preserving each anchor's mandatory closure."""
    source_round = {source["sourceId"]: group["id"] for group in groups for source in group["sources"]}
    returned = {source["recordId"]: source for source in evidence["sources"]}
    costs = {}
    for anchor in evidence["anchors"]:
        source = returned[anchor["recordId"]]
        closure = [source, *(returned[record] for record in source["premiseRecordIds"])]
        cost = len({source_round[member["sourceId"]] for member in closure})
        sid = source["sessionId"]
        costs[sid] = min(costs.get(sid, cost), cost)
    eligible = sorted(costs[sid] for sid in gold_sessions if sid in costs)
    result = {"candidate_session_recall": len(eligible) / len(gold_sessions) if gold_sessions else None,
              "candidate_turn_recall": len({source["sourceId"] for source in returned.values()} & gold_turns) / len(gold_turns) if gold_turns else None}
    for budget in (5, 10, 20):
        spent, covered = 0, 0
        for cost in eligible:
            if spent + cost > budget:
                break
            spent += cost
            covered += 1
        result[f"session_oracle@{budget}"] = covered / len(gold_sessions) if gold_sessions else None
    return result


def native(entries, binary, work):
    work.mkdir(parents=True, exist_ok=True)
    arms = defaultdict(list)
    baseline = {"roles": ["user", "assistant"], "k": 32, "candidateLimit": 128,
                "maxSources": 64, "maxBytes": 65536,
                "maxRounds": None, "assistantWeight": 1.0, "sessionDiversity": 0.0}
    variants = {"baseline": baseline}
    for weight in (0.0, 0.25, 0.5, 0.96, 1.0, 2.0):
        variants[f"assistant_{weight:g}"] = {**baseline, "assistantWeight": weight}
    for strength in (0.1, 0.25, 0.5, 1.0):
        variants[f"diversity_{strength:g}"] = {**baseline, "sessionDiversity": strength}
    for budget in (5, 10, 20):
        variants[f"budget_{budget}"] = {**baseline, "maxRounds": budget}
    for weight in (0.0, 0.25, 0.5, 1.0):
        for budget in (10, 20):
            variants[f"combined_w{weight:g}_n{budget}"] = {
                **baseline, "assistantWeight": weight, "sessionDiversity": 0.25, "maxRounds": budget}
    variants["candidate_ceiling"] = {**baseline, "k": 1000, "candidateLimit": 1000,
                                      "maxSources": 4096, "maxBytes": 2 * 1024 * 1024}
    variants["selected"] = {**baseline, "assistantWeight": 0.96, "maxRounds": 10}
    oracle = []
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
                if variants[name]["maxRounds"] is not None:
                    require(len(selected) <= variants[name]["maxRounds"], "native round budget exceeded")
                arms[name].append({"qid": qid, "qtype": entry["question_type"],
                                   "metrics": measure(selected, gold_sessions, gold_turns),
                                   "round_ids": [group["id"] for group in selected],
                                   "dropped": len(reply["evidence"]["dropped"]),
                                   "anchors": len(reply["evidence"]["anchors"])})
            oracle.append({"qid": qid, "qtype": entry["question_type"], "metrics": budget_ceiling(
                replies["candidate_ceiling"]["evidence"], groups, gold_sessions, gold_turns)})
            print(json.dumps({"completed": len(arms["baseline"]), "qid": qid}), flush=True)
    finally:
        client.close()
    return {"identity": identity, "arms": {name: {"summary": summarize(rows), "rows": rows} for name, rows in arms.items()},
            "oracle": {"summary": summarize(oracle), "rows": oracle}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--dev-export", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--native-binary", type=Path)
    parser.add_argument("--native-report", type=Path)
    parser.add_argument("--native-limits", type=Path)
    parser.add_argument("--native-work", type=Path, default=ROOT / "benchmarks/results/longmemeval_retrieval_native_v1")
    parser.add_argument("--assistant-weight", type=float, default=0.5)
    parser.add_argument("--session-diversity", type=float, default=0.0)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--subquery-mode", choices=("off", "fused", "expansion"), default="off")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/longmemeval_retrieval_dev_v1.json")
    args = parser.parse_args()
    require(1 <= args.max_rounds <= 4096, "invalid reference round budget")
    require(not (args.native_binary and args.native_report), "choose a native run or a native report")
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
        report["reference"] = reference(entries, args.run, args.assistant_weight, args.session_diversity,
                                          args.max_rounds, args.subquery_mode)
        report["artifacts"] = {name: file_digest(args.run / name) for name in ("scores.json", "targets.json", "queries.json", "rerank.json")}
        report["defaults"] = {"reference": {"assistant_weight": args.assistant_weight,
                                             "session_diversity": args.session_diversity,
                                             "max_rounds": args.max_rounds, "subqueries": args.subquery_mode}}
    if args.native_report:
        previous = json.loads(args.native_report.read_text())
        require(previous["dev_sha256"] == report["dev_sha256"] and previous["split_sha256"] == report["split_sha256"]
                and previous["heldout_records_decoded"] == 0 and previous["llm_api_calls"] == 0,
                "native report protocol differs")
        require({row["qid"] for row in previous["native"]["arms"]["baseline"]["rows"]} == dev,
                "native report membership differs")
        report["native"] = previous["native"]
        report["artifacts"]["native_report_sha256"] = file_digest(args.native_report)
        require("selected" in report["native"]["arms"], "native report lacks the selected default confirmation")
        report["defaults"]["native"] = {**report["native"]["identity"]["variants"]["selected"],
                                          "model_stage_subquery_weight": 0.0}
    if args.native_limits:
        require("native" in report, "native limits require native measurements")
        proof = json.loads(args.native_limits.read_text())
        require({row["qid"] for row in proof["rows"]} == dev, "limit proof membership differs")
        require(proof["native_identity_sha256"] == digest(report["native"]["identity"])
                and proof["dev_sha256"] == digest(entries), "limit proof native identity differs")
        report["native"]["limit_proof"] = proof
        report["artifacts"]["native_limits_sha256"] = file_digest(args.native_limits)
    if "reference" in report and "native" in report:
        primary = "session_recall@10"
        baseline = report["native"]["arms"]["baseline"]["summary"]
        selected = report["native"]["arms"]["selected"]["summary"]
        require(all(selected[kind][primary] >= baseline[kind][primary] for kind in baseline),
                "selected native default regresses a question type")
        report["acceptance"] = {"target": 0.95, "native_measured": selected["overall"][primary],
                                "no_question_type_session_regression": True,
                                "native_qwen_model_stages_measured": False,
                                "native_lexical_fusion_bound": report["native"].get("limit_proof", {}).get("summary", {}).get("overall", {}).get("role_fusion_upper_bound@10")}
        report["default_rationale"] = [
            "Ten native rounds improve session recall@10 without per-type session regression; five rounds regress and twenty adds context without improving recall@10.",
            "Native assistant weight 0.96 was chosen on dev in a 0.90..1.30 sweep at 0.01 steps with no per-type session regression; the selected configuration was confirmed through the native API. Assistant sources were already indexed in the native baseline.",
            "Reference assistant weight 0.5 improves multi-session coverage without changing any per-type turn recall@10 or round-turn recall@10.",
            "Session MMR stays opt-in: it improves session coverage but lowers evidence-turn coverage on dev.",
            "Subqueries stay off: fusion and cached expansion regress a question type; expansion candidates lack complete main-query reranker coverage.",
            "The budget-only native ablation lowers knowledge-update turn recall@10 from 0.8444 to 0.8111; full selected per-type effects are reported separately."]
        report["per_type_effects"] = {}
        for path, arms, choices in (
                ("native", report["native"]["arms"], {"assistant_weight_0.96": "assistant_0.96", "assistant_weight_0.5": "assistant_0.5", "session_mmr_0.25": "diversity_0.25", "round_budget_10": "budget_10"}),
                ("reference", report["reference"], {"assistant_index_0.5": "assistant_0.5", "assistant_reserved_2": "assistant_reserved_2", "session_mmr_0.25": "diversity_0.25", "round_budget_10": "budget_10", "subquery_fusion": "subquery_fusion", "subquery_expansion": "subquery_expansion"})):
            base = arms["baseline"]["summary"]
            report["per_type_effects"][path] = {feature: {
                kind: {metric: arms[arm]["summary"][kind][metric] - values[metric]
                       for metric in values if ("recall@" in metric or "all@" in metric)
                       and not metric.endswith("_n") and values[metric] is not None}
                for kind, values in base.items()} for feature, arm in choices.items()}
    report["artifacts"]["analysis_scripts"] = {Path(__file__).name: file_digest(__file__),
                                                "longmemeval_retrieval_limits.py": file_digest(Path(__file__).with_name("longmemeval_retrieval_limits.py"))}
    for path in ("native", "reference"):
        if path not in report:
            continue
        arms = report[path]["arms"] if path == "native" else report[path]
        for name, arm in arms.items():
            for row in arm["rows"]:
                if name in ("baseline", "selected"):
                    row["round_ids"] = row["round_ids"][:20]
                else:
                    row.pop("round_ids", None)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
