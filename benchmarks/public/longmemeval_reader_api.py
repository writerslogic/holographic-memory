# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = ["openai==3.27.0", "python-dotenv==1.2.4", "tiktoken==0.14.0", "tokenizers==0.23.3"]
# ///
"""Frozen, resumable LongMemEval dev readers and blinded official judging."""
import argparse
import copy
import hashlib
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from tokenizers import Tokenizer

from longmemeval_answer_evidence import build_evidence, cap_sources, reader_payload

PROTOCOL = Path("benchmarks/public/reader_protocol_v1.json")
PREREGISTRATION = Path("benchmarks/results/preregistration_2026-10.json")
RESULT = Path("benchmarks/results/longmemeval_dev_answers_v1.json")
PREPARED = Path("target/superiority-validation/prepared_answers.json")
LEDGER = Path("benchmarks/results/spend_2026-10.json")
SOURCE = Path.home() / ".cache/hms-bench/downloads/longmemeval_s_cleaned.json"
SPLIT = Path("benchmarks/public/longmemeval_split.json")
EVIDENCE = Path("benchmarks/results/evidence_budget_dev.json")
CACHE = Path.home() / ".cache/hms-bench/answers-v1"
MUTEX = threading.Lock()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def load_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    assert sha(PROTOCOL) == json.loads(PREREGISTRATION.read_text())["reader_protocol_sha256"]
    return protocol


def client():
    load_dotenv(Path.home() / ".env", override=False)
    return OpenAI(timeout=180, max_retries=0)


def retained(groups):
    return sorted([s for g in groups for s in g["sources"]], key=lambda s: (s["observation_time"], s["session_index"], s["turn_index"]))


def request(protocol, prompt, stage):
    settings = protocol["systems"]["matched_gpt54"]
    return {"model": settings["reader_model" if stage == "reader" else "judge_model"], "input": prompt, "reasoning": {"effort": settings["reasoning_effort"]},
            "max_output_tokens": settings["reader_max_output_tokens" if stage == "reader" else "judge_max_output_tokens"],
            "store": False, "truncation": "disabled"}


def prepare():
    protocol = load_protocol()
    tokenizer = tiktoken.get_encoding("o200k_base")
    weak = Tokenizer.from_file(protocol["systems"]["weak_qwen4b"]["tokenizer_path"])
    weak.no_padding()
    weak.no_truncation()
    template = protocol["systems"]["weak_qwen4b"]["chat_template"]
    def counts(prompt):
        return len(tokenizer.encode(prompt)), len(weak.encode(template.format(reader_input=prompt), add_special_tokens=False).ids)
    questions = build_evidence(SOURCE, SPLIT, EVIDENCE)
    rows = []
    api = client()
    for q in questions:
        row = {k: q[k] for k in ("qid", "qtype", "question", "question_date")}
        row["systems"] = {s: {"arms": {}} for s in protocol["systems"]}
        for name in protocol["arms"]:
            original = q["arms"][name]
            def serialize(groups):
                arm = {"source_groups": groups}
                return protocol["reader_prompt"] + canonical(reader_payload(q, arm))
            arm = cap_sources(original, serialize, lambda p: max(counts(p)), protocol["token_cap"])
            checks = []
            while not arm["reader_cap"]["selection_failure"]:
                prompt = serialize(arm["source_groups"])
                req = {"model": protocol["systems"]["matched_gpt54"]["reader_model"], "input": prompt}
                token_response = api.responses.input_tokens.count(**req).model_dump(mode="json")
                checks.append({"request": req, "request_sha256": digest(canonical(req)), "response": token_response,
                               "group_ids": [g["group_id"] for g in arm["source_groups"]]})
                if token_response["input_tokens"] <= protocol["token_cap"]:
                    break
                if len(arm["source_groups"]) == 1:
                    arm["reader_cap"].update(selection_failure=True, failure_reason="anchor exceeds framed reader token cap")
                    arm["source_groups"] = []
                    break
                group = max(arm["source_groups"], key=lambda g: g["selection_rank"])
                arm["source_groups"].remove(group)
                arm["reader_cap"]["dropped_group_ids"].append(group["group_id"])
            failed = arm["reader_cap"]["selection_failure"]
            prompt = "" if failed else serialize(arm["source_groups"])
            primary_count, weak_count = counts(prompt) if not failed else (0, 0)
            arm["reader_cap"]["input_tokens"] = max(primary_count, weak_count)
            for system, token_count in zip(protocol["systems"], (primary_count, weak_count)):
                value = copy.deepcopy(arm)
                value.update(retained_sources=retained(arm["source_groups"]), reader_input=prompt,
                             reader_input_sha256=digest(prompt), input_tokens=token_count,
                             uncapped_input_tokens=counts(serialize(original["source_groups"]))[0 if system == "matched_gpt54" else 1],
                             framing_checks=checks)
                value["reader"] = {"status": "selection_failure" if failed else "pending", "raw_text": "", "parsed": None}
                value["judge"] = {"status": "pending", "raw_text": "", "score": False}
                value["scores"] = scores(value)
                row["systems"][system]["arms"][name] = value
        rows.append(row)
        print("prepared", len(rows), flush=True)
    paths = [PROTOCOL, PREREGISTRATION, SPLIT, SOURCE, EVIDENCE, Path(__file__),
             Path("benchmarks/public/longmemeval_answer_evidence.py"), Path("src/core/operand_retriever.rs"),
             Path("src/bin/operand-retriever.rs"), Path(protocol["judge_source"]["path"])]
    report = {"schema_version": 1, "dev_only": True, "heldout_run": False,
              "protocol_path": str(PROTOCOL), "protocol_sha256": sha(PROTOCOL),
              "preregistration_path": str(PREREGISTRATION), "preregistration_sha256": sha(PREREGISTRATION),
              "source_path": str(SOURCE), "split_path": str(SPLIT), "evidence_report_path": str(EVIDENCE),
              "evidence_report_sha256": sha(EVIDENCE), "artifacts": [{"path": str(p), "sha256": sha(p)} for p in paths],
              "rows": rows, "summary": {}, "instability": {"entries": []}, "spend_ledger_path": str(LEDGER), "complete": False}
    write(PREPARED, report)
    write(RESULT, report)
    print("prepared_all", len(rows), flush=True)


def api_call(api, protocol, req, call_id):
    cache_key = digest(canonical({"request": req, "call_id": call_id, "protocol_sha256": sha(PROTOCOL)}))
    cache_path = CACHE / (cache_key + ".json")
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    estimated_input = len(tiktoken.get_encoding("o200k_base").encode(req["input"])) + 512
    reserve = estimated_input * 2.5e-6 + req["max_output_tokens"] * 15e-6
    with MUTEX:
        ledger = json.loads(LEDGER.read_text())
        existing = next((e for e in ledger["entries"] if e["cache_key"] == cache_key), None)
        if existing:
            raise RuntimeError("unsettled call already reserved; resolve ledger before retry")
        if ledger["total_usd"] + reserve > 250 or ledger["by_category_usd"]["reader_judge_dev"] + reserve > 120:
            return {"status": "budget_exhausted", "raw_text": "", "request": req, "request_sha256": digest(canonical(req)),
                    "response": None, "usage": None, "cost_usd": 0, "cache_key": cache_key}
        entry = {"cache_key": cache_key, "call_id": call_id, "category": "reader_judge_dev", "status": "reserved",
                 "reserved_usd": reserve, "cost_usd": reserve, "unix_time": time.time(), "usage": None}
        ledger["entries"].append(entry)
        ledger["total_usd"] += reserve
        ledger["by_category_usd"]["reader_judge_dev"] += reserve
        write(LEDGER, ledger)
    try:
        response = api.responses.create(**req)
        usage = response.usage.model_dump(mode="json") if response.usage else None
        if usage is None:
            raise RuntimeError("response lacks usage; reservation remains charged")
        cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
        cost = (usage["input_tokens"] - cached) * 2.5e-6 + cached * .25e-6 + usage["output_tokens"] * 15e-6
        output = {"status": "ok" if response.status == "completed" else "incomplete", "raw_text": response.output_text,
                  "response": response.model_dump(mode="json"), "usage": usage, "cost_usd": cost}
    except Exception as error:
        cost = reserve
        output = {"status": "error", "raw_text": "", "response": None, "usage": None, "cost_usd": cost,
                  "error_type": type(error).__name__, "http_status": getattr(error, "status_code", None),
                  "cost_is_conservative_reservation": True}
    output.update(request=req, request_sha256=digest(canonical(req)), cache_key=cache_key)
    with MUTEX:
        ledger = json.loads(LEDGER.read_text())
        entry = next(e for e in ledger["entries"] if e["cache_key"] == cache_key)
        ledger["total_usd"] += cost - reserve
        ledger["by_category_usd"]["reader_judge_dev"] += cost - reserve
        entry.update(status=output["status"], cost_usd=cost, usage=output["usage"])
        write(LEDGER, ledger)
        write(cache_path, output)
    return output


def parse_answer(raw):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"answer", "abstain", "claims"}:
        return None
    if not isinstance(value["answer"], str) or not isinstance(value["abstain"], bool) or not isinstance(value["claims"], list):
        return None
    if value["abstain"] and value["claims"]:
        return None
    for claim in value["claims"]:
        if not isinstance(claim, dict) or set(claim) != {"text", "citations"} or not isinstance(claim["text"], str) or not isinstance(claim["citations"], list):
            return None
        for citation in claim["citations"]:
            if not isinstance(citation, dict) or set(citation) != {"source_id", "quote"} or not all(isinstance(v, str) for v in citation.values()):
                return None
    return value


def scores(arm):
    value = arm["reader"].get("parsed")
    source = {s["source_id"]: s["text"] for s in arm["retained_sources"]}
    claims = value["claims"] if arm["reader"]["status"] == "ok" and value else []
    valid = sum(bool(c["text"].strip()) and bool(c["citations"]) and all(
        x["source_id"] in source and bool(x["quote"]) and x["quote"] in source[x["source_id"]]
        for x in c["citations"]) for c in claims)
    abstain = bool(value and value["abstain"])
    return {"correct": bool(arm["judge"]["status"] == "ok" and arm["judge"]["score"]),
            "citation_valid_claims": valid, "citation_total_claims": len(claims),
            "all_claims_supported": bool(claims and valid == len(claims) and not abstain), "abstain": abstain}


def percentile(values, fraction):
    ordered = sorted(values)
    at = fraction * (len(ordered) - 1)
    low = int(at)
    return ordered[low] + (at - low) * (ordered[min(low + 1, len(ordered) - 1)] - ordered[low])


def summary(report):
    result = {}
    for system in ("matched_gpt54", "weak_qwen4b"):
        result[system] = {}
        for arm in ("first_five", "knapsack", "operand"):
            values = [r["systems"][system]["arms"][arm]["scores"] for r in report["rows"]]
            n, correct = len(values), sum(v["correct"] for v in values)
            p, z = correct / n, 1.959963984540054
            denominator = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denominator
            width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
            valid, total = sum(v["citation_valid_claims"] for v in values), sum(v["citation_total_claims"] for v in values)
            item = {"n": n, "correct_count": correct, "answer_accuracy": p, "answer_accuracy_ci": [center - width, center + width],
                    "citation_valid_claims": valid, "citation_total_claims": total, "citation_support": valid / total if total else 0,
                    "all_claims_supported_count": sum(v["all_claims_supported"] for v in values), "paired_deltas": {}}
            for control in ("first_five", "knapsack"):
                differences = [int(v["correct"]) - int(r["systems"][system]["arms"][control]["scores"]["correct"]) for r, v in zip(report["rows"], values)]
                rng = random.Random(20261009)
                samples = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000)]
                item["paired_deltas"][control] = {"mean": sum(differences) / n, "ci95": [percentile(samples, .025), percentile(samples, .975)]}
            result[system][arm] = item
    return result


def save_report(report):
    report["summary"] = summary(report)
    report["spend_ledger_sha256"] = sha(LEDGER)
    report["spend_usd"] = json.loads(LEDGER.read_text())["total_usd"]
    write(RESULT, report)


def run_readers(report, protocol):
    tasks = [(r, a) for r in report["rows"] for a in protocol["arms"]
             if r["systems"]["matched_gpt54"]["arms"][a]["reader"]["status"] == "pending"]
    random.Random(20261009).shuffle(tasks)
    api = client()
    def run(task):
        row, name = task
        arm = row["systems"]["matched_gpt54"]["arms"][name]
        return row, name, api_call(api, protocol, request(protocol, arm["reader_input"], "reader"), "reader:" + row["qid"] + ":" + name)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for number, future in enumerate(as_completed([executor.submit(run, t) for t in tasks]), 1):
            row, name, call = future.result()
            arm = row["systems"]["matched_gpt54"]["arms"][name]
            parsed = parse_answer(call["raw_text"]) if call["status"] in ("ok", "completed") else None
            arm["reader"] = {"status": "ok" if parsed else "error", "raw_text": call["raw_text"], "parsed": parsed, "call": call}
            arm["scores"] = scores(arm)
            save_report(report)
            print("reader_completed", number, "status", arm["reader"]["status"], "spend_usd", round(report["spend_usd"], 4), flush=True)


def import_weak(report, output_path):
    outputs = { (x["qid"], x["arm"]): x for x in (json.loads(line) for line in Path(output_path).read_text().splitlines()) }
    for row in report["rows"]:
        for name, arm in row["systems"]["weak_qwen4b"]["arms"].items():
            call = outputs.get((row["qid"], name))
            if call is None or arm["reader"]["status"] == "selection_failure":
                continue
            assert call["request"]["input"] == arm["reader_input"]
            assert call["request_sha256"] == digest(canonical(call["request"]))
            parsed = parse_answer(call["raw_text"]) if call["status"] in ("ok", "completed") else None
            arm["reader"] = {"status": "ok" if parsed else "error", "raw_text": call["raw_text"], "parsed": parsed, "call": call}
            arm["scores"] = scores(arm)
    save_report(report)


def judge_prompt(protocol, row, answer, reference):
    official = protocol["judge_source"]
    template = official["abstention_template"] if "_abs" in row["qid"] else official["templates"][row["qtype"]]
    return template.format(question=row["question"], answer=reference, response=answer)


def run_judges(report, protocol):
    dev = {r["qid"] for r in report["rows"]}
    raw = {q["question_id"]: q for q in json.loads(SOURCE.read_text()) if q["question_id"] in dev}
    tasks = [(r, s, a) for r in report["rows"] for s in protocol["systems"] for a in protocol["arms"]
             if r["systems"][s]["arms"][a]["judge"]["status"] == "pending" and r["systems"][s]["arms"][a]["reader"]["status"] != "pending"]
    random.Random(20261009).shuffle(tasks)
    api = client()
    def run(task):
        row, system, name = task
        arm = row["systems"][system]["arms"][name]
        blind_id = digest("judge-blind-v1:" + row["qid"] + ":" + system + ":" + name)
        if arm["reader"]["status"] != "ok":
            return row, system, name, {"status": "reader_failure", "raw_text": "", "score": False, "blind_id": blind_id}
        prompt = judge_prompt(protocol, row, arm["reader"]["parsed"]["answer"], raw[row["qid"]]["answer"])
        call = api_call(api, protocol, request(protocol, prompt, "judge"), blind_id)
        text = call["raw_text"].strip().lower()
        return row, system, name, {"status": "ok" if call["status"] == "ok" and text in ("yes", "no") else "error",
                                  "raw_text": call["raw_text"], "score": bool(call["status"] == "ok" and text == "yes"), "blind_id": blind_id, "call": call}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for number, future in enumerate(as_completed([executor.submit(run, t) for t in tasks]), 1):
            row, system, name, judged = future.result()
            arm = row["systems"][system]["arms"][name]
            arm["judge"] = judged
            arm["scores"] = scores(arm)
            save_report(report)
            print("judge_completed", number, "status", judged["status"], "spend_usd", round(report["spend_usd"], 4), flush=True)


def instability(report, protocol):
    rows = {r["qid"]: r for r in report["rows"]}
    if not report["instability"]["entries"]:
        for i, qid in enumerate(protocol["instability"]["sample_qids"]):
            name = protocol["arms"][i % 3]
            report["instability"]["entries"].append({"qid": qid, "arm": name, "system": "matched_gpt54", "outcomes": []})
    tasks = [(e, i) for e in report["instability"]["entries"] for i in range(20)
             if i not in {o["repeat_index"] for o in e["outcomes"]}]
    random.Random(20261009).shuffle(tasks)
    api = client()
    dev = set(rows)
    raw = {q["question_id"]: q for q in json.loads(SOURCE.read_text()) if q["question_id"] in dev}
    def run(task):
        entry, i = task
        row = rows[entry["qid"]]
        arm = row["systems"]["matched_gpt54"]["arms"][entry["arm"]]
        primary = arm["judge"]
        if "call" in primary:
            req = primary["call"]["request"]
        else:
            answer = arm["reader"]["parsed"]["answer"] if arm["reader"]["parsed"] else "Reader failure: no valid answer."
            req = request(protocol, judge_prompt(protocol, row, answer, raw[row["qid"]]["answer"]), "judge")
        call = api_call(api, protocol, req, primary["blind_id"] + ":repeat:" + str(i))
        text = call["raw_text"].strip().lower()
        return entry, {"repeat_index": i, "status": "ok" if call["status"] == "ok" and text in ("yes", "no") else "error",
                       "raw_text": call["raw_text"], "score": call["status"] == "ok" and text == "yes", "call": call}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for number, future in enumerate(as_completed([executor.submit(run, t) for t in tasks]), 1):
            entry, outcome = future.result()
            entry["outcomes"].append(outcome)
            entry["outcomes"].sort(key=lambda x: x["repeat_index"])
            save_report(report)
            print("repeat_completed", number, "spend_usd", round(report["spend_usd"], 4), flush=True)
    disagreements = questions = failures = 0
    for entry in report["instability"]["entries"]:
        primary = rows[entry["qid"]]["systems"][entry["system"]]["arms"][entry["arm"]]["judge"]["score"]
        differences = sum(o["score"] != primary for o in entry["outcomes"])
        disagreements += differences
        questions += differences > 0
        failures += sum(o["status"] != "ok" for o in entry["outcomes"])
    report["instability"]["summary"] = {"n_questions": 20, "n_calls": 400, "disagreements": disagreements,
        "disagreement_rate": disagreements / 400, "questions_with_disagreement": questions,
        "question_instability_rate": questions / 20, "failures": failures}
    save_report(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "readers", "judges", "instability", "import-weak", "finalize"))
    parser.add_argument("--weak-output", type=Path, default=Path("target/superiority-validation/weak_reader_outputs.jsonl"))
    args = parser.parse_args()
    protocol = load_protocol()
    if args.stage == "prepare":
        prepare()
        return
    report = json.loads(RESULT.read_text())
    assert report["protocol_sha256"] == sha(PROTOCOL)
    if args.stage == "readers":
        run_readers(report, protocol)
    elif args.stage == "judges":
        run_judges(report, protocol)
    elif args.stage == "instability":
        instability(report, protocol)
    elif args.stage == "import-weak":
        import_weak(report, args.weak_output)
    else:
        assert all(a["reader"]["status"] != "pending" and a["judge"]["status"] != "pending" for r in report["rows"] for s in r["systems"].values() for a in s["arms"].values())
        assert sum(len(e["outcomes"]) for e in report["instability"]["entries"]) == 400
        for entry in report["instability"]["entries"]:
            assert {o["repeat_index"] for o in entry["outcomes"]} == set(range(20))
        frozen_ledger = RESULT.with_name("longmemeval_dev_answers_v1_spend.json")
        write(frozen_ledger, json.loads(LEDGER.read_text()))
        report["spend_ledger_path"] = str(frozen_ledger)
        report["complete"] = True
        save_report(report)
    print(json.dumps({"stage": args.stage, "spend_usd": report["spend_usd"]}), flush=True)


if __name__ == "__main__":
    main()
