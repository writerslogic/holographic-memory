# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen, resumable LongMemEval dev readers and blinded judging for the v2 operand arm.

Same protocol, prompts, reader and judge as v1 (benchmarks/public/longmemeval_reader_api.py, whose
request, caching, scoring and judging functions are reused unchanged). What differs:
  evidence   benchmarks/results/longmemeval_v2/evidence_budget_dev.json: the operand arm selects
             from the obligation-decomposed candidate pool (lever 1B.1); first_five and knapsack
             are the frozen v1 selections, so their reader and judge calls are byte-identical to
             v1 and are served from the v1 cache at no cost
  systems    matched_gpt54 only (the report declares reader_systems); the local weak reader is
             a separate row (lever 1B.5)
  ledger     benchmarks/results/longmemeval_dev_answers_v2_spend.json: the v1 entries of the reused
             calls plus every new v2 call, so each paid call the report depends on is recorded once
  result     benchmarks/results/longmemeval_dev_answers_v2.json, with the paired delta of the v2
             operand arm over the v1 operand arm (same 100 questions) and its bootstrap 95% CI
"""
import argparse
import copy
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import longmemeval_reader_api as v1  # noqa: E402
from longmemeval_answer_evidence import build_evidence, cap_sources, reader_payload  # noqa: E402

SYSTEM = "matched_gpt54"
VARIANT = "v2"
EVIDENCE = Path("benchmarks/results/longmemeval_v2/evidence_budget_dev.json")
RESULT = Path("benchmarks/results/longmemeval_dev_answers_v2.json")
PREPARED = Path("target/superiority-validation/prepared_answers_v2.json")
LEDGER = Path("benchmarks/results/longmemeval_dev_answers_v2_spend.json")


def select_variant(name):
    """v2: lever 1B.1 alone; v2b: 1B.1 with twice the byte budget for multi-session and
    temporal-reasoning questions (the reader token cap is unchanged)."""
    global VARIANT, EVIDENCE, RESULT, PREPARED, LEDGER, PREV_RESULT, PREV_LEDGER, PREV_NAME
    VARIANT = name
    if name == "v3":
        PREV_NAME = "v2b"
        PREV_RESULT = Path("benchmarks/results/longmemeval_dev_answers_v2b.json")
        PREV_LEDGER = Path("benchmarks/results/longmemeval_dev_answers_v2b_spend.json")
    EVIDENCE = Path(f"benchmarks/results/longmemeval_{name}/evidence_budget_dev.json")
    RESULT = Path(f"benchmarks/results/longmemeval_dev_answers_{name}.json")
    PREPARED = Path(f"target/superiority-validation/prepared_answers_{name}.json")
    LEDGER = Path(f"benchmarks/results/longmemeval_dev_answers_{name}_spend.json")
V1_RESULT = Path("benchmarks/results/longmemeval_dev_answers_v1.json")
V1_LEDGER = Path("benchmarks/results/longmemeval_dev_answers_v1_spend.json")
# The report whose byte-identical arms are carried over (v1 for v2 and v2b; v2b for v3).
PREV_RESULT, PREV_LEDGER, PREV_NAME = V1_RESULT, V1_LEDGER, "v1"
REUSED_ARMS = ("first_five", "knapsack")
NEW_ARM = "operand"


def seed_ledger(report, protocol):
    """The variant's ledger: the v1 entries of every call whose record the report copies (reader
    attempts, judges and instability repeats of arms with byte-identical evidence) plus the new
    calls already recorded for this variant, so each paid call the report depends on appears once."""
    v1_ledger = json.loads(PREV_LEDGER.read_text())
    v1_report = json.loads(PREV_RESULT.read_text())
    keys = set()
    for row in report["rows"]:
        for name, arm in row["systems"][SYSTEM]["arms"].items():
            if not arm.get("copied_from_v1"):
                continue
            reader = arm["reader"]
            for attempt in reader.get("prior_attempts", []) + ([reader["call"]] if reader.get("call") else []):
                keys.add(attempt["cache_key"])
            if arm["judge"].get("call"):
                keys.add(arm["judge"]["call"]["cache_key"])
    v1_instability = {(e["qid"], e["arm"]): e for e in v1_report["instability"]["entries"]}
    rows = {r["qid"]: r for r in report["rows"]}
    for i, qid in enumerate(protocol["instability"]["sample_qids"]):
        name = protocol["arms"][i % 3]
        if rows[qid]["systems"][SYSTEM]["arms"][name].get("copied_from_v1"):
            keys.update(o["call"]["cache_key"] for o in v1_instability[(qid, name)]["outcomes"])
    v1_keys = {e["cache_key"] for e in v1_ledger["entries"]}
    entries = [e for e in v1_ledger["entries"] if e["cache_key"] in keys]
    if LEDGER.exists():
        entries += [e for e in json.loads(LEDGER.read_text())["entries"] if e["cache_key"] not in v1_keys]
    by_category = {c: 0.0 for c in v1_ledger["caps_usd"] if c != "total"}
    for e in entries:
        by_category[e["category"]] += e["cost_usd"]
    ledger = {"schema_version": 1, "caps_usd": v1_ledger["caps_usd"], "entries": entries,
              "by_category_usd": by_category, "total_usd": sum(by_category.values()),
              "seeded_from": {"path": str(PREV_LEDGER), "sha256": v1.sha(PREV_LEDGER), "copied_call_keys": len(keys), "copied_entries": sum(e["cache_key"] in keys for e in entries)}}
    v1.write(LEDGER, ledger)


def prepare():
    protocol = v1.load_protocol()
    tokenizer = tiktoken.get_encoding("o200k_base")
    weak = Tokenizer.from_file(protocol["systems"]["weak_qwen4b"]["tokenizer_path"])
    weak.no_padding()
    weak.no_truncation()
    template = protocol["systems"]["weak_qwen4b"]["chat_template"]

    def counts(prompt):
        return len(tokenizer.encode(prompt)), len(weak.encode(template.format(reader_input=prompt), add_special_tokens=False).ids)
    questions = build_evidence(v1.SOURCE, v1.SPLIT, EVIDENCE)
    v1_rows = {r["qid"]: r for r in json.loads(PREV_RESULT.read_text())["rows"]}
    rows = []
    api = v1.client()
    for q in questions:
        row = {k: q[k] for k in ("qid", "qtype", "question", "question_date")}
        row["systems"] = {SYSTEM: {"arms": {}}}
        for name in protocol["arms"]:
            original = q["arms"][name]

            def serialize(groups):
                return protocol["reader_prompt"] + v1.canonical(reader_payload(q, {"source_groups": groups}))
            arm = cap_sources(original, serialize, lambda p: max(counts(p)), protocol["token_cap"])
            checks = []
            while not arm["reader_cap"]["selection_failure"]:
                prompt = serialize(arm["source_groups"])
                req = {"model": protocol["systems"][SYSTEM]["reader_model"], "input": prompt}
                token_response = api.responses.input_tokens.count(**req).model_dump(mode="json")
                checks.append({"request": req, "request_sha256": v1.digest(v1.canonical(req)), "response": token_response,
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
            value = copy.deepcopy(arm)
            value.update(retained_sources=v1.retained(arm["source_groups"]), reader_input=prompt,
                         reader_input_sha256=v1.digest(prompt), input_tokens=primary_count,
                         uncapped_input_tokens=counts(serialize(original["source_groups"]))[0], framing_checks=checks)
            value["reader"] = {"status": "selection_failure" if failed else "pending", "raw_text": "", "parsed": None}
            value["judge"] = {"status": "pending", "raw_text": "", "score": False}
            value["scores"] = v1.scores(value)
            previous = v1_rows[q["qid"]]["systems"][SYSTEM]["arms"][name]
            if name in REUSED_ARMS:
                assert previous["reader_input"] == prompt, f"{q['qid']} {name}: frozen arm input differs from {PREV_NAME}"
            if previous["reader_input"] == prompt and previous["uncapped_input_tokens"] == value["uncapped_input_tokens"] and not failed:
                # Byte-identical evidence: the v1 reader attempts, judge and scores are the
                # measurement of this exact input; they are carried over, never re-sampled.
                assert previous["source_groups"] == value["source_groups"], f"{q['qid']} {name}: groups differ"
                value = copy.deepcopy(previous)
                value["copied_from_v1"] = True
                value["copied_from"] = PREV_NAME
            row["systems"][SYSTEM]["arms"][name] = value
        rows.append(row)
        print("prepared", len(rows), flush=True)
    paths = [v1.PROTOCOL, v1.PREREGISTRATION, v1.SPLIT, v1.SOURCE, EVIDENCE, Path(__file__), Path(v1.__file__),
             Path("benchmarks/public/longmemeval_answer_evidence.py"), Path("benchmarks/public/longmemeval_obligations.py"),
             Path("benchmarks/public/resume_longmemeval_readers.py"),
             Path("benchmarks/results/longmemeval_dev_obligations_v2.json"),
             Path("src/core/experimental/operand_retriever.rs"), Path("src/bin/research/operand-retriever.rs"), Path(protocol["judge_source"]["path"])]
    evidence_report = json.loads(EVIDENCE.read_text())
    report = {"schema_version": 1, "dev_only": True, "heldout_run": False, "reader_systems": [SYSTEM], "variant": VARIANT,
              "lever": ("1B.3 abstention calibration (operand arm second pass under a widened byte budget for flagged abstentions, on v2b); " if VARIANT == "v3" else "1B.1 obligation decomposition (operand arm); first_five and knapsack frozen from v1; ") + evidence_report["protocol"],
              "budget_factor": evidence_report["budget_factor"], "selector_decision": evidence_report["decision"],
              "calibration": evidence_report.get("calibration"), "previous_variant": PREV_NAME,
              "previous_result_path": str(PREV_RESULT), "previous_result_sha256": v1.sha(PREV_RESULT),
              "v1_result_path": str(V1_RESULT), "v1_result_sha256": v1.sha(V1_RESULT),
              "protocol_path": str(v1.PROTOCOL), "protocol_sha256": v1.sha(v1.PROTOCOL),
              "preregistration_path": str(v1.PREREGISTRATION), "preregistration_sha256": v1.sha(v1.PREREGISTRATION),
              "source_path": str(v1.SOURCE), "split_path": str(v1.SPLIT), "evidence_report_path": str(EVIDENCE),
              "evidence_report_sha256": v1.sha(EVIDENCE), "artifacts": [{"path": str(p), "sha256": v1.sha(p)} for p in paths],
              "rows": rows, "summary": {}, "instability": {"entries": []}, "spend_ledger_path": str(LEDGER), "complete": False,
              "copied_from_v1": {name: sum(r["systems"][SYSTEM]["arms"][name].get("copied_from_v1", False) for r in rows) for name in protocol["arms"]}}
    seed_ledger(report, protocol)
    v1.write(PREPARED, report)
    save_report(report)
    print("prepared_all", len(rows), flush=True)


def paired_vs_v1(report, against=None):
    """Per-arm paired delta of this variant's correctness over the same arm of `against`
    (the v1 report by default) on the same questions."""
    v1_rows = {r["qid"]: r for r in json.loads((against or V1_RESULT).read_text())["rows"]}
    out = {}
    for name in report["rows"][0]["systems"][SYSTEM]["arms"]:
        differences = [int(r["systems"][SYSTEM]["arms"][name]["scores"]["correct"])
                       - int(v1_rows[r["qid"]]["systems"][SYSTEM]["arms"][name]["scores"]["correct"]) for r in report["rows"]]
        n = len(differences)
        rng = random.Random(20261016)
        samples = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000)]
        out[name] = {"mean": sum(differences) / n, "ci95": [v1.percentile(samples, .025), v1.percentile(samples, .975)],
                     "wins": sum(d > 0 for d in differences), "losses": sum(d < 0 for d in differences),
                     "v1_correct": sum(v1_rows[r["qid"]]["systems"][SYSTEM]["arms"][name]["scores"]["correct"] for r in report["rows"]),
                     "v2_correct": sum(r["systems"][SYSTEM]["arms"][name]["scores"]["correct"] for r in report["rows"])}
    return out


def summary(report):
    result = {SYSTEM: {}}
    for arm in report["rows"][0]["systems"][SYSTEM]["arms"]:
        values = [r["systems"][SYSTEM]["arms"][arm]["scores"] for r in report["rows"]]
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
            differences = [int(v["correct"]) - int(r["systems"][SYSTEM]["arms"][control]["scores"]["correct"]) for r, v in zip(report["rows"], values)]
            rng = random.Random(20261009)
            samples = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000)]
            item["paired_deltas"][control] = {"mean": sum(differences) / n, "ci95": [v1.percentile(samples, .025), v1.percentile(samples, .975)]}
        result[SYSTEM][arm] = item
    return result


def abstention_accounting(report):
    """Lever 1B.3: per arm, abstentions split by the judge's verdict (a false abstention is
    one the judge marks wrong) and answers that are wrong or carry a claim without a valid
    citation (unsupported)."""
    out = {}
    for arm in report["rows"][0]["systems"][SYSTEM]["arms"]:
        values = [r["systems"][SYSTEM]["arms"][arm]["scores"] for r in report["rows"]]
        out[arm] = {"abstentions": sum(v["abstain"] for v in values),
                    "false_abstentions": sum(v["abstain"] and not v["correct"] for v in values),
                    "true_abstentions": sum(v["abstain"] and v["correct"] for v in values),
                    "wrong_answers": sum((not v["abstain"]) and not v["correct"] for v in values),
                    "unsupported_answers": sum((not v["abstain"]) and not (v["correct"] and v["all_claims_supported"]) for v in values)}
    return out


def context_tokens(report):
    """Median reader input tokens per arm (the complete framed prompt, o200k_base)."""
    out = {}
    for arm in report["rows"][0]["systems"][SYSTEM]["arms"]:
        values = sorted(r["systems"][SYSTEM]["arms"][arm]["input_tokens"] for r in report["rows"])
        out[arm] = {"median": v1.percentile(values, .5), "p95": v1.percentile(values, .95), "max": values[-1]}
    return out


def save_report(report):
    report["summary"] = summary(report)
    report["paired_vs_v1"] = paired_vs_v1(report)
    if PREV_RESULT != V1_RESULT:
        report["paired_vs_" + PREV_NAME] = paired_vs_v1(report, PREV_RESULT)
    report["reader_context_tokens"] = context_tokens(report)
    report["abstention_accounting"] = abstention_accounting(report)
    report["spend_ledger_sha256"] = v1.sha(LEDGER)
    report["spend_usd"] = json.loads(LEDGER.read_text())["total_usd"]
    v1.write(RESULT, report)


def run_readers(report, protocol):
    tasks = [(r, a) for r in report["rows"] for a in protocol["arms"]
             if r["systems"][SYSTEM]["arms"][a]["reader"]["status"] == "pending"]
    random.Random(20261009).shuffle(tasks)
    api = v1.client()

    def run(task):
        row, name = task
        arm = row["systems"][SYSTEM]["arms"][name]
        return row, name, v1.api_call(api, protocol, v1.request(protocol, arm["reader_input"], "reader"), "reader:" + row["qid"] + ":" + name)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for number, future in enumerate(as_completed([executor.submit(run, t) for t in tasks]), 1):
            row, name, call = future.result()
            arm = row["systems"][SYSTEM]["arms"][name]
            parsed = v1.parse_answer(call["raw_text"]) if call["status"] in ("ok", "completed") else None
            arm["reader"] = {"status": "ok" if parsed else "error", "raw_text": call["raw_text"], "parsed": parsed, "call": call}
            arm["scores"] = v1.scores(arm)
            save_report(report)
            print("reader_completed", number, name, "status", arm["reader"]["status"], "spend_usd", round(report["spend_usd"], 4), flush=True)


def run_judges(report, protocol):
    dev = {r["qid"] for r in report["rows"]}
    raw = {q["question_id"]: q for q in json.loads(v1.SOURCE.read_text()) if q["question_id"] in dev}
    tasks = [(r, a) for r in report["rows"] for a in protocol["arms"]
             if r["systems"][SYSTEM]["arms"][a]["judge"]["status"] == "pending" and r["systems"][SYSTEM]["arms"][a]["reader"]["status"] != "pending"]
    random.Random(20261009).shuffle(tasks)
    api = v1.client()

    def run(task):
        row, name = task
        arm = row["systems"][SYSTEM]["arms"][name]
        blind_id = v1.digest("judge-blind-v1:" + row["qid"] + ":" + SYSTEM + ":" + name)
        if arm["reader"]["status"] != "ok":
            return row, name, {"status": "reader_failure", "raw_text": "", "score": False, "blind_id": blind_id}
        prompt = v1.judge_prompt(protocol, row, arm["reader"]["parsed"]["answer"], raw[row["qid"]]["answer"])
        call = v1.api_call(api, protocol, v1.request(protocol, prompt, "judge"), blind_id)
        text = call["raw_text"].strip().lower()
        return row, name, {"status": "ok" if call["status"] == "ok" and text in ("yes", "no") else "error",
                           "raw_text": call["raw_text"], "score": bool(call["status"] == "ok" and text == "yes"), "blind_id": blind_id, "call": call}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for number, future in enumerate(as_completed([executor.submit(run, t) for t in tasks]), 1):
            row, name, judged = future.result()
            arm = row["systems"][SYSTEM]["arms"][name]
            arm["judge"] = judged
            arm["scores"] = v1.scores(arm)
            save_report(report)
            print("judge_completed", number, name, "status", judged["status"], "spend_usd", round(report["spend_usd"], 4), flush=True)


def instability(report, protocol):
    rows = {r["qid"]: r for r in report["rows"]}
    if not report["instability"]["entries"]:
        v1_entries = {(e["qid"], e["arm"]): e for e in json.loads(PREV_RESULT.read_text())["instability"]["entries"]}
        for i, qid in enumerate(protocol["instability"]["sample_qids"]):
            name = protocol["arms"][i % 3]
            entry = {"qid": qid, "arm": name, "system": SYSTEM, "outcomes": []}
            if rows[qid]["systems"][SYSTEM]["arms"][name].get("copied_from_v1"):
                entry = copy.deepcopy(v1_entries[(qid, name)])
            report["instability"]["entries"].append(entry)
    tasks = [(e, i) for e in report["instability"]["entries"] for i in range(20) if i not in {o["repeat_index"] for o in e["outcomes"]}]
    random.Random(20261009).shuffle(tasks)
    api = v1.client()
    raw = {q["question_id"]: q for q in json.loads(v1.SOURCE.read_text()) if q["question_id"] in set(rows)}

    def run(task):
        entry, i = task
        row = rows[entry["qid"]]
        arm = row["systems"][SYSTEM]["arms"][entry["arm"]]
        primary = arm["judge"]
        if "call" in primary:
            req = primary["call"]["request"]
        else:
            answer = arm["reader"]["parsed"]["answer"] if arm["reader"]["parsed"] else "Reader failure: no valid answer."
            req = v1.request(protocol, v1.judge_prompt(protocol, row, answer, raw[row["qid"]]["answer"]), "judge")
        call = v1.api_call(api, protocol, req, primary["blind_id"] + ":repeat:" + str(i))
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


def rebuild_ledger(report, protocol):
    """Finalize the variant ledger as exactly the paid calls the report references: each entry
    comes from the variant ledger, else from the v1 ledger (a judge or repeat whose prompt
    matched a v1 call was served from the shared cache), else is reconstructed from the cached
    call record with the cache file's modification time as its timestamp."""
    tokenizer = tiktoken.get_encoding("o200k_base")
    known = {}
    for path in (V1_LEDGER, PREV_LEDGER, LEDGER):
        if path.exists():
            for e in json.loads(path.read_text())["entries"]:
                known.setdefault(e["cache_key"], e)
    referenced = []  # (call, call_id)
    for row in report["rows"]:
        for name, arm in row["systems"][SYSTEM]["arms"].items():
            reader = arm["reader"]
            base = "reader:" + row["qid"] + ":" + name
            attempts = reader.get("prior_attempts", []) + ([reader["call"]] if reader.get("call") else [])
            for index, attempt in enumerate(attempts):
                call_id = base if index == 0 else f"{base}:restored-credit-attempt:{index}"
                if attempt["status"] != "budget_exhausted":
                    referenced.append((attempt, call_id))
            if arm["judge"].get("call"):
                referenced.append((arm["judge"]["call"], arm["judge"]["blind_id"]))
    for entry in report["instability"]["entries"]:
        blind = v1.digest("judge-blind-v1:" + entry["qid"] + ":" + entry["system"] + ":" + entry["arm"])
        for outcome in entry["outcomes"]:
            referenced.append((outcome["call"], blind + ":repeat:" + str(outcome["repeat_index"])))
    entries, seen, reconstructed = [], set(), 0
    for call, call_id in referenced:
        key = call["cache_key"]
        if key in seen:
            continue
        seen.add(key)
        if key in known:
            entries.append(known[key])
            continue
        cache_path = v1.CACHE / (key + ".json")
        estimated = len(tokenizer.encode(call["request"]["input"])) + 512
        reserve = estimated * 2.5e-6 + call["request"]["max_output_tokens"] * 15e-6
        entries.append({"cache_key": key, "call_id": call_id, "category": "reader_judge_dev", "status": call["status"],
                        "reserved_usd": reserve, "cost_usd": call["cost_usd"], "unix_time": cache_path.stat().st_mtime,
                        "usage": call["usage"], "reconstructed_from_cache": True})
        reconstructed += 1
    entries.sort(key=lambda e: e["unix_time"])
    caps = json.loads(V1_LEDGER.read_text())["caps_usd"]
    by_category = {c: 0.0 for c in caps if c != "total"}
    for e in entries:
        by_category[e["category"]] += e["cost_usd"]
    v1_keys = {e["cache_key"] for e in json.loads(V1_LEDGER.read_text())["entries"]}
    prev_keys = v1_keys | {e["cache_key"] for e in json.loads(PREV_LEDGER.read_text())["entries"]}
    new_spend = sum(e["cost_usd"] for e in entries if e["cache_key"] not in prev_keys)
    v1.write(LEDGER, {"schema_version": 1, "caps_usd": caps, "entries": entries, "by_category_usd": by_category,
                      "total_usd": sum(by_category.values()),
                      "provenance": {"v1_ledger_sha256": v1.sha(V1_LEDGER), "referenced_calls": len(entries),
                                     "from_v1": sum(e["cache_key"] in v1_keys for e in entries),
                                     "from_previous": sum(e["cache_key"] in prev_keys for e in entries), "previous": PREV_NAME,
                                     "reconstructed": reconstructed, "new_spend_usd": new_spend}})
    report["new_spend_usd"] = new_spend


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("prepare", "readers", "judges", "instability", "finalize"))
    parser.add_argument("--variant", choices=("v2", "v2b", "v3"), default="v2")
    args = parser.parse_args()
    select_variant(args.variant)
    protocol = v1.load_protocol()
    v1.LEDGER = LEDGER  # api_call records every new call in the variant's ledger
    if args.stage == "prepare":
        prepare()
        return
    report = json.loads(RESULT.read_text())
    assert report["protocol_sha256"] == v1.sha(v1.PROTOCOL) and report["reader_systems"] == [SYSTEM] and report["variant"] == VARIANT
    if args.stage == "readers":
        run_readers(report, protocol)
    elif args.stage == "judges":
        run_judges(report, protocol)
    elif args.stage == "instability":
        instability(report, protocol)
    else:
        assert all(a["reader"]["status"] != "pending" and a["judge"]["status"] != "pending" for r in report["rows"] for a in r["systems"][SYSTEM]["arms"].values())
        assert sum(len(e["outcomes"]) for e in report["instability"]["entries"]) == 400
        for entry in report["instability"]["entries"]:
            assert {o["repeat_index"] for o in entry["outcomes"]} == set(range(20))
        rebuild_ledger(report, protocol)
        report["complete"] = True
        save_report(report)
    print(json.dumps({"stage": args.stage, "spend_usd": report["spend_usd"]}), flush=True)


if __name__ == "__main__":
    main()
