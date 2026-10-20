# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Lever 1B.5: the local open-weight reader row on the frozen LongMemEval dev 100.

The report starts as the v3 report (matched gpt-5.4 arms, instability and paid calls carried
over byte for byte, so they cost nothing) and adds the reader system `local_qwen30b`: the same
three frozen reader inputs per question, answered by Qwen3-30B-A3B-Instruct-2507 served on Modal
(benchmarks/public/longmemeval_local_modal.py), judged by the official judge exactly as the
matched system is (same prompts, blinding, caching and ledger through longmemeval_reader_api.py).
The two systems are never blended: every summary block is per system.

  uv run python -I benchmarks/public/longmemeval_local_reader.py prepare
  uv run --with modal python -I benchmarks/public/longmemeval_local_reader.py readers --cap 20
  uv run python -I benchmarks/public/longmemeval_local_reader.py judges
  uv run python -I benchmarks/public/longmemeval_local_reader.py finalize
"""
import argparse
import copy
import json
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_longmemeval_answers as checker  # noqa: E402
import longmemeval_reader_api as v1  # noqa: E402

SYSTEM = "local_qwen30b"
MATCHED = "matched_gpt54"
VARIANT = "local30b"
PREV_NAME = "v3"
PREV_RESULT = Path("benchmarks/results/longmemeval_dev_answers_v3.json")
PREV_LEDGER = Path("benchmarks/results/longmemeval_dev_answers_v3_spend.json")
V1_LEDGER = Path("benchmarks/results/longmemeval_dev_answers_v1_spend.json")
RESULT = Path("benchmarks/results/longmemeval_dev_answers_local30b_v1.json")
LEDGER = Path("benchmarks/results/longmemeval_dev_answers_local30b_v1_spend.json")
WORK = Path("target/superiority-validation/local30b")
PROMPTS = WORK / "prompts.json"
OUTPUTS = WORK / "reader_outputs.jsonl"
MODAL_DRIVER = Path("benchmarks/public/longmemeval_local_modal.py")


def spec():
    import longmemeval_local_modal as local

    value = local.system_spec()
    weak = v1.load_protocol()["systems"]["weak_qwen4b"]
    # The pinned 30B revision ships the same tokenizer.json as the frozen 4B reader.
    value.update(tokenizer_path=weak["tokenizer_path"], tokenizer_sha256=weak["tokenizer_sha256"],
                 tokenizer_package=weak["tokenizer_package"], count_template=weak["chat_template"])
    return value


def prepare():
    protocol = v1.load_protocol()
    report = json.loads(PREV_RESULT.read_text())
    assert report["complete"] and report["reader_systems"] == [MATCHED], "v3 report must be the complete matched row"
    system = spec()
    assert v1.sha(Path(system["tokenizer_path"])) == system["tokenizer_sha256"], "tokenizer fingerprint differs"
    qwen = Tokenizer.from_file(system["tokenizer_path"])
    qwen.no_padding()
    qwen.no_truncation()

    def count(text):
        return len(qwen.encode(system["count_template"].format(reader_input=text), add_special_tokens=False).ids)
    # The local system counts its inputs with its own tokenizer, as the weak reader did in v1.
    evidence = json.loads(Path(report["evidence_report_path"]).read_text())
    frozen = {entry["qid"]: entry for entry in evidence["rows"]}
    historical_path = Path(report["evidence_report_path"]).with_name("evidence_completeness_dev.json")
    historical = {entry["qid"]: set(entry["ids"]) for entry in json.loads(historical_path.read_text())["rows"]}
    questions = {q["question_id"]: q for q in json.loads(v1.SOURCE.read_text()) if q["question_id"] in frozen}
    for row in report["rows"]:
        groups = checker.reconstruct_groups(questions[row["qid"]], frozen[row["qid"]], historical[row["qid"]])
        arms = {}
        for name, matched in row["systems"][MATCHED]["arms"].items():
            arm = copy.deepcopy(matched)
            for key in ("copied_from_v1", "copied_from"):
                arm.pop(key, None)
            arm["input_tokens"] = count(arm["reader_input"])
            arm["uncapped_input_tokens"] = count(checker.reader_input(questions[row["qid"]], groups[name], protocol))
            failed = arm["reader"]["status"] == "selection_failure"
            arm["reader"] = {"status": "selection_failure" if failed else "pending", "raw_text": "", "parsed": None}
            arm["judge"] = {"status": "pending", "raw_text": "", "score": False}
            arm["scores"] = v1.scores(arm)
            arms[name] = arm
        row["systems"][SYSTEM] = {"arms": arms}
    report.update(reader_systems=[MATCHED, SYSTEM], variant=VARIANT, lever=(
        "1B.5 local reader row: Qwen3-30B-A3B-Instruct-2507 on Modal over the v3 evidence selection; "
        "the matched gpt-5.4 system is carried over from v3 unchanged and reported separately"),
        previous_variant=PREV_NAME, previous_result_path=str(PREV_RESULT), previous_result_sha256=v1.sha(PREV_RESULT),
        local_reader_system=system, spend_ledger_path=str(LEDGER), complete=False)
    for key in ("paired_vs_v1", "paired_vs_v2b", "reader_context_tokens", "abstention_accounting", "local_reader"):
        report.pop(key, None)
    paths = {artifact["path"] for artifact in report["artifacts"]}
    for path in (Path(__file__), MODAL_DRIVER, Path("benchmarks/public/longmemeval_modal.py"), PREV_RESULT):
        if str(path) not in paths:
            report["artifacts"].append({"path": str(path), "sha256": v1.sha(path)})
    WORK.mkdir(parents=True, exist_ok=True)
    prompts = [{"key": f"{row['qid']}:{name}", "input": arm["reader_input"]}
               for row in report["rows"] for name, arm in row["systems"][SYSTEM]["arms"].items()
               if arm["reader"]["status"] == "pending"]
    v1.write(PROMPTS, prompts)
    seed_ledger(report)
    save_report(report)
    print(json.dumps({"prepared": len(report["rows"]), "prompts": len(prompts)}), flush=True)


def seed_ledger(report):
    """Start the variant ledger from the v3 ledger: every paid call the carried-over matched
    system references is already recorded there."""
    previous = json.loads(PREV_LEDGER.read_text())
    v1.write(LEDGER, {"schema_version": 1, "caps_usd": previous["caps_usd"], "entries": previous["entries"],
                      "by_category_usd": previous["by_category_usd"], "total_usd": previous["total_usd"],
                      "provenance": {"previous": PREV_NAME, "previous_ledger_sha256": v1.sha(PREV_LEDGER)}})


def run_readers(cap):
    report = json.loads(RESULT.read_text())
    command = [sys.executable, "-I", str(MODAL_DRIVER), "--prompts", str(PROMPTS), "--out", str(OUTPUTS),
               "--cap", str(cap), "--batch", "100"]
    subprocess.run(command, check=True)
    import_outputs(report)


def import_outputs(report):
    import longmemeval_local_modal as local

    records = {}
    for line in OUTPUTS.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["key"]] = record
    imported = 0
    for row in report["rows"]:
        for name, arm in row["systems"][SYSTEM]["arms"].items():
            record = records.get(f"{row['qid']}:{name}")
            if arm["reader"]["status"] != "pending" or record is None:
                continue
            request = local.request_for(arm["reader_input"])
            assert record["request_sha256"] == v1.digest(v1.canonical(request)), "reader output for a different request"
            parsed = v1.parse_answer(record["raw_text"]) if record["status"] == "completed" else None
            call = {"status": record["status"], "raw_text": record["raw_text"], "response": None, "usage": None,
                    "cost_usd": record["cost_usd"], "gpu_seconds": record["gpu_seconds"],
                    "batch_gpu_seconds": record["batch_gpu_seconds"], "batch_wall_seconds": record["batch_wall_seconds"],
                    "batch_size": record["batch_size"], "unix_time": record["unix_time"], "request": request,
                    "request_sha256": record["request_sha256"], "cache_key": record["request_sha256"]}
            arm["reader"] = {"status": "ok" if parsed else "error", "raw_text": record["raw_text"], "parsed": parsed,
                             "call": call}
            arm["scores"] = v1.scores(arm)
            imported += 1
    save_report(report)
    print(json.dumps({"imported": imported}), flush=True)


def run_judges():
    report = json.loads(RESULT.read_text())
    protocol = v1.load_protocol()
    dev = {row["qid"] for row in report["rows"]}
    raw = {q["question_id"]: q for q in json.loads(v1.SOURCE.read_text()) if q["question_id"] in dev}
    tasks = [(row, name) for row in report["rows"] for name in protocol["arms"]
             if row["systems"][SYSTEM]["arms"][name]["judge"]["status"] == "pending"
             and row["systems"][SYSTEM]["arms"][name]["reader"]["status"] != "pending"]
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
                           "raw_text": call["raw_text"], "score": bool(call["status"] == "ok" and text == "yes"),
                           "blind_id": blind_id, "call": call}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for number, future in enumerate(as_completed([executor.submit(run, t) for t in tasks]), 1):
            row, name, judged = future.result()
            arm = row["systems"][SYSTEM]["arms"][name]
            arm["judge"] = judged
            arm["scores"] = v1.scores(arm)
            if number % 25 == 0 or number == len(tasks):
                save_report(report)
            print("judge_completed", number, name, judged["status"], flush=True)
    save_report(report)


def rebuild_ledger(report):
    """The variant ledger is exactly the paid API calls the report references (the matched
    system's from v1 and v3, the local system's judges from the session ledger)."""
    known = {}
    for path in (V1_LEDGER, PREV_LEDGER, v1.LEDGER):
        if path.exists():
            for entry in json.loads(path.read_text())["entries"]:
                known.setdefault(entry["cache_key"], entry)
    referenced = []
    for row in report["rows"]:
        for system in (MATCHED, SYSTEM):
            for name, arm in row["systems"][system]["arms"].items():
                reader = arm["reader"]
                if system == MATCHED:
                    base = "reader:" + row["qid"] + ":" + name
                    attempts = reader.get("prior_attempts", []) + ([reader["call"]] if reader.get("call") else [])
                    for index, attempt in enumerate(attempts):
                        if attempt["status"] != "budget_exhausted":
                            referenced.append((attempt, base if index == 0 else f"{base}:restored-credit-attempt:{index}"))
                if arm["judge"].get("call"):
                    referenced.append((arm["judge"]["call"], arm["judge"]["blind_id"]))
    for entry in report["instability"]["entries"]:
        blind = v1.digest("judge-blind-v1:" + entry["qid"] + ":" + entry["system"] + ":" + entry["arm"])
        for outcome in entry["outcomes"]:
            referenced.append((outcome["call"], blind + ":repeat:" + str(outcome["repeat_index"])))
    entries, seen = [], set()
    for call, call_id in referenced:
        key = call["cache_key"]
        if key in seen:
            continue
        seen.add(key)
        assert key in known, f"paid call {call_id} missing from every ledger"
        entries.append(known[key])
    entries.sort(key=lambda entry: entry["unix_time"])
    previous = json.loads(PREV_LEDGER.read_text())
    caps = previous["caps_usd"]
    by_category = {category: 0.0 for category in caps if category != "total"}
    for entry in entries:
        by_category[entry["category"]] += entry["cost_usd"]
    prev_keys = {entry["cache_key"] for entry in previous["entries"]}
    new_spend = sum(entry["cost_usd"] for entry in entries if entry["cache_key"] not in prev_keys)
    v1.write(LEDGER, {"schema_version": 1, "caps_usd": caps, "entries": entries, "by_category_usd": by_category,
                      "total_usd": sum(by_category.values()),
                      "provenance": {"previous": PREV_NAME, "previous_ledger_sha256": v1.sha(PREV_LEDGER),
                                     "referenced_calls": len(entries),
                                     "from_previous": sum(entry["cache_key"] in prev_keys for entry in entries),
                                     "new_spend_usd": new_spend}})
    report["new_spend_usd"] = new_spend


def local_row(report):
    """The local system's own line: accuracy, citations, abstentions, context and cost."""
    tokenizer = tiktoken.get_encoding("o200k_base")
    system = report["local_reader_system"]
    qwen = Tokenizer.from_file(system["tokenizer_path"])
    qwen.no_padding()
    qwen.no_truncation()
    out = {}
    for name in report["rows"][0]["systems"][SYSTEM]["arms"]:
        arms = [row["systems"][SYSTEM]["arms"][name] for row in report["rows"]]
        values = [arm["scores"] for arm in arms]
        o200k = sorted(len(tokenizer.encode(arm["reader_input"], disallowed_special=())) for arm in arms)
        counts = sorted(len(qwen.encode(system["count_template"].format(reader_input=arm["reader_input"]),
                                        add_special_tokens=False).ids) for arm in arms)
        calls = [arm["reader"].get("call") for arm in arms if arm["reader"].get("call")]
        outputs = [len(tokenizer.encode(arm["reader"]["raw_text"])) for arm in arms if arm["reader"].get("raw_text")]
        matched = [row["systems"][MATCHED]["arms"][name]["scores"]["correct"] for row in report["rows"]]
        deltas = [int(v["correct"]) - int(m) for v, m in zip(values, matched)]
        out[name] = {
            "n": len(values), "correct_count": sum(v["correct"] for v in values),
            "reader_failures": sum(arm["reader"]["status"] == "error" for arm in arms),
            "selection_failures": sum(arm["reader"]["status"] == "selection_failure" for arm in arms),
            "abstentions": sum(v["abstain"] for v in values),
            "false_abstentions": sum(v["abstain"] and not v["correct"] for v in values),
            "true_abstentions": sum(v["abstain"] and v["correct"] for v in values),
            "wrong_answers": sum((not v["abstain"]) and not v["correct"] for v in values),
            "unsupported_answers": sum((not v["abstain"]) and not (v["correct"] and v["all_claims_supported"]) for v in values),
            "reader_context_tokens_o200k": {"median": v1.percentile(o200k, .5), "p95": v1.percentile(o200k, .95), "max": o200k[-1]},
            "reader_context_tokens_qwen": {"median": v1.percentile(counts, .5), "p95": v1.percentile(counts, .95), "max": counts[-1]},
            "reader_output_tokens_o200k": {"median": v1.percentile(sorted(outputs), .5) if outputs else None},
            "gpu_seconds_per_answer_amortized": sum(c["gpu_seconds"] for c in calls) / len(calls) if calls else None,
            "modal_cost_usd": sum(c["cost_usd"] for c in calls),
            "paired_vs_matched_gpt54_same_arm": checker.paired_interval(deltas),
        }
    batches = {}
    for row in report["rows"]:
        for arm in row["systems"][SYSTEM]["arms"].values():
            call = arm["reader"].get("call")
            if call:
                # One entry per Modal batch: its prompts share the batch's recorded timing.
                batches[(round(call["batch_gpu_seconds"], 3), round(call["batch_wall_seconds"], 3), call["batch_size"])] = call
    out["modal_batches"] = [{"batch_size": c["batch_size"], "gpu_seconds": c["batch_gpu_seconds"],
                             "wall_seconds": c["batch_wall_seconds"], "usd": c["batch_gpu_seconds"] / 3600
                             * system["usd_per_hour"] * system["overhead"]} for _, c in sorted(batches.items())]
    out["modal_cost_usd_total"] = sum(b["usd"] for b in out["modal_batches"])
    out["note"] = ("one system per line: these values describe the local reader only; the matched gpt-5.4 "
                   "system is carried over from v3 and reported under its own name in summary")
    return out


def save_report(report):
    report["summary"] = checker.recompute_summary(report["rows"], tuple(report["reader_systems"]))
    report["local_reader"] = local_row(report)
    report["spend_ledger_sha256"] = v1.sha(LEDGER)
    report["spend_usd"] = json.loads(LEDGER.read_text())["total_usd"]
    v1.write(RESULT, report)


def finalize():
    report = json.loads(RESULT.read_text())
    protocol = v1.load_protocol()
    pending = [(row["qid"], name) for row in report["rows"] for name in protocol["arms"]
               if row["systems"][SYSTEM]["arms"][name]["reader"]["status"] == "pending"
               or row["systems"][SYSTEM]["arms"][name]["judge"]["status"] == "pending"]
    assert not pending, f"{len(pending)} local arms still pending"
    rebuild_ledger(report)
    report["complete"] = True
    # The driver files are fingerprinted as they are at completion (they may change between stages).
    own = {str(Path(__file__)), str(MODAL_DRIVER), "benchmarks/public/check_longmemeval_answers.py"}
    report["artifacts"] = [artifact for artifact in report["artifacts"] if artifact["path"] not in own]
    report["artifacts"].extend({"path": path, "sha256": v1.sha(Path(path))} for path in sorted(own))
    save_report(report)
    print(json.dumps({"complete": True, "local": {name: value for name, value in report["local_reader"].items()
                                                     if name in protocol["arms"]}, "new_spend_usd": report["new_spend_usd"]},
                     default=str)[:2000], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "readers", "import", "judges", "finalize"))
    parser.add_argument("--cap", type=float, default=20.0)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "readers":
        run_readers(args.cap)
    elif args.stage == "import":
        import_outputs(json.loads(RESULT.read_text()))
    elif args.stage == "judges":
        run_judges()
    else:
        finalize()


if __name__ == "__main__":
    main()
