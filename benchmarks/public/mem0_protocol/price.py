# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Price a memory-benchmarks run from its --predict-only output, without calling a model.

For every saved prediction the answer prompt is rebuilt with the harness's own prompt
builder (imported from the pinned checkout) on exactly the memories the answerer would
see, and counted with o200k_base. Judge prompts are rebuilt with a stand-in answer of
the calibrated mean length. Completion tokens (gpt-5 bills hidden reasoning as output)
come from a proxy call log (openai_calls.jsonl) of a real run, split into answer and
judge calls; ``--calibrate N`` makes that log for N saved predictions of this benchmark
(answer + judge through the harness's own LLMClient, so the same parameters).

Run with the harness interpreter:
    python price.py --clone CLONE --benchmark longmemeval --predict DIR \
        --calls RUN/openai_calls.jsonl [--calibrate 2 --proxy http://127.0.0.1:PORT/v1]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

PRICE = {"input": 1.25, "output": 10.0}  # gpt-5 list price, USD per 1M tokens (no cache, no batch)


def load(predict: Path) -> list[dict]:
    rows = []
    for p in sorted(predict.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text())
        if "retrieval" in d and "question" in d:
            rows.append(d)
    return rows


def builders(benchmark: str):
    """(answer_prompt(row) -> str, judge_prompts(row, answer) -> list[str]) from the harness."""
    if benchmark == "locomo":
        from benchmarks.locomo import prompts as P

        def answer(row):
            return P.get_answer_generation_prompt(row["question"], row["retrieval"]["search_results"][:200],
                                                  reference_date=row.get("reference_date"))

        def judge(row, ans):
            gold = P.preprocess_answer(row["category"], str(row["ground_truth_answer"]))
            return [P.JUDGE_SYSTEM_PROMPT + P.get_judge_prompt(row["category"], row["question"], gold, ans)]
        return answer, judge
    if benchmark == "longmemeval":
        from benchmarks.longmemeval import prompts as P
        from benchmarks.longmemeval import run as R

        def qdate(row):
            return R.parse_longmemeval_date_human(row["question_date"]) if row.get("question_date") else ""

        def answer(row):
            mems = sorted(row["retrieval"]["search_results"][:200], key=lambda x: x.get("created_at") or "")
            return P.get_answer_generation_prompt(question=row["question"], search_results=mems,
                                                  question_date=qdate(row))

        def judge(row, ans):
            return [P.get_judge_prompt(question_type=row["question_type"], question_id=row["question_id"],
                                       question=row["question"], answer=row["ground_truth_answer"],
                                       response=ans, question_date=qdate(row))]
        return answer, judge
    from benchmarks.beam import prompts as P
    from benchmarks.beam import run as R

    def answer(row):
        mems = sorted(row["retrieval"]["search_results"][:200], key=lambda m: m.get("created_at", "") or "")
        return P.get_beam_answer_generation_prompt(row["question"], mems, top_k=200)

    def judge(row, ans):
        nuggets = R.extract_rubric_nuggets(row) or [""]
        calls = [P.BEAM_JUDGE_SYSTEM_PROMPT + P.get_beam_nugget_judge_prompt(row["question"], n, ans) for n in nuggets]
        if row.get("question_type") == "event_ordering":
            # one fact extraction plus one alignment per extracted event (assumed = nugget count)
            calls.append(P.get_beam_fact_extraction_prompt(ans))
            calls += [P.get_beam_event_alignment_prompt(n, nuggets) for n in nuggets]
        return calls
    return answer, judge


def calibration(calls: Path) -> tuple[float, float, int, int]:
    """Mean completion tokens of answer and judge calls in a proxy log.

    A call tagged by --calibrate carries its kind; otherwise structured (json_object) calls
    are judge calls, which holds for the LoCoMo and BEAM harnesses (their answerer is plain
    text). LongMemEval's yes/no judge is plain text too, so its log must come from --calibrate.
    """
    ans, jud = [], []
    for line in calls.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") != 200:
            continue
        kind = r.get("kind") or ("judge" if "response_format" in r["request"] else "answer")
        (jud if kind == "judge" else ans).append(r["usage"]["completion_tokens"])
    if not ans or not jud:
        raise SystemExit(f"{calls}: need answer and judge calls, got {len(ans)} and {len(jud)}")
    return statistics.mean(ans), statistics.mean(jud), len(ans), len(jud)


async def calibrate(benchmark: str, rows: list[dict], n: int, base_url: str, log: Path) -> None:
    """Answer and judge n saved predictions through the harness LLMClient, tagging each call."""
    import os

    from benchmarks.common.llm_client import LLMClient

    os.environ["OPENAI_BASE_URL"] = base_url
    llm = LLMClient(model="gpt-5", provider="openai")
    answer, judge = builders(benchmark)
    for row in rows[:n]:
        before = log.read_text().count("\n") if log.exists() else 0
        ans = await llm.generate(system="", user=answer(row))
        _tag(log, before, "answer")
        for prompt in judge(row, ans or "x")[:3]:
            before = log.read_text().count("\n")
            if benchmark == "longmemeval":
                await llm.judge_yes_no(prompt)
            else:
                await llm.generate_structured(system="", user=prompt)
            _tag(log, before, "judge")


def _tag(log: Path, before: int, kind: str) -> None:
    lines = log.read_text().splitlines()
    for i in range(before, len(lines)):
        r = json.loads(lines[i])
        r["kind"] = kind
        lines[i] = json.dumps(r, sort_keys=True)
    log.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clone", required=True, type=Path)
    ap.add_argument("--benchmark", required=True, choices=["locomo", "longmemeval", "beam"])
    ap.add_argument("--predict", required=True, type=Path, help="harness predicted_<project> directory")
    ap.add_argument("--calls", required=True, type=Path, help="proxy openai_calls.jsonl for completion lengths")
    ap.add_argument("--calibrate", type=int, default=0, help="first make the call log from N predictions")
    ap.add_argument("--proxy", default=None, help="running openai_proxy base URL (with --calibrate)")
    ap.add_argument("--scale-to", type=int, default=None, help="extrapolate to this many questions")
    args = ap.parse_args()
    sys.path.insert(0, str(args.clone.resolve()))
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
    rows = load(args.predict)
    if args.calibrate:
        if not args.proxy:
            raise SystemExit("--calibrate needs --proxy")
        asyncio.run(calibrate(args.benchmark, rows, args.calibrate, args.proxy, args.calls))
    ans_out, judge_out, n_ans, n_jud = calibration(args.calls)
    stand_in = "x " * int(min(ans_out, 400))
    answer, judge = builders(args.benchmark)
    a_in, j_in, j_calls, mems = [], [], [], []
    for row in rows:
        a_in.append(len(enc.encode(answer(row))))
        prompts = judge(row, stand_in)
        j_calls.append(len(prompts))
        j_in.append(sum(len(enc.encode(p)) for p in prompts))
        mems.append(len(row["retrieval"]["search_results"]))
    n = len(rows)
    scale = (args.scale_to / n) if args.scale_to else 1.0
    tot_in = (sum(a_in) + sum(j_in)) * scale
    tot_out = (n * ans_out + sum(j_calls) * judge_out) * scale
    usd = (tot_in * PRICE["input"] + tot_out * PRICE["output"]) / 1e6
    print(json.dumps({
        "benchmark": args.benchmark, "questions_measured": n, "scaled_to": args.scale_to or n,
        "memories_per_question_mean": statistics.mean(mems),
        "answer_prompt_tokens": {"mean": statistics.mean(a_in), "median": statistics.median(a_in), "max": max(a_in)},
        "judge_calls_per_question": statistics.mean(j_calls),
        "judge_prompt_tokens_per_question_mean": statistics.mean(j_in),
        "calibration": {"answer_completion_tokens_mean": ans_out, "judge_completion_tokens_mean": judge_out,
                        "answer_calls": n_ans, "judge_calls": n_jud, "log": str(args.calls)},
        "total_input_tokens": round(tot_in), "total_output_tokens": round(tot_out),
        "usd_list_price": round(usd, 2), "usd_if_batch_50pct": round(usd / 2, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
