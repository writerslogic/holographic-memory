# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "tiktoken==0.14.0", "openai==2.8.1", "tqdm==4.67.1", "backoff==2.2.1"]
# ///
"""LongMemEval dev QA under the official generation and judge protocol.

Generation mirrors xiaowu0162/LongMemEval src/generation/run_generation.py at 9e0b455 with the
flat-turn retriever: each ranked user turn expands to its round (that turn and the next one), the
top-k rounds are stripped of `has_answer`, sorted by session date, serialized as JSON, and given to
the reader with the chain-of-note template (the official default reading method), temperature 0,
800 output tokens, retrieval history truncated to 128000 - 800 - 1000 tokens. The judge is the
official `get_anscheck_prompt` from src/evaluation/evaluate_qa.py, temperature 0, 10 tokens,
correct iff the reply contains "yes".

The ranking is the reference pipeline (longmemeval_pipeline.rank_question) over cached Qwen3 scores
with the tuned configuration that the native StageParams defaults mirror. Exact score ties are broken
by a hash of the turn id, never by the id text (official ids encode evidence labels). The source file
is parsed once and filtered to the 100 frozen dev ids before any ranking or prompt is built; held-out
questions never reach retrieval, the reader or the judge.

`--batch` sends the same request bodies through the OpenAI Batch API (24h window, half price): the
reader stage first, then the judge stage over the reader texts. Each stage persists its input file,
batch id and custom-id to cache-key map before polling, so rerunning the same command resumes it;
results land in the same prompt-hash call cache the synchronous path reads. Failed or missing lines
are reported and stop the run unless `--sync-retry` sends them synchronously at the full rate.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import longmemeval_pipeline as P  # noqa: E402

SPLIT = HERE / "longmemeval_split.json"
SOURCE = Path.home() / ".cache/hms-bench/downloads/longmemeval_s_cleaned.json"
SOURCE_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
EVALUATE_QA = Path.home() / ".cache/hms-bench/downloads/longmemeval_evaluate_qa_9e0b455f4ef0.py"
RUN_DIR = Path.home() / ".cache/hms-bench/longmemeval_s/runs/s-dev-small-all2"
CACHE = Path.home() / ".cache/hms-bench/longmemeval_official_calls"
LEDGER = ROOT / "benchmarks/results/longmemeval_official_spend.json"
SPEND_CAP_USD = 60.0
JUDGE_MODEL = "gpt-4o-2024-08-06"
READERS = {"gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"}
# USD per token: input, cached input, output.
PRICES = {"gpt-4o-2024-08-06": (2.5e-6, 1.25e-6, 10e-6), "gpt-4o-mini-2024-07-18": (0.15e-6, 0.075e-6, 0.6e-6)}
# Batch API: half the input and output rates. Cached input tokens are billed here at the batch input
# rate, an upper bound that does not depend on whether a batch discount stacks with prompt caching.
BATCH_PRICES = {model: (pin / 2, pin / 2, pout / 2) for model, (pin, _, pout) in PRICES.items()}
BATCH_STATE = Path.home() / ".cache/hms-bench/longmemeval_official_batches"
BATCH_DONE = {"completed", "failed", "expired", "cancelled"}
CHAT_URL = "/v1/chat/completions"
GEN_LENGTH = 800
MAX_RETRIEVAL_TOKENS = 128000 - GEN_LENGTH - 1000
TEMPLATE = ("I will give you several history chats between you and a user. Please answer the question based on the "
            "relevant chat history. Answer the question step by step: first extract all the relevant information, "
            "and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\n"
            "Question: {}\nAnswer (step by step):")
LOCK = threading.Lock()
ENC = tiktoken.get_encoding("o200k_base")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dev_ids() -> set[str]:
    split = json.loads(SPLIT.read_text())
    dev, heldout = set(split["dev"]), set(split["heldout"])
    if len(dev) != 100 or len(heldout) != 400 or dev & heldout:
        raise ValueError("invalid frozen dev100/heldout400 split")
    return dev


def load_dev(source: Path = SOURCE) -> list[dict]:
    """Raw dev entries only; the held-out questions are never decoded into the result."""
    data = source.read_bytes()
    if sha_bytes(data) != SOURCE_SHA256:
        raise ValueError("LongMemEval-S source differs from the pinned release")
    dev = dev_ids()
    entries = [e for e in json.loads(data) if e["question_id"] in dev]
    if {e["question_id"] for e in entries} != dev or len(entries) != 100:
        raise ValueError("dev questions missing from source")
    return entries


def _label_free_ranks(scores: dict[str, float]) -> dict[str, int]:
    def key(t):
        return -scores[t], hashlib.sha256(t.encode()).digest()

    return {t: r for r, t in enumerate(sorted(scores, key=key))}


def rankings(entries: list[dict], sub_weight: float, round_weight: float = 0.0) -> dict[str, list[str]]:
    """Reference turn ranking per dev question; ids are official corpus ids used only as labels."""
    P._ranks = _label_free_ranks
    dev = {e["question_id"] for e in entries}
    if not dev <= dev_ids():
        raise ValueError("ranking requested for a non-dev question")
    questions = P.load(SOURCE, dev)
    scores = P.load_scores(RUN_DIR / "scores.json")
    targets = json.loads((RUN_DIR / "targets.json").read_text())
    qfile = json.loads((RUN_DIR / "queries.json").read_text())
    rerank = json.loads((RUN_DIR / "rerank.json").read_text())
    cfg = json.loads((HERE / "longmemeval_config.json").read_text())
    cfg["turn"]["variants"]["sub"] = sub_weight
    if round_weight:
        cfg["turn"]["kinds"]["round"] = round_weight
    out = {}
    for q in questions:
        trange = P.time_range(q, qfile["by_key"], qfile["llm"])
        session = P.rank_question(q, scores[q.qid], targets[q.qid], cfg["session"], "session", trange,
                                  rerank.get("session", {}).get(q.qid))
        out[q.qid] = P.rank_question(q, scores[q.qid], targets[q.qid], cfg["turn"], "turn", trange,
                                     rerank.get("turn", {}).get(q.qid), session)
    if set(out) != dev:
        raise ValueError("ranking does not cover the dev questions")
    return out


def official_prompt(entry: dict, ranked: list[str], topk: int) -> str:
    """run_generation.prepare_prompt with flat-turn, json, cot=true, merge none, useronly false."""
    by_id, dates = {}, {}
    for date, sid, session in zip(entry["haystack_dates"], entry["haystack_session_ids"], entry["haystack_sessions"]):
        by_id[sid], dates[sid] = session, date
    chunks = []
    for corpus_id in ranked:
        converted = corpus_id.replace("noans_", "answer_")
        sid, turn = "_".join(converted.split("_")[:-1]), int(converted.split("_")[-1]) - 1
        if sid not in by_id or turn >= len(by_id[sid]):
            continue
        round_ = [by_id[sid][turn]]
        if turn + 1 < len(by_id[sid]):
            round_.append(by_id[sid][turn + 1])
        chunks.append((dates[sid], copy.deepcopy(round_)))
    chunks = chunks[:topk]
    for _, session in chunks:
        for turn in session:
            turn.pop("has_answer", None)
    chunks.sort(key=lambda chunk: chunk[0])
    history = ""
    for i, (date, session) in enumerate(chunks):
        history += "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(
            i + 1, date, "\n" + json.dumps(session))
    tokens = ENC.encode(history, allowed_special={"<|endoftext|>"})
    if len(tokens) > MAX_RETRIEVAL_TOKENS:
        history = ENC.decode(tokens[:MAX_RETRIEVAL_TOKENS])
    if "has_answer" in history:
        raise ValueError("evidence label reached the reader input")
    return TEMPLATE.format(history, entry["question_date"], entry["question"])


def load_judge_prompt():
    spec = importlib.util.spec_from_file_location("lme_evaluate_qa", EVALUATE_QA)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_anscheck_prompt


def _ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"schema_version": 1, "cap_usd": SPEND_CAP_USD, "prices_usd_per_token": PRICES,
            "total_usd": 0.0, "calls": 0, "by_model_usd": {}, "by_run_usd": {}}


def request_body(model: str, prompt: str, max_tokens: int) -> dict:
    """The chat completion request both the synchronous and the batch path send."""
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "n": 1,
            "temperature": 0, "max_tokens": max_tokens}


def cache_key(request: dict, repeat: int = 0) -> str:
    # A repeat index asks for a fresh sample of the same request (temperature 0 is not deterministic).
    return sha_bytes(json.dumps({**request, **({"repeat": repeat} if repeat else {})}, sort_keys=True).encode())


def _reserve(usd: float) -> None:
    with LOCK:
        ledger = _ledger()
        if ledger["total_usd"] + usd > SPEND_CAP_USD:
            raise RuntimeError("spend cap reached")
        ledger["total_usd"] += usd
        LEDGER.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n")


def _settle(run: str, model: str, reserve: float, cost: float, calls: int, batch: dict | None = None) -> None:
    with LOCK:
        ledger = _ledger()
        ledger["total_usd"] += cost - reserve
        ledger["calls"] += calls
        ledger["by_model_usd"][model] = ledger["by_model_usd"].get(model, 0.0) + cost
        ledger["by_run_usd"][run] = ledger["by_run_usd"].get(run, 0.0) + cost
        if batch is not None:
            ledger.setdefault("batches", []).append(batch)
        LEDGER.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n")


def _worst_case(prices: dict, model: str, prompt: str, max_tokens: int) -> float:
    price_in, _, price_out = prices[model]
    return (len(ENC.encode(prompt)) + 16) * price_in + max_tokens * price_out


def call(client, run: str, model: str, prompt: str, max_tokens: int, repeat: int = 0) -> dict:
    request = request_body(model, prompt, max_tokens)
    path = CACHE / f"{cache_key(request, repeat)}.json"
    if path.exists():
        return json.loads(path.read_text())
    price_in, price_cached, price_out = PRICES[model]
    reserve = _worst_case(PRICES, model, prompt, max_tokens)
    _reserve(reserve)
    cost = reserve
    try:
        for attempt in range(4):
            try:
                completion = client.chat.completions.create(**request)
                break
            except Exception:
                if attempt == 3:
                    raise
                threading.Event().wait(2 ** attempt * 5)
        usage = completion.usage
        cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        cost = (usage.prompt_tokens - cached) * price_in + cached * price_cached + usage.completion_tokens * price_out
        out = {"text": completion.choices[0].message.content.strip(), "model": completion.model,
               "prompt_tokens": usage.prompt_tokens, "cached_tokens": cached,
               "completion_tokens": usage.completion_tokens, "cost_usd": cost}
    finally:
        _settle(run, model, reserve, cost, 1)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def batch_lines(model: str, prompts: dict[str, str], max_tokens: int, repeat: int) -> tuple[list[str], dict[str, str]]:
    """JSONL request lines for the uncached requests and their custom-id to cache-key map."""
    pending = {}
    for prompt in prompts.values():
        key = cache_key(request_body(model, prompt, max_tokens), repeat)
        if not (CACHE / f"{key}.json").exists():
            pending[key] = prompt
    ids = {f"r{i:04d}": key for i, key in enumerate(sorted(pending))}
    lines = [json.dumps({"custom_id": cid, "method": "POST", "url": CHAT_URL,
                         "body": request_body(model, pending[key], max_tokens)}) for cid, key in ids.items()]
    return lines, ids


def _write_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    tmp.replace(path)


def _line_result(model: str, record: dict) -> tuple[dict | None, object]:
    """(cache entry, None) for a successful output line, else (None, its error)."""
    response = record.get("response") or {}
    if record.get("error") or response.get("status_code") != 200:
        return None, record.get("error") or (response.get("body") or {}).get("error") or response.get("status_code")
    body = response["body"]
    usage = body["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    price_in, price_cached, price_out = BATCH_PRICES[model]
    cost = (usage["prompt_tokens"] - cached) * price_in + cached * price_cached + usage["completion_tokens"] * price_out
    return {"text": body["choices"][0]["message"]["content"].strip(), "model": body["model"],
            "prompt_tokens": usage["prompt_tokens"], "cached_tokens": cached,
            "completion_tokens": usage["completion_tokens"], "cost_usd": cost, "batch": True}, None


def _submit(client, state_path: Path, state: dict, data: bytes) -> dict:
    """Upload the input file and create the batch, persisting each id as soon as it exists."""
    if not state.get("input_file_id"):
        upload = client.files.create(file=(state_path.stem + ".jsonl", data, "application/jsonl"), purpose="batch")
        state["input_file_id"] = upload.id
        _write_state(state_path, state)
    batch = client.batches.create(input_file_id=state["input_file_id"], endpoint=CHAT_URL, completion_window="24h",
                                  metadata={"run": state["run"], "stage": state["stage"]})
    state["batch_id"] = batch.id
    _write_state(state_path, state)
    return state


def _poll(client, batch_id: str, max_wait: float):
    """Retrieve the batch with bounded backoff until it reaches a final status or the wait runs out."""
    deadline, delay = time.monotonic() + max_wait, 15.0
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(json.dumps({"batch_id": batch_id, "status": batch.status,
                          "completed": counts.completed if counts else None,
                          "failed": counts.failed if counts else None,
                          "total": counts.total if counts else None}), flush=True)
        if batch.status in BATCH_DONE:
            return batch
        if time.monotonic() + delay > deadline:
            raise SystemExit(f"batch {batch_id} is still {batch.status} after {max_wait:.0f}s; "
                             "rerun the same command to resume it")
        time.sleep(delay)
        delay = min(delay * 2, 300.0)


def batch_stage(client, run: str, stage: str, model: str, prompts: dict[str, str], max_tokens: int,
                repeat: int, max_wait: float, sync_retry: bool) -> dict[str, dict]:
    """Answer every prompt through one batch and the call cache; returns cache entries keyed like `prompts`."""
    lines, ids = batch_lines(model, prompts, max_tokens, repeat)
    if lines:
        data = ("\n".join(lines) + "\n").encode()
        BATCH_STATE.mkdir(parents=True, exist_ok=True)
        state_path = BATCH_STATE / f"{stage}_{sha_bytes(data)[:20]}.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state is None:
            state_path.with_suffix(".jsonl").write_bytes(data)
            reserve = sum(_worst_case(BATCH_PRICES, model, json.loads(line)["body"]["messages"][0]["content"],
                                      max_tokens) for line in lines)
            _reserve(reserve)
            state = {"run": run, "stage": stage, "model": model, "repeat": repeat, "custom_ids": ids,
                     "reserve_usd": reserve, "input_sha256": sha_bytes(data), "settled": False}
            _write_state(state_path, state)
        if not state.get("batch_id"):
            try:
                state = _submit(client, state_path, state, data)
            except Exception:
                _settle(run, model, state["reserve_usd"], 0.0, 0)
                state_path.unlink()
                raise
        print(json.dumps({"stage": stage, "batch_id": state["batch_id"], "requests": len(ids),
                          "state": str(state_path)}), flush=True)
        batch = _poll(client, state["batch_id"], max_wait)
        results, failures = {}, {}
        for file_id in (batch.output_file_id, batch.error_file_id):
            if not file_id:
                continue
            for raw in client.files.content(file_id).text.splitlines():
                if raw.strip():
                    record = json.loads(raw)
                    out, error = _line_result(model, record)
                    if out is None:
                        failures[record["custom_id"]] = error
                    else:
                        results[record["custom_id"]] = out
        for cid in ids:
            if cid not in results and cid not in failures:
                failures[cid] = f"no output line (batch {batch.status})"
        if batch.status == "failed" and batch.errors:
            print(json.dumps({"batch_errors": [e.model_dump() for e in batch.errors.data or []]}), flush=True)
        if not state["settled"]:
            cost = sum(out["cost_usd"] for out in results.values())
            _settle(run, model, state["reserve_usd"], cost, len(results),
                    {"batch": True, "batch_id": state["batch_id"], "run": run, "stage": stage, "model": model,
                     "status": batch.status, "requests": len(ids), "succeeded": len(results),
                     "failed": len(failures), "usd": cost, "rates_usd_per_token": BATCH_PRICES[model]})
            state["settled"] = True
            _write_state(state_path, state)
        CACHE.mkdir(parents=True, exist_ok=True)
        for cid, out in results.items():
            (CACHE / f"{ids[cid]}.json").write_text(json.dumps(out))
        if failures:
            print(json.dumps({"stage": stage, "batch_id": state["batch_id"], "failed": len(failures),
                              "errors": {cid: str(err) for cid, err in sorted(failures.items())}}), flush=True)
            if not sync_retry:
                raise SystemExit(f"{len(failures)} {stage} requests failed in batch {state['batch_id']}; rerun to "
                                 "resubmit them as a new batch, or add --sync-retry to send them synchronously")
            by_key = {cache_key(request_body(model, p, max_tokens), repeat): p for p in prompts.values()}
            for cid in sorted(failures):
                call(client, run, model, by_key[ids[cid]], max_tokens, repeat)
    return {qid: json.loads((CACHE / f"{cache_key(request_body(model, p, max_tokens), repeat)}.json").read_text())
            for qid, p in prompts.items()}


def summarize(rows: list[dict]) -> dict:
    by_type = {}
    for row in rows:
        group = by_type.setdefault(row["qtype"], {"n": 0, "correct": 0})
        group["n"] += 1
        group["correct"] += row["correct"]
    abstention = [row for row in rows if "_abs" in row["qid"]]
    return {"n": len(rows), "correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "by_type": dict(sorted(by_type.items())),
            "abstention": {"n": len(abstention), "correct": sum(row["correct"] for row in abstention)}}


def write_summary() -> dict:
    """Pool the committed dev runs by (reader, topk, subquery weight) and settle spend from the call cache."""
    arms = {}
    for path in sorted((ROOT / "benchmarks/results").glob("longmemeval_official_dev_*.json")):
        if path.name.endswith(("_summary.json", "_subquery_recall.json")):
            continue
        run = json.loads(path.read_text())
        key = (f'{run["reader"]["model"]}|k{run["retrieval"]["topk_context"]}|sub{run["retrieval"]["subquery_weight"]}'
               f'|round{run["retrieval"].get("round_key_weight", 0.0)}')
        arm = arms.setdefault(key, {"reader": run["reader"]["model"], "topk_context": run["retrieval"]["topk_context"],
                                    "subquery_weight": run["retrieval"]["subquery_weight"], "runs": []})
        arm["runs"].append({"file": path.name, "repeat": run.get("repeat", 0), **run["summary"]})
    for arm in arms.values():
        correct = [r["correct"] for r in arm["runs"]]
        arm["mean_correct"] = sum(correct) / len(correct)
        arm["min_correct"], arm["max_correct"] = min(correct), max(correct)
        types = sorted({t for r in arm["runs"] for t in r["by_type"]})
        arm["mean_by_type"] = {t: {"n": arm["runs"][0]["by_type"][t]["n"],
                                   "mean_correct": sum(r["by_type"][t]["correct"] for r in arm["runs"]) / len(arm["runs"])}
                               for t in types}
    calls = [json.loads(p.read_text()) for p in CACHE.glob("*.json")]
    spend = {"calls": len(calls), "total_usd": sum(c["cost_usd"] for c in calls), "by_model_usd": {}}
    for c in calls:
        spend["by_model_usd"][c["model"]] = spend["by_model_usd"].get(c["model"], 0.0) + c["cost_usd"]
    summary = {"schema_version": 1, "split": "dev", "n_questions": 100, "heldout_evaluated": 0,
               "reader_protocol": "official run_generation.py 9e0b455: flat-turn rounds, json history, cot (con), "
                                  "temperature 0, max_tokens 800",
               "judge": {"model": JUDGE_MODEL, "protocol": "official evaluate_qa.py get_anscheck_prompt, temperature 0"},
               "retrieval": "reference pipeline turn ranking over cached Qwen3 scores (longmemeval_config.json)",
               "arms": dict(sorted(arms.items())), "spend_settled_from_call_cache": spend,
               "limitations": [
                   "Retrieval is the reference ranking, not native ConversationMemory recall; the native path "
                   + "was not run end to end (about 3 minutes per question locally).",
                   "The reference ranking indexes user turns; evidence held only in assistant turns is reached "
                   + "through round expansion alone.",
                   "Reader and judge at temperature 0 are not deterministic: repeated k=10 runs differ on about "
                   + "6 of 100 questions, so single runs are data points and arms are compared by their run ranges.",
                   "Replacing label-bearing tie-breaks with hashed ids leaves every top-10/15/50 set unchanged.",
                   "topk_context was chosen on dev; there is no held-out claim."]}
    k10 = [json.loads(p.read_text())["rows"] for p in sorted((ROOT / "benchmarks/results").glob(
        "longmemeval_official_dev_gpt4o_k10*.json"))]
    passes = {}
    for rows in k10:
        for row in rows:
            passes.setdefault(row["qid"], [row["qtype"], 0])[1] += row["correct"]
    summary["gpt4o_k10_failures"] = {
        "runs": len(k10),
        "failed_every_run": sorted([q, t] for q, (t, n) in passes.items() if n == 0),
        "failed_some_runs": sorted([q, t, n] for q, (t, n) in passes.items() if 0 < n < len(k10))}
    out = ROOT / "benchmarks/results/longmemeval_official_dev_summary.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    ledger = _ledger()
    ledger.update(total_usd=spend["total_usd"], calls=spend["calls"], by_model_usd=spend["by_model_usd"],
                  settled_from="call cache; by_run_usd is the live ledger and can differ by failed reservations")
    LEDGER.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n")
    return summary


def write_subquery_recall() -> dict:
    """Official recall_all@k of the reference ranking per subquery weight; labels used only for scoring."""
    entries = load_dev()
    types = {e["question_id"]: e["question_type"] for e in entries}
    report = {"schema_version": 1, "split": "dev", "metric": "recall_all@k over scored dev questions",
              "ks": [5, 10, 30, 50], "by_subquery_weight": {}}
    for weight in (0.0, 0.5, 1.0, 2.0):
        groups = {}
        for qid, ranked in rankings(entries, weight).items():
            gold = {t for t in ranked if t.startswith("answer_")}
            if not gold or "_abs" in qid:
                continue
            hits = [float(gold <= set(ranked[:k])) for k in report["ks"]]
            for group in ("overall", types[qid]):
                groups.setdefault(group, []).append(hits)
        report["by_subquery_weight"][str(weight)] = {
            group: {"n": len(rows), "recall_all": [round(sum(col) / len(rows), 4) for col in zip(*rows)]}
            for group, rows in sorted(groups.items())}
    out = ROOT / "benchmarks/results/longmemeval_official_dev_subquery_recall.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", choices=sorted(READERS))
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--sub", type=float, default=0.0, help="reference subquery variant weight")
    parser.add_argument("--round", type=float, default=0.0, help="weight of user+reply round keys")
    parser.add_argument("--tag")
    parser.add_argument("--estimate", action="store_true", help="count reader input tokens only")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=0, help="fresh reader and judge samples of the same inputs")
    parser.add_argument("--summarize", action="store_true", help="pool committed runs and settle spend")
    parser.add_argument("--subquery-recall", action="store_true", help="recall_all@k per subquery weight")
    parser.add_argument("--qids", help="comma-separated dev question ids to run instead of all 100")
    parser.add_argument("--batch", action="store_true", help="send reader and judge requests via the Batch API")
    parser.add_argument("--max-wait", type=float, default=24 * 3600.0, help="seconds to poll one batch")
    parser.add_argument("--sync-retry", action="store_true", help="resend failed batch lines synchronously")
    args = parser.parse_args()
    if args.subquery_recall:
        report = write_subquery_recall()
        print(json.dumps({w: v["overall"]["recall_all"] for w, v in report["by_subquery_weight"].items()}))
        return
    if args.summarize:
        summary = write_summary()
        print(json.dumps({k: (v["mean_correct"], v["min_correct"], v["max_correct"], len(v["runs"]))
                          for k, v in summary["arms"].items()}), flush=True)
        return
    if not args.reader or not args.tag:
        parser.error("--reader and --tag are required for a run")
    entries = load_dev()
    if args.qids:
        wanted = set(args.qids.split(","))
        entries = [e for e in entries if e["question_id"] in wanted]
        if len(entries) != len(wanted):
            parser.error("--qids names a question outside the dev split")
    ranked = rankings(entries, args.sub, args.round)
    prompts = {e["question_id"]: official_prompt(e, ranked[e["question_id"]], args.topk) for e in entries}
    tokens = sum(len(ENC.encode(p)) for p in prompts.values())
    rates = BATCH_PRICES if args.batch else PRICES
    estimate = tokens * rates[args.reader][0] + len(entries) * GEN_LENGTH * rates[args.reader][2]
    print(json.dumps({"reader_input_tokens": tokens, "max_reader_cost_usd": round(estimate, 2)}), flush=True)
    if args.estimate:
        return
    from openai import OpenAI

    client = OpenAI()
    judge_prompt = load_judge_prompt()
    run = f"{args.tag}:{args.reader}"

    def judge_input(entry, hyp):
        return judge_prompt(entry["question_type"], entry["question"], entry["answer"], hyp["text"],
                            abstention="_abs" in entry["question_id"])

    def row(entry, hyp, verdict):
        return {"qid": entry["question_id"], "qtype": entry["question_type"], "hypothesis": hyp["text"],
                "reader_model": hyp["model"], "reader_prompt_tokens": hyp["prompt_tokens"],
                "judge_model": verdict["model"], "judge_reply": verdict["text"],
                "correct": "yes" in verdict["text"].lower()}

    if args.batch:
        hyps = batch_stage(client, run, "reader", args.reader, prompts, GEN_LENGTH, args.repeat, args.max_wait,
                           args.sync_retry)
        judge_prompts = {e["question_id"]: judge_input(e, hyps[e["question_id"]]) for e in entries}
        verdicts = batch_stage(client, run, "judge", JUDGE_MODEL, judge_prompts, 10, args.repeat, args.max_wait,
                               args.sync_retry)
        rows = sorted((row(e, hyps[e["question_id"]], verdicts[e["question_id"]]) for e in entries),
                      key=lambda r: r["qid"])
    else:
        def answer(entry):
            hyp = call(client, run, args.reader, prompts[entry["question_id"]], GEN_LENGTH, args.repeat)
            return row(entry, hyp, call(client, run, JUDGE_MODEL, judge_input(entry, hyp), 10, args.repeat))

        with ThreadPoolExecutor(args.workers) as pool:
            rows = sorted(pool.map(answer, entries), key=lambda r: r["qid"])
    summary = summarize(rows)
    report = {"schema_version": 1, "split": "dev", "heldout_evaluated": 0, "tag": args.tag,
              "reader": {"model": args.reader, "snapshots_observed": sorted({r["reader_model"] for r in rows}),
                         "protocol": "official run_generation.py flat-turn, json history, cot (con), temperature 0, "
                                     f"max_tokens {GEN_LENGTH}, topk_context {args.topk}"},
              "judge": {"model": JUDGE_MODEL, "snapshots_observed": sorted({r["judge_model"] for r in rows}),
                        "protocol": "official evaluate_qa.py get_anscheck_prompt, temperature 0, max_tokens 10"},
              "retrieval": {"ranking": "reference pipeline turn ranking over cached Qwen3 scores "
                                       "(longmemeval_config.json, the native StageParams defaults)",
                            "subquery_weight": args.sub, "round_key_weight": args.round,
                            "topk_context": args.topk},
              "repeat": args.repeat, "api": "batch" if args.batch else "sync",
              "official_commit": "9e0b455f4ef0",
              "evaluate_qa_sha256": sha_bytes(EVALUATE_QA.read_bytes()),
              "source_sha256": SOURCE_SHA256, "summary": summary, "rows": rows}
    out = ROOT / f"benchmarks/results/longmemeval_official_{args.tag}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"tag": args.tag, "reader": args.reader, **summary,
                      "ledger_total_usd": round(_ledger()["total_usd"], 4)}), flush=True)


if __name__ == "__main__":
    main()
