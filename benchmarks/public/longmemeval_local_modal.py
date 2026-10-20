# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run frozen LongMemEval reader prompts through the local open-weight reader on Modal.

  uv run --with modal python -I benchmarks/public/longmemeval_local_modal.py \
      --prompts <json: [{"key": ..., "input": ...}]> --out <jsonl> --cap 20 --batch 100

Reuses the GPU function of longmemeval_modal.py (vLLM 0.11.0, greedy decoding, max 1536 new
tokens, the tokenizer's chat template with thinking disabled) with the `large` reader
(Qwen3-30B-A3B-Instruct-2507 at its pinned revision) on one H100. Every batch is one call whose
wall time (model load included) is charged at Modal's published H100 rate plus the 15% overhead
of longmemeval_modal.py and shared equally by the batch's prompts; the driver stops before a
batch that would pass --cap. Outputs are appended per batch, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import longmemeval_modal as lm  # noqa: E402

MODEL, REVISION, GPU = lm.MODELS["large"]["llm"]
ENGINE = "vllm==0.11.0"
MAX_MODEL_LEN = 12288
MAX_NEW_TOKENS = 1536
PROBE_SECS_PER_ITEM = 30.0


def system_spec() -> dict:
    """The frozen generation settings of the local reader, as a report records them."""
    return {
        "name": "local_qwen30b",
        "reader_model": MODEL,
        "revision": REVISION,
        "engine": ENGINE,
        "gpu": GPU,
        "max_model_len": MAX_MODEL_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0,
        "chat_template": "tokenizer apply_chat_template, single user turn, add_generation_prompt, enable_thinking false",
        "judge_model": "gpt-5.4-2026-03-05",
        "usd_per_hour": lm.USD_PER_HOUR[GPU],
        "overhead": lm.OVERHEAD,
        "modal_app": lm.app.name,
        "modal_function": "llm_h100",
    }


def request_for(prompt: str) -> dict:
    spec = system_spec()
    return {key: spec[key] for key in ("reader_model", "revision", "engine", "gpu", "max_model_len",
                                       "max_new_tokens", "temperature", "chat_template")} | {"input": prompt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", type=float, default=20.0)
    parser.add_argument("--batch", type=int, default=100)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])
    pending = [p for p in prompts if p["key"] not in done]
    print(json.dumps({"prompts": len(prompts), "done": len(done), "pending": len(pending)}), flush=True)
    if not pending:
        return
    budget = lm.Budget(args.cap)
    rate = lm.USD_PER_HOUR[GPU] * lm.OVERHEAD
    secs_per_item = PROBE_SECS_PER_ITEM
    with lm.app.run():
        for start in range(0, len(pending), args.batch):
            batch = pending[start:start + args.batch]
            projected = len(batch) * secs_per_item / 3600 * rate
            budget.check(projected, f"reader batch of {len(batch)}")
            t = time.time()
            texts, secs = lm.llm_h100.remote(MODEL, REVISION, GPU, [p["input"] for p in batch])
            wall = time.time() - t
            budget.add("reader", GPU, secs, len(batch))
            secs_per_item = max(secs / len(batch), 1.0)
            if not (isinstance(texts, list) and len(texts) == len(batch) and all(isinstance(x, str) for x in texts)):
                raise RuntimeError("reader output shape mismatch")
            share = secs / len(batch)
            with args.out.open("a") as output:
                for p, text in zip(batch, texts, strict=True):
                    req = request_for(p["input"])
                    record = {"key": p["key"], "request_sha256": hashlib.sha256(
                        json.dumps(req, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                              "raw_text": text, "status": "completed", "gpu_seconds": share,
                              "batch_gpu_seconds": secs, "batch_wall_seconds": wall, "batch_size": len(batch),
                              "cost_usd": share / 3600 * rate, "unix_time": time.time()}
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({"batch": len(batch), "gpu_seconds": round(secs, 1), "wall_seconds": round(wall, 1),
                              "usd": round(secs / 3600 * rate, 4), "total_usd": round(budget.total(), 4)}), flush=True)


if __name__ == "__main__":
    main()
