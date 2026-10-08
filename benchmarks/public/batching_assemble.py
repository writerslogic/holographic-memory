"""Usage: uv run python -I benchmarks/public/batching_assemble.py <work dir> <out.json>

Writes benchmarks/results/local_models_batching.json from the parity outputs (<work>/out/par_*,
par2_* after the KV-contiguity fix) and the paired timing files from batching_timing.py:
<work>/timing/{enc,facts}.json (commit 6004d40, before the fix), enc_sweep.json (encoder batch
sweep; encoders are unaffected by the fix) and v2.json (after the fix)."""
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

W, out_path = Path(sys.argv[1]), Path(sys.argv[2])
OUT, T = W / "out", W / "timing"
here = Path(__file__).resolve().parent


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def compare(kind, a, b, *extra):
    res = subprocess.run([sys.executable, "-I", str(here / "batching_compare.py"), kind,
                          str(W / a), str(W / b), *extra], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def untimed(tag):
    if not (OUT / f"{tag}.log").exists():
        return None
    txt = (OUT / f"{tag}.log").read_text()
    j = next(json.loads(x) for x in txt.splitlines() if x.startswith("{"))
    peak = re.search(r"(\d+)\s+peak memory footprint", txt)
    j["peak_memory_footprint_bytes"] = int(peak.group(1)) if peak else None
    return j


parity_pre_fix = {
    "facts_main_vs_new_batch1": compare("facts", "out/par_f_base.out", "out/par_f_b1.out"),
    "facts_new_batch1_vs_batch8": compare("facts", "out/par_f_b1.out", "out/par_f_b8.out"),
}
parity = {
    "embed_docs_main_vs_new_batch1": compare("embed", "out/par_ed_base.out", "out/par_ed_b1.out", "1024"),
    "embed_docs_batch1_vs_batch4": compare("embed", "out/par_ed_b1.out", "timing/sweep/ed_b4_r1.out", "1024"),
    "embed_docs_batch1_vs_batch16": compare("embed", "out/par_ed_b1.out", "out/par_ed_b16.out", "1024"),
    "embed_queries_main_vs_new_batch1": compare("embed", "out/par_eq_base.out", "out/par_eq_b1.out", "1024"),
    "embed_queries_batch1_vs_batch16": compare("embed", "out/par_eq_b1.out", "out/par_eq_b16.out", "1024"),
    "rerank_main_vs_new_batch1": compare("rerank", "out/par_rr_base.out", "out/par_rr_b1.out"),
    "rerank_batch1_vs_batch16": compare("rerank", "out/par_rr_b1.out", "out/par_rr_b16.out"),
    "embed_docs_main_vs_new_default_timed_run": compare("embed", "timing/v2/ed_base_r0.out", "timing/v2/ed_default_r0.out", "1024"),
    "rerank_main_vs_new_default_timed_run": compare("rerank", "timing/v2/rr_base_r0.out", "timing/v2/rr_default_r0.out"),
}
if (OUT / "par2_f_b8.out").exists():
    parity |= {
        "facts_main_vs_new_batch1": compare("facts", "out/par_f_base.out", "out/par2_f_b1.out"),
        "facts_main_vs_new_batch8": compare("facts", "out/par_f_base.out", "out/par2_f_b8.out"),
        "facts_new_batch8_pre_fix_vs_post_fix": compare("facts", "out/par_f_b8.out", "out/par2_f_b8.out"),
        "facts_batch1_vs_batch8": compare("facts", "out/par2_f_b1.out", "out/par2_f_b8.out"),
    }
    sys.path.insert(0, str(here))
    from batching_compare import parse
    b1 = json.loads((OUT / "par2_f_b1.out").read_text())
    b8 = json.loads((OUT / "par2_f_b8.out").read_text())
    parity["facts_batch1_vs_batch8"]["fact_differences"] = [
        {"item": i, "facts_batch1": len(parse(b1[i])), "facts_batch8": len(parse(b8[i])),
         "only_batch1": [f for f in parse(b1[i]) if f not in parse(b8[i])],
         "only_batch8": [f for f in parse(b8[i]) if f not in parse(b1[i])]}
        for i in parity["facts_batch1_vs_batch8"]["different_text"]]


def summarize(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    arms = {}
    for r in d["runs"]:
        a = arms.setdefault(r["arm"], {"wall_secs": [], "run_secs": [], "generated_tokens": [],
                                       "items": None, "peak_memory_footprint_bytes": [],
                                       "load_gate_met": [], "exit": []})
        pb = r["public_bench"]
        a["wall_secs"].append(round(r["wall_secs"], 2))
        a["exit"].append(r["exit"])
        a["load_gate_met"].append(r["load_gate_met"])
        a["peak_memory_footprint_bytes"].append(r["peak_memory_footprint_bytes"])
        a["items"] = pb.get("items")
        if "run_secs" in pb:
            a["run_secs"].append(round(pb["run_secs"], 2))
        if pb.get("generated_tokens"):
            a["generated_tokens"].append(pb["generated_tokens"])
    for a in arms.values():
        a["median_wall_secs"] = statistics.median(a["wall_secs"])
        if a["run_secs"]:
            a["median_run_secs"] = statistics.median(a["run_secs"])
            a["items_per_run_sec"] = a["items"] / a["median_run_secs"]
            if a["generated_tokens"]:
                a["generated_tokens_per_run_sec"] = statistics.median(a["generated_tokens"]) / a["median_run_secs"]
    return {"plan": {k: v for k, v in d["plan"].items() if k != "arms"},
            "arms_argv": {x["name"]: x["argv"] for x in d["plan"]["arms"]},
            "runs": [{k: r[k] for k in ("arm", "round", "exit", "wall_secs", "load_avg_before",
                                        "load_avg_after", "load_gate_met",
                                        "peak_memory_footprint_bytes", "max_rss_bytes")}
                     | {"public_bench": r["public_bench"]} for r in d["runs"]],
            "summary": arms}


doc = {
    "description": "Batched Qwen3 decoding (vendored Qwen3 forward with per-row positions and a "
                   "left-padding mask, continuous batching) and right-padded encoder batches, "
                   "local-models feature on Apple M4 Metal. Parity runs were untimed. Timed runs are "
                   "paired: one /Volumes/A/.hms-target/timed.sh invocation per plan, arms "
                   "alternating in plan order for 2 rounds, gate 1-minute load < 8 with one 900 s "
                   "wait budget shared by the whole invocation (batching_timing.py); once it is "
                   "spent, runs start regardless and carry load_gate_met false. Arms named "
                   "*_default pass no --batch (embedder 4, re-ranker 1, decoder 1 on Metal). The baseline binary (main dde6cd2) "
                   "reports only total secs, so it is compared by wall time; new-binary arms also "
                   "report run_secs (excluding model load). CUDA was not built or run.",
    "date": sh("date", "-u", "+%Y-%m-%dT%H:%M:%SZ"),
    "environment": {
        "machine": sh("sysctl", "-n", "hw.model"),
        "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": int(sh("sysctl", "-n", "hw.memsize")),
        "os": sh("sw_vers", "-productVersion"),
        "rustc": sh("rustc", "+1.98.0", "--version"),
        "device": "Metal", "dtypes": "encoders f32, LLM bf16",
        "build": "cargo +1.98.0 build --release --features local-models --bin public-bench",
        "shared_machine": "other sessions ran benchmarks and builds concurrently; see per-run load averages",
    },
    "models": {
        "Qwen/Qwen3-4B-Instruct-2507": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "Qwen/Qwen3-Reranker-0.6B": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    },
    "inputs": {
        "facts": "first 64 distinct haystack sessions of longmemeval_s_cleaned.json in first-seen "
                 "order, user turns only, prompts::fact_prompt; greedy, max 1536 new tokens "
                 "(batching_inputs.py). Timed facts runs use the first 16 of them.",
        "session_ids": json.loads((W / "data/facts64_ids.json").read_text()),
        "embed_docs": "96 user turns from those sessions (every k-th), document mode",
        "embed_queries": "first 96 LongMemEval_S questions, query mode",
        "rerank_pairs": "96 (question, user turn) pairs",
        "fact_parse": "raw text compared; facts = the JSON object's facts list; normalized = "
                      "lowercase with non-word runs collapsed (batching_compare.py)",
    },
    "baseline": "main dde6cd2 (candle-transformers 0.11 Qwen3, one sequence per forward on Metal)",
    "protocol_deviations": [
        "The timing lock was held longer than the 15-minute limit: v2.json about 21 min "
        "(all runs met the load gate) and the pre-fix facts.json about 33 min (two runs "
        "started above the gate: f_b1 round 0 at load 21.77, f_base round 1 at 8.75). "
        "A compliant re-time of the decoder arms (plan_v3: batch 1 vs 8, about 11 min) was "
        "attempted on 2026-10-07 from 13:00 to 14:00 PDT but never started: the 1-minute load "
        "average stayed between 32 and 190 from other sessions' jobs.",
    ],
    "defaults_decision": {
        "decoder": "batch 1 kept as the Metal/CUDA default: batch 8 left 56 of 64 fact outputs "
                   "(87.5%) byte-identical to batch 1, below the 95% bar, and all 8 differing "
                   "outputs parse to different fact lists. Batching is opt-in via --batch / "
                   "Generator::set_batch.",
        "embedder": "batch 4 on Metal/CUDA (was 16 in 8a76400): fastest in the paired sweep; "
                    "output differs from batch 1 by at most 3.6e-7 per component.",
        "reranker": "batch 1 on Metal/CUDA: padding was slower at every batch size measured.",
        "cuda": "not built or run; the CUDA defaults copy the Metal measurements and are unverified.",
    },
    "parity": parity,
    "parity_pre_fix_6004d40": parity_pre_fix,
    "parity_runs_untimed": {t: untimed(t) for t in
                            ["par_f_base", "par_f_b1", "par_f_b8", "par2_f_b1", "par2_f_b8"]},
    "timed": summarize(T / "v2.json"),
    "timed_encoder_batch_sweep": summarize(T / "enc_sweep.json"),
    "timed_pre_fix_6004d40": {
        "note": "new batch-1 decoding was slower than main: the KV cache was concatenated from a "
                "strided value tensor, so every decode step re-copied the cache element-wise. "
                "Fixed by making k and v contiguous before the concatenation (as candle does).",
        "encoders": summarize(T / "enc.json"),
        "facts": summarize(T / "facts.json"),
    },
}
out_path.write_text(json.dumps(doc, indent=1) + "\n")
for k, v in (("timed", doc["timed"]), ("sweep", doc["timed_encoder_batch_sweep"])):
    if v:
        print(k, json.dumps({a: {x: s.get(x) for x in ("wall_secs", "run_secs",
              "items_per_run_sec", "generated_tokens_per_run_sec", "load_gate_met", "exit",
              "peak_memory_footprint_bytes")} for a, s in v["summary"].items()}, indent=0))
