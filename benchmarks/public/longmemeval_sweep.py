# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
"""Ablate and tune the LongMemEval fusion config on the frozen dev split only.

  uv run --script benchmarks/public/longmemeval_sweep.py <run dir> <longmemeval_s_cleaned.json>

<run dir> holds scores.json, targets.json, queries.json and rerank.json of a Modal dev run made
with --all-kinds (`modal volume get hms-lme runs/<tag>`). Each lever is measured alone against the
baseline (user-turn keys for turns, session keys for sessions, original question, document-API
hybrid RRF, max aggregation), then greedy coordinate ascent keeps a value only if it raises the
objective (mean of recall_all@5/10 and ndcg_any@5/10 at that level) by at least MIN_GAIN, and a
drop-one pass records each kept lever's gain in combination. Writes longmemeval_config.json and
benchmarks/results/longmemeval_dev_ablation.json. Held-out questions are never loaded.
"""

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import longmemeval_pipeline as P  # noqa: E402
from evaluate import lme_eval_utils  # noqa: E402

MIN_GAIN = 0.005  # one of 84 scored questions moves a recall by 0.012

LEVERS = {
    "session": [
        ("w_lex", [0.0, 0.5, 1.0, 2.0]),
        ("fuse", ["rrf", "cvx"]),
        ("kinds.session_exp", [0.0, 0.5, 1.0, 2.0]),
        ("kinds.session", [0.0, 0.5, 1.0]),
        ("kinds.turn", [0.0, 0.5, 1.0, 2.0]),
        ("kinds.round", [0.0, 0.5, 1.0]),
        ("kinds.fact", [0.0, 0.5, 1.0]),
        ("fact_as", [None, "session", "session_exp", "turn", "turn_exp"]),
        ("kinds.turn_exp", [0.0, 0.5, 1.0]),
        ("agg", ["max", "sum2", "sum3", "rrf"]),
        ("variants.rewrite", [0.0, 0.5, 1.0]),
        ("variants.sub", [0.0, 0.25, 0.5, 1.0]),
        ("time_w", [0.0, 0.5, 1.0, 2.0]),
        ("k0", [10.0, 30.0, 60.0]),
        ("rerank_w", [0.0, 1.0, 2.0, 4.0, 8.0]),
        ("rerank_n", [10, 20, 30]),
    ],
    "turn": [
        ("w_lex", [0.0, 0.5, 1.0, 2.0]),
        ("fuse", ["rrf", "cvx"]),
        ("kinds.turn", [0.0, 0.5, 1.0]),
        ("kinds.round", [0.0, 0.5, 1.0, 2.0]),
        ("kinds.fact", [0.0, 0.5, 1.0, 2.0]),
        ("fact_as", [None, "turn", "turn_exp", "round"]),
        ("kinds.turn_exp", [0.0, 0.5, 1.0, 2.0]),
        ("agg", ["max", "sum2", "sum3", "rrf"]),
        ("variants.rewrite", [0.0, 0.5, 1.0]),
        ("variants.sub", [0.0, 0.25, 0.5, 1.0]),
        ("sess_prior", [0.0, 0.5, 1.0, 2.0, 4.0]),
        ("time_w", [0.0, 0.5, 1.0, 2.0]),
        ("k0", [10.0, 30.0, 60.0]),
        ("rerank_w", [0.0, 1.0, 2.0, 4.0, 8.0]),
        ("rerank_n", [10, 20, 30, 50]),
    ],
}


def setv(cfg: dict, path: str, value) -> dict:
    c = copy.deepcopy(cfg)
    if "." in path:
        a, b = path.split(".")
        c[a][b] = value
    else:
        c[path] = value
    return c


def getv(cfg: dict, path: str):
    if "." in path:
        a, b = path.split(".")
        return cfg[a].get(b, 0.0)
    return cfg[path]


def main(run_dir: str, data: str) -> None:
    rd = Path(run_dir)
    ev, ev_sha = lme_eval_utils()
    dev = P.split_ids(HERE / "longmemeval_split.json", "dev")
    qs = P.load(data, dev)
    scores = P.load_scores(rd / "scores.json")
    targets = json.loads((rd / "targets.json").read_text())
    qfile = json.loads((rd / "queries.json").read_text()) if (rd / "queries.json").exists() else {}
    queries, llm = qfile.get("by_key"), qfile.get("llm", "")
    rerank = json.loads((rd / "rerank.json").read_text()) if (rd / "rerank.json").exists() else None
    tr = {q.qid: P.time_range(q, queries, llm) for q in qs}

    def rank(cfg, level, sess_rank=None):
        return {q.qid: P.rank_question(q, scores[q.qid], targets[q.qid], cfg, level, tr[q.qid],
                                       (rerank or {}).get(level, {}).get(q.qid),
                                       (sess_rank or {}).get(q.qid)) for q in qs}

    def metrics(cfg, level, sess_rank=None):
        return P.evaluate(qs, rank(cfg, level, sess_rank), level, ev)

    base = {
        "session": {**copy.deepcopy(P.DEFAULT), "kinds": {"session": 1.0}},
        "turn": {**copy.deepcopy(P.DEFAULT), "kinds": {"turn": 1.0}},
    }
    report = {"objective": "mean of recall_all@5, recall_all@10, ndcg_any@5, ndcg_any@10 (official eval_utils)",
              "eval_sha256": ev_sha, "n_dev": len(qs), "n_scored": sum(q.scored for q in qs),
              "min_gain": MIN_GAIN, "levels": {}}
    best_sess_rank = None
    final = {}
    for level in ("session", "turn"):
        sr = best_sess_rank if level == "turn" else None
        b = base[level]
        b_obj = P.objective(metrics(b, level, sr))
        alone = []
        for path, values in LEVERS[level]:
            for v in values:
                if v == getv(b, path):
                    continue
                if path == "rerank_n":
                    c = setv(setv(b, "rerank_w", 2.0), path, v)
                else:
                    c = setv(b, path, v)
                alone.append({"lever": path, "value": v, "objective": P.objective(metrics(c, level, sr)),
                              "gain": None})
                alone[-1]["gain"] = alone[-1]["objective"] - b_obj
        cur, cur_obj = copy.deepcopy(b), b_obj
        trace = []
        for _ in range(2):
            changed = False
            for path, values in LEVERS[level]:
                best_v, best_o = getv(cur, path), cur_obj
                for v in values:
                    o = P.objective(metrics(setv(cur, path, v), level, sr))
                    if o >= best_o + MIN_GAIN:
                        best_v, best_o = v, o
                if best_v != getv(cur, path):
                    trace.append({"lever": path, "value": best_v, "objective": best_o, "gain": best_o - cur_obj})
                    cur, cur_obj, changed = setv(cur, path, best_v), best_o, True
            # Re-rank weight, depth and the merge constant interact; search them jointly.
            best_c, best_o = None, cur_obj
            for w in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0):
                for n in dict(LEVERS[level])["rerank_n"]:
                    for k0 in (10.0, 30.0, 60.0):
                        c = setv(setv(setv(cur, "rerank_w", w), "rerank_n", n), "k0", k0)
                        o = P.objective(metrics(c, level, sr))
                        if o >= best_o + MIN_GAIN:
                            best_c, best_o = c, o
            if best_c is not None:
                trace.append({"lever": "rerank_w+rerank_n+k0", "value": [best_c["rerank_w"], best_c["rerank_n"],
                              best_c["k0"]], "objective": best_o, "gain": best_o - cur_obj})
                cur, cur_obj, changed = best_c, best_o, True
            if not changed:
                break
        pruned = []
        while True:
            drop_one = []
            for path, _ in LEVERS[level]:
                bv = getv(b, path)
                if getv(cur, path) != bv:
                    o = P.objective(metrics(setv(cur, path, bv), level, sr))
                    drop_one.append({"lever": path, "reverted_to": bv, "objective": o, "loss": cur_obj - o})
            weakest = min(drop_one, key=lambda d: d["loss"], default=None)
            if weakest is None or weakest["loss"] >= MIN_GAIN:
                break
            # Keep only levers that still pay for themselves in combination.
            pruned.append(weakest)
            cur, cur_obj = setv(cur, weakest["lever"], weakest["reverted_to"]), weakest["objective"]
        trace += [{"pruned": p["lever"], "objective": p["objective"]} for p in pruned]
        base_m, final_m = metrics(b, level, sr), metrics(cur, level, sr)
        report["levels"][level] = {"baseline": b, "baseline_objective": b_obj, "baseline_metrics": base_m,
                                   "alone": alone, "greedy": trace, "final": cur, "final_objective": cur_obj,
                                   "final_metrics": final_m, "drop_one": drop_one}
        final[level] = cur
        if level == "session":
            best_sess_rank = rank(cur, "session")
        print(f"{level}: baseline {b_obj:.4f} -> {cur_obj:.4f}", flush=True)
    (HERE / "longmemeval_config.json").write_text(json.dumps(final, indent=1) + "\n")
    out = HERE.parent / "results" / "longmemeval_dev_ablation.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
