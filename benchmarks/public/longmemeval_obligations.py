# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy==2.2.6"]
# ///
"""Obligation-decomposed candidate pools for the LongMemEval dev operand arm (lever 1B.1).

A multi-session question carries several obligations (entities, relations, a date range) that
one query rarely satisfies at once. This stage retrieves per obligation, unions whole sources,
re-ranks the union and hands it to the operand selector under the frozen per-question byte
budget. Obligations come from the cached dev run only: the LLM sub-queries (entity and relation
decompositions, `sub0..sub3`, scored on the same keys as the question) and the parsed time range
(the date obligation). No model is called and no held-out question is loaded.

Stages (dev 100 only):
  candidates  reproduce the tuned baseline turn ranking from the cached run (identity with
              evidence_completeness_dev.json is asserted), build the per-obligation rankings,
              fuse them with the baseline by weighted reciprocal ranks, sweep the fusion weights
              and pool size on dev, and write benchmarks/results/longmemeval_dev_obligations_v2.json
              with coverage before and after
  budget      run the operand retriever over the v2 pools under the frozen budgets and write
              benchmarks/results/longmemeval_v2/evidence_budget_dev.json (the v1 first_five and
              knapsack selections are carried over unchanged; only the operand arm changes)
  check       recompute every packet, byte count and completeness of the v2 report from the raw
              source and write benchmarks/results/longmemeval_v2/evidence_budget_summary.json
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

variant = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import longmemeval_pipeline as P  # noqa: E402

ROOT = HERE.parents[1]
CACHE = Path.home() / ".cache/hms-bench"
SOURCE = CACHE / "downloads/longmemeval_s_cleaned.json"
SOURCE_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
SPLIT = HERE / "longmemeval_split.json"
COMPLETENESS = ROOT / "benchmarks/results/evidence_completeness_dev.json"
COMPLETENESS_SUMMARY = ROOT / "benchmarks/results/evidence_completeness_summary.json"
BUDGET_V1 = ROOT / "benchmarks/results/evidence_budget_dev.json"
BUDGET_V1_SUMMARY = ROOT / "benchmarks/results/evidence_budget_summary.json"
OBLIGATIONS = ROOT / "benchmarks/results/longmemeval_dev_obligations_v2.json"
VARIANTS = {
    "v2": {"dir": "longmemeval_v2", "budget_factor": {}, "note": "lever 1B.1 alone: frozen per-question byte budget"},
    "v3": {"dir": "longmemeval_v3", "budget_factor": {"multi-session": 2, "temporal-reasoning": 2}, "qid_factor": 4,
           "calibration": {"previous": "longmemeval_dev_answers_v2b.json", "lexical_top1_min": 15.0},
           "note": "lever 1B.3 abstention calibration: v2b budgets, and four times the byte budget (under the unchanged reader token cap) for the questions whose v2b operand reader abstained while the top turn candidate's lexical retrieval score was at least 15; the rule was tuned on dev, where it separates 6 of the 7 false abstentions from all 6 true ones"},
    "v2b": {"dir": "longmemeval_v2b", "budget_factor": {"multi-session": 2, "temporal-reasoning": 2},
            "note": "lever 1B.1 plus the budget part of 1B.2: twice the frozen byte budget for multi-session and temporal-reasoning questions, the 8192-token reader cap unchanged"},
}
V2_DIR = ROOT / "benchmarks/results/longmemeval_v2"
BUDGET_V2 = V2_DIR / "evidence_budget_dev.json"
BUDGET_V2_SUMMARY = V2_DIR / "evidence_budget_summary.json"


def select_variant(name):
    global V2_DIR, BUDGET_V2, BUDGET_V2_SUMMARY
    V2_DIR = ROOT / "benchmarks/results" / VARIANTS[name]["dir"]
    BUDGET_V2 = V2_DIR / "evidence_budget_dev.json"
    BUDGET_V2_SUMMARY = V2_DIR / "evidence_budget_summary.json"
    return VARIANTS[name]


def calibration_flags(variant):
    """Lever 1B.3: the questions whose previous-arm operand reader abstained although retrieval
    was confident (top turn candidate's lexical score at or above the threshold), with the
    features of every abstention so the rule's separation is on record. Gold labels are not
    consulted; the threshold was chosen on dev."""
    cal = variant.get("calibration")
    if not cal:
        return {"flagged": []}
    if "result" in cal:
        return cal["result"]
    previous = json.loads((ROOT / "benchmarks/results" / cal["previous"]).read_text())
    scores = P.load_scores(RUN / "scores.json")
    features, flagged = {}, []
    for row in previous["rows"]:
        arm = row["systems"]["matched_gpt54"]["arms"]["operand"]
        parsed = arm["reader"].get("parsed") or {}
        if not parsed.get("abstain"):
            continue
        turns = [r for r in scores[row["qid"]]["orig"] if r[0].startswith("turn|")]
        lexical = sorted((r[1] for r in turns), reverse=True)
        dense = sorted((r[2] for r in turns), reverse=True)
        f = {"qtype": row["qtype"], "lexical_top1": lexical[0], "lexical_margin": lexical[0] - lexical[1],
             "dense_top1": dense[0], "dense_margin": dense[0] - dense[1], "input_tokens": arm["input_tokens"]}
        f["flagged"] = f["lexical_top1"] >= cal["lexical_top1_min"]
        features[row["qid"]] = f
        if f["flagged"]:
            flagged.append(row["qid"])
    cal["result"] = {"rule": "second pass with a widened byte budget when the operand reader abstained and the top turn candidate's lexical score >= lexical_top1_min",
                     "lexical_top1_min": cal["lexical_top1_min"], "previous_report": cal["previous"],
                     "previous_report_sha256": sha(ROOT / "benchmarks/results" / cal["previous"]),
                     "abstentions": len(features), "flagged": sorted(flagged), "features": features}
    return cal["result"]


def variant_budget(variant, qtype, budget, qid=None):
    factor = variant["budget_factor"].get(qtype, 1)
    if qid is not None and qid in calibration_flags(variant)["flagged"]:
        factor *= variant["qid_factor"]
    return budget * factor
RUN = CACHE / "longmemeval_s/runs/s-dev-small-all2"
FINAL = ROOT / "benchmarks/results/longmemeval_dev_final.json"
TURNS = CACHE / "holo_dev/turns_dev.json"
SCORER = CACHE / "downloads/longmemeval_eval_utils_9e0b455f4ef0.py"
KERNEL = ROOT / "src/core/operand_retriever.rs"
HARNESS = ROOT / "src/bin/operand-retriever.rs"
MAX_RANK = 64  # the operand kernel rejects cached ranks above this
SWEEP = {"w_sub": (0.0, 0.5, 1.0), "w_date": (0.0, 1.0), "pool": (24, 32, 48), "mode": ("turn", "session_union")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def encode_groups(groups):
    return canonical(groups).encode()


def load_scorer():
    import numpy as np
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a: np.asarray(a, dtype=float)
    spec = importlib.util.spec_from_file_location("lme_eval", SCORER)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev


def load_questions():
    tx = json.loads(TURNS.read_text())
    frozen = json.loads(SPLIT.read_text())
    assert {q["qid"] for q in tx["questions"]} == set(frozen["dev"]) and len(frozen["dev"]) == 100
    return [P.Question(q["qid"], q["qtype"], q["question"], q["date"], P.parse_date(q["date"]), [
        P.Session(s["sid"], P.parse_date(s["date"]), s["date"], [(t, "", "") for t in s["tids"]]) for s in q["sessions"]])
        for q in tx["questions"]]


def metrics(ranking, ids, ev):
    positions = {tid: i for i, tid in enumerate(ids)}
    correct = [tid for tid in ids if "answer" in tid]
    result = {}
    for k in (5, 10):
        any_recall, all_recall, ndcg = ev.evaluate_retrieval([positions[t] for t in ranking], correct, ids, k)
        result.update({f"recall_any@{k}": any_recall, f"recall_all@{k}": all_recall, f"ndcg_any@{k}": ndcg})
    return result


def obligation_rankings(q, scores, targets, queries, llm, cfg_turn):
    """Per-obligation full turn rankings: one per cached sub-query, one for the date range."""
    facets, seen = [], set()
    for variant, text in P.query_variants(q, queries["by_key"], llm):
        if not variant.startswith("sub") or text.casefold().strip() in seen:
            continue
        seen.add(text.casefold().strip())
        cfg = copy.deepcopy(cfg_turn)
        cfg.update(variants={"sub": 1.0}, rerank_w=0.0)
        facets.append(("sub:" + variant, P.rank_question(q, {variant: scores[q.qid][variant]}, targets[q.qid], cfg, "turn")))
    trange = P.time_range(q, queries["by_key"], llm)
    in_range = set()
    if trange is not None:
        for s in q.sessions:
            if s.date is not None and trange[0] <= s.date <= trange[1]:
                in_range.update(tid for tid, _, _ in s.turns)
    return facets, in_range


def fuse(base, facets, in_range, w_sub, w_date, k0=60.0):
    """Weighted reciprocal-rank union of the baseline and the obligation rankings."""
    score = {t: 1.0 / (k0 + r + 1) for r, t in enumerate(base)}
    for _, ranking in facets:
        for r, t in enumerate(ranking):
            score[t] += w_sub / (k0 + r + 1)
    for t in in_range:
        score[t] += w_date / (k0 + 1)
    pos = {t: i for i, t in enumerate(base)}
    return sorted(base, key=lambda t: (-score[t], pos[t]))


def pool_of(base, fused, size, mode="turn", sessions=None):
    """The v1 top-20 (order kept: frozen arms and the anchor stay valid) plus new candidates up
    to `size`: the best turns of the fused ranking (`turn`), or whole sources (`session_union`:
    every user turn of each session in session-rank order, turns within a session in fused
    order, so the selector sees a complete source and picks the turns that witness the
    obligations)."""
    pool = list(dict.fromkeys(base))[:20]
    if mode == "turn":
        order = fused
    else:
        fpos = {t: i for i, t in enumerate(fused)}
        order = [t for sid in sessions["ranking"] for t in sorted(sessions["turns"][sid], key=lambda t: fpos[t])]
        order += [t for t in fused if t not in set(order)]
    for t in order:
        if len(pool) >= size:
            break
        if t not in pool:
            pool.append(t)
    return pool


def stage_candidates(args):
    ev = load_scorer()
    questions = load_questions()
    canonical_final = json.loads(FINAL.read_text())
    assert sha(SCORER) == canonical_final["evaluation"]["sha256"]
    scores = P.load_scores(RUN / "scores.json")
    targets = json.loads((RUN / "targets.json").read_text())
    queries = json.loads((RUN / "queries.json").read_text())
    rerank = json.loads((RUN / "rerank.json").read_text())
    config = canonical_final["config"]
    base, _ = P.run(questions, scores, targets, config["turn"], config["session"], ev, queries["by_key"], queries["llm"], rerank)
    completeness = json.loads(COMPLETENESS.read_text())
    assert sha(COMPLETENESS) == json.loads(COMPLETENESS_SUMMARY.read_text())["report_sha256"]
    historical = {r["qid"]: r for r in completeness["rows"]}
    per_q = {}
    for q in questions:
        ranking = base["turn"][q.qid]
        assert ranking == historical[q.qid]["baseline_ranking"], f"baseline ranking differs for {q.qid}"
        ids = [tid for s in q.sessions for tid, _, _ in s.turns]
        assert ids == historical[q.qid]["ids"]
        facets, in_range = obligation_rankings(q, scores, targets, queries, queries["llm"], config["turn"])
        sessions = {"ranking": base["session"][q.qid], "turns": {s.sid: [tid for tid, _, _ in s.turns] for s in q.sessions}}
        per_q[q.qid] = (q, ranking, ids, facets, in_range, sessions)
    scored = [q for q in questions if historical[q.qid]["scored"]]
    assert len(scored) == 84

    def coverage(pool_fn):
        return sum(all(t in pool_fn(q.qid) for t in gold_of(q)) for q in scored)

    def gold_of(q):
        return [tid for s in q.sessions for tid, _, _ in s.turns if "answer" in tid]

    def mean_metric(rank_fn, field):
        return statistics.mean(metrics(rank_fn(q.qid), per_q[q.qid][2], ev)[field] for q in scored)

    baseline_pool = {qid: pool_of(v[1], v[1], 20) for qid, v in per_q.items()}
    before = {"coverage_pool20": coverage(lambda qid: baseline_pool[qid]),
              "recall_all@5": mean_metric(lambda qid: per_q[qid][1], "recall_all@5"),
              "recall_all@10": mean_metric(lambda qid: per_q[qid][1], "recall_all@10"),
              "ndcg_any@10": mean_metric(lambda qid: per_q[qid][1], "ndcg_any@10")}
    sweep = []
    for w_sub in SWEEP["w_sub"]:
        for w_date in SWEEP["w_date"]:
            fused = {qid: fuse(v[1], v[3], v[4], w_sub, w_date) for qid, v in per_q.items()}
            for pool, mode in ((p, m) for p in SWEEP["pool"] for m in SWEEP["mode"]):
                pools = {qid: pool_of(per_q[qid][1], fused[qid], pool, mode, per_q[qid][5]) for qid in per_q}
                row = {"w_sub": w_sub, "w_date": w_date, "pool": pool, "mode": mode,
                       "coverage_pool": coverage(lambda qid: pools[qid]),
                       "coverage_pool20_fused": coverage(lambda qid: fused[qid][:20]),
                       "recall_all@5": mean_metric(lambda qid: fused[qid], "recall_all@5"),
                       "recall_all@10": mean_metric(lambda qid: fused[qid], "recall_all@10"),
                       "ndcg_any@10": mean_metric(lambda qid: fused[qid], "ndcg_any@10"),
                       "mean_pool_size": statistics.mean(len(pools[qid]) for qid in pools)}
                sweep.append(row)
    best = max(sweep, key=lambda r: (r["coverage_pool"], r["recall_all@10"], r["recall_all@5"], -r["pool"], -r["w_sub"], r["mode"] == "turn"))
    fused = {qid: fuse(v[1], v[3], v[4], best["w_sub"], best["w_date"]) for qid, v in per_q.items()}
    pools = {qid: pool_of(per_q[qid][1], fused[qid], best["pool"], best["mode"], per_q[qid][5]) for qid in per_q}
    rows = []
    for q in questions:
        _, ranking, ids, facets, in_range, _ = per_q[q.qid]
        gold = gold_of(q) if historical[q.qid]["scored"] else []
        f = fused[q.qid]
        rows.append({"qid": q.qid, "qtype": q.qtype, "scored": historical[q.qid]["scored"], "n_obligations": len(facets) + bool(in_range),
                     "obligations": [name for name, _ in facets] + (["date_range"] if in_range else []),
                     "gold_ranks_baseline": [ranking.index(t) for t in gold], "gold_ranks_fused": [f.index(t) for t in gold],
                     "covered_pool20_baseline": all(t in baseline_pool[q.qid] for t in gold) if gold else None,
                     "covered_pool_v2": all(t in pools[q.qid] for t in gold) if gold else None,
                     "pool_v2": pools[q.qid], "ranks_v2": [min(f.index(t), MAX_RANK) for t in pools[q.qid]],
                     "metrics_baseline": metrics(ranking, ids, ev) if gold else None,
                     "metrics_fused": metrics(f, ids, ev) if gold else None})
    by_type = {}
    for kind in sorted({r["qtype"] for r in rows if r["scored"]}):
        rr = [r for r in rows if r["scored"] and r["qtype"] == kind]
        by_type[kind] = {"n": len(rr), "covered_pool20_baseline": sum(r["covered_pool20_baseline"] for r in rr),
                         "covered_pool_v2": sum(r["covered_pool_v2"] for r in rr)}
    out = {"schema_version": 1, "lever": "1B.1 obligation decomposition: per-obligation retrieval, union of whole sources, reciprocal-rank re-rank, operand selection",
           "obligation_sources": {"entity_relation": "cached LLM sub-queries sub0..sub3 of " + queries["llm"] + ", scored on the same keys as the question",
                                  "date": "cached parsed time range: sessions inside it get one reciprocal-rank unit"},
           "dev_only": True, "run": str(RUN), "config_sha256": sha(FINAL), "completeness_sha256": sha(COMPLETENESS),
           "n_dev": len(rows), "n_scored": len(scored), "before": before, "sweep": sweep, "chosen": best,
           "after": {"coverage_pool": best["coverage_pool"], "recall_all@5": best["recall_all@5"],
                     "recall_all@10": best["recall_all@10"], "ndcg_any@10": best["ndcg_any@10"]},
           "by_type": by_type, "rows": rows}
    OBLIGATIONS.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"before": before, "chosen": best, "by_type": by_type}, indent=1))


def legacy_groups(question):
    groups, ordinals = {}, {}
    for sid, date, turns in zip(question["haystack_session_ids"], question["haystack_dates"], question["haystack_sessions"]):
        users = [(i + 1, t) for i, t in enumerate(turns) if t["role"] == "user"]
        observed_sid = sid.replace("answer", "noans") if "answer" in sid and not any(t.get("has_answer") for _, t in users) else sid
        for ordinal, t in users:
            tid = f"{sid}_{ordinal}"
            if "answer" in sid and not t.get("has_answer"):
                tid = tid.replace("answer", "noans")
            groups.setdefault(tid, []).append([hashlib.sha256(tid.encode()).hexdigest(), hashlib.sha256(observed_sid.encode()).hexdigest(), date, "user", t["content"]])
            ordinals[tid] = ordinal
    return groups, ordinals


def stage_budget(args):
    from evidence_budget_diagnostic import aggregate
    obligations = json.loads(OBLIGATIONS.read_text())
    v1 = json.loads(BUDGET_V1.read_text())
    assert sha(BUDGET_V1) == json.loads(BUDGET_V1_SUMMARY.read_text())["report_sha256"]
    assert sha(SOURCE) == SOURCE_SHA256 == v1["source_sha256"]
    completeness = json.loads(COMPLETENESS.read_text())
    assert sha(COMPLETENESS) == v1["input_sha256"] == obligations["completeness_sha256"]
    historical = {r["qid"]: r for r in completeness["rows"]}
    frozen = json.loads(SPLIT.read_text())
    dev = set(frozen["dev"])
    raw = {q["question_id"]: q for q in json.loads(SOURCE.read_text()) if q["question_id"] in dev}
    pools = {r["qid"]: r for r in obligations["rows"]}
    requests, prepared = [], []
    for row in v1["rows"]:
        q = raw[row["qid"]]
        groups, ordinals = legacy_groups(q)
        v2 = pools[row["qid"]]
        candidates = v2["pool_v2"]
        assert candidates[:20] == row["candidates"], "v1 candidates must stay the prefix of the v2 pool"
        ranks = v2["ranks_v2"]
        costs = [len(encode_groups(groups[t])) + 1 for t in candidates]
        assert costs[:20] == row["costs"]
        budget = variant_budget(variant, row["qtype"], row["budget"], row["qid"])
        requests.append({"question": q["question"], "budget": budget, "candidates": [
            {"observations": groups[t], "ordinal": ordinals[t], "rank": r} for t, r in zip(candidates, ranks)]})
        prepared.append((row, groups, candidates, ranks, costs, budget))
    executable = Path(args.binary).resolve(strict=True)
    proc = subprocess.run([str(executable)], input="\n".join(canonical(r) for r in requests) + "\n", text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"operand retriever failed ({proc.returncode}): {proc.stderr.strip()[:2000]}")
    outputs = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(outputs) == 100
    report = json.loads(canonical(v1))
    for new_row, (row, groups, candidates, ranks, costs, budget), request, output in zip(report["rows"], prepared, requests, outputs):
        result = output["retrieval"]
        selected = result["selected"]
        selected_ids = [candidates[i] for i in selected]
        packet = result["packet"].encode()
        gold = {tid for tid in historical[row["qid"]]["ids"] if "answer" in tid}
        assert len(packet) <= budget
        assert json.loads(packet)["q"] == [request["candidates"][i]["observations"] for i in selected]
        assert selected and selected[0] == 0, "operand selection must keep the anchor first"
        fused_top = pools[row["qid"]]
        new_row.update(candidates=candidates, costs=costs, ranks=ranks, budget=budget, budget_v1=row["budget"])
        new_row["top_k_complete"] = {str(k): gold <= set(candidates[:k]) for k in (5, 10, 20, 64)}
        needed = sum(c for t, c in zip(candidates, costs) if t in gold)
        limit = budget - len(b'{"v":1,"evidence":[]}') + 1
        new_row["oracle"] = {"five_item_ceiling": len(gold) <= 5,
                             "unanchored_source_budget": gold <= set(candidates) and needed <= limit,
                             "anchored_source_budget": gold <= set(candidates) and needed + (0 if candidates[0] in gold else costs[0]) <= limit}
        for name in ("first_five", "rank_skip", "shortest", "knapsack"):
            assert set(row["arms"][name]["selected"]) <= set(candidates[:20])
        new_row["arms"]["operand"] = {"selected": selected_ids, "packet_bytes": len(packet), "packet_sha256": hashlib.sha256(packet).hexdigest(), "complete": gold <= set(selected_ids)}
        new_row["operand"] = {"packet": result["packet"], "required": result["required"], "witnessed": result["witnessed"],
                              "operand_ns": output["operand_ns"], "control_ns": output["control_ns"], "index_ns": output["index_ns"],
                              "request_sha256": hashlib.sha256(canonical(request).encode()).hexdigest(),
                              "control_v2pool": [candidates[i] for i in output["control"]],
                              "pool_size": len(candidates), "covered_pool_v2": fused_top["covered_pool_v2"]}
    scored = [r for r in report["rows"] if r["scored"]]
    count = sum(r["arms"]["operand"]["complete"] for r in scored)
    v1_count = sum(r["arms"]["operand"]["complete"] for r in v1["rows"] if r["scored"])
    report.update({"protocol": "lever 1B.1 obligation-decomposed candidate pool (longmemeval_dev_obligations_v2.json) + operand selector; v1 first_five and knapsack arms carried over unchanged; " + variant["note"],
                   "attempt": "obligations_" + args.variant, "variant": args.variant, "budget_factor": variant["budget_factor"],
                   "calibration": calibration_flags(variant) if variant.get("calibration") else None,
                   "code_sha256": sha(__file__), "kernel_sha256": sha(KERNEL), "harness_sha256": sha(HARNESS),
                   "binary_sha256": sha(executable), "obligations_sha256": sha(OBLIGATIONS), "v1_report_sha256": sha(BUDGET_V1),
                   "input_sha256": sha(COMPLETENESS), "validated_input_sha256": sha(COMPLETENESS_SUMMARY), "source_sha256": sha(SOURCE),
                   "split_sha256": sha(SPLIT), "summary": aggregate(report["rows"]),
                   "decision": {"complete": count, "n": len(scored), "complete_v1": v1_count,
                                "paired_delta_complete": paired([int(a["arms"]["operand"]["complete"]) - int(b["arms"]["operand"]["complete"])
                                                                  for a, b in zip(scored, [r for r in v1["rows"] if r["scored"]])]),
                                "equal_budget": not variant["budget_factor"],
                                "changed_operand_selections": sum(a["arms"]["operand"]["selected"] != b["arms"]["operand"]["selected"] for a, b in zip(report["rows"], v1["rows"]))},
                   "overall_superiority_established": False})
    V2_DIR.mkdir(exist_ok=True)
    shutil.copyfile(COMPLETENESS, V2_DIR / "evidence_completeness_dev.json")
    assert sha(V2_DIR / "evidence_completeness_dev.json") == report["input_sha256"]
    BUDGET_V2.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(json.dumps(report["decision"]))
    for new_row, old_row in zip(scored, [r for r in v1["rows"] if r["scored"]]):
        if new_row["arms"]["operand"]["complete"] != old_row["arms"]["operand"]["complete"]:
            print(new_row["qid"], new_row["qtype"], "gain" if new_row["arms"]["operand"]["complete"] else "loss", new_row["arms"]["operand"]["selected"])


def paired(values):
    import random
    rng = random.Random(20261016)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    def quantile(p):
        f = p * (len(draws) - 1)
        i = int(f)
        return draws[i] + (f - i) * (draws[min(i + 1, len(draws) - 1)] - draws[i])
    return {"mean": statistics.mean(values), "ci95": [quantile(.025), quantile(.975)], "n": n}


def stage_check(args):
    report = json.loads(BUDGET_V2.read_text())
    v1 = json.loads(BUDGET_V1.read_text())
    assert sha(SOURCE) == report["source_sha256"] == SOURCE_SHA256
    assert sha(SPLIT) == report["split_sha256"]
    assert sha(V2_DIR / "evidence_completeness_dev.json") == sha(COMPLETENESS) == report["input_sha256"]
    frozen = json.loads(SPLIT.read_text())
    assert {r["qid"] for r in report["rows"]} == set(frozen["dev"]) and not {r["qid"] for r in report["rows"]} & set(frozen["heldout"])
    completeness = {r["qid"]: r for r in json.loads(COMPLETENESS.read_text())["rows"]}
    raw = {q["question_id"]: q for q in json.loads(SOURCE.read_text()) if q["question_id"] in set(frozen["dev"])}
    failures = []
    for row, old in zip(report["rows"], v1["rows"]):
        assert row["qid"] == old["qid"] and row["budget"] == variant_budget(variant, row["qtype"], old["budget"], row["qid"]) and row["budget_v1"] == old["budget"]
        groups, _ = legacy_groups(raw[row["qid"]])
        assert set(groups) == set(completeness[row["qid"]]["ids"])
        candidates = row["candidates"]
        assert candidates[:20] == old["candidates"] and len(set(candidates)) == len(candidates)
        costs = [len(encode_groups(groups[t])) + 1 for t in candidates]
        assert costs == row["costs"]
        gold = {t for t in completeness[row["qid"]]["ids"] if "answer" in t}
        for name in ("first_five", "rank_skip", "shortest", "knapsack"):
            assert row["arms"][name] == old["arms"][name], f"{row['qid']} {name} changed"
            blob = json.dumps({"v": 1, "evidence": [groups[t] for t in row["arms"][name]["selected"]]}, ensure_ascii=False, separators=(",", ":")).encode()
            assert len(blob) == row["arms"][name]["packet_bytes"] <= row["budget"] and hashlib.sha256(blob).hexdigest() == row["arms"][name]["packet_sha256"]
        op = row["arms"]["operand"]
        packet = json.dumps({"v": 2, "q": [groups[t] for t in op["selected"]], "i": []}, ensure_ascii=False, separators=(",", ":")).encode()
        if not (packet.decode() == row["operand"]["packet"] and len(packet) == op["packet_bytes"] <= row["budget"]
                and hashlib.sha256(packet).hexdigest() == op["packet_sha256"] and op["selected"][0] == candidates[0]
                and set(op["selected"]) <= set(candidates) and op["complete"] == (gold <= set(op["selected"]))):
            failures.append(row["qid"])
        assert row["top_k_complete"] == {str(k): gold <= set(candidates[:k]) for k in (5, 10, 20, 64)}
    scored = [r for r in report["rows"] if r["scored"]]
    complete = sum(r["arms"]["operand"]["complete"] for r in scored)
    assert complete == report["decision"]["complete"] and not failures, failures
    from evidence_budget_diagnostic import aggregate
    assert report["summary"] == json.loads(json.dumps(aggregate(report["rows"])))
    result = {"report_sha256": sha(BUDGET_V2), "validator_sha256": sha(__file__), "protocol_sha256": report["code_sha256"],
              "scored_dev_questions": len(scored), "heldout_questions_untouched": len(frozen["heldout"]),
              "validation_gates_passed": ["frozen source, split and historical identity fingerprints", "v1 arms byte-identical",
                                         "operand packets recomputed from raw source within budget with the anchor first",
                                         "completeness and top-k recomputed"],
              "decision": report["decision"], "summary": report["summary"], "overall_superiority_established": False}
    BUDGET_V2_SUMMARY.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["decision"]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("candidates", "budget", "check"))
    p.add_argument("--binary", type=Path, default=Path("/Volumes/A/.hms-target-1b/release/operand-retriever"))
    p.add_argument("--variant", choices=tuple(VARIANTS), default="v2")
    args = p.parse_args()
    global variant
    variant = select_variant(args.variant)
    {"candidates": stage_candidates, "budget": stage_budget, "check": stage_check}[args.stage](args)


if __name__ == "__main__":
    main()
