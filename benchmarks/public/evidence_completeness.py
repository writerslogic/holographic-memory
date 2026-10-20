# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["numpy==2.2.6"]
# ///
"""Frozen cached-score quality experiment; research record section 15."""

import argparse
import copy
import importlib.util
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

import longmemeval_pipeline as P
from geometry_experiments import digest, recover

ALPHAS = (.25, .50, .75)
FAMILIES = ("minimum_beam", "mean_beam", "mean_greedy")
SEED = 20261008


def choose(base, facets, alpha, family):
    if any(Counter(f) != Counter(base) for f in facets):
        raise ValueError("facet membership must match the observed corpus")
    if len(facets) < 2:
        return base
    candidates = list(dict.fromkeys(base[:64] + [t for facet in facets for t in facet[:10]]))
    position = {}
    for i, tid in enumerate(base):
        position.setdefault(tid, i)
    prior = [1 / math.sqrt(1 + position[t]) for t in candidates]
    facet_positions = []
    for facet in facets:
        positions = {}
        for i, tid in enumerate(facet):
            positions.setdefault(tid, i)
        facet_positions.append(positions)
    compatibility = [[1 / math.sqrt(1 + f[t]) for f in facet_positions] for t in candidates]

    def objective(selected):
        maxima = [max(compatibility[i][j] for i in selected) for j in range(len(facets))]
        coverage = min(maxima) if family == "minimum_beam" else statistics.mean(maxima)
        return alpha * sum(prior[i] for i in selected) / 5 + (1 - alpha) * coverage

    beam = [(0,)]
    width = 1 if family == "mean_greedy" else 32
    for _ in range(min(5, len(candidates)) - 1):
        expanded = {tuple(sorted((*state, i))) for state in beam for i in range(len(candidates)) if i not in state}
        beam = sorted(expanded, key=lambda state: (-objective(state), state))[:width]
    selected = {candidates[i] for i in beam[0]}
    head = sorted(selected, key=position.get)
    removed = set()
    tail = []
    for tid in base:
        if tid in selected and tid not in removed:
            removed.add(tid)
        else:
            tail.append(tid)
    return head + tail


def interval(values):
    rng = random.Random(SEED)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    def quantile(p):
        f = p * (len(draws) - 1)
        i = int(f)
        return draws[i] + (f - i) * (draws[min(i + 1, len(draws) - 1)] - draws[i])
    return {"mean": statistics.mean(values), "ci95": [quantile(.025), quantile(.975)]}


def metrics(ranking, ids, ev):
    positions = {tid: i for i, tid in enumerate(ids)}
    correct = [tid for tid in ids if "answer" in tid]
    result = {}
    for k in (5, 10):
        any_recall, all_recall, ndcg = ev.evaluate_retrieval([positions[t] for t in ranking], correct, ids, k)
        result.update({f"recall_any@{k}": any_recall, f"recall_all@{k}": all_recall, f"ndcg_any@{k}": ndcg})
    return result


def split(rows):
    rng = random.Random(SEED)
    result = {"calibration": [], "confirmation": []}
    for kind in sorted({r["qtype"] for r in rows if r["scored"]}):
        ids = sorted(r["qid"] for r in rows if r["qtype"] == kind and r["scored"])
        rng.shuffle(ids)
        result["calibration"].extend(ids[:len(ids) // 2])
        result["confirmation"].extend(ids[len(ids) // 2:])
    return result


def summarize(rows, partition):
    groups = {**partition, "all_dev": [r["qid"] for r in rows if r["scored"]]}
    result = {}
    for group, members in groups.items():
        rr = [r for r in rows if r["qid"] in members]
        result[group] = {}
        for variant in rr[0]["variants"]:
            result[group][variant] = {}
            for kind in ["overall"] + sorted({r["qtype"] for r in rr}):
                selected = [r for r in rr if kind == "overall" or r["qtype"] == kind]
                result[group][variant][kind] = {"n": len(selected), **{
                    field: interval([r["variants"][variant]["metrics"][field] for r in selected])
                    for field in ("recall_all@5", "ndcg_any@10")}}
    chosen = {}
    for family in FAMILIES:
        candidates = [f"{family}_{alpha:.2f}" for alpha in ALPHAS]
        chosen[family] = max(candidates, key=lambda v: (
            result["calibration"][v]["overall"]["recall_all@5"]["mean"],
            result["calibration"][v]["overall"]["ndcg_any@10"]["mean"], -float(v.rsplit("_", 1)[1])))
    control = max(["baseline", chosen["mean_beam"], chosen["mean_greedy"]], key=lambda v: (
        result["calibration"][v]["overall"]["recall_all@5"]["mean"],
        result["calibration"][v]["overall"]["ndcg_any@10"]["mean"], v == "baseline"))
    candidate = chosen["minimum_beam"]
    comparisons = {}
    for comparator in dict.fromkeys(("baseline", control)):
        rr = [r for r in rows if r["qid"] in partition["confirmation"]]
        comparisons[comparator] = {}
        for kind in ["overall"] + sorted({r["qtype"] for r in rr}):
            sel = [r for r in rr if kind == "overall" or r["qtype"] == kind]
            comparisons[comparator][kind] = interval([
                r["variants"][candidate]["metrics"]["recall_all@5"] - r["variants"][comparator]["metrics"]["recall_all@5"]
                for r in sel])
    passed = all(c["overall"]["mean"] >= .05 and c["overall"]["ci95"][0] > 0 and
                 c["multi-session"]["mean"] >= .10 and
                 all(v["mean"] > -.02 for k, v in c.items() if k not in ("overall", "multi-session"))
                 for c in comparisons.values())
    return {"metrics": result, "selected": chosen, "selected_control": control,
            "confirmation_comparisons": comparisons, "quality_leap_proxy_pass": passed,
            "overall_superiority_established": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cache = Path.home() / ".cache/hms-bench"
    p.add_argument("--run", type=Path, default=cache / "longmemeval_s/runs/s-dev-small-all2")
    p.add_argument("--cache", type=Path, default=cache / "holo_dev")
    p.add_argument("--output", type=Path, default=Path("benchmarks/results/evidence_completeness_dev.json"))
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--work", type=Path, default=Path("target/conversation-geometry-validation/models"))
    args = p.parse_args()
    if args.self_test:
        base = list("abcdefg")
        for size in (4, 5, 6):
            b = base[:size]
            result = choose(b, [b, list(reversed(b))], .25, "minimum_beam")
            assert result[0] == b[0] and set(result) == set(b) and b[-1] in result[:5]
        assert choose(base, [base], .25, "minimum_beam") == base
        aliases = base + [base[0]]
        result = choose(aliases, [aliases, list(reversed(aliases))], .25, "minimum_beam")
        assert Counter(result) == Counter(aliases) and len(set(result[:5])) == 5
        try:
            choose(base, [base, list("abcdefx")], .25, "minimum_beam")
        except ValueError:
            pass
        else:
            raise ValueError("unobserved facet identity accepted")
        print("anchor, evidence-budget boundaries, single-facet fallback and membership passed")
        return
    canonical_file = Path("benchmarks/results/longmemeval_dev_final.json")
    canonical = json.loads(canonical_file.read_text())
    ablation_file = Path("benchmarks/results/longmemeval_dev_ablation.json")
    ablation = json.loads(ablation_file.read_text())
    scorer_file = cache / "downloads/longmemeval_eval_utils_9e0b455f4ef0.py"
    assert digest(scorer_file) == canonical["evaluation"]["sha256"]
    import numpy as np
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a: np.asarray(a, dtype=float)
    spec = importlib.util.spec_from_file_location("lme_eval", scorer_file)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    tx = json.loads((args.cache / "turns_dev.json").read_text())
    split_file = Path(__file__).with_name("longmemeval_split.json")
    frozen = json.loads(split_file.read_text())
    assert {q["qid"] for q in tx["questions"]} == set(frozen["dev"])
    questions = [P.Question(q["qid"], q["qtype"], q["question"], q["date"], P.parse_date(q["date"]), [
        P.Session(s["sid"], P.parse_date(s["date"]), s["date"], [(t, "", "") for t in s["tids"]]) for s in q["sessions"]])
        for q in tx["questions"]]
    scores = P.load_scores(args.run / "scores.json")
    targets = json.loads((args.run / "targets.json").read_text())
    queries = json.loads((args.run / "queries.json").read_text())
    rerank = json.loads((args.run / "rerank.json").read_text())
    config = canonical["config"]
    base, baseline_metrics = P.run(questions, scores, targets, config["turn"], config["session"], ev,
                                   queries["by_key"], queries["llm"], rerank)
    differences = []
    for level, types in canonical["metrics"]["dev"].items():
        for kind, values in types.items():
            for metric, value in values.items():
                if not math.isclose(value, baseline_metrics[level][kind][metric], abs_tol=1e-10):
                    differences.append({"level": level, "type": kind, "metric": metric,
                                        "canonical": value, "reconstructed": baseline_metrics[level][kind][metric]})
    if args.baseline_only:
        print(json.dumps({"baseline_differences": differences, "current": baseline_metrics,
                          "duplicate_ids": [{"qid": q.qid,
                              "corpus": sum(len(s.turns) for s in q.sessions),
                              "unique": len({tid for s in q.sessions for tid, _, _ in s.turns}),
                              "ranking": len(base["turn"][q.qid])}
                              for q in questions if len(base["turn"][q.qid]) != len(set(base["turn"][q.qid]))]}, indent=1))
        return
    for level in ("turn", "session"):
        if config[level] != ablation["levels"][level]["final"]:
            raise ValueError("cached baseline configuration mismatch")
        for kind, values in ablation["levels"][level]["final_metrics"].items():
            for metric, value in values.items():
                if not math.isclose(value, baseline_metrics[level][kind][metric], abs_tol=1e-10):
                    raise ValueError(f"cached baseline metric mismatch: {level}/{kind}/{metric}")
    print("tuned cached-run baseline reproduced; separate Modal-final differences recorded", flush=True)
    source_manifest = json.loads((args.work / "manifest.json").read_text())
    assert source_manifest["version"] == 1
    rows = []
    recovered_count = 0
    for qi, q in enumerate(questions):
        ids = [tid for s in q.sessions for tid, _, _ in s.turns]
        facets, seen = [], set()
        for variant, text in P.query_variants(q, queries["by_key"], queries["llm"]):
            if not variant.startswith("sub") or text.casefold().strip() in seen:
                continue
            seen.add(text.casefold().strip())
            cfg = copy.deepcopy(config["turn"])
            cfg.update(variants={"sub": 1.}, rerank_w=0.)
            facets.append(P.rank_question(q, {variant: scores[q.qid][variant]}, targets[q.qid], cfg, "turn"))
        ranking = base["turn"][q.qid]
        variants = {"baseline": {"top10": ranking[:10], "metrics": metrics(ranking, ids, ev)}}
        for family in FAMILIES:
            for alpha in ALPHAS:
                result = choose(ranking, facets, alpha, family)
                assert len(result) == len(ids) and set(result) == set(ids)
                variants[f"{family}_{alpha:.2f}"] = {"top10": result[:10], "metrics": metrics(result, ids, ev)}
        entry = source_manifest["questions"][qi]
        assert q.qid == entry["qid"] and ids == entry["tids"]
        with np.load(args.work / f"q{qi:03}.npz", allow_pickle=False) as archive:
            for result in variants.values():
                selected_ids = set(result["top10"][:5])
                slots = [i for i, tid in enumerate(ids) if tid in selected_ids]
                result["source_slots"] = slots
                recovered_count += recover(archive, entry, slots)
        rows.append({"qid": q.qid, "qtype": q.qtype, "scored": q.scored, "ids": ids,
                     "baseline_ranking": ranking, "facet_rankings": facets, "variants": variants})
        if (qi + 1) % 10 == 0:
            print(f"evidence {qi + 1}/100", flush=True)
    partition = split(rows)
    summary = summarize(rows, partition)
    report = {"protocol": "IDEAS-2026-10 section 15; retrieval only, dev internal confirmation",
              "code_sha256": digest(__file__), "script_lock_sha256": digest(Path(str(__file__) + ".lock")),
              "canonical_sha256": digest(canonical_file), "split_sha256": digest(split_file),
              "cached_baseline_sha256": digest(ablation_file), "modal_final_differences": differences,
              "source_manifest_sha256": digest(args.work / "manifest.json"),
              "source_turn_readouts_verified": recovered_count,
              "input_sha256": {name: digest(args.run / name) for name in
                               ("scores.json", "targets.json", "queries.json", "rerank.json")},
              "export_sha256": digest(args.cache / "turns_dev.json"),
              "partition": partition, "rows": rows, "summary": summary,
              "faceted_scored_questions": sum(r["scored"] and len(r["facet_rankings"]) >= 2 for r in rows)}
    args.output.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"partition_sizes": {k: len(v) for k, v in partition.items()},
                      "faceted_scored": report["faceted_scored_questions"],
                      **{k: v for k, v in summary.items() if k != "metrics"}}), flush=True)


if __name__ == "__main__":
    main()
