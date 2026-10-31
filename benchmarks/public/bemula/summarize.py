"""Summarize a Bemula run of WARM into benchmarks/results/bemula_v1.json.

Reads the run directories written by run_warm.py (metrics.json, items.jsonl, run_meta.json),
the WARM query/forget log, and the published baseline metrics in the Bemula checkout. Each
failed forget-variant efficacy or attack item is attributed from the logged recall:
`engine` when recall returned a purged id, `policy` when the request's target value (the
retention gold answer) appears in the retrieved context, `model` otherwise (the value was
not retrieved; the answer came from inference over surviving facts or a guess). The value
match is a case-insensitive substring test, a diagnostic rather than a grade.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SUITE_KEYS = ["RA", "FE", "PR", "RLA", "AR", "FA", "UG", "bemula_score", "request_pass_rate"]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _pick(metrics: dict) -> dict:
    out = {k: metrics.get(k) for k in SUITE_KEYS}
    out["by_forget_type"] = {
        t: {k: v.get(k) for k in SUITE_KEYS if k in v} for t, v in metrics.get("by_forget_type", {}).items()
    }
    out["suite_pass_rates"] = metrics.get("suite_pass_rates")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bemula", type=Path, required=True, help="Bemula checkout")
    ap.add_argument("--warm-run", type=Path, required=True)
    ap.add_argument("--rag-run", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--dialogues", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True, help="JSON with suite commit, models, WARM SHA")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    warm = json.loads((args.warm_run / "metrics.json").read_text())
    rag = json.loads((args.rag_run / "metrics.json").read_text())
    items = _jsonl(args.warm_run / "items.jsonl")
    log = _jsonl(args.log)

    # Retention gold per request: the stored value a forget request targets.
    target = {}
    for r in items:
        if r["suite"] == "retention" and r.get("request_id"):
            target.setdefault(r["request_id"], []).append(str(r.get("gold_answer", "")).lower())

    contexts = defaultdict(list)
    purged_returned = 0
    for e in log:
        if e["kind"] == "query":
            contexts[(e["instance_id"], e["variant"], e["question"])].append(e)
            purged_returned += len(e.get("purged_returned") or [])

    attribution = Counter()
    for r in items:
        if r["variant"] == "control" or r["correct"] or r["suite"] not in ("efficacy", "attack"):
            continue
        golds = [g for g in target.get(r.get("request_id"), []) if g and len(g) > 2]
        hits = contexts.get((r["instance_id"], "forget", r["question"]), [])
        if any(h.get("purged_returned") for h in hits):
            attribution["engine"] += 1
        elif any(g in h["context"].lower() for h in hits for g in golds):
            attribution["policy"] += 1
        else:
            attribution["model"] += 1

    requests = sum(len(d["forget_requests"]) for d in _jsonl(args.dialogues))
    forgets = [e for e in log if e["kind"] == "forget"]
    review = {
        "forget_events_forget_copy": sum(e["variant"] == "forget" for e in forgets),
        "forget_events_control_copy": sum(e["variant"] == "control" for e in forgets),
        "facts_purged": sum(len(e["purged"]) for e in forgets),
        "facts_whose_review_failed": sum(e.get("scan_failures", 0) for e in forgets),
        "classifier_failures": sum(e["kind"] == "classify_failure" for e in log),
    }

    published = {}
    for run in sorted((args.bemula / "results").glob("*_full_deepseek")):
        published[run.name.removesuffix("_full_deepseek")] = _pick(json.loads((run / "metrics.json").read_text()))

    result = json.loads(args.meta.read_text())
    result.update(
        {
            "warm": _pick(warm),
            "warm_items": warm.get("num_items"),
            "same_day_rag": _pick(rag),
            "published_deepseek": published,
            "forget_requests": {"in_suite": requests, **review},
            "purged_ids_returned_by_recall": purged_returned,
            "failed_forget_items_attribution": dict(attribution),
        }
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
