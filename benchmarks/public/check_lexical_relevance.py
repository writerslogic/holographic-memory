# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Independent check of lexical_relevance_v1.json: recompute every recorded nDCG@10 from the stored
top-10 ids and the BEIR qrels, and confirm the selected setting is the SciFact-train argmax."""
import json, math, statistics, sys
from pathlib import Path


def ndcg10(ranked, graded):
    gains = lambda ids: sum(graded.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ids))
    ideal = gains(sorted(graded, key=graded.get, reverse=True)[:10])
    return gains(ranked[:10]) / ideal if ideal else 0.0


def qrels_for(dataset, split):
    rows = {}
    with (Path.home() / ".cache/hms-bench/downloads" / dataset / "qrels" / f"{split}.tsv").open() as f:
        next(f)
        for line in f:
            q, d, g = line.rstrip().split("\t")
            rows.setdefault(q, {})[d] = int(g)
    return rows


def main(path):
    r = json.loads(Path(path).read_text())
    failures = []
    for dataset, grid in r["grids"].items():
        qrels = qrels_for(dataset, grid["split"])
        if len(qrels) != grid["queries"]:
            failures.append(f"{dataset}: {len(qrels)} qrels queries, recorded {grid['queries']}")
        for key, rows in r["top10"][dataset].items():
            value = statistics.mean(ndcg10(rows[q], qrels[q]) for q in qrels if q in rows)
            if abs(value - grid["ndcg10"][key]) > 1e-12 or len(rows) != len(qrels):
                failures.append(f"{dataset} {key}: recomputed {value:.6f} vs recorded {grid['ndcg10'][key]:.6f} over {len(rows)} queries")
    tune = r["grids"]["scifact"]["ndcg10"]
    best = max((k for k in tune if "|" in k and k != "null|null"), key=tune.get)
    if "title_weights" in r:
        titles = [k for k in tune if k.startswith("title")]
        best_title = max(titles, key=tune.get)
        sel = r["title_weights"]["selected_on_scifact_train"]
        if best_title != sel["setting"] or abs(r["grids"]["nfcorpus"]["ndcg10"][best_title] - sel["nfcorpus_dev_ndcg10"]) > 1e-12:
            failures.append(f"title weight selection {sel['setting']} is not the SciFact-train argmax {best_title} or its dev value differs")
    sel = r["selected_on_scifact_train"]
    if best != f"{sel['k1']}|{sel['b']}":
        failures.append(f"selected {sel['k1']}|{sel['b']} is not the SciFact-train argmax {best}")
    if abs(r["grids"]["nfcorpus"]["ndcg10"][best] - sel["nfcorpus_dev_ndcg10"]) > 1e-12:
        failures.append("reported NFCorpus dev value does not match the grid")
    print(json.dumps({"ok": not failures, "failures": failures,
                      "nfcorpus_dev_default": r["default"]["nfcorpus_dev_ndcg10"],
                      "nfcorpus_dev_selected": sel["nfcorpus_dev_ndcg10"]}))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
