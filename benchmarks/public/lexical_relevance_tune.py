# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""BM25 k1/b for the WARM lexical scorer: tune on SciFact train, report on NFCorpus dev (2C.1).

Usage: lexical_relevance_tune.py --package <dir with index.js and writerslogic-warm.node>
       --out benchmarks/results/lexical_relevance_v1.json [--work <folder>]

The grid is scored on both datasets through the Node package (k=100, candidateLimit=100, lexical
only). The row that counts is NFCorpus dev at the (k1, b) that maximizes SciFact-train nDCG@10;
the NFCorpus-dev-optimal setting is recorded for information only, since it is tuned on the
evaluation queries. Top-10 ids per query are kept for the default and the selected settings so
check_lexical_relevance.py can recompute every nDCG from the BEIR qrels.
"""
import argparse, json, math, statistics, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from document_competition import prepare  # noqa: E402

K1_GRID = [0.6, 0.9, 1.2, 1.5, 2.0]
B_GRID = [0.3, 0.5, 0.75, 0.9, 1.0]
TITLE_WEIGHTS = [1, 2, 3, 4]

RUNNER = r"""
'use strict'
const fs = require('node:fs'); const path = require('node:path')
const { WarmMemorySystem } = require(path.join(process.argv[2], 'index.js'))
;(async () => {
  const input = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'))
  const grid = JSON.parse(process.argv[4])
  const titleWeight = Number(process.argv[7] || 1)
  const dir = fs.mkdtempSync(path.join(process.argv[5], 'hmsrel-'))
  const warm = new WarmMemorySystem(1024, dir, {})
  const docs = titleWeight > 1 ? input.documents.map(d => ({ ...d, titleWeight })) : input.documents
  for (let at = 0; at < docs.length; at += 4096) await warm.memorizeDocuments(docs.slice(at, at + 4096))
  const out = {}
  for (const [k1, b] of grid) {
    const options = { k: 100, candidateLimit: 100, lexicalWeight: 1, semanticWeight: 0 }
    if (k1 !== null) { options.bm25K1 = k1; options.bm25B = b }
    const rows = {}
    for (const q of input.queries) rows[q.id] = (await warm.searchDocuments(q.text, options)).slice(0, 10).map(h => h.documentId)
    out[`${k1}|${b}`] = rows
  }
  fs.writeFileSync(process.argv[6], JSON.stringify(out))
})().catch(e => { console.error(e); process.exit(1) })
"""


def ndcg10(ranked, graded):
    gains = lambda ids: sum(graded.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ids))
    ideal = gains(sorted(graded, key=graded.get, reverse=True)[:10])
    return gains(ranked[:10]) / ideal if ideal else 0.0


def evaluate(rows, qrels):
    return statistics.mean(ndcg10(rows[q], qrels[q]) for q in qrels if q in rows)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--package", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--work")
    a = a.parse_args()
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="lexrel-"))
    prepare(work)
    (work / "runner.cjs").write_text(RUNNER)
    grid = [[None, None]] + [[k1, b] for k1 in K1_GRID for b in B_GRID]
    results = {}
    for dataset, split in [("scifact", "train"), ("nfcorpus", "dev")]:
        qrels = json.loads((work / f"{dataset}-{split}-qrels.json").read_text())
        runs = {}
        for weight in TITLE_WEIGHTS:
            out = work / f"{dataset}-{split}-runs-t{weight}.json"
            # the BM25 grid only at the default title weight; title weights only at the default BM25
            subgrid = grid if weight == 1 else [[None, None]]
            subprocess.run(["node", str(work / "runner.cjs"), str(Path(a.package).resolve()),
                            str(work / f"{dataset}-{split}-input.json"), json.dumps(subgrid), str(work), str(out),
                            str(weight)], check=True)
            for key, rows in json.loads(out.read_text()).items():
                runs[key if weight == 1 else f"title{weight}"] = rows
        results[dataset] = {"split": split, "queries": len(qrels), "runs": runs, "qrels": qrels,
                            "ndcg10": {key: evaluate(rows, qrels) for key, rows in runs.items()}}
    tune = results["scifact"]["ndcg10"]
    bm25_keys = [k for k in tune if "|" in k and k != "null|null"]
    selected = max(bm25_keys, key=tune.get)
    title_keys = [k for k in tune if k.startswith("title")]
    selected_title = max(title_keys, key=tune.get)
    dev = results["nfcorpus"]["ndcg10"]
    dev_optimal = max(bm25_keys, key=dev.get)
    k1, b = (float(x) for x in selected.split("|"))
    report = {
        "schema_version": 1, "category": "2C_lexical_relevance", "date": "2026-10-19",
        "protocol": "WARM lexical-only BM25 through the Node package (k=100, candidateLimit=100), one chunk per document, title + text; k1 and b (at title weight 1) and the title weight (at default BM25) selected by nDCG@10 on SciFact train, reported on NFCorpus dev (324 queries); nDCG@10 computed in Python from the BEIR qrels in ~/.cache/warm-bench/downloads",
        "grid": {"k1": K1_GRID, "b": B_GRID},
        "default": {"k1": 1.2, "b": 0.75, "scifact_train_ndcg10": tune["null|null"], "nfcorpus_dev_ndcg10": dev["null|null"]},
        "selected_on_scifact_train": {"k1": k1, "b": b, "scifact_train_ndcg10": tune[selected], "nfcorpus_dev_ndcg10": dev[selected]},
        "nfcorpus_dev_optimal_for_information_only": {"setting": dev_optimal, "nfcorpus_dev_ndcg10": dev[dev_optimal]},
        "title_weights": {"grid": TITLE_WEIGHTS, "scifact_train_ndcg10": {k: tune[k] for k in title_keys},
                          "nfcorpus_dev_ndcg10": {k: dev[k] for k in title_keys},
                          "selected_on_scifact_train": {"setting": selected_title, "scifact_train_ndcg10": tune[selected_title], "nfcorpus_dev_ndcg10": dev[selected_title]}},
        "grids": {d: {"split": r["split"], "queries": r["queries"], "ndcg10": r["ndcg10"]} for d, r in results.items()},
        "top10": {d: {key: r["runs"][key] for key in ("null|null", selected, selected_title)} for d, r in results.items()},
    }
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("default", "selected_on_scifact_train", "nfcorpus_dev_optimal_for_information_only", "title_weights")}))


if __name__ == "__main__":
    main()
