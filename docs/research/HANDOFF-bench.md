# Handoff: comparative benchmark (hms-bench)

Done and verified: harness `benchmarks/competitive/` (runner, pinned per-comparator environments,
ANN and text workers, Lucene JVM comparator, matched-recall analysis, Bonferroni-adjusted bootstrap
verdicts, table/GAPS/README-block generator, traceability tests), `examples/competitive_text.rs`,
`--latency` and RSS fields in `public-bench ann-qgraph`, `.github/workflows/bench-refresh.yml`,
`land.py` (signed bot-branch PR), `modal_run.py`, docs in `docs/benchmarks/README.md`.
Verified through smoke runs on the Mac (all 12 ANN systems, 5 text systems end to end) and a
Modal smoke run (Linux path; two Linux defects found and fixed).

Not done: no committed results file yet, so COMPARISON.md and GAPS are empty. The Mac was at load
50-60 from other agents, so its numbers were not committed. The Modal `light` run was stopped
unfinished to save budget (about $2-3 of the $30 cap spent).

Indicative smoke findings (Mac, loaded, 5k vectors / NFCorpus; NOT claims): text QPS HMS far ahead
of tantivy/SQLite/Lucene; nDCG tied; HMS loses ingest time (about 10 s vs 0.1-0.6 s), serving RSS,
cold start and per-op replace; ANN QPS at recall 0.95 below FAISS HNSW; no ANN cold start or
update path. These must be re-measured before they become gaps.

Next command (writes results into the repo, then regenerate and commit):

    uv run --with modal modal run --detach benchmarks/competitive/modal_run.py --suite light --cap-usd 8
    ./benchmarks/competitive/hms-bench table      # also writes docs/benchmarks/GAPS.md and benchmarks/competitive/GAPS.json
    uv run --no-project --python 3.13 --with pytest --with numpy==2.5.4 pytest -q benchmarks/competitive/tests

`--detach` needs the results fetched from the app afterwards; without it keep the terminal open.
Then run `--suite heavy` the same way (SIFT has no HMS row: qgraph is angular only). Set the
repository secret BENCH_BOT_TOKEN for auto-merge of the weekly PR. Known gaps in the harness: no
Node-API arm for HMS (the accepted serving_memory_v40 protocol drives both arms from Node), ANN
index-size and install-dependency rows are informational (different quantities).
