# Comparative benchmark (hms-bench)

One command reproduces the HMS comparison against the vector, text and embedded-search systems
people actually use. The generated results are in [COMPARISON.md](COMPARISON.md); the rule that
governs everything below is that **every number in that file traces to a committed results file
under `benchmarks/results/competitive/`, and a comparator that beats HMS is reported as beating
HMS.**

```sh
./benchmarks/competitive/hms-bench list                    # systems, suites, what was not run
./benchmarks/competitive/hms-bench run --suite light       # the hosted-runner subset (about an hour)
./benchmarks/competitive/hms-bench run --suite standard    # a local workstation
./benchmarks/competitive/hms-bench table                   # regenerate COMPARISON.md from committed results
./benchmarks/competitive/hms-bench table --check           # fail if the committed table is stale
uv run --no-project --python 3.13 --with pytest --with numpy==2.5.4 pytest benchmarks/competitive/tests
```

`hms-bench run` builds nothing itself: it needs `public-bench` and the `competitive_text` example
(`cargo build --profile dist --bin public-bench --example competitive_text`, with
`CARGO_TARGET_DIR` if you use a private target directory; `dist` is the release build without the
symbol table, the binary the install-size row measures), `uv`, a JVM (17+) for the Lucene row,
and network access for the first run (datasets, pinned wheels, two Lucene jars with pinned sha256).

## What is compared

| Family | Systems (version pinned in `benchmarks/competitive/requirements/*.txt`) |
|---|---|
| Vector / ANN | hnswlib, FAISS (HNSW, IVF-PQ with and without exact refine, Flat), USearch (f32, int8), Annoy, sqlite-vec, LanceDB (IVF-PQ, IVF-HNSW-SQ), Chroma, Qdrant embedded local mode, Milvus Lite (HNSW, IVF-Flat), DuckDB vss, NGT (ONNG), PyNNDescent, ScaNN and DiskANN (Linux only), Voyager |
| Text (BM25) | SQLite FTS5, tantivy, Apache Lucene 9.12.3 (JVM), Whoosh, bm25s |
| Agent-memory systems | Mem0, Zep, Letta, LangMem: not run here; see "Plugging in more systems" |
| Not run, with reasons | pgvector (needs a PostgreSQL server; not installable from pip/cargo/npm) |

HMS is measured through its own binaries: `public-bench ann-qgraph` (the vertex graph with the
accepted configuration recorded in `benchmarks/results/qgraph_pool_paired_x86_glove_v4.json`,
tuned on a train holdout, never on the benchmark queries) and `examples/competitive_text.rs`
(the document API, lexical only, one chunk per document).

## Datasets

* **glove-100-angular** (1.18M x 100, ann-benchmarks): `light` uses the first 100,000 vectors and
  1,000 queries, `standard` 250,000 and 2,000, `heavy` all 1,183,514 and 10,000. For every subset
  the exact top-100 truth is recomputed (`hmsbench/data.py`); the files HMS and every comparator
  read are byte-identical, and their sha256 are recorded.
* **sift-128-euclidean** (1M x 128, ann-benchmarks): `standard` uses the first 250,000 vectors
  and 2,000 queries, `heavy` all 1,000,000 and 10,000. HMS runs `public-bench ann-qgraph
  --metric euclidean` (vertex index on the vectors as given, Euclidean truth).
* **BEIR SciFact (train qrels) and NFCorpus (dev qrels)**, the existing document-competition
  protocol: lexical only, top 100, one chunk per document, test qrels never read.
* **BEIR FiQA (dev qrels)**, `standard` and `heavy` only: 57,600 documents (the 38 empty ones
  are left out for every system), 500 queries, same protocol.
* **memory** (agent-memory-shaped): all sessions of the 100 LongMemEval_S dev questions as one
  searchable memory (about 5,000 sessions, truncated to 4,000 words), relevant documents are the
  answer sessions. The 400 held-out questions are never read.

## Protocol

* Single client thread, one query at a time, per-query `perf_counter_ns` (or the Rust/JVM
  equivalent) around the search call. Builds may use `threads` cores (4 light/standard, 8 heavy)
  for every system that can; HMS builds with the same count.
* Every system runs each phase (build, serve, ops) in a fresh process; the orchestrator reads the
  child's peak RSS from `wait4`. Cold start is the wall time from spawning the serving process to
  the first answered query (interpreter, imports, open, first query; the file cache is warm).
* Serve rounds are interleaved: each round runs HMS and every comparator once, in rotated order,
  so a load spike lands on all systems. Each phase takes a timing lock while it runs, never across a whole suite; the lock is a
  no-op on CI. Load averages are recorded per configuration; a load gate is not enforced on
  shared machines (`load_gate_met` is `null`, not `true`).
* **Matched recall.** Each comparator sweeps its documented good settings (HNSW M 16 and 32 with
  efConstruction 200 and ef 10..1024; IVF nprobe 1..256; Annoy 100 trees with search_k up to
  51,200; and so on, see `ann_worker.py`). For each target recall@10 (0.90, 0.95, 0.99) the row
  with recall at or above the target and the highest pooled median QPS over all rounds is chosen;
  that row's per-round values carry the confidence interval. A system that never reaches the
  target is reported as "not reached" with its best recall, never dropped.
  Comparators are never tuned below their documented defaults.
* **Intervals and verdicts.** 95% bootstrap intervals over rounds (or queries, for quality),
  Bonferroni-adjusted over the family of verdict metrics of the file ("family guard", as in the
  superiority ledger). The advantage factor is oriented so that above 1 means HMS is better; WIN or
  LOSS needs the interval of the factor to exclude 1; deterministic byte counts use a 5% band;
  everything else is TIE. `index size` and `install distributions` are printed without a verdict
  because the two sides measure different things (HMS reports its in-memory index, comparators the
  persisted directory; Python wheels versus statically linked crates).
* **Durability rows.** Replace/delete cost is one durable commit per operation on a persisted store
  (on macOS every system's commit ends in `fcntl(F_FULLFSYNC)`, which flushes the drive cache:
  HMS after each journal frame, tantivy through Rust's `sync_all`/`sync_data`, Lucene through the
  JDK's `FileChannel.force`, and SQLite because it runs with `fullfsync`; libraries that need an
  explicit save to be durable include that save in the figure).
* **Hardware classes.** Each file records its class (`hardware.class`) and the table has one
  section per file; rows from different classes are never put side by side. Hosted-runner rows are
  shared virtual machines: read the ratios, not the absolute numbers.

## Known measurement boundaries (read before quoting a ratio)

* HMS is timed through its Rust API; Python comparators through their bindings; Lucene through the
  JVM after a full untimed warm-up pass. Each section prints the boundary per system. The existing
  accepted serving rows (`benchmarks/results/serving_memory_v40.json`) drive both arms from Node,
  which narrows HMS's lead against SQLite; the two protocols are not interchangeable.
* Comparator query preparation (word splitting, SQL expression building) is done outside the timer
  to favour the comparators; each engine's own parsing stays inside it.
* LanceDB, Chroma and Milvus Lite run Rust or C++ runtimes with their own threads; QPS is still one
  client at a time.
* Qdrant's embedded local mode is an exact scan (its documentation points to a server for indexed
  search); it is included because it is the only Qdrant that needs no service.
* HMS has no persisted ANN index, so ANN cold start and ANN replace/delete are "n/m", and its ANN
  serving RSS is that of the process that also built the index (an upper bound on serving memory).
* Above 200k vectors the slow exact-scan systems answer a prefix of the query file
  (`protocol.query_caps`); recall and QPS are over that prefix.

## Self-updating table

`.github/workflows/bench-refresh.yml`: the `test` job runs the generator tests on every PR that
touches the harness, results or docs. The `run` job re-runs the `light` suite weekly, on demand and
on release tags on a GitHub-hosted runner (hardware class `github-hosted-ubuntu-24.04-x86_64`),
uploads the raw results as an artefact, and the `land` job regenerates `COMPARISON.md`, the README
block (between `<!-- BENCH:START -->` and `<!-- BENCH:END -->`) and `comparison.cells.json`, then
lands them through a `bot/bench-refresh-<run>` branch created with GraphQL `createCommitOnBranch`
(GitHub signs the commit), opens a PR and squash-merges it, deleting the branch. Merging past the
review requirement needs the repository secret `BENCH_BOT_TOKEN` (a token allowed to bypass it);
without it the PR is opened with the default token and left for a maintainer, and PRs opened with the
default token do not trigger CI. All actions are pinned by SHA.

## Heavy suite

Full GloVe-100 and SIFT-1M do not fit a hosted runner. Two supported ways to run them:

* Self-hosted or workstation: `./benchmarks/competitive/hms-bench run --suite heavy --only ann`.
  Set `HMS_BENCH_HW_CLASS` if the derived class is not descriptive. Commit the resulting `heavy-*.json` files.
* Modal: `uv run --with modal modal run --detach benchmarks/competitive/modal_run.py --suite heavy --cap-usd 12`
  ships the working tree to one 8 vCPU / 32 GiB container, builds HMS there, runs the suite and
  writes the results back with hardware class `modal-x86_64-8vcpu-32g`. The cap kills the run before
  the published rate would exceed it.

## Plugging in more systems

* **ANN or vector store:** add an `Adapter` subclass in `hmsbench/ann_worker.py` (`build`, `open`,
  `configure`, `query`, optional `replace`/`delete`/`flush`), register it in `ann_worker.SYSTEMS`
  and `hmsbench/registry.py`, add its pinned requirement to `hmsbench/envs.py` (`PINS`) and run
  `./benchmarks/competitive/hms-bench lock`.
* **Text or memory retrieval:** add a class with `build(documents)`, `open()`, `prepare(text)`,
  `search(prepared)`, `replace(doc)`, `delete(doc)` in `hmsbench/text_worker.py`. The `memory`
  dataset already has the agent-memory shape (sessions as documents, answer sessions as relevant),
  so Mem0, Zep, Letta and LangMem plug in as text systems with their own pinned environment; the
  stage that runs them is a separate piece of work and none of them is run here.
* Facts that are not measurements (service needed, encryption at rest) live in
  `registry.FACTS` with their basis; only HMS's carries a measured source.

## Files

| Path | Purpose |
|---|---|
| `benchmarks/competitive/hms-bench` | entry point |
| `benchmarks/competitive/hmsbench/` | runner, workers, analysis, table generator |
| `benchmarks/competitive/lucene/LuceneBm25.java` | Lucene BM25 comparator |
| `benchmarks/competitive/requirements/` | pinned, locked comparator environments |
| `benchmarks/competitive/tests/` | generator and traceability tests |
| `benchmarks/results/competitive/*.json` | raw results (one file per suite, dataset and hardware class) |
| `docs/benchmarks/COMPARISON.md`, `GAPS.md`, `comparison.cells.json`, `benchmarks/competitive/README-block.md` | generated |
