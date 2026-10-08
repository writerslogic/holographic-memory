# Next sessions: executable specs (written 2026-10-07)

The Opus session (prompt 2) and the Sonnet session (prompt 3) follow these specs verbatim. Each
spec names its files, commands, parameters, acceptance test, cap and results file. Where a spec is
silent, the defaults in the session prompt apply. `docs/CONTINUATION.md` records what is done and
the exact next command; this file records what to do.

## Goal and arenas (maintainer, 2026-10-07)

HMS outperforms every competitor on every metric except build time. Three arenas, each with the
metric that decides it:

- ANN index, vs FAISS HNSW, hnswlib, Glass, SymphonyQG, RaBitQ, DiskANN/Vamana, ScaNN, NGT-QG,
  Lucene/Vespa HNSW: single-thread QPS at recall@10 0.90 and 0.95, and index bytes, same machine
  and session. Anything published must be measured on x86.
- Text retrieval, vs SPLADE v2/v3, BM25, ColBERTv2 and dense instruction encoders
  (Qwen3-Embedding-8B, E5-Mistral): BEIR nDCG@10 with official pytrec_eval.
- Agent memory, vs Supermemory, Exabase M-1, Mem0, CortexDB, Zep: LongMemEval end-to-end answer
  accuracy on the frozen split, fixed reader and judge, bootstrap CIs, plus one run with the
  paper's own reader so the vendor comparison is like for like.

Build time is the only metric where losing is acceptable; it is still reported.

## What would falsify "dominant" (report it as found)

Any ANN system beats HMS on QPS at either recall target on either dataset on x86, or has a
smaller index at equal recall; the HMS hybrid retriever scores below SPLADE v3 on the BEIR average
or below any listed published encoder; HMS answer accuracy on LongMemEval S dev is below the best
vendor number with the same reader, or any question type regresses against the tuned pipeline; a
research item fails its go/no-go test (published as a negative result, not dropped).

## Maintainer's decisions (everything else is the session's)

Modal caps: $40 for the LongMemEval dev reader runs, $20 of OpenAI API for the one comparability
run, $150 for the encoder programme (teacher scored once on Modal, tuning local on this M4, one
full run), $60 for the 10M ANN run, $15 for the x86 comparison run; the billion-scale run is a
separate decision for which the design and estimate are written (spec A5). The Sonnet session
submits the ann-benchmarks PR after x86 numbers exist; the Sonnet session drafts the arXiv paper
and the maintainer submits it. OPENAI_API_KEY is in ~/.env (mode 600): sessions load it with
python-dotenv or `set -a; . ~/.env; set +a`, call the API from a local script (never inside Modal,
no Modal secret), cache every response by content hash, and never print, log, commit or pass the
key on a command line.

## State the specs start from

- `main` holds the checked graph cache (format HMSGRF02), the merge verification
  (`benchmarks/results/qgraph_merge_verification_heldout.json`: the new default search path is
  kept, x1.30-1.58 over the old one; the 4-bit opt-in encoding is a memory-only gain), the
  competitor harness (`benchmarks/public/evaluate.py ann --extra`, `ann_extra.py`,
  `lucene_hnsw/LuceneHnsw.java`) and the M4 test-set comparison
  (`benchmarks/results/public_qgraph_<set>.json`, section "Quantized graph index" of
  `docs/PUBLIC-BENCHMARKS.md`, labelled "Apple M4, NEON; x86 numbers pending").
- Held-out tuning: `public-bench ann-qgraph --index vertex --holdout 2000` (train vectors held
  out as queries; the test set is never read for a decision). Chosen parameters: nytimes degree
  64, glove degree 32, build_ef 200, alpha 1.0, `--residual` (8-bit codes); rerank 16 at 0.90 and
  64 at 0.95 on nytimes, 16 on glove.
- Timing protocol: single thread, one query at a time, 3 repeats (median and min), load recorded,
  every timed command under `/Volumes/A/.hms-target/timed.sh` (hold at most 15 minutes, which is
  why `evaluate.py` runs one system per invocation and `ann-merge` joins the parts).
- The machine is a shared 10-core Apple M4; load is often 5-250. All SIMD kernels are NEON.
- Build: `export CARGO_TARGET_DIR=/Volumes/A/.hms-target CARGO_BUILD_JOBS=6`; the gate is
  `cargo fmt -- --check; cargo +1.99.0 clippy --all-targets -- -D warnings; cargo +1.99.0 clippy
  --all-targets --all-features -- -D warnings; cargo test --locked; cargo test --locked
  --all-features; cargo +1.89.0 check --all-targets; cargo deny check; uvx ruff check
  benchmarks/public`.

## (X) x86 port and run: the blocker for every public number

Files: `src/core/qgraph/kernels.rs`, `src/core/qgraph/vertex.rs`, `src/core/qgraph/build_kernels.rs`
(every `#[cfg(target_arch = "aarch64")]` NEON kernel gets an AVX2 and an AVX-512 version behind
`cfg(target_arch = "x86_64")` with runtime detection via `is_x86_feature_detected!`, keeping the
scalar fallback); one test `kernels_agree_on_every_path` that runs every available path on random
inputs (seeded, 1,000 vectors, dims 24/100/128/256/961) and asserts bit-identical results
(integer kernels) and identical f32 results (float kernels) against the scalar path. On the M4 the
test exercises NEON vs scalar; the x86 paths are exercised by the Modal run.

Modal run (cap $15): `benchmarks/public/ann_x86_modal.py`, a CPU-only Modal function (image: Rust
stable, Python 3.12, faiss-cpu, hnswlib, ngt, rabitqlib, Java 17; the ann datasets pulled from
`~/.cache/hms-bench` via a Modal volume `hms-ann`), `cpu=16, memory=65536`, records `lscpu` and
the Modal container id in the results file. Inside: build `public-bench` (release, `--features`
none), then the step-4 protocol verbatim (HMS VGraph default encoding and the 4-bit opt-in rows,
then `evaluate.py ann` builtin, then `--extra symphonyqg rabitq ngt lucene_hnsw glass diskann
scann`, repeats 3, single-threaded search, then `ann-merge`), for nytimes-256-angular and
glove-100-angular. Output: `benchmarks/results/public_qgraph_<set>_x86.json`. Glass, DiskANN and
ScaNN build on x86: pin `glass` from `git+https://github.com/zilliztech/pyglass` (record the
commit), `diskannpy==0.7.0`, `scann==1.4.2`; SymphonyQG's own repository needs AVX-512 and is
run from source if the container has it (record `lscpu` flags), otherwise `rabitqlib.SymqgIndex`
stands in as it does on the M4 and the row says so. Acceptance: both results files exist with
`load_gate_met` true for every reported row, versions pinned, and the doc's x86 table is primary
with the M4 table second. The paper and the ann-benchmarks PR use these numbers.

## (A) ANN

### Systems that beat HMS on the M4 (from `public_qgraph_nytimes-256-angular.json`, preliminary)

Glove was not run and every nytimes row is above the load gate, so this list is provisional and
is re-derived after the rerun (see `docs/CONTINUATION.md`).

- SymphonyQG (rabitqlib `SymqgIndex`, raw refinement): 5,913 vs 4,409 QPS at recall 0.90 with the
  8-bit VGraph (0.75x). Technique (Gou et al., SIGMOD 2025): 1-bit RaBitQ codes of every neighbour
  stored with the vertex and estimated for all neighbours in one FastScan-style SIMD pass, with the
  raw vector read only to refine the few candidates that pass; plus a graph refined for that
  estimator. VGraph change: the 1-bit screen of (A0) is the adoption (skip the 8-bit estimate when
  the 1-bit bound cannot enter the pool); its scalar form lost (A0), so batch the screen over all neighbours of a
  vertex (one pass over the row) and keep the 8-bit code only for the survivors, which also cuts
  bytes toward RaBitQ's. The paired `timed.sh` run decides.
- RaBitQ IVF 1+4 / 1+8 and RaBitQ HNSW 1+4 (rabitqlib): 55-95 MB against 172-226 MB for VGraph
  (0.25-0.42x of HMS bytes) at 0.2-0.5x of its QPS. Technique (Gao and Long, SIGMOD 2024; extended
  RaBitQ 2025): a 1-bit code plus 4 or 8 extra bits per coordinate with a provable error bound.
  VGraph change: an extended-RaBitQ vertex code (1+4 bits, a rotated 1-bit sign code and a 4-bit
  residual) in place of the 8-bit LVQ code and 8-bit residual, behind `--vertex-code rabitq`;
  acceptance: index bytes within 1.3x of RaBitQ HNSW 1+4 at no QPS loss against the 4-bit opt-in
  encoding (paired, both sets). Spec it under (A3) if the screen lands first.
- Build time: SymphonyQG 17 s, RaBitQ 12-29 s, FAISS 51-93 s against VGraph 111 s (nytimes, load
  47): spec (E).

### (A0) Committed with paired timing (ideas-tree session, 2026-10-07)

Both search changes are on `main` (6b714bf; `docs/research/IDEAS-2026-10.md` checkpoint 3,
`benchmarks/results/qgraph_stop_screen_heldout.json`, paired coordinator rounds on held-out
train vectors, nytimes 5 / glove 4, load 11-185 so only the paired ratios count):

- Patience stop rule (`VSearchParams::patience`, `--patience`): 17-29% fewer code estimates at
  equal recall; paired QPS x1.33 at 0.95 on nytimes (4/5 rounds) and x1.42 on glove (4/4),
  neutral at 0.90. Library default stays 0. The test-set rerun adds HMS rows with
  `--patience 256` (nytimes) / `--patience 384` (glove) beside the fixed-ef rows.
- 1-bit screen (`from_graph_with(.., screen)`, `--screen`, `--screen-sigmas`): NEGATIVE as
  implemented, x0.65-0.87 in every arm although 76% of the 8-bit estimates were skipped: the
  scalar per-neighbour screen costs as much as the SDOT estimate it replaces. Stays opt-in and
  off. The NEON kernel's microbenchmark (`benchmarks/results/qgraph_screen_microbench.json`)
  reaches 0.25 of the estimate per neighbour, yet every screened loop model is slower than
  estimating all 64 fresh neighbours: the 8-bit path is bound by overlapping row fetches, and
  the screen serializes the survivors' fetches behind its result. The SymphonyQG adoption is
  therefore not a kernel but a loop that overlaps those fetches (pipelined expansions); only
  that version is worth an end-to-end run.

### (A1) Learned traversal (research item)

Prior art to cite and surpass: Baranchuk et al., ICML 2019 (learned routing in graph search);
Li et al., SIGMOD 2020 (learned early termination). Ours replaces the fixed beam (ef) with a learned
expand/stop policy over the quantized-graph search state. The baseline is the patience stop rule
of (A0) once it lands, not fixed ef: a learned stop must cut code estimates at recall 0.95 by at
least a further 10% against patience (and 15% against fixed ef) to pass.

- Files: `public-bench ann-qgraph --oracle-trace <file>` (new flag) logs, per query and per
  expansion step, the state features and the oracle decision; `benchmarks/public/traversal_policy.py`
  trains the scorer (numpy / scikit-learn gradient boosting, no Modal) and exports it as a flat
  f32 table; `src/core/qgraph/policy.rs` loads it behind `--policy <file>` and
  `VSearchParams::policy`; results in `benchmarks/results/qgraph_learned_traversal_heldout.json`.
- State features (per step): the query code's scale, pool size, best-so-far estimate, the k-th
  best estimate, the candidate's estimate and its rank in the pool, the mean and min of the
  candidate's neighbour code estimates (already computed when the row is scanned), hops taken,
  code estimates so far.
- Training signal: offline oracle. For each `--holdout 2000` query on both datasets, run the fixed
  search at ef 512 recording every expansion; the oracle expand/stop labels are the minimal prefix
  of expansions that reaches the exact top-10 (stop = first step after the last needed expansion).
- Policy model: a small MLP (2x32) or a 64-tree gradient-boosted scorer; inference must cost under
  5% of one code-estimate evaluation, measured with the same `timed.sh` harness (report ns per
  call beside ns per estimate).
- Kill test, first: early termination alone (fixed expansion order, learned stop). Go only if the
  mean number of code estimates per query at recall 0.95 on nytimes held-out falls by >= 15%
  against the tuned fixed-ef row of `qgraph_merge_verification_heldout.json`. Then the full
  expand/stop policy.
- Acceptance: paired A/B under `timed.sh` against the tuned fixed-ef search on both sets, QPS at
  0.90 and 0.95 (interpolated as in `evaluate.py`), 11 rounds, same sign-test rule as the merge
  verification. A regression or a failed kill test is a negative result: report it, keep fixed ef
  as the default. Budget: 1 day. No Modal.

### (A2) Learned graph (research item)

Prior art: DiskANN/Vamana alpha-pruning (Subramanya et al., NeurIPS 2019); Dong et al. 2020
(learned partitions). Objective: edge selection that minimises greedy hops to the true neighbours
of held-out queries. Parameterisation: per-vertex selection over the candidate set Vamana already
computes (`Rows::insert_pass` candidates), by iterated local search (swap one edge, keep if the
held-out greedy-hop objective on a 2,000-query sample improves); a differentiable relaxation only
if the local search is too slow.

- Files: `BuildParams::learned: bool` (default false) and `src/core/qgraph/learned.rs`;
  `public-bench ann-qgraph --learned`; results `benchmarks/results/qgraph_learned_graph_heldout.json`.
- Kill test: re-prune a 100k-vector nytimes subset (first 100k train rows, held-out 2000 of them);
  go if recall@10 at fixed ef (96 and 384, rerank 16/64) rises >= 0.5 point at equal degree. 1-day
  budget.
- If go: both datasets; build time may rise up to 10x; search speed and memory may not fall
  (paired against the Vamana graph at equal degree and memory, `timed.sh`, 11 rounds).

### (A3) Quantized-distance construction with exact re-check in prune

Files: `src/core/qgraph/build_kernels.rs`, `mod.rs` (`Rows::insert_pass`, `robust_prune`);
results `benchmarks/results/qgraph_build_quantized_heldout.json`. Construction search uses the
8-bit build codes for ordering and the exact distance only for the final prune decision; recall at
fixed ef, QPS and index_bytes must not regress (same graph-hash or paired recall rows, `timed.sh`).
Also: the build-bounds screen is on only at dim >= 128 and was measured at 100 and 256 dims;
measure 128 and 192 (synthetic Gaussian-mixture sets of 200k vectors via `prepare.py --synthetic`)
or keep the cutoff documented as untested there. Glove construction is still about 85% candidate
search: compute the bounds for four candidates at once (one NEON pass over four code rows).

### (A4) The 10M run (cap $60)

`benchmarks/public/prepare.py deep1b-10m` (first 10M base vectors of Deep1B, 96 dims, with the
public 10k queries and ground truth; BigANN-10M, 128 dims, as the alternative), sha256 recorded.
VGraph (degree 64, build_ef 200, default encoding, held-out tuning on 2,000 train vectors first)
vs `diskannpy` 0.7.0 in-memory at its recommended parameters (R 64, L 100, alpha 1.2), same
protocol as step 4, on a Modal x86 container (`cpu=32, memory=131072`) or locally if RAM allows
(10M x 96 x 4 = 3.8 GB of vectors plus the graph: local is feasible on the 32 GB M4). Output:
`benchmarks/results/public_qgraph_10m.json`; a loss is written as a loss.

### (A5) Billion-scale design (document only)

`docs/BILLION-SCALE.md`: on-disk layout (vertex codes and adjacency in one 128-byte-aligned row
per vertex, raw vectors on SSD for the final re-rank), per-vector memory budget (code bytes +
4 x degree + ids), sharded Vamana merge as in DiskANN (k-means into 40 overlapping shards, build
each, merge by union and re-prune), and a cost estimate at Modal's rates (CPU $0.0473/core-hour,
memory $0.008/GiB-hour): dataset staging of 120-384 GB about $5 plus hours of transfer; build on
64 cores and ~512 GiB for 6-12 h, $150-350; search evaluation $20-40; a DiskANN baseline under the
same protocol $150-350; plus 50% for one failed attempt; total $500-1,200, +-50%, to be refined
after the 10M run. Execution needs the maintainer's separate go-ahead.

## (B) Index-aware learned encoder (research item)

Prior art to cite and position against: JPQ (Zhan et al., CIKM 2021) and RepCONC (Zhan et al.,
WSDM 2022), which jointly train an encoder with product quantization for IVF; ours is the first
against a quantized graph with LVQ and RaBitQ codes. SPLADE v2/v3 for the sparse objective.

- Backbone: a 110M-parameter encoder with a public licence (`bert-base-uncased` or
  `distilbert-base-uncased` first; larger only if budget remains). Training data: MS MARCO
  passage train triples with hard negatives mined by Qwen3-Embedding-8B (top-50 minus positives);
  no BEIR test data.
- Teacher: Qwen3-Reranker-8B scores computed ONCE on Modal
  (`benchmarks/public/sparse_teacher_modal.py`, H100, about $30), cached on the `hms-lme` volume
  under `sparse/teacher/<sha256 of (query, passage)>.json`.
- Loss: MarginMSE(student margin, teacher margin) + lambda_F * FLOPS(q, d) + lambda_I * L_index,
  where for a batch of documents D with exact similarities s(q, d) and VGraph-code-estimated
  similarities s~(q, d) (8-bit LVQ query code against the document's LVQ code and RaBitQ edge
  code), L_index = mean over in-batch neighbours of (s(q, d) - s~(q, d))^2 - mu * (min over the
  true top-k set N of s~(q, d) - max over D \ N of s~(q, d)). Give lambda_F, lambda_I and mu
  as swept values in the results file.
- Kill test (local, M4 Metal, `benchmarks/public/sparse_train.py`): 100k-triple subset, with and
  without L_index; go if the exact-vs-estimated similarity error on MS MARCO dev top-100
  candidates drops >= 20% at no MS MARCO dev MRR@10 loss (within 0.002). Record every setting in
  `benchmarks/results/sparse_local_sweep.json`.
- Local tuning protocol: 2-4k steps per setting, MS MARCO dev only. Then the single full run on
  Modal (`benchmarks/public/sparse_train_modal.py`, H100, resumable checkpoints on the volume,
  $150 cap total including the teacher), checkpoint exported as safetensors + tokenizer with sha256.
- Inference: `src/core/models/sparse.rs` (candle) encodes text to the sparse vector and writes it
  into the existing inverted index; the `sparse` kind in the document API and in `public-bench
  beir`.
- Published comparison (`evaluate.py beir`, official pytrec_eval, ONCE): HMS sparse, HMS sparse
  without the term (ablation), HMS hybrid (sparse + Qwen3-Embedding-8B dense, RRF, and single-pass
  fused scoring once both share the index), vs BM25 and the released SPLADE checkpoint through our
  harness, ColBERTv2 and the strongest dense encoders by published numbers (labelled), on SciFact,
  NFCorpus, FiQA, ArguAna, SciDocs, TREC-COVID (at least four beyond the first two). Results:
  `benchmarks/results/public_beir_sparse.json`. If the kill test fails, train the plain distilled
  encoder under the same cap so the BEIR comparison still exists, and report the term as a
  negative result.

## (C) LongMemEval end-to-end answer score

- Reader: `Qwen/Qwen3-30B-A3B-Instruct-2507` @ `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`
  (`MODELS["large"]` in `benchmarks/public/longmemeval_modal.py`), greedy, max 256 new tokens.
  Judge for the open runs: the same model with the official judge prompt from
  `~/.cache/hms-bench/downloads/longmemeval_eval_utils_9e0b455f4ef0.py`
  (`get_anscheck_prompt`), so reader and judge are fixed.
- Fixed answer prompt (verbatim, the `{}` fields filled):

  ```
  You answer questions about a user from excerpts of their earlier chat sessions.
  Use only the excerpts. If they do not contain the answer, reply exactly: I don't know.
  Answer in one short sentence with the specific value, name, date or list asked for.
  If several excerpts give different values for the same thing, the most recent dated
  excerpt is current unless the question asks about the past or about a change.
  If the question needs facts from several excerpts, combine them.

  Today's date: {question_date}
  Excerpts (oldest first; each starts with its session date):
  {excerpts}

  Question: {question}
  Answer:
  ```

  `{excerpts}` are the top-K items of the tuned pipeline (config
  `benchmarks/results/longmemeval_dev_final.json`, its cached rank output on the `hms-lme`
  volume), each as `[session date] <turn or fact text>`.
- New `reader` stage in `longmemeval_modal.py`: inputs the cached top-K ranking, K in {10, 20,
  50}; levers, each off and on: (i) reader-side aggregation for multi-session and
  temporal-reasoning questions (group excerpts by session and prefix each group with its date and
  a count), (ii) event-date extraction beside the session date (the date mentioned inside the
  excerpt, extracted by the same model in a `dates` stage, cached), (iii) abstention calibration
  (answer "I don't know" only when the top-1 score is below a threshold tuned on dev). Cache keys:
  sha256 of (model id, revision, prompt text, decoding parameters) per call, reader and judge
  outputs cached on the volume like the other stages; `Budget` and its reservation logic
  unchanged; `--cap 40` for the whole set of runs.
- Command: `uvx modal run benchmarks/public/longmemeval_modal.py --stage reader --dataset s
  --part dev --models large --k 10,20,50 --levers all --cap 40 --tag s-dev-reader --out
  benchmarks/results/longmemeval_dev_answer.json`. Never `--dataset m`, never held-out ids.
- Comparability run: `benchmarks/public/longmemeval_reader_api.py` reads the cached top-K
  rankings (`modal volume get hms-lme ...`), loads OPENAI_API_KEY from ~/.env with python-dotenv,
  calls gpt-4o as reader and judge with the paper's prompts, caches responses by content hash under
  a gitignored `benchmarks/results/cache/`, enforces the $20 cap from the `usage` field, never
  prints the key. Output appended to `longmemeval_dev_answer.json` under `comparability_gpt4o`.
- Report: accuracy overall and per type with bootstrap 95% CIs (1,000 resamples, seed 20261007),
  dev-only label, cost per run; `docs/PUBLIC-BENCHMARKS.md` table placing HMS beside the paper's
  Table 3 and each vendor's self-reported number with its K and reader where known, same-reader
  rows marked.

### (C2) Compositional retrieval and supersession edges (research items)

Gated 1-day dev experiments in `benchmarks/public/holographic_memory_proto.py` (numpy; no Rust
port unless dev justifies it).

- Compositional: VSA binding of entity, role and time-bucket codes to each atom's sign code;
  unbind the query, then resolve by exact nearest-neighbour search over the atom codes (numpy
  first; VGraph traversal only after the kill test passes) instead of the single trace that
  failed on capacity before. The slot-emitting query stage uses the cached LLM outputs, adding a
  `slots` stage to `longmemeval_modal.py` only if the cache lacks it, inside the $40 cap.
  Kill test, three seeds: go if multi-session or temporal-reasoning turn recall rises >= 0.02 over
  the holographic flat Hamming arm in `benchmarks/results/holographic_dev.json` and is >= dense
  exact (max-over-turns cosine), with no other question type down >= 0.01.
- Supersession: fact B "replaces" fact A when entity and attribute match with a later date;
  retrieval returns the newest unless the question asks for history. Go if knowledge-update
  accuracy rises with no other type down.
- Results: `benchmarks/results/compositional_dev.json` and `supersession_dev.json`, scored per
  type on S dev with the official metrics (`rank_turns` in the prototype) and CIs against the
  tuned pipeline. The dev set is 100 questions (per type 6-27), so every CI is wide; state the
  widths and never claim a separation the n cannot support.

### (C3) Grassmannian session subspace (numpy, 3-hour cap, no Modal)

Never while the timing lock is held. For each S-dev session take the rank-k right singular basis
U (d x k) of its L2-normalised cached turn embeddings (`~/.cache/hms-bench/holo_dev/turn_emb.f16`,
rows mapped to questions, sessions and turns as `rank_turns()` in `holographic_memory_proto.py`
does; read it, do not assume), k in {1, 2, 4, 8}, stored as float16 and int8. Score a unit query
by projection energy ||U^T q||^2 and by singular-value-weighted energy. Controls at equal bytes:
the mean-pooled session vector and the top-m turns kept so bytes match. Two-stage variant:
subspace shortlist of m in {5, 10} sessions, then exact turn cosine inside them; report shortlist
recall@10 and turn metrics. Official metrics (session and turn R_all@5, nDCG@10) overall and per
type with bootstrap CIs (1,000 resamples, seed 20261007), bytes per session and per question,
numpy query cost labelled indicative with `uptime` load. Selection rule, stated before running: the
smallest bytes whose session R_all@5 and nDCG@10 are within 0.005 of dense exact. Pass only if
that setting exists at >= 4x fewer bytes per session than dense float16 and the two-stage
shortlist recall@10 is >= 0.99; otherwise a negative result, written up, not tuned past. Report
every setting tried, dev-only. Script `benchmarks/public/geometry_experiments.py` (subcommand
`subspace`), results `benchmarks/results/geometry_dev.json`, and a `docs/PUBLIC-BENCHMARKS.md`
subsection "Representation experiments (S dev)".

## (D) On-device LLM speed

- Part 1, prefix KV caching (`src/core/models/llm.rs`, `qwen3.rs`; the fact and query prompts in
  `src/core/models/prompts.rs` share a fixed instruction prefix): compute the prefix KV cache once
  per `Generator` and reuse it for every call, batch-1 and batched paths. Acceptance: byte-identical
  outputs to the current batch-1 path on the 64 prompts from `benchmarks/public/batching_inputs.py`,
  checked with `batching_compare.py` (Metal; weights in `/Volumes/A/hms-models`). Paired tokens/s
  and sessions/s before and after under `timed.sh`. Results
  `benchmarks/results/local_models_prefix_cache.json`; on-device doc section updated.
- Part 2, 4-bit weights behind a config option (candle quantized Qwen3, GGUF Q4_K_M from the
  pinned revision, sha256 recorded) or an MLX / llama.cpp backend behind a feature flag.
  Acceptance: fact outputs on the first 300 S-dev sessions compared with the cached bf16 outputs
  via `benchmarks/public/compare_fact_caches.py`, then S-dev retrieval re-ranked with those facts
  via `longmemeval_sweep.py` on cached scores (no Modal) within 0.004 of
  `benchmarks/results/longmemeval_dev_final.json`. Report tokens/s and peak memory (paired, under
  `timed.sh`). If it fails, bf16 stays the default and the gap is reported. Results
  `benchmarks/results/local_models_q4.json`.

## (E) Build-time parity

A later session. VGraph build time against FAISS HNSW and hnswlib under the bit-identical-graph
rule (`HMS_QGRAPH_BUILD_PROFILE=1` hash unchanged); the only metric where losing is acceptable;
still reported in every table.

## (F) Publication

- One arXiv paper (cs.IR), `docs/paper/` (LaTeX; every table generated from
  `benchmarks/results/` by `docs/paper/tables.py`). Outline: (1) the index: LVQ vertex codes,
  RaBitQ edge codes, Vamana construction with build-bound screening, the checked cache and the
  measurement protocol; (2) x86 results vs every ANN competitor (primary table), M4 results
  second; (3) learned traversal and learned graph, positive or negative, positioned against
  Baranchuk et al. 2019, Li et al. 2020 and DiskANN/Vamana; (4) the index-aware encoder vs its
  ablation, SPLADE, BM25 and the published dense and late-interaction numbers, positioned against
  JPQ and RepCONC; (5) LongMemEval end-to-end with CIs beside the paper's Table 3 and the vendor
  claims, same-reader rows marked; (6) compositional, supersession and subspace dev results; (7)
  protocol: held-out tuning, load, interpolation, caps; (8) limitations: build time, anything not
  run, every metric lost, ColBERTv2 and 7B dense encoders compared by published numbers only, the
  held-out LongMemEval_M run not executed. Every claim ties to a results file.
- Limitations also carry a no-compute "Representations considered" paragraph: hyperbolic
  (Poincare) embedding of the session/turn hierarchy, cellular-sheaf coherence for fact updates,
  ultrametric/p-adic codes, holonomic/non-abelian bundles, non-commutative motives, knot and link
  invariants; for each a concrete retrieval construction or "no construction", and why it was not
  tested (the dev set is 100 questions, so 0.01 margins cannot separate CIs; knot invariants change
  discontinuously under small input changes and the Jones polynomial is #P-hard in general).
- The README states plainly, from `benchmarks/results/holographic_dev.json`, which HMS components
  are holographic (superposed sign-code traces with graceful degradation) and which are not (the
  dense + rerank pipeline that carries the accuracy). Renaming is the maintainer's decision, not a
  session's.
- Venues: the ann-benchmarks PR after spec X (`benchmarks/public/ann_benchmarks/`: Dockerfile
  building `public-bench` for x86, a Python wrapper implementing the upstream `BaseANN` interface
  through a subprocess or a thin pyo3 surface, a config YAML for both sets at the parameters in
  the x86 results files; one full fit+query pass locally through the upstream harness first); a
  big-ann-benchmarks PR after A4; SISAP 2027 only if inner-product search is added; MTEB once the
  retriever exists. Missing per venue: ann-benchmarks needs x86 numbers; big-ann needs the 10M
  (and ideally 100M) run; MTEB needs the retriever of spec B.

## Checkpoint rule

After every numbered task, commit (signed), write its results file, and update
`docs/CONTINUATION.md` with what is done and the exact next command. If context, time or budget
runs short, stop at the last checkpoint; the same prompt is run again and continues from
`docs/CONTINUATION.md`. Never leave a task half-done without a note there. Do not re-measure what
a committed results file already records unless the code changed.

## Standing rules

Decisions never use test queries, neighbors.i32, BEIR test qrels for tuning, or the held-out
LongMemEval ids in `benchmarks/public/longmemeval_split.json`; no claim without a file in
`benchmarks/results/`; negative results are reported plainly and a metric where HMS loses is
written as a loss with the margin; recall gets bootstrap 95% CIs over queries (1,000 resamples,
fixed seed), QPS reports median and min of the repeats, paired ratios report their spread; record
load averages for every timing and run every timed command under `/Volumes/A/.hms-target/timed.sh`
(a machine-wide lock; hold it at most 15 minutes; never while a local training job runs); the only
cargo, clippy and test runs are one incremental `CARGO_BUILD_JOBS=6 cargo build --release` per
code change that must be measured, one module-scoped `cargo test --release --lib <module>` before
that module's final commit, and the full gate exactly once at the end of the session with targeted
reruns only for failed steps; Modal and API spend only under the stated caps, checked after every
run with `uvx modal billing report --for today` (the ledger is an estimate and has under-reported);
commit with `git -c user.email=david@writerslogic.com commit -S`; regenerate CHANGELOG.md with
`git cliff -c cliff.toml -o CHANGELOG.md` as `chore: update changelog`; push main only; never tag,
publish, yank, or run the single LongMemEval_M job without the maintainer's explicit go-ahead. Use
`export CARGO_TARGET_DIR=/Volumes/A/.hms-target`.
