# Persistent-memory campaign, version 2

The frozen contract is `benchmarks/results/superiority_v2/contract.json`, SHA256
`811c35bbb0410019f890412bf3af0f3900218dd02e7d346346265a3f6a4bd013`.
It is a target, not a success claim. The inherited branch is
`research/2026-10-superiority`, base `648ac99`; the checkout was already dirty.
The baseline patch and source hashes preserve that distinction. No inherited
cloud budget authorizes new spending in this campaign. No protected held-out
evaluation, publication, commit, or push is authorized.

## Architecture and evidence ledger

| Status | Evidence | Consequence |
|---|---|---|
| Frozen baseline source | `engine/document_api.rs::search_documents` takes a shared mutation gate and reads the document map; `documents.rs::search` filters documents, scans eligible chunks for term DF and BM25, scores stored f64 embeddings exactly, sorts both lists, then blends or uses RRF | Concurrent readers can coexist; mutations/compaction exclude them. Filters change BM25 collection statistics. An index must preserve this contract, mixed stemmed/legacy chunks, IDs and versions. |
| Frozen baseline source | `engine/mod.rs::compact` takes the exclusive mutation gate; document mutations use the persistent transaction path | A faster benchmark bridge does not improve document search or compaction interference. |
| Source verified | `qgraph/vertex.rs` owns quantized codes, graph, optional residual codes; search refines codes with a float query, not original vectors | Candidate coverage and final ranking must be measured separately. Exact reranking cannot repair absent candidates. |
| Source verified | `engine/multi_hop.rs` dispatches single-hop algebraic recovery, two-relation rule rewrite, or explicit triple-store chaining | Graph diffusion is a complementary operator, not equivalent relation execution. Confidence is heuristic, not a proof or calibrated probability. |
| Source verified | Optional local-model stages in `document_api.rs`; separate phase/experimental modules | Shared embeddings/readers/budgets are required in matched retrieval tests. Algebraic results cannot establish document or QA superiority. |
| Artifact verified, not re-executed | `ann_extended_diagnostic_v1/report.md`, `acceptance.json`, retained HMS report: full GloVe, GCE C4 Intel Emerald Rapids, October 11–12; 540 accepted rows, nine complete methods, five query repetitions, one build each | Supplied 1,023/965/1,699/1,288/1,240 QPS values trace to this virtualized-x86 diagnostic. Not M4, not bare metal, not an accepted original-campaign run. |
| Provenance conflict resolved | `COMPETITORS.md` and scoreboard say no x86 run as of October 9; continuation records later C3/C4 runs | Older absence statements are stale globally; matched bare-metal/preregistered x86 cells remain unknown. Do not merge C3, C4, and M4 timings. |
| Artifact verified, limited scope | Extended report rejects IVF-PQ m100 because nprobe was recorded but not applied; Lucene/NGT timed out | Excluded rows stay excluded; incomplete methods do not get inferred performance. |
| Supplied claim with existing artifacts | `longmemeval_dev_answers_v1.json` and summary report 87/100 and 201/204 citation spans | Repeated-use development only. Membership is not entailment; vendor conditions differ. No held-out or vendor margin established. |
| Authoritative boundary | `SECURITY.md`, `PRIVACY.md` | Encryption at rest, signing, query plaintext, DP bundling and provenance have separate scopes. No PoSME/RATS verification/encrypted search/complete-erasure claim. |
| Unknown | Current serving RSS isolated from construction, modern high-dimensional ANN, product filtered concurrency/update/compaction frontiers, matched QA competitors | Empty measurements stay empty. Broad superiority is not established. |

## Ranked portfolio and three-route comparisons

| Priority / bottleneck | Conventional route | Mechanism-driven exploration | Simplification/replacement | Discriminating experiment and decision |
|---|---|---|---|---|
| A1 ANN misses | Wider beam and bounded original-vector rerank | Residual-feedback second query: perturb against query quantization error, union candidates, rank originals | Exact graph traversal and flat exact scan | Same graph, train-only disjoint queries, ef32/64/128/256; measure pool coverage, ranking loss, distances and time. Feedback must beat double-ef by 0.005 recall on both sets within 10% cost; otherwise reject default adoption. Final-ranking gap >0.005 prioritizes reranking; exact-traversal gain >0.005 prioritizes traversal precision; neither redirects to topology. |
| A2 Document scans | Filter-aware inverted lists then block-max WAND | Query-conditioned lexical/dense candidate allocation using measured shortlist uncertainty | Flat exact fallback for small/selective sets | Preserve filtered collection statistics and result-score ties; compare all three at 1/10/0.1% eligibility and across lexical/semantic/hybrid. Accept >=20% p95 improvement with exact lexical parity and <=0.005 quality loss; kill on stale/filter violations. |
| A3 Construction peak | Reuse scratch, reduce copies, compact adjacency | Staged quantization/rate allocation driven by observed graph coverage | Partitioned RaBitQ IVF | Profile allocations by phase before editing. Same total bytes and build threads. Require >=20% peak reduction and no primary recall/QPS regression; reject representation complexity without net benefit. |
| A4 Maintenance exclusion | Immutable searchable segments + update buffer | Query-preserving publication driven by versions/provenance | Retain exact map with snapshot readers | Stress acknowledged updates, deletes, restart, corruption and compaction concurrently. Zero stale/version/filter failures; <=10% p99 regression. Reject if transaction semantics weaken. |
| A5 Temporal/multi-hop quality | Explicit chains + validity intervals | Query-conditioned PPR supplemental candidates with source reconciliation | Direct triple indexes and exact source packets | Synthetic contradictions, supersession, rare facts first; later authorized QA at identical models/budgets. Preserve relation semantics; reject diffusion that silently changes operators. |

A1 is an information-gathering experiment, not proof of competitive advantage.
Its exact-score control uses original normalized float32 vectors with float64
cosine truth; rounding distinctions and original-ID overlap must be disclosed.
Existing LVQ and author-supported RaBitQ/SymphonyQG implementations are the
mechanism references; no novelty claim is made for query feedback.

## Envelope and subsystem decisions before new measurements

| Category | Win | Loss / boundary | Unknown | Recommendation |
|---|---|---|---|---|
| ANN | Historical C4 bytes vs FAISS HNSW | C4 fastest >=95% HMS loses QPS/bytes/build/RSS to RaBitQ; sampled HMS fails 99% | Current matched M4/x86 contract | Retain as control; augment only if ablations pass; partition replacement remains live. |
| Document pipeline | No matched efficiency win established | Exact scanning costs grow with eligible chunks | Filter/concurrency/update frontier | Augment candidate generation, retain exact fallback. |
| Structured memory | Explicit chains preserve relation operators | Existing holographic dev compression loses accuracy against flat controls | Calibrated uncertainty, temporal/contradiction benefit | Retain symbolic chains; experimental algebra must earn storage/cost. |
| QA | Operand selection improves reused development evidence coverage | Not entailment, not held-out, not a vendor comparison | Matched answer-quality advantage | Retain source provenance; no superiority claim. |
| Operations | Existing transactions and security boundaries | Exclusive compaction and large historical construction RSS | Failure/recovery/maintenance performance | Retain durability; measure before redesign. |

Immediate resource limits: local ARM only, current load above 3, zero new paid
budget, no untouched QA authorization. Quality diagnostics may run under load;
latency then remains descriptive and cannot pass a timing gate. Dedicated x86,
an idle host, matched competitor access and separate held-out authorization are
needed to close the corresponding cells, not to start independent local work.

## A1 measured result: reject feedback, retain topology investigation

Both datasets have a deterministic, disjoint training-only split of 10,000
database rows and 200 queries, seed 24301. No benchmark test file was opened.
The revised producer records all reranking pools. The checker verifies 19,200
rows per dataset against independently computed float64 raw-input cosine truth,
pool membership, prefix identity, wider-beam equivalence, and repeatability.
Six corruptions are rejected. The original less-complete evidence is retained
in each dataset's `attempt1/`; it is not silently overwritten.

| Sample / ef | Residual16 native recall | Original16 | Full-pool exact | Exact graph traversal | Feedback union | Double-ef single probe |
|---|---:|---:|---:|---:|---:|---:|
| GloVe / 64 | .9610 | .9610 | .9610 | .9605 | .9620 | .9935 |
| nytimes / 128 | .9445 | .9445 | .9445 | .9460 | .9470 | .9885 |

Native recall uses angular-distance threshold tolerance .001; strict .000001
recall and original-ID overlap are separately reported. Residual and original
reranking agree on recall at every sampled beam; exact traversal makes only
small positive/negative changes. The observed sample losses are principally
candidate discovery, not final ranking. This does **not** identify the cause of
the full-GloVe plateau or establish an algorithmic ceiling. Feedback fails the
preregistered native-recall gain at every beam on both sets, so it is rejected
without needing a favorable timing result. The implementation remains an
experimental diagnostic only. No production ANN default changed.

Timings are ineligible: the ARM host was shared and loaded; the first diagnostic
had no exclusive timing lock. No confidence interval shown here is corrected
confirmatory evidence. Exact control latency includes scalar float64 scoring,
and original-vector memory is charged separately; its runtime is not a tuned
competitor comparison. No ANN superiority was established.

Reproduce the diagnostic (output directories must be fresh; the runner refuses
to overwrite evidence):

```sh
uv run --script benchmarks/public/ann_failure_map.py build glove-100-angular --binary /Volumes/C/rust-target/release/ann-failure-map
uv run --script benchmarks/public/ann_failure_map.py run glove-100-angular --binary /Volumes/C/rust-target/release/ann-failure-map
uv run --script benchmarks/public/ann_failure_map.py run nytimes-256-angular --binary /Volumes/C/rust-target/release/ann-failure-map
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --script benchmarks/public/ann_failure_map.py check glove-100-angular
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --script benchmarks/public/ann_failure_map.py check nytimes-256-angular
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --script benchmarks/public/test_ann_failure_map.py
```

The build manifest binds the executable, source hashes, Cargo.lock and toolchain
before execution and verifies source/executable stability afterward. Later
document edits change the source snapshot; do not confuse a fresh build with
that frozen binary. Resolve the target directory from `cargo metadata` on other
machines. All inputs, mappings, compressed raw rows and summaries are under
`benchmarks/results/superiority_v2/{glove-100-angular,nytimes-256-angular}`.

## A2 product prototype: incremental lexical postings

Protocol: `superiority_v2/document_protocol.json`, SHA256
`5a686d645107fa0b0039f03aadd0ded9daaf63183c4e060ce2b13d3acaad2bbd`.
The source-level cost diagnosis is two full eligible-chunk term lookup passes,
for filtered document frequencies and BM25 scores. The prototype replaces these
with postings when a simple work estimate favors them; selective queries keep
the exact scan. Eligible chunks are still enumerated, dense scoring is still
exact and exhaustive, and both lists still sort fully. This is **not** bounded
candidate generation or WAND, and no durable dense index is integrated.

Postings are derived state inside the document store's existing lock. The same
transaction/replay paths update them; no on-disk format changes. Stemmed and
legacy terms stay separate. Sorted numeric slot postings preserve the existing
filtered collection statistics and floating-point summation order. Deletions
shrink sparse posting allocations; a sparse slot registry rebuilds when more
than half its slots are free and its high-water mark exceeds 1,024. That bounds
retained slot state by live data plus a fixed floor, but the rebuild and common
term shifts can pause readers. Sustained-update tail latency is unmeasured.

Independent critique identified and prompted the sparse-survivor reclamation
test, an assertion that a filtered differential fixture actually uses postings,
and synchronized reader/writer phases. Exact score/source/ID/version parity is
the correctness gate. Corpus quality and tuned competitor comparisons remain
separate work. A synthetic paired scan comparison cannot establish the frozen
product-superiority contract.

### Retained A2/A2b measurements

Apple M4, 10 CPUs, 32 GiB; 11 paired rounds per corpus size. Every round missed the load gate. Ratios below are descriptive candidate/baseline QPS ratios (baseline/candidate mean latency), not accepted speed claims. Each revision independently passed 12,672 exact query-output comparisons.

| N | mode | 1% filter | postings-only A2 | filter-index A2b |
|---:|---|---|---:|---:|
| 1000 | lexical | False | 1.795 | 1.527 |
| 1000 | semantic | False | 1.008 | 0.919 |
| 1000 | hybrid | False | 1.497 | 1.397 |
| 1000 | lexical | True | 0.968 | 2.953 |
| 1000 | semantic | True | 0.856 | 2.432 |
| 1000 | hybrid | True | 0.911 | 1.839 |
| 10000 | lexical | False | 1.319 | 1.191 |
| 10000 | semantic | False | 1.011 | 0.914 |
| 10000 | hybrid | False | 1.307 | 1.236 |
| 10000 | lexical | True | 0.945 | 5.595 |
| 10000 | semantic | True | 0.925 | 6.635 |
| 10000 | hybrid | True | 0.945 | 5.143 |

The selective regression motivated scalar-metadata and source-URI candidate indexes. A2b removes that observed loss, but unfiltered semantic p95 at N=10,000 has a baseline/candidate ratio of 0.896 (candidate latency 1.116x), outside the 10% regression budget. A2b unfiltered lexical QPS ratio 1.191 also misses 1.20. These results are not hidden or used to lower the contract. A2c tests removal of boxed per-document iteration as a specific overhead hypothesis; causality remains unproven pending the ablation.

A2b median process peak RSS at N=10,000 is 154,992,640 bytes versus 150,126,592; these include ingest and allocator history, not isolated serving RSS. Median ingest is 1.894 versus 1.972 seconds under this run order/load; it is not a construction speed claim. Common-term update tails, sparse-registry rebuild stalls, corruption/recovery timing and realistic corpus scalability remain unknown.

Artifact directories preserve all compressed raw reports, raw OS time logs, source/executable hashes, independent output checks and five corruption rejections. The first two gate attempts were intentionally stopped before completion when measurements required revisions; their exit 255 records are not passed gates.

### A2c and current acceptance status

Static iterator dispatch replaces the boxed iterator; this is a tested simplification, not proof that dispatch caused the large-N semantic regression. All 12,672 output pairs match again; the independent source/ID/filter checker rejects five corruptions. No round meets the load gate.

| N | mode | 1% filter | descriptive QPS ratio | p95 baseline/candidate |
|---:|---|---|---:|---:|
| 1000 | lexical | False | 1.754 | 1.728 |
| 1000 | semantic | False | 0.985 | 0.968 |
| 1000 | hybrid | False | 1.478 | 1.484 |
| 1000 | lexical | True | 2.646 | 3.239 |
| 1000 | semantic | True | 2.470 | 2.234 |
| 1000 | hybrid | True | 1.875 | 1.767 |
| 10000 | lexical | False | 1.220 | 1.094 |
| 10000 | semantic | False | 0.913 | 0.905 |
| 10000 | hybrid | False | 1.193 | 1.044 |
| 10000 | lexical | True | 5.897 | 5.897 |
| 10000 | semantic | True | 6.724 | 7.598 |
| 10000 | hybrid | True | 4.869 | 6.951 |

At N=10,000, unfiltered hybrid QPS ratio 1.193 is below the 1.20 target, and semantic p95 ratio 0.905 corresponds to candidate latency 1.105x, slightly above the 10% budget. Removing boxing did not remove that large-N boundary. **Do not declare A2 accepted or the mission complete.** The code is an uncommitted product prototype with exact functional parity in the tested cases. Profile that cell on an idle host before further changes. Selected-filter gains cannot conceal it.

The first two gate passes were stopped intentionally for measured revisions. The third found one new test-only Clippy lint (`i % 2 == 0`), fixed to `is_multiple_of`; it did not change measured production behavior. The final complete gate record will supersede only validation status, not any measurement or negative result.

Reproduce the final paired study after building/copying the frozen scan and candidate binaries (use a fresh output directory):

```sh
uv run --script benchmarks/public/document_search_compare.py --baseline benchmarks/results/superiority_v2/bin/document-search-baseline --candidate benchmarks/results/superiority_v2/bin/document-search-candidate-v3 --out benchmarks/results/superiority_v2/document_static_paired
uv run --script benchmarks/public/check_document_search_compare.py benchmarks/results/superiority_v2/document_static_paired
```

The scan binary is pinned by `document-baseline-binary.sha256`; candidate builds have source/toolchain manifests. `source-current.tar.gz` preserves the measured source snapshot (before the test-only lint correction), and `baseline-document-source.tar.gz` restores the two baseline implementation files. Overlay these on base commit `648ac99`; retain the initial dirty-source distinction. `envelope.csv` keeps every primary category unknown rather than promoting descriptive wins.

Next in order: (1) finish all five Rust gates; (2) idle-host profile/paired rerun of the unfiltered semantic boundary and check query/ingest/RSS regression budgets; (3) measure common-term replacements, sparse-registry rebuild, compaction and recovery with concurrent readers; (4) run real-corpus matched document baselines and bounded candidate generation; (5) enlarge the train-only ANN decomposition to identify whether the full-size coverage gap is topology or traversal. VGraph is still not the document backend. Current blockers to final claims: idle ARM timings, dedicated/pinned x86 execution, strong matched pipeline competitors, and separate protected QA authorization. No new paid compute, held-out QA, commit, push or external disclosure occurred.

### Modal pilot preparation — cancelled before execution

**Superseded by user instruction:** do not use Modal; connect with
`ssh -l davidcondrey -i ~/.ssh/david@writerslogic writerslogic-03`.
The provider runner and its lockfile are archived as text under `modal_pilot/`;
no Modal source upload, compute launch or new paid spend occurred.

The user offered Modal and the name `writerslogic-03` after the M4 load failures.
`modal_document_pilot.py` and `superiority_v2/modal_pilot/proposal.json` prepare
one synthetic-only, CPU-only scan/A2c replication. A read-only identity request
authenticates workspace `dcondrey`, which differs from the proposed destination;
destination clarification and explicit paid-compute approval are pending.
Preparation does not establish that Modal
provides a dedicated physical host. Its results must stay separate from M4 and
the historical GCE diagnostic; the existing load gate is unchanged.

The proposal fixes one container, four requested/limited physical CPU cores,
8 GiB requested/limited memory, a 3,600-second function timeout, zero retries,
and a single pinned logical CPU for paired timing after compilation. It requests
a $2 allowance; estimated function cost at those limits is $0.253/hour plus a
$0.75 preparation/startup/egress allowance, not a bill or provider-enforced cap.
Rates and semantics were checked against [Modal pricing](https://modal.com/pricing)
and [resource documentation](https://modal.com/docs/guide/resources).

Independent review prompted authenticated-workspace verification before upload
and process-group cleanup for timed-out workloads. Linux GNU-time RSS is
converted from KiB; Darwin byte counters stay unchanged. All 132 retained
Darwin logs reproduce their recorded RSS; malformed counters fail explicitly.
Three portable timing-harness tests pass, including actual descendant cleanup
on timeout. Source payload hashes, baseline overlay identity and manifest target
presence were checked locally. The identity RPC was verified using the pinned
SDK's synchronization context; the earlier cross-context attempt was stopped.
No compute was launched and no source was uploaded. Earlier preparations and
the proposal preceding that SDK fix are preserved separately.

### Existing SSH host

SSH succeeds with user `davidcondrey`; the alias's configured `David Condrey`
user fails authentication. Initial read-only inspection found Windows 11 Pro
Insider Preview (10.0.26340), Intel Core Ultra 9 185H, 16 physical cores / 22
logical processors, approximately 31.5 GiB visible RAM and 1% CPU load.
Following the restart, `writerslogic-02` is verified as Ubuntu 24.04 WSL2 on
Windows host `writerslogic-03`, not a second independent machine. WSL exposes
22 logical CPUs and about 15.4 GiB RAM; initial load was 0.20. Rust 1.96.0 is already
installed without a default; the runner selects it using `RUSTUP_TOOLCHAIN`.
Python 3.13.9 was installed with the host's existing uv. No new OS was installed.

The user requested sleep prevention and then announced a restart.
`ssh_host/keep-awake.ps1` implements a temporary Windows SYSTEM_REQUIRED
request with an eight-hour failsafe and stop-file release. Its first launch
could not connect during the restart; the subsequent launch registered the
request and a live heartbeat was verified before benchmarking. The deadline is
2026-10-13T03:52:04Z. [The API prevents idle sleep](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate), not manual sleep or lid closure.
Do not install a new OS, reboot again, change permanent power settings or use
Modal as an implicit fallback. Existing-host setup and isolated benchmark work
are authorized; new paid resources remain unauthorized.

The frozen source capsule, build protocol and input hashes are under
`superiority_v2/ssh_host/`. The bounded synthetic scan/A2c pilot completed at `/home/davidcondrey/hms-superiority-v2/attempt-1`; both binaries built
and all paired outputs passed independent checking, as reported below.
WSL virtual-CPU affinity does not pin a Windows physical P/E core; retain this
operating-model limitation and keep these results separate from ARM and GCE.

### Verified x86/WSL synthetic pilot

All 44 process runs satisfy the frozen Linux load<3 check. All 12,672 query
pairs match exactly, and the independent checker rejects five corruptions.
The returned capsule is 3,645,205 bytes, SHA256
`3a531969e266d31a4fff2341ebfb8fac37dea08b766b563efd77db7265704434`;
all 102 listed file hashes verify, and the local checker reproduces the remote
result without changing any verified file. Raw reports, build/source manifests,
OS peak-RSS counters and commands remain under `ssh_host/returned-1/`.

These are comparisons against the frozen HMS scan, not tuned competitors.
Intervals below are paired-round bootstrap 95% intervals without familywise
correction; they do not establish the primary success contract. The synthetic
sine embeddings lie in a two-dimensional sine/cosine subspace up to floating-point error and are not a modern
semantic embedding workload. The test establishes exact implementation parity
and exposes serving costs for this fixture; it does not measure answer quality.

| N | mode | 1% filter | QPS ratio vs scan | exploratory 95% interval | p95 baseline/candidate |
|---:|---|---|---:|---|---:|
| 1000 | lexical | False | 1.567 | [1.511, 1.661] | 1.660 |
| 1000 | semantic | False | 0.991 | [0.944, 1.040] | 0.959 |
| 1000 | hybrid | False | 1.179 | [1.140, 1.253] | 1.067 |
| 1000 | lexical | True | 2.772 | [2.630, 2.875] | 2.399 |
| 1000 | semantic | True | 2.529 | [2.505, 2.621] | 3.318 |
| 1000 | hybrid | True | 2.015 | [1.891, 2.049] | 2.383 |
| 10000 | lexical | False | 1.877 | [1.770, 1.914] | 1.829 |
| 10000 | semantic | False | 0.994 | [0.939, 1.003] | 1.000 |
| 10000 | hybrid | False | 1.317 | [1.242, 1.346] | 1.294 |
| 10000 | lexical | True | 7.135 | [6.679, 7.544] | 5.848 |
| 10000 | semantic | True | 12.548 | [12.065, 13.341] | 11.072 |
| 10000 | hybrid | True | 11.359 | [11.082, 12.456] | 9.304 |

The N=10,000 semantic regression seen descriptively on loaded M4 did not
reproduce here: QPS ratio 0.994 and p95 ratio 1.000. This does not resolve the ARM
boundary. N=1,000 hybrid improves 1.179x but misses the 1.20 point target.
At N=10,000, median process peak RSS is 156,778,496 bytes versus 152,064,000
(1.031x), and median ingest is 24.989 s versus 24.006 s (1.041x). These RSS values
include ingestion and allocator history, not isolated serving memory.

Retain the prototype for operational and real-corpus evaluation. Query-speed
gains over the old scan are now reproducible on this WSL fixture; competitor
superiority, real-corpus quality, update/compaction tails, isolated serving RSS,
and the ARM regression budget remain open.

### Oracle hardening and prepared scale diagnostic

NumPy 2.2.6's local matrix multiply emitted divide/overflow/invalid warnings
despite finite unit-vector operands. A full independent reduction on the same
2,000,000 GloVe similarities remained finite and differed by at most
7.78e-16 (`truth-backend-check.json`). The checker now uses explicit NumPy
einsum reduction without BLAS optimization and rejects nonfinite inputs or
invalid cosine output. No cause for the warning is asserted. Rechecking all
38,400 legacy rows leaves every numeric summary and contrast unchanged; the
original artifacts stay intact (`legacy-oracle-recheck/comparison.json`).

`ann_scale_100k/protocol.json` freezes a diagnostic expansion preserving both
old 200-query sets and each old 10,000-row database prefix, adding 90,000 disjoint
training vectors per corpus. The completed expansion is reported below. No benchmark test or protected
QA rows are read. Its purpose is to test whether ranking/traversal precision
loss emerges at scale, not to revive the rejected feedback mechanism or claim
a performance gain.

### Verified source validation and 100k decomposition

Remote Rust 1.96.0 validation passes all five project gates. The initial
all-feature Clippy failure was caused by omitted source-capsule example fixtures;
the exact two local COSE files are bound by `ssh_host/fixtures-supplement.json`.
The original benchmark manifest is unchanged. Original validation logs and the
failure remain in `ssh_host/gates*`; `ssh_host/validation-summary.json` combines
the successful original-source gates. Local ARM's interrupted build gates remain
unverified; local formatting passes.

The 100k training-only expansion is complete. Both remote and local independent
checkers validate 6,400 rows per corpus; every summary field reproduces exactly.
The returned capsule has 25 verified hashes and preserves the prepared inputs.
Native recall for residual16/full-pool exact is identical at all four beams:

| Corpus | ef32 | ef64 | ef128 | ef256 | Double-ef at 256 |
|---|---:|---:|---:|---:|---:|
| GloVe 100k | .7795 | .8625 | .9270 | .9635 | .9850 |
| nytimes 100k | .7875 | .8445 | .8945 | .9360 | .9680 |

Exact-traversal native-recall deltas versus full-pool reranking are GloVe
[-.0010, +.0015, +.0005, -.0005] and nytimes [-.0040, +.0045, -.0005, -.0005].
None exceeds the preregistered .005 point threshold; uncertainty remains wide
(the nytimes ef64 exploratory interval is [-.005, .016]). Prioritize candidate
routing/topology investigation, not final-rank precision. This is one graph
seed and reused 200-query development sets, not a full-corpus ceiling or a
competitive/confirmatory result. All timings remain ineligible. Evidence:
`ann_scale_100k/{returned,local-recheck,local-verification.json}`.

### Operations diagnostic and staged-flush improvement

The first lexical operations diagnostic completes 18 processes across
N=2,047/2,048/2,049, three alternating scan/A2c pairs each, four readers.
Independent local checking validates 112,691 queries; 41,472 static-query and
before/after-restart output pairs match exactly. Five corruptions per remote
report are rejected; two additional local mutations specifically verify
acknowledged replacement/deletion rejection. There are no stale, version,
source or filter violations in this fixture. Source/metadata filters are not
an authorization system.

Compaction overlaps queries lasting up to 2.84 seconds for the scan control and
2.81 seconds for A2c. This is measured temporal overlap; lock waiting is inferred
from the exclusive mutation gate, not directly instrumented. Phase sample counts
vary with writer duration and scheduling, making compaction p99 ratios unstable.
Process peak RSS includes retained observer rows and cannot establish serving
memory or a regression-budget pass. All losses and raw outputs remain under
`document_operations/returned/`.

Source inspection identifies one synchronous mmap flush per staged mutation.
An instrumented A2c process records 4,175 msync calls taking 6.863 seconds across
the whole workload (8.336 seconds through compaction queries); this is not
isolated phase attribution. `strace 6.8-0ubuntu2` was downloaded and unpacked into
a user-owned remote tools directory; no system package installation occurred.
Profile evidence and tool identity are in `compaction_batch/operations-profile/`.

**Rejected alternative:** grouping transactions into one CRC frame enlarges
the record set discarded when recovery reaches a corrupt frame. Its source,
two passed boundary tests and deliberately interrupted gates remain retained;
no grouped-frame implementation remains in the working tree.

**Retained uncommitted change:** `storage::StagedArena` preserves every original
CRC/encryption frame and batches only flushes in a fresh, empty staging directory.
It flushes after 1 MiB of serialized frame bytes or 1,024 frames, on segment
rotation, and before publication. An oversized existing-valid frame is flushed
immediately. Normal live writes remain synchronous. Final staged validation and
atomic CURRENT publication are unchanged. This preserves the existing corrupt-
tail prefix recovery policy; it does not newly promise explicit rejection of
every durable corruption.

The frozen follow-up is `compaction_batch/staged-flush-protocol.json`. All five
Rust gates pass for its exact source capsule, including encrypted compaction
and interrupted-publication recovery. Four new tests cover byte/frame-count
boundaries at limit-1/limit/limit+1, restart, refusal to open an existing store,
and valid-prefix preservation after staged-frame corruption.

Eighteen fresh paired processes compare the staged-flush implementation against
frozen A2c, not the older scan. Local independent checking validates 106,985
lifecycle queries and 41,472 exact static/restart output pairs; all five standard
corruptions per report and two targeted acknowledgement corruptions are rejected.
All 144 returned file hashes verify; all 187 manifest source files match the
current local checkout. Source and binary identities are retained with the logs.

| Documents | A2c median compaction (ms) | Staged-flush median (ms) | Median paired candidate/baseline | Maximum overlapping query, A2c / staged (ms) |
|---:|---:|---:|---:|---:|
| 2047 | 1888.260 | 52.739 | .02681 | 2808.168 / 63.619 |
| 2048 | 2373.839 | 56.299 | .02146 | 2760.456 / 70.122 |
| 2049 | 1809.620 | 55.629 | .03058 | 2773.839 / 63.029 |

All nine pairs improve compaction duration, passing the frozen exploratory
20% reduction signal. A second instrumented process reduces msync calls from
4,175 to 3,152, exactly 1,023 fewer calls for 1,024 compacted records; fsync calls
remain 19. This supports the intended mechanism. Instrumented elapsed times
are not adoption evidence. Both instrumented reports pass the independent
lifecycle checker.

Retain the staged-flush change for further product evaluation. These are three-
pair synthetic lexical WSL diagnostics, with variable query denominators,
observer memory, and unpinned hybrid-core scheduling. They do not satisfy a
primary p95/p99, serving-RSS, competitor, or cross-architecture contract. The
exclusive compaction gate remains; the measured stalls are reduced, not removed.
Next: fixed-work concurrency and sustained-update measurements with isolated
serving RSS, larger/realistic stores and ARM replication; then matched real-corpus
competitors and train-only routing/topology ablations. Protected held-out QA,
new paid compute, commits, pushes and external disclosure remain unauthorized.

### Comprehensive compaction and invention campaign (completed)

The user requested more compatible improvements and original-invention exploration
per pass. Evidence is under `superiority_v2/compaction_comprehensive/`. The broad
diagnostic covers N=2049 at 1/4/8 readers, N=8193 at 4 readers, encryption, trained
NSG, and three compactions per case. All 41 returned hashes and 52,982 lifecycle
queries verify locally. Phase wall times, calling-thread allocation requests,
CPU/fault snapshots, process/thread samples, I/O, query lock waits and timestamped
filesystem/futex traces are retained. Nested times overlap; instrumentation and
observer memory preclude primary timing or serving-RSS claims. The failed
pre-build instrumentation attempt remains in remote `comprehensive-1`; completed
diagnostic source/evidence are in `comprehensive-2`.

The retained uncommitted candidate separates maintenance exclusion from reader
exclusion: queued writers wait before entering the reader lock, allowing queries
during compaction. It hands verified staged mappings directly to publication,
avoiding two recovery reopens, reuses serialization storage, and borrows plaintext
payloads. Strict staged-length checks reject early zero headers as well as CRC
failures. Per-record recovery boundaries, encryption and synchronous live writes
remain intact. Tests cover queued mutations with concurrent queries, durable
restart, and corrupt/truncated first/middle/last staged records. All five required
Rust gates pass on WSL/Rust 1.96.0 (`validation/gates/`). Four focused ARM compaction
tests pass (`arm-focused-compaction.log`); this is not a full ARM gate result.

Fresh uninstrumented validation against the frozen staged-flush implementation
uses identical probes with 512 compaction queries per reader and two repeat
compactions. All 36 runs / 18 pairs, alternating arm order, are retained. Local
independent checking verifies 273 artifact hashes, 188 current source hashes,
350,744 lifecycle queries and 158,226 exact output pairs, including the fixed
compaction-query outputs. Capsule SHA256:
`6c719533b83f4c648151b7f0604964e2d281cbb69c76adb58b15617b35e9b1ec`.
Reproduce with `uv run --no-project --python 3.13.9 python benchmarks/results/superiority_v2/compaction_comprehensive/analyze_validation.py`.

| Documents / readers | Median compaction, baseline → candidate (ms) | Maximum overlapping query, baseline → candidate (ms) |
|---|---:|---:|
| 2047 / 4 | 60.631 → 46.457 | 81.206 → 7.459 |
| 2048 / 4 | 57.098 → 51.541 | 92.423 → 6.796 |
| 2049 / 4 | 53.762 → 49.985 | 82.177 → 7.673 |
| 2049 / 1 | 53.496 → 44.601 | 81.507 → 4.724 |
| 2049 / 8 | 54.525 → 52.480 | 73.533 → 5.528 |
| 8193 / 4 | 108.324 → 90.454 | 138.091 → 14.712 |

Reader availability improves, but losses remain: N2047 pair0 compaction is
82.336 ms versus 60.631 ms baseline; eight-reader median compaction-phase p99
increases from .066338 to .082259 ms (about 24%). N2048 median ingest increases
3928.071 → 4551.527 ms; N8193 ingest increases 15152.953 → 16655.151 ms and static
p95 .504196 → .560356 ms. Three pairs do not establish causal attribution for
these variations. This synthetic, unpinned WSL closed-loop diagnostic does not
establish competitor superiority, the primary performance envelope, open-loop
p99, or isolated serving RSS. Retain the candidate and every losing run for
further evaluation; do not claim a universal latency improvement.

The original-invention feasibility pass tests keyed XOR superposition with exact
corrections, shared modal bases, and bounded-depth residual forests against
compact delta-varint and LZ4 sparse-vector baselines. All 54 configurations lose
to delta-varint in projected bytes. A separate Python set-based oracle verifies
13,824 exact reconstructions, projected byte counts and residual/depth counts
from raw witnesses (`reconstructive-independent-check.json`). These are projected
framing sizes, not implemented durable codecs or serving-memory measurements.
The first prototype/report remain preserved alongside the witness extension.
Regeneration has a small incremental descriptor only when source already exists;
for 256 short-template vectors at dimension16384, eager regeneration takes
127.277 ms versus .175 ms for varint encode/decode. No production codec change is
supported by this result.

A more fundamental finding is verified: **255 of 256 longer varied synthetic
texts encode to an empty sparse vector at both dimensions1024 and16384**, with
only two unique vectors; short templates and random controls have no empties and
256 unique vectors. Raw witnesses and the independent checker substantiate this
synthetic counterexample, not a real-corpus quality estimate. `encode_text_internal`
uses `EntangledHVec::bundle` majority voting within sparse character/word/phrase
representations; final frequency selection cannot recover an empty candidate
set. The existing Bloom-bundle alternative already acknowledges sparse-majority
failure. The compaction probe uses short templates; float qgraph ANN is a separate
substrate. Neither its recall nor compaction performance is explained by this
counterexample. Stored-vector compatibility and the production encoder are unchanged.

Eight hypotheses, controls and kill conditions are in `invention-register.json`:
access-weighted generative records; failure-trained associative routing;
query-equivalent physical generations; replay-budgeted segment evacuation;
reversible shared holographic representations; a joint storage/routing algebraic
code; semantic-invariant publication certificates; and query-span memory with
exact residual escalation. Reader separation is implemented, but persistent
certificates and the other speculative mechanisms remain unproven. No originality
relative to prior art is established. Generic selective recomputation is already
in [Knockoff (FAST 2017)](https://www.usenix.org/conference/fast17/technical-sessions/presentation/dou),
learned routing in [Learning to Route (ICML 2019)](https://arxiv.org/abs/1905.10987),
and probabilistic screening in [Probabilistic Routing](https://arxiv.org/abs/2402.11354).
Previously losing residual-query feedback must not be renamed as a new invention.

Next research should first discriminate density-aware, information-preserving
composition against existing Bloom union and fixed-density frequency controls,
measuring distinctness, perturbation similarity, rare-feature preservation,
operators and migration/version compatibility. Random backfill alone cannot
establish retained information. Continue train-only topology work separately,
and assess realistic concurrency, isolated serving RSS and ARM before primary
or competitor claims. Batch discriminating measurements rather than narrow
incremental tuning passes.

Remote validation and witness work are complete; no benchmark/gate workers were
found before guard shutdown. The bounded Windows sleep guard was verified
released at 2026-10-12T21:52:09.4408483Z, before its 2026-10-13T05:26:58.3594575Z
deadline (`sleep-guard-release.json`). Remote roots `comprehensive-{1,2,3}` and
`comprehensive-input/validation-run.log` remain preserved. No permanent power
settings changed. No paid compute, Modal, protected held-out evaluation, commit,
push or external disclosure occurred.

### Sparse composition experiment design (2026-10-12; not executed)

The comprehensive design is
[`sparse_composition_design/protocol.md`](../../benchmarks/results/superiority_v2/sparse_composition_design/protocol.md).
It extends the existing invention register's verified encoder diagnostic with
H-SC1: retain per-level counts until projection and reserve bounded support for
low-frequency evidence. This is a falsifiable hypothesis, not established novelty
or lossless text encoding. The original register and raw evidence are unchanged.

One frozen campaign compares production majority, uncapped Bloom union, capped
union, raw and normalized fixed-density frequency, support sampling, two reserve
fractions, existing counting Bloom and a random distinctness control. It specifies
four dimensions, three density budgets, five seeds, length/capacity boundaries,
rare-feature and perturbation pairs, exact operators, joint byte/CPU/memory costs,
version compatibility and disposable migration failure cases. Independent raw
witness checking, tamper tests and explicit rejection criteria precede any result.
No synthetic result can establish real-corpus or competitor superiority.

Design validation checks links, parent-register identity, required sections and
preservation of the pre-existing document prefixes and production diff; receipt:
`sparse_composition_design/design-validation.json`. No experiment, SSH compute,
production encoding change or migration was performed. The five WSL gates and
four ARM tests remain inherited evidence for the prior compaction candidate;
this documentation change does not expand their scope. Preserve the eight-reader
phase-p99 regression and every earlier loss. Float ANN routing remains separate.

The requested design is complete. Future execution begins with an isolated
harness/checker and a source/input/command freeze. A fresh bounded sleep guard is
mandatory before SSH compute. No Modal, paid compute, protected held-out evaluation,
commits, pushes or external disclosure is authorized.

### Sparse composition campaign (bounded diagnostic completed)

Evidence and the checked report:
[`sparse_composition_run/attempt-2/report.md`](../../benchmarks/results/superiority_v2/sparse_composition_run/attempt-2/report.md).
The source capsule SHA256 is
`2a14d77003b512557a61708fb2a2b391a159f207b2378926fd204bdd269df535`.
`acceptance.json` binds 199 source files, raw/checked artifacts and analysis code.
Independent reconstruction checks 25,596 input/dimension rows, 588,708 sparse
outputs, 1,638,144 native counting-Bloom probe scores, 1,024 parent text vectors,
512 parent random vectors and680 operator-capacity cases. Native XOR/permutation/
JSON inverse checks pass; production-majority tie choices are checked for rank
eligibility, with exact production identity additionally checked on golden data.
The parent255/256 empty varied-text finding is reproduced at both original dimensions.

All five Rust gates now pass on **local ARM/Rust1.96.0**, extending the earlier
four focused ARM tests (`sparse_composition_run/gates/results.json`). Three focused
native test methods cover bounds, support and existing-version document restart;
three Python methods cover reconstruction, 15 tamper cases and18 edge inputs.
No production encoder or stored-vector meaning changed. Existing document prefixes
are byte-for-byte preserved in `attempt-2/preservation-check.json`.

Both reserved-coordinate hypotheses fail the frozen advancement criteria. At
D16384/B64, R25 rare-feature AUC trails the strongest fixed-budget control by
.2065 (97.5% interval[-.2242,-.1883]); R50 trails by.1987
([-.2163,-.1806]). All eight native-budget dimension/candidate quality signals
fail; these are negative results for the tested variants, not a proof against
all information-retaining composition. `sparse_composition_run/invention-register.json`
extends the unchanged parent register and records the rejection without a novelty claim.

The useful lead is the **existing capped Bloom-union control (UC)**: union within
levels, then the existing cross-level frequency cap. At D16384/B64 its rare-token
AUC is.8171 versus.5977 for production; perturbation ordering is.9197 versus.6626.
It has480/480 distinct vectors in separate fixed-seed base cohorts and zero empties,
versus production's245 empty vectors among480 bases. These are synthetic lexical
structure/primitive-membership measurements, not real-document relevance or a
competitor benchmark. Keep the dimension boundary and cost tradeoffs:

| D / native budget | Production rare AUC | Capped-union rare AUC | Production / capped-union mean raw varint bytes |
|---:|---:|---:|---:|
|1024 /4|.5397|.5378|2.24 /5.33|
|4096 /16|.5837|.6427|8.98 /20.98|
|16384 /64|.5977|.8171|35.60 /87.10|
|65536 /256|.6212|.8965|146.34 /352.57|

At D1024 capped union loses slightly on rare AUC and has only118 distinct vectors
among256 original varied-text witnesses despite no empties. No universal quality
claim is supported. At D16384 uncapped union reaches rare AUC.9727 but uses
9976.81 mean raw varint bytes versus87.10 for capped union. Raw payload sizes
exclude framing, versions, source and indexes; production's smaller payload is
partly empty vectors. Random full-input hashing is480/480 distinct at D16384 but
rare AUC.4885 and perturbation ordering.4999: distinctness alone is insufficient.

Projection before merging changes231/680 operator results; merging full counts
is exact. This constrains a future compositional state representation; it is not
an implemented associative merge API or proof about text concatenation boundaries.

This completes the **bounded diagnostic**, not every measurement in the original
design. The pre-measurement amendment retained all dimensions, lengths, seeds,
controls, budgets and perturbations but used8 rather than32 families, replacing
trivial Recall@10 over8 items with Recall@1/MRR. Literal primitive dumps were
replaced by texts, frozen feature-generation code and exact counts reconstructed
independently. Position/level ablations, exclusive rare-coordinate attribution,
equal-total-byte frontiers, gated five-round end-to-end costs and a versioned
migration/crash implementation remain undone. Existing-version restart passes;
new-version compatibility is unverified. No candidate is eligible for adoption.

Collection finished in2817.202 seconds. Sampled research-producer RSS reached
1,319,927,808 bytes, below the8GiB sampled guard; this is not isolated serving RSS.
Initial/final host load exceeds3. All component timings are descriptive; neither
a product speed improvement nor the primary performance envelope is established.
The native producer and closed-shard checker ran concurrently for part of the
campaign; `verification-scheduling.json` records this, and `validation-lineage.json`
binds every read to the final immutable native hashes. The original frozen checker,
inputs and quality rules were unchanged. `attempt-1` preserves a preparation-only
path error; no measurements came from it.

Next substantive work should evaluate capped union under a distinct encoder
version with matched end-to-end costs, realistic allowed-source quality and
migration/recovery behavior, retaining D1024 and every losing cell. Do not revive
R25/R50 under new names or infer improvements to float ANN routing. Prior compaction
losses, including the approximately24% eight-reader phase-p99 regression, remain
unchanged. No competitor/primary-envelope contract is met.

All local campaign workers have exited; all gates and checked reporting completed.
No SSH, Modal, paid compute, protected held-out evaluation, commits, pushes or
external disclosure occurred. The remote sleep guard remains in its previously
released state; establish and verify a fresh bounded guard before future SSH compute.

### Versioned capped-union prototype and train-only diagnostic (2026-10-13)

The existing capped-union control now has an explicit experimental store identity,
`multiscale-capped-union-v1`, selected through `HmsConfig.text_encoder` or the
`hms-admin reencode --text-encoder` flag. The default remains `multiscale-words-v1`.
Both direct text encoding and document chunks use the configured version. Schema
mismatch rejects mixing configurations before loading arena data; original source
documents are re-encoded into a new, verified, atomically published store.

Evidence and reproduction:
[`capped_union_version_v1/report.md`](../../benchmarks/results/superiority_v2/capped_union_version_v1/report.md).
The frozen pilot uses 64 complete SciFact abstracts and 16 training queries,
three dimensions and five paired repetitions. All 30 runs are independently
checked, including 2,400 vector readouts, 1,920 retained CRC/LZ4 document frames,
480 initial and 480 restart queries, source metadata/offsets, exact Jaccard ranks
and full retained store file lengths. The original producer/source/input freezes,
181-file source capsule, prechange native binary and 240 legacy witnesses remain
preserved. Amended analysis and tests have their own freeze; no rows were changed.

| D | Majority / capped-union nDCG@10 | Capped-union Recall@10 | Majority / capped-union empty abstracts | Majority / capped-union logical frame bytes |
|---:|---:|---:|---:|---:|
|1024|.0000 / .0000|.000|64 / 0|130786 / 131711|
|4096|.0000 / .0848|.125|64 / 0|130794 / 135090|
|16384|.0000 / .0867|.125|64 / 0|130863 / 149883|

Capped union has 64 distinct vectors at every dimension, but absolute relevance
is low and D1024 still has no relevant result in the first ten. The legacy baseline
has identical empty vectors and 64-way top-score ties; these results do not establish
a strong retrieval system. BM25 document-search outputs match across encoders and
restarts; dense embeddings and float ANN routing are unchanged. No novelty or
default-adoption claim follows from this pilot.

All 30 pre/post load gates failed (pre-run 1-minute load 3.78–6.50); timings are
descriptive. At D16384, median ingest is 1692.390 / 1751.916 ms and encode-plus-query
is 1.677 / 1.985 ms for majority / capped union. Compaction medians near 4.45–4.50 s
in this small ARM fixture are retained; earlier WSL measurements concern different
workloads. A fixed 1 GiB segment dominates file allocation, so this is not an
equal-total-byte frontier. OS process peak RSS includes validation work and does
not establish isolated serving RSS or the primary concurrency envelope.

All five Rust gates pass on ARM/Rust1.96.0; default features also pass Rust1.89.
Regression checks cover 49 prechange vectors, density boundaries, feature rejection,
both-direction encoder mismatch, compaction/restart, acknowledged writes/deletion
after abrupt exit, invalid migration input and existing-destination refusal.
Actual CLI reencode/verify preserves its source; four Python oracle/tamper tests pass.
Two checker corrections are archived: precompaction segments are removed by normal
publication, and small-store exact scan includes zero-score neighbors and
empty/empty identity. Only retained postcompaction frames are independently checked;
precompaction lengths remain producer observations. Migration-publication crash
injection is still unmeasured.

Retain the explicit experimental option. Next advancement requires broader
training-corpus relevance/selectivity, gated costs, equal-total-byte comparisons,
isolated serving RSS and migration crash injection. Preserve D1024, prior reserve
failures, operator projection losses and the earlier eight-reader compaction-p99
regression. No remote compute, paid compute, Modal, protected QA evaluation, commit,
push or external disclosure occurred; the remote sleep guard remains released.

### Parallel competitor campaign (2026-10-13)

The user authorized simultaneous work on vector search, document retrieval and
agent-memory evidence. Local train/dev comparisons now include FAISS HNSW,
hnswlib, author RaBitQ IVF, SymphonyQG, FAISS IVF scalar quantization and SQLite
FTS5. This expands controls, not a claim of broad superiority. No paid compute,
protected held-out QA evaluation, commit or push is authorized by this campaign.
The previously authorized SSH hosts may run document controls after establishing
and verifying a fresh bounded Windows sleep guard; no permanent power settings
may change.

The document implementation avoids visiting unmatched chunks for unfiltered
lexical queries, caches collection statistics and term frequencies, and selects
the candidate head before sorting. Selective filters keep the existing adaptive
path. All 5,665 public train/dev pairs retain exact fields, scores and ordering;
the independent follow-up checks 3,399 candidate queries and 42 total runs.
Mean local speedups versus previous HMS are 32.8x on SciFact train and 22.0x on
NFCorpus dev. All load gates fail, and final selective-filter/zero-frequency
guards were checked for correctness but not separately timed. SQLite's optimized
native rank control remains slower descriptively; SQLite has better SciFact
Recall@100. Different tokenizers and IDF prevent an identical-scoring competitor
claim. Evidence:
[`document_competition_v1/report.md`](../../benchmarks/results/superiority_v2/document_competition_v1/report.md).

Optional SDK source packets factor repeated verbatim turns and session metadata
while preserving every occurrence, role and observation time. Same selections
use 0.870756x the bytes; 98 more complete groups fit the existing byte ceilings.
Support stays 79/84, and no reader/answer-quality measurement was made. An
independent raw-source checker verifies all 100 reused development questions,
608 retained occurrences, exact reproduction and nine tamper rejections. The
codec has packet/source/group/occurrence bounds and 18 SDK tests pass; five
live-server tests skip. Evidence:
[`memory_competition_v1/checked_summary.json`](../../benchmarks/results/memory_competition_v1/checked_summary.json).

The experimental angular IVF replacement stores contiguous centroid-relative
codes without graph edges or retained original vectors. Shared spherical
centroids/assignments are identical across IVF competitors; raw float64 cosine
and original ids provide independent truth. GloVe10k's initial 8/0 calibration
reaches .99375 recall in 1,411,744 serialized bytes, but the frozen .99 point
misses on fresh validation at .98917. RaBitQ9's validation anchor reaches .99583
in 1,803,554 bytes; HMS8/4 reaches 1.0 in 2,091,744 bytes. No .99 memory win follows
from that initial candidate. Its nytimes10k build rejects eight zero database
rows, retained as a failure rather than removed from the dataset.

The second candidate explicitly permits finite zero database rows, retains
nonzero queries/unit centroids, adds a 5-bit residual arm, caches centroid query
scores and selects ARM SDOT for 4-bit codes at runtime. Six integration tests
cover padding, quantization error bounds, zero rows, empty lists, ties, original
ids, shape/limit boundaries, malformed serialization and reload equivalence.
Its 100k query split uses training-only queries disjoint from the initial
10k validation; no .99 acceptance is inferred without checked results. Frozen
candidates, input membership, failed points and protocol amendments remain in
[`superiority_v3/ann_competition/`](../../benchmarks/results/superiority_v3/ann_competition/).
ANN timings remain ineligible because of host load and unequal query-call
boundaries; supplemental native batch loops are descriptive only. Consolidated
Rust gates and source capsules are retained under
`superiority_v3/integration_gates/`. All five required Rust gates pass on ARM /
Rust1.96.0, and the default library passes Rust1.89.0; the frozen source remained
unchanged throughout. The packed signed 5-bit code and runtime integer-kernel
parity regressions pass in the all-feature suite.

The attempted matched previous/final-candidate/native-SQLite document comparison
on the existing Windows/WSL host could not begin: both SSH hosts were offline,
and the fresh guard connection timed out before execution. No guard, transfers
or remote compute began; no cleanup remains. Hashed failure receipts are retained
under `superiority_v3/document_remote_v1/`. Old remote work and local failed
timing gates remain preserved. Future remote compute requires a fresh verified
guard, heartbeat, deadline and verified release.

The user's subsequent availability update triggered a second attempt, preserved
separately in `document_remote_v1/attempt-2/`. Fresh Tailscale status and bounded
overlay pings still showed both peers unreachable before SSH authentication.
No guard or remote work began. The third attempt reached both hosts and
established a supervised bounded sleep guard. The user then chose improvements
before broader benchmarking; the guard was released and cleanup verified, with
no transfers or remote compute. `document_remote_v1/attempt-3/cleanup-receipt.json`
records the release.

The ANN campaign is parked at that same request. Both 10k datasets are checked;
all 25 GloVe100k calibration arms are checked and their selections frozen. HMS4/5
reaches .99625 calibration recall in 15,806,880 bytes, versus current author
RaBitQ9 at .9925 in 17,290,366 bytes. This 8.58% storage difference is calibration
only. The frozen HMS .99 point scans a mean 52,216 of 100,000 vectors; 128 probes
reach at most .97125 in the measured calibration configurations. Candidate
routing and scan volume remain material limitations.

Twelve GloVe100k validation workers completed predictions before the controller
was interrupted; those outputs are quarantined, unscored and uninspected. That
split has been queried and cannot be described as untouched. NYTimes100k inputs
and shared centroids are ready, but construction/query workers never started.
The checkpoint and tested resume scripts are under
`superiority_v3/ann_competition/checkpoint.md` and `campaign-source-checkpoint-1/`.
Do not score quarantined validation, resume broad benchmarks or start 250k work
while choosing implementation changes.

### Architecture priorities (2026-10-13)

The production document API computes exact dense cosine over eligible in-memory
f64 embeddings; the separate experimental ANN implementation does not accelerate
this path. A fast standalone index is therefore insufficient evidence of a fast
end-to-end document system. A document candidate backend needs schema-bound
embeddings, stable chunk/source identities, metadata-filter correctness and
durable replacement/deletion before adoption; exact search remains a reference
and a useful fallback for small or selective workloads.

The current mutation gate already lets readers proceed during compaction.
Document queries previously retained the read gate and document lock while
scoring, delaying mutations; independently captured model query variants could
also mix document revisions. The implementation now separates query preparation,
coherent capture and execution, sharing immutable document owners without
cloning dense buffers or the full vocabulary. Moving compaction to immutable
generations with replayed writes is a separate future change to the durability
protocol, not part of this step.

The text encoder's 64/64 empty-abstract diagnostic and capped-union low relevance
remain reasons to separate lexical retrieval, learned semantic embeddings and
VSA composition rather than equate sparse text composition with semantic
retrieval. No default encoder change, validated ANN win or end-to-end speedup
is inferred from these architecture decisions.

Snapshot and model-stage integrity integration is complete. Single and model
searches score outside the read gate; all variants capture membership, filtered
statistics and source owners together. Retained views survive replacement,
deletion, slot reuse, registry rebuild, durable compaction and restart. Model
configuration validates native limits and finite weight sums before installation
or search; fusion rejects duplicate/conflicting chunk sources and invalid scores.
No journal or persisted document format changed.

All five required Rust gates pass on ARM/Rust1.96.0, and the default library
passes Rust1.89.0. Sixteen snapshot/model checks pass in the integrated all-feature
suite; the frozen source remains unchanged throughout the gates. Commands, logs,
source capsule and acceptance receipt are in
`superiority_v3/architecture_gates/`. Root independently verifies 49 delegated
artifact/source hashes and the five parked ANN harness tests. The previous document
timing evidence concerns the postings implementation before this snapshot change;
no new end-to-end speedup, corpus comparison or competitor result was measured.
Broader benchmarks remain parked; no quarantined validation predictions were
opened, and no new remote compute, commit or push occurred.

### Dense retrieval, atomic ingest and bounded service work (2026-10-13)

The production document path now has an optional exact f64 box-tree candidate
index. Training captures immutable source owners, builds outside query/mutation
locks and publishes only the captured generation; concurrent same-generation
publication is idempotent. Unfiltered semantic and hybrid queries share the
trained index without copying dense payloads. Filters, unsupported shapes and
invalidated caches retain the exact scan. Replacement, deletion, compaction and
restart preserve source identity and score/order parity; restart requires explicit
retraining. Rust/Node maintenance and the authenticated tenant HTTP maintenance
endpoint expose the backend. Limits and fallback behavior are documented in
[`production-readiness.md`](../production-readiness.md).

Two controlled synthetic fixtures demonstrate less scoring work, not faster
end-to-end retrieval: the exact document tree evaluates 64 of 512 f64 dots, and
experimental IVF block bounds evaluate 64 of 512 selected coarse rows. IVF still
reports all selected rows as `scanned`; `evaluated` and auxiliary bytes are separate
telemetry. Its 7,432 paired searches retain bitwise results and eight code-width
configurations retain identical serialized bytes. The 512-row bounds use 680 or
1,192 auxiliary bytes plus per-searcher scratch and two bound-dot calls per block.
Overlapping/high-dimensional boxes or tied rows may still require a full scan.
Evidence: `superiority_v3/dense_document_v1/` and `ivf_routing_v1/`. No real-corpus
latency, recall improvement or competitor win follows from these fixtures.

Native document batches now prepare bounded inputs and commit once, so a late
validation failure cannot partially replace documents. Native IDs must be unique;
HTTP validates every incoming record before retaining each repeated ID's last
occurrence, preserving its existing response count and replacement semantics.
Aggregate source/metadata/embedding/chunk and prepared-transaction limits apply
before commit. The server admits at most four body readers or blocking operations
by default, returns immediate overload responses, times out stalled bodies after
30 seconds and retains work permits until cancelled handlers' blocking work ends.
Tenant initialization reserves capacity atomically and performs I/O outside the
global registry lock; failures wake waiters and allow retry. These are functional
concurrency guarantees, not throughput measurements.

Exact restart assertions exposed 28 f64 score differences in the previous JSON
decoder. The existing `serde_json/float_roundtrip` feature fixes that precision
boundary without a dependency/version or persisted-format change; its parser
cost is unmeasured. HTTP validation now accepts finite nonzero f64 vectors rather
than rejecting magnitudes outside f32 range. Actual process tests cover cosine
scores for the smallest f64 subnormal through `f64::MAX`, including restart.
The original failures and frozen delegated deliveries remain retained.

Root focused checks pass: 18 server checks, 24 Python SDK tests with 46 subtests
against a real server (none skipped), four Node process tests, TypeScript checking
and packed CommonJS/ESM/native-package checks. Commands, log hashes, delegated
artifact verification and root precision delta are under
`superiority_v3/improvements_v1/`. Broad benchmarks remain parked; no quarantined
ANN validation, external model download, production-model benchmark, remote
compute, commit or push occurred.

The final integrated gates pass: formatting, default and all-feature all-target
Clippy with warnings denied, 381 default and 600 all-feature tests, and the default
library on Rust1.89.0. The existing raw kernel microbenchmark remains ignored in
both test configurations. Frozen source stayed unchanged throughout all six
commands; the initial experimental test-only Clippy failure is retained separately.
Source archive, command logs and acceptance receipt are in
`superiority_v3/improvements_v1/gates-final/` and `improvements_v1/acceptance.json`.


### Concurrent-write compaction (2026-10-14)

Compaction now captures a coherent live-state snapshot and arena checkpoint, then
serializes and verifies an unpublished generation without holding the mutation gate.
Document payloads and dense embeddings retain their immutable owners; sparse vectors
and ancillary records are copied. A separate compactor mutex prevents generation
replacement by another compactor during journal replay. Two finite catch-up passes
copy acknowledged journal frames outside mutation exclusion. The final tail is
copied while mutations are excluded, before the existing atomic `CURRENT` publication
and trained-index checkpoint persistence. Journal/schema meaning is unchanged, and
replayed batches and encrypted payloads retain their transaction boundaries.

Reading the live journal across rotation required consistent mmap selection. Arena
reads now acquire active and historical-segment locks in the same order as rotation
and publication, and reject overflowing lengths or reads across segment boundaries.
Staging verification is separated from publication through an owned prepared arena;
appended frames are read back and CRC-checked before publication. Corrupt or shortened
initial staging still fails without replacing the acknowledged generation.

Focused lifecycle tests pass on local ARM, including document replacement/deletion,
source/filter/embedding identity, intact batches, late writes after both catch-up
passes, concurrent compactors, forced segment rotation, and actual-core process exits
at the three existing publication phases. The replay and process-exit fixtures cover
plaintext and encryption when security is enabled. All six final Rust gates pass
on unchanged frozen source: 392 default and 611 all-feature tests, formatting, both
all-target Clippy configurations, and the Rust1.89 default-library check. The existing
raw kernel microbenchmark remains ignored. The rebuilt Node addon passes all four
process tests; binding sources and the previous native binary are preserved. Evidence
and separately frozen rejected gate attempts are under
`superiority_v3/compaction_replay_v1/`. The old borrowed
iterator remains test-only for the independent scan reference; compaction shares
source owners. Only three prior Rust source files changed, with one new module.

These are functional checks, not workload measurements or a competitor result.
Snapshot capture still excludes writers, sparse/ancillary snapshots add memory, and
the final tail, flush, publication and cache persistence still delay mutations.
No fixed pause bound or end-to-end speedup is established. Fixed-work serving RSS,
sustained updates, realistic corpora and matched competitors remain outstanding.
Broad ANN campaigns and quarantined predictions remain parked, and no remote work,
paid compute, model download, commit or push occurred. Existing sparse-encoder losses
and earlier ARM/concurrency regressions remain part of the ledger.

A final frame-boundary regression exposed a malformed header that retained a valid
payload CRC while understating the raw payload length. The original replay accepted
it; the failing witness and source supplement are retained as
`length-regression-before.{json,log}` and its source archive. Replay now validates
both raw and compressed lengths before reading or allocating the payload, rejecting
compressed lengths above the declared raw length and raw lengths above the existing
50 MiB frame limit. The previously passing six-gate snapshot remains in
`gates-final-3/`; the final boundary fix passes all six gates in `gates-final-4/`.
Limit-1, limit and limit+1 cases cover both payload-length consistency and the existing
50 MiB raw-frame ceiling. `acceptance.json` binds current source, complete gate logs,
native artifacts, the retained rejected attempts and the intermediate passing source;
reproduce integrity verification with
`uv run --no-project --python 3.13.9 python benchmarks/results/superiority_v3/compaction_replay_v1/verify_acceptance.py`.

### Fixed-work serving and sustained updates (2026-10-14)

The bounded local ARM diagnostic is complete in
`superiority_v3/compaction_updates_v1/`. It compares the accepted compactor with
its reconstructed predecessor in three alternating-order pairs at 1,024 and
5,182 original cached SciFact abstracts. Separate seed and serving processes
exclude ingestion from sampled serving RSS. Four synchronized readers execute
fixed counts with eight-millisecond closed-loop pacing while one writer performs
256 atomic two-document replacements and 64 deletions. All readers extend beyond
publication; the workload tests filtered hot pairs, not relevance or saturation.

Independent checks validate 61,440 queries, 3,840 operations, 37,236 exact restart
pairs and 43,380 exact cross-arm static/final/restart result pairs, including full
text, metadata, source URI, offsets, versions, batch coherence and acknowledged
deletions. Eleven corruptions of an actual witness are rejected. Only seven of
twelve full sampled load<3 gates pass; all timing medians remain descriptive.

At 1,024 documents, median longest writes are 4,410.2→38.4 ms, query p99 is
1.444→20.553 ms, and sampled compaction maximum RSS is 26.422→28.422 MiB. At
5,182, these are 4,654.7→376.8 ms, 1.489→21.082 ms and 104.094→105.422 MiB.
Longest writes improve in every pair, but write p99, query p99, maximum query
delay and sampled compaction RSS worsen in every pair. Three compaction-duration
pairs regress. The candidate's largest write/query delays, 1,266.7/1,256.7 ms,
occur in a large-store run that passes the load gate. These losses are retained;
no serving latency improvement or primary-envelope claim follows.

Verified implementation behavior: `commit` holds reader exclusion over durable
arena append and live-state application. Inference: the additional concurrent
writes can expose that existing exclusion more often, consistent with the
observed query regressions. This is not a profile attributing each pause to a
flush. A future split between writer serialization/durable append and atomic
live-state publication needs independent durability and concurrency design;
snapshot checkpoints, transaction boundaries, failure handling and acknowledged
restart identity must remain coherent. No such product change was made here.

Serving RSS includes 32 fixture records and bounded output buffers; sampled
maxima can miss peaks. Learned embeddings, direct publication-pause attribution,
open-loop tails, larger stores, other hosts and matched competitors remain open.
The full protocol, source/input freezes, binaries, raw query outputs, memory/load
samples and separately retained preparation failures are preserved. The complete
grid is `attempt-6`; rejected earlier attempts do not enter the medians.
All six final Rust gates pass on unchanged frozen source: formatting, both
all-target Clippy configurations, 392 default and 611 all-feature tests, and the
Rust1.89 default-library check. The old raw kernel microbenchmark stays ignored.
Four oracle unit checks and eleven actual-witness tamper checks pass.
`compaction_updates_v1/acceptance.json` binds sources, outputs, memory/load
observations, retained attempts and gate logs; its verifier independently rechecks
the corpus-to-input mapping and all run/cross-arm results.
Reproduce analysis with
`uv run --no-project --python 3.13.9 python benchmarks/results/superiority_v3/compaction_updates_v1/analyze.py`.
Broad ANN and quarantined predictions remain parked; no remote/paid compute,
model download, commit, push or external disclosure occurred.

### Durable append with atomic live publication (2026-10-14)

The diagnosed reader-exclusion mechanism is now changed. A FIFO writer lease
serializes durable append and complete live-state publication; its queue mutex
is held only for ticket allocation/release. Queries can capture the old live
revision during encryption and synchronous arena append. Reader exclusion starts
when the transaction consumes its lease to publish the whole batch and revision,
before acknowledgment. Compaction snapshots/checkpoints and maintenance share
the same writer queue. Journal encoding, encryption and public API meaning are
unchanged; only `engine/mutation.rs` and `mutation_gate.rs` changed.

The retained prechange regression fails because pending append blocks document
queries. Actual-core checks now pass before and after real append, including
vector queries, coherent old document/source/embedding snapshots, competing
writer and compactor exclusion, cleanup/retry, and exact postcompaction restart.
Abrupt child exits before append, after append and after publication recover
whole batches in plaintext/encrypted modes. FIFO ordering through a blocked
publication and counter boundary checks pass. Unacknowledged durable frames may
replay, as before; disk-controller or power failure was not simulated.

`superiority_v3/reader_publication_v1/` retains both complete same-protocol local
ARM campaigns: intermediate mutex-held writer ownership (`attempt-1`) and final
queued ownership (`attempt-2`). Each independently verifies 61,440 queries,
3,840 operations, 37,236 exact restart pairs and 43,380 cross-arm result pairs.
The final grid passes eleven of twelve load<3 gates; one small-store baseline
misses its gate. Both runs pass in five of six pairs. All medians are descriptive.

Final median query p99 is 20.502→0.0606 ms at 1,024 documents and
19.280→0.0648 ms at 5,182. Query p99 and maximum query delay improve in all six
pairs. The largest candidate query is 0.482 ms while the largest write remains
1,332.0 ms. Retain losses: three pairs worsen longest writes, two write p99,
one compaction duration, three sampled compaction maximum RSS and two large-store
static RSS. Live-state publication and existing arena/maintenance costs still
have no fixed latency bound. No saturation, learned-embedding memory, competitor
or primary-envelope result is established.

All six final Rust gates pass on unchanged corrected source: 399 default and
618 all-feature tests, formatting, both all-target Clippy configurations and the
Rust1.89 library check. The old raw kernel microbenchmark stays ignored. Rebuilt
Node passes all four process checks with unchanged binding sources. The original
Clippy failure, test-only cast correction and identical measured production
prefix are bound separately by `lint-delta.json`; final gates are in
`reader_publication_v1/gates-final-2/`. Source/native archives, raw workloads,
independent tamper checks, gate logs and preserved intermediate results are bound
by `reader_publication_v1/acceptance.json`. Reproduce with its verifier.

Open-loop serving, larger/bulk/learned-embedding workloads, direct pause
attribution, other platforms and matched competitors remain open. Broad ANN and
quarantined validation stay parked. No remote/paid compute, model download,
commit, push or external disclosure occurred; preserve all previous losses.


### Incremental replacement publication and broader research (2026-10-14)

The user requires substantial competitive improvements, authorizes exploratory
constraint relaxation, and requires no sacrifice in adoption. The working
candidate retains document slots and unchanged term postings for equal chunk
counts, updates frequency/filter/embedding ownership, and preserves changed-count
fallback. Surviving shard chunks are overwritten; obsolete IDs are removed.
Journal, synchronous durability, encryption and search contracts are unchanged.
Three frozen source files differ, including regression coverage.

The 24-cell open-loop experiment is complete under
`superiority_v3/posting_publication_v1/`. The oracle checks 115,200 offers,
113,043 completions, 2,157 explicit rejections, 3,840 operations, 3,517 whole-bulk
queries, 310,920 exact restart pairs and 310,920 cross-arm pairs. Publication and
API-service p99 improve descriptively in all twelve pairs. Bulk publication
median p99 is 19.806→5.723 ms at 5,182 and 48.808→9.075 ms at 20,728 documents.
The larger fixture is four explicit text copies, embeddings are synthetic one-hot
controls, and arrivals are absolute with bounded queues. No learned relevance,
independent large-corpus scaling or competitor advantage is inferred.

All 24 load<3 gates fail; exploratory thresholds 4/6 also accept none. The original
numerical gates pass, but the stricter user-requested audit fails in three groups.
Retain every loss: six sampled-RSS pairs, four write-p99 pairs, five append-p99
pairs, four compaction pairs, seven final-maintenance pairs, two response-p99 pairs,
one rejection-count pair and other producer/queue/reservation losses. Producer
lag reaches seconds in some observations; query percentiles exclude rejected
requests while counts retain them. Their causes are not established. The candidate
has not achieved the no-sacrifice or competitive-superiority requirement.

All six final Rust gates pass on unchanged source: 401 default and 620 all-feature
tests, both Clippy configurations, formatting and the Rust1.89 library check.
Five focused replacement checks and eight actual-record adversarial oracle methods
pass. Plain/encrypted recovery and chunk-count transitions are covered. The rebuilt
Node addon passes four process checks; binding sources are unchanged. Rejected
preparation compiles and prior negative results remain preserved.

The user also requested creative model translations and worldwide third-party
research through October14. `research-shortlist.json` records twenty primary or
first-party leads, dates, verified institutional affiliations, selective review
depth, proposed experiments and transfer limits. Examples include
[DBSP incremental computation](https://doi.org/10.1007/s00778-025-00922-y),
[SPatch random-walk/star-mesh deletion](https://arxiv.org/html/2512.18060v1),
[India-affiliated sorted graph analysis](https://proceedings.mlr.press/v267/gollapudi25a.html),
[Chinese-language LSMDiskANN](https://www.jos.org.cn/jos/article/abstract/7513),
[France-affiliated DARTH](https://arxiv.org/html/2505.19001v1), and
[Japan's IVF++ repository](https://ir.library.osaka-u.ac.jp/repo/ouka/all/105173//?lang=1).
Paper mechanisms and reported results are external evidence, not reproduced HMS
wins; predicted recall is not an exact-result certificate.

The next implementation hypothesis is immutable prepared document ownership
through publication, avoiding deep vocabulary/text/embedding clones under reader
exclusion. Treat this as a compiler/ownership experiment: require identical wire
payloads, exact source/rank/filter behavior, old-owner lifetime, encrypted/plain
journal and compacted recovery, and allocation/RSS checks before measuring.
Separately diagnose I/O/maintenance/scheduler regressions instead of assuming
publication gains explain them. No protected validation access, model download,
new remote/paid compute, git commit, push or publication occurred.


### Shared document ownership and retained replay capacity (2026-10-14)

The completed local diagnostic in `superiority_v3/shared_document_publication_v1/`
preserves the incremental-posting parent and shares immutable prepared mutation/
live document owners. Four frozen files differ, including existing serde rc
support; Cargo.lock and bindings remain unchanged. Exact legacy bytes, shared
owner identity/lifetime and plain/encrypted uncompacted/compacted recovery pass.
All six final Rust gates pass (402 default/621 all-feature tests), rebuilt Node
passes four checks and eight actual-record adversarial oracle methods pass.

All24 cells preserve exact behavior across115,200 offers,107,868 completions,
7,332 drops,3,840 operations,3,208 whole-bulk queries and310,920 exact restart and
cross-arm pairs each. All load<3 gates fail; exploratory load<6 admits nine cells.
Original numerical and strict adoption audits fail. Publication/service/response
p99 regress in8/10/8 pairs; write/append in7 each, serving max RSS in8, compaction/
final maintenance in5 each, drops in4. Static max/median and serving median RSS
regress in every pair. The expanded audit retains p50/max losses. All timing is
descriptive; observed unrelated CPU work does not causally explain every tail.
No recovery or competitive-superiority claim is accepted.

Four isolated replay profiles identify retained capacity: the old clone stores
one chunk slot per one-chunk document; shared deserialized owners retain four.
All51,820 document/clone control pairs preserve exact wire bytes. Spare reserved
chunk bytes add2.254/9.014 MiB at5,182/20,728 records, excluding metadata/BTree
nodes and allocator rounding. This is a deterministic allocation cost consistent
with increased static RSS, not an attribution of all losses. Next test length-
sized deserialized chunk storage at the3/4/5 growth boundary, with exact bytes and
plain/encrypted recovery; then remeasure. Separately, direct commit serialization
may eliminate its complete intermediate encoded buffer; that is unmeasured.

Five additional primary/first-party leads through October14 are retained in
`research-recovery.json`: [German BtrLog](https://arxiv.org/html/2606.27051v1),
[South Korean ITLogging](https://pure.seoultech.ac.kr/en/publications/improving-database-performance-by-leveraging-network-assisted-log/),
[Greece/Canada/US snapshot work](https://arxiv.org/html/2007.02372v3),
[Purdue replay tails](https://cs.purdue.edu/homes/csjgwang/pubs/SIGMOD26_RaaS.pdf)
and [Turso group commit](https://turso.tech/blog/turso-group-commit).
Institutional affiliations, dates, selective review depths and transfer limits
are recorded; no nationality is inferred from names or external win reproduced.
Do not replace prepared mutation recovery with nondeterministic request/model
replay, weaken persistence or change the fixed single-writer workload to claim
recovery. All authorization, protected-data and negative-evidence limits remain.


### Tight replay chunks: partial memory recovery (2026-10-14)

`superiority_v3/replay_capacity_v1/` completes the corrective experiment. An
actual-core3→4 capacity failure is retained; length-sized chunk deserialization
now passes3/4/5 plain/encrypted direct journal and compacted recovery with exact
wire bytes, source/filter/embedding/rank identity. Arc publication and incremental
postings stay intact. Only documents.rs and mutation regression coverage change.
All six Rust gates pass (403 default/622 all-feature checks); rebuilt Node passes
four checks, with eight actual-record adversarial oracle methods also passing.
Cargo.lock and bindings remain unchanged.

The same24-cell workload checks115,200 offers,113,708 completions,1,492 drops,
3,840 operations,3,600 whole-bulk queries and310,920 exact restart and cross-arm
pairs each. Static max/median and serving median RSS improve in every pair.
Median static maxima are104.063→101.703/103.641→100.953 MiB at5,182 records and
378.313→368.891/378.563→368.281 MiB at20,728 (batch2/64). Two serving peaks regress;
publication/response p99 regress in6 pairs, service/write/append in5, compaction/
final maintenance in3, drops in1. All producer/queue/reservation and p50/max losses
remain. All load<3 gates fail; exploratory load<6 admits ten cells. Original
numerical and strict audits fail; no full no-sacrifice or competitive win follows.

Four isolated profiles prove chunk capacity4→1 and exact clone/control JSON bytes
for51,820 documents. Recovered reserved slot bytes are2.254/9.014 MiB at5,182/
20,728, excluding other structures and allocator rounding; these are not RSS or
latency measurements. Paired RSS observations are consistent with capacity repair,
without attributing every resource/tail effect. Source freezes, binaries, exact
oracles, rejected pre-fix regression and gate logs remain durable and bound.

`next-recovery-hypotheses.json` preserves the next discriminating tests: power-of-
two fixtures conceal inner-vector or metadata-array spare capacity. Test127/128/
129 embedding values, a supported non-power-of-two VSA active count, metadata
arrays3/4/5 and caller-overreserved buffers before generalizing ownership changes.
Direct commit serialization and finer isolated append/maintenance profiling remain
separate unmeasured mechanisms. All prior authorization/data/compute limits stand.


### Wider immutable-owner capacities (2026-10-14)

The six-case synthetic diagnostic in `superiority_v3/owned_buffer_capacity_v1/`
leaves production unchanged and compares the pre-Arc posting parent with current
Arc/tight-chunk ownership. Prepared sparse vectors keep16/24 slots for4/6 indices,
while replay keeps8 for6; embeddings127/129 replay with128/256 capacity. Caller
metadata arrays reserved64 retain64 for3/4/5 elements, then4/8 for3/5 on replay.
The old deep clone stores length-sized buffers. All36 clone-wire,24 recovery and
six cross-arm query checks are exact. Both isolated Clippy/build checks pass;
root403/622 gates and Node4 remain valid. Reserved counts do not prove RSS/latency
or learned-quality effects; no performance or competitive win is claimed.

Next normalize owning buffers during common preparation (including model stages)
and deserialization before immutable sharing, retaining all bytes, values, map
ordering, missing/default field behavior, old views and durability. Actual-core
boundary regressions and complete gates/serving-resource audit are required before
adoption. Root code is not changed by this diagnostic. All prior authorization,
data/compute boundaries, failures and negative evidence remain preserved.


### Compact all owning document buffers (2026-10-14)

`superiority_v3/compact_owned_buffers_v1/` completes common preparation and field-
deserialization buffer compaction while retaining Arc sharing/incremental posting
publication. Only documents.rs and mutation regression coverage differ; values,
bytes/order/types and durability/security remain. Actual-core127/128/129, VSA1536,
metadata3/4/5, Unicode/empty reserved buffers, snapshots and plain/encrypted recovery
pass. Pre-fix capacity64/length7 and two test-only shadowing gate failures are
retained. Final gates-final-3 passes404 default/623 all-feature checks; Node4,
eight actual-record oracle methods, legacy bytes/defaults pass. Bindings/lockfile
are unchanged. Repeated six-shape controls prove every measured owning buffer
capacity=length in all states, with36 exact clone-wire,24 recovery and six cross-
arm query pairs. Reserved capacity excludes allocator rounding/BTree nodes/RSS;
no runtime or learned-quality claim follows from these profiles.

All24 serving cells preserve exact results across115,200 offers,115,191 completions,
9 baseline-only drops,3,840 operations,3,598 whole-bulk queries and310,920 restart/
cross-arm pairs each. Ten original load<3 gates pass; exploratory4/6 admit all24.
Retain publication/service p99 losses in6/8 pairs, response/write/append in4 each,
serving max RSS in7, static/serving median RSS in9, final maintenance in8 and
compaction in2, plus all p50/max and producer/queue/reservation losses. Numerical/
strict audits fail; no unconditional no-sacrifice or competitive win is accepted.

The next isolated hypothesis removes commit's complete intermediate JSON buffer
using existing direct compaction serialization, requiring exact transaction bytes
and complete failure/recovery/resource checks. Append/maintenance phase attribution
remains separate. All previous authorization, data/compute and evidence limits
persist; no broader source, paid/remote compute, model download or git/publication
operation occurs.


### Direct commit serialization (2026-10-15)

`superiority_v3/direct_commit_serialization_v1/` completes this bounded hypothesis.
The compact-owner parent independently reverified before source/native edits.
Only mutation.rs production code changes: direct JSON into the prefixed payload
before writer reservation, keeping locked serde_json's128-byte initial JSON
allowance. Compact immutable owners, incremental postings, validation, synchronous
append, encryption and atomic publication remain. All21 full payload pairs are
exact; allocations 2→1, reallocations equal, cumulative requested bytes lower and
requested peaks down 22.5–33.8%. Payload-only capacity increases in all 21; this
is requested allocation, excluding allocator rounding/transient relocation/RSS.

The initial Vec::new direct candidate remains frozen:21 exact pairs and lower
peaks, but3–4 extra reallocations. Its correction trades more cumulative requested
bytes in 3 and more peak/payload capacity in 4 cases relative to that initial
candidate; both complete comparisons and every tooling failure remain preserved.
All six final Rust gates pass 406/625 tests on fixed source; Node 4 and eight actual-
record oracle methods pass. All-variant complete bytes, transaction-size/validation
boundaries, identical retry payloads, coherent snapshots, interrupted publication
and plain/encrypted direct/compacted recovery pass. Bindings/lockfile are unchanged.

The unchanged 24-cell serving grid checks 115,200 offers, 111,097 accepted completions, 4,103 drops, 3,840 operations, 3,323 whole-bulk queries and 310,920 exact restart/310,920 cross-arm pairs. 0/24 original load<3 gates pass; exploratory load4/6 pass 18/24 cells. Numerical gates=False; strict no-regression=False; all-metric pairs without losses=False. Keep every loss in analysis.json, including append_p99=5, compaction_ns=7, dropped=2, final_maintenance_lease_ns=6, producer_lag_p99=6, queue_delay_p99=6, reader_exclusion_p99=7, sampled_serving_max_rss=5, scheduled_response_p99=7, service_p99=3, writer_lease_wait_p99=3, writes_p99=7. Static/serving median RSS, producer/queue/reservation and p50/max losses remain in the full audit. No unconditional recovery or competitive superiority follows.

Serialization-only allocation repair does not attribute append or maintenance
tails. CRC/compression/copy/flush and final-maintenance isolation remain the next
bounded mechanism tests. All previous source/evidence, data/compute limits and
losses persist; no broad/protected/remote/paid work, model download or git/publication
operation occurs. Historical receipts are not weakened for the current checkout.

### Growable arena capacity (2026-10-15)

`superiority_v3/append_stage_recovery_v1/` freezes the direct-serialization parent,
attributes the1GiB full-sync cost, retains the rejected fixedv3 packing cliff and
completes bounded format4 correction. Read report/analysis/growth-costs and the
independent verifier. New-store file capacity starts64MiB on Unix while logical
1GiB packing/offsets remain; other platforms retain full allocation. Existing
formats1/2/3 retain their geometry, embedding schema2 remains, and all synchronous
flushes stay. Metadata version/capacity change explicitly. Transaction/frame
bytes, query results, encryption and compact incremental publication remain exact.
Two older binaries fail closed before data writes (lock PID update remains), and
both legacy layouts pass actual new-binary write/compaction/restart controls.

All six final Rust gates pass418/637 checks; Node4 and adversarial oracle8 pass.
Twelve new integrity checks include actual growth failure/retry, partial allocation
bounds/recovery, waiting readers and plain/encrypted legacy interruption replay.
The final three packing boundaries preserve60 complete frame/offset pairs and120
restart reads, with identical used address space and file capacities640MiB,640MiB,704MiB
versus1024MiB. Fixedv3's25% capacity loss and all earlier failures remain. Growth
costs remain explicit: 50/60 descriptive write pairs are slower, including
27/28 growing writes. These counts/clocks do not prove RSS or serving latency.

Final compaction medians: N5182/b2=4903.1→571.0ms; N5182/b64=5221.7→581.2ms; N20728/b2=6907.0→1705.5ms; N20728/b64=5690.3→1218.0ms.
The final24-cell grid checks 115,200 offers, 109,374 accepted completions, 5,826 drops, 3,840 operations, 3,348 whole-bulk queries and 310,920 restart/310,920 cross-arm pairs. 0/24 original load<3 gates pass. Numerical gates=False; strict no-regression=False; all-metric pairs without losses=False. Retain pair losses: append_p99=1, final_maintenance_lease_ns=1, producer_lag_p99=4, queue_delay_p99=4, reader_exclusion_p99=7, sampled_serving_max_rss=2, scheduled_response_p99=4, service_p99=8, writer_lease_wait_p99=8, writes_p99=1. All p50/max, producer/queue/reservation and static/serving median RSS losses remain in analysis.json.

Every source, native artifact, failure and loss is retained. Overall recovery and
competitive superiority remain unestablished. User requests acceleration; the
decision artifact prioritizes one bounded public production document comparison
with a cached native control after this seal, preserving the broad contract and
all quality/resource/timing gates. ANN/QA breadth remains parked; source/configuration
bound gate reuse and focused pre-freeze checks reduce repeated validation work.
A bounded stack attempt timed out during symbol processing. Subsequent paired
lookup controls reproduce53.6/89.8 seconds with the historical dependency directory
versus.41/.55 seconds with an empty one, producing identical metadata. Default/all-feature
intermediate directories are isolated inside the required target, retaining all
old cache files and identical source, commands, profiles and assertions. These
lookup controls do not establish a full-build or product speedup.
Further implementation work must attribute remaining
append/final-maintenance and growth-boundary costs before any new I/O/scheduling
change. All original authorization, data/compute and no-publication limits persist.

### Common Node diagnostic checkpoint (2026-10-15)

`superiority_v3/document_node_control_v1/` completes a bounded same-runtime public
NFCorpus dev control after the independently sealed format4 experiment. Source,
bindings and root native remain unchanged; all six Rust gates418/637 and Node4 stay
valid, and the isolated release package passes4 original API checks. Ten thousand
three hundred sixty-eight calls complete;5184 HMS calls preserve full historical
results (score fields bitwise), and9072 repeat pairs remain exact. Only the three
declared absent NAPI Option fields map to null.

The two alternating pairs give descriptive HMS/SQLite throughput ratio8.226 and
sampled serving-max RSS ratio1.693. Both pairs lose on all static/serving max+median
RSS metrics. All four original load gates fail; the11-round/statistical contract is
unmet. Different result richness, tokenizer/IDF and client/retention/validation
costs remain explicit. No competitive or relevance win follows. Initial observer
shutdown timeout and timed-audit adapter/source/records remain separately retained.
Read report/analysis/protocol and independently reverify the pilot.

Next attribute memory with empty/loaded native processes, client heaps/retained
results and actual terms/postings/registry ownership before a new representation
or wider sweep. VSA vectors already use sparse u32 indices. All exactness,
durability/security, publication and authorization limits remain unchanged.

### Client-retention memory attribution (2026-10-15)

`superiority_v3/document_memory_attribution_v1/` completes six bounded processes
in alternating empty/discard/retain order:15,552 calls and10,368 loaded exact
reference calls. All11 result fields remain bound, f64 scores bitwise; native and
root source remain unchanged. Empty static RSS~70.9MiB, loaded~124.3MiB; discard
serving~138MiB versus retained first/current passes~207–217MiB. Both paired
retention differences (+68.3/+79.0MiB) remain. Raw memoryUsage snapshots, external
samples and full digests are retained. RSS is not native requested heap; no forced
GC/cache eviction, new timing acceptance or competitive loss exemption applies.
The earlier1.693RSS ratio and0/4load gates stand. Next attribute actual Rust
allocations and terms/posting/registry capacity in a source clone before a core
representation change. All original integrity/security/authorization limits remain.

### Isolated native allocation attribution (2026-10-15)

`superiority_v3/document_native_allocations_v1/` completes six release native
processes/15,552calls and10,368exact loaded queries. Quiescent live requested Rust
heap~43.5MiB remains unchanged after warmup through all calls in all six cells.
454,312forward term entries,19,211posting terms; posting capacity10,771,648bytes
versus7,268,992used leaves3,502,656slack. Term-string capacity2,285,548bytes.
320/416byte blocks total25,327,104requested bytes but are not field-attributed by
this counter; denser BTree reconstruction is the next bounded hypothesis.
All-thread System forwarding and coherent allocation-free counter snapshots
exclude allocator rounding/mmap/V8/transient relocation. Self-check capacity
deltas and histogram sums pass;4original Node checks pass for both builds. The
initial unlocked build ran no attribution cells and is retained. Root source and
native stay unchanged; all prior losses and authorization limits persist. No
production memory or competitive win is established by this diagnostic.

### Terms-tree node compaction (2026-10-15)

`superiority_v3/compact_term_nodes_v1/` completes the bounded unconditional
consuming term-map reconstruction hypothesis. Only documents.rs production logic
changes; test module covers complete bytes/legacy-frequency boundaries. All21
transaction payload pairs are exact; six Rust gates419/638, rebuilt Node4,
production release Node4 and oracle8 pass. Publication/durability/security stay.
Instrumented loaded requested heap~43.5→33.7MiB;10,279,296fewer node-sized bytes.
Recovery allocation requests rise50,697(~4.7%); every cost stays.
Uninstrumented Node pairs preserve10,368complete historical calls and648cross-arm
rows; static/serving-max RSS ratios 0.9205/
0.9484,QPS ratio 0.9867.
4/4load gates pass;2/11rounds remain insufficient. All API/open/RSS losses stay.
The original24-cell grid retains115,200offers, 115,200accepted,
0drops, 3,840operations and
310,920restart/310,920cross-arm exact pairs;
13/24load gates pass, numerical=False, strict=False.
Pair losses:{"append_p99": 5, "compaction_ns": 5, "final_maintenance_lease_ns": 5, "producer_lag_p99": 4, "queue_delay_p99": 4, "reader_exclusion_p99": 7, "scheduled_response_p99": 4, "service_p99": 6, "writer_lease_wait_p99": 6, "writes_p99": 7}.
All p50/max, producer/queue/reservation and static/serving RSS losses remain.
Every failed tooling attempt, source/native predecessor and receipt remains.
The initial sealing audit was interrupted after an accidental whole-workspace
scan; explicit frozen-manifest verification replaces it, with exit130/source/log
retained. No product measurement or successful gate repeats.
No recovery or competitive superiority follows. Next prioritize a bounded matched
output-contract SQLite control while retaining the original control/gates and
quality differences; do not multiply historical-source ratios into current claims.
No authority/data/compute/publication limit changes; ANN/QA breadth stays parked.

### Full-output SQLite diagnostic control (2026-10-15)

`superiority_v3/document_output_contract_v1/` retains the original IDs-only native
SQLite rank control and adds8present DocumentResult fields from a persisted
metadata table/common chunking preparation. SQLite serving does not load HMS;
query/rank semantics remain identical to the frozen earlier SQL control. All
15,552calls complete,5184HMS historical complete calls/10368SQL historical rank
calls/5184SQL complete projected calls exact,13,608repeat pairs exact.6/6load
gates pass;2/11rounds remain insufficient for acceptance. Current same-runtime
HMS/SQL-full QPS ratio 8.3504, serving-max RSS ratio
1.2222. Both source pairs retain every HMS static/serving RSS/open loss;
tokenizer/IDF and wider functionality differ. No relevance/overall superiority.
Root source/native/bindings stay identical to sealed419/638Rust and Node4 source;
no product build/gate repeats. Unused duplicate-cache adapter is retained and
revised before cells. Next isolate a compact UTF8/usize/u32 forward-term table
with exact legacy map bytes/parsing and unchanged ownership, before any root edit.
All authorization/evaluation/compute limits and historical losses persist.

### Packed forward-term hypothesis, isolated only (2026-10-15)

`superiority_v3/packed_forward_terms_v1/` has no root product edit. Optional boxed
short-key UTF8/usize/u32 tables preserve sorted map serialization and delegate old
parsing; byte-heavy fallback keeps existing BTree reconstruction. Packing budget
is a layout-derived temporary allowance heuristic, not an unmeasured peak promise.
Release node-api/security lib Clippy -Dwarnings and Node4 pass after retained
interface/auto-deref corrections.15,552instrumented calls/10,368loaded full exact
results finish with requested heap~33.7→24.7MiB; chunk header capacities shrink,
posting counts/capacities stay.39shape/12parser/error JSON+bincode controls and
allocation costs are pending; preflight fixture failures remain. No adoption,
accepted latency/RSS or superiority claim. Root419/638/Node4source remains fixed.
Next inspect all boundary losses before considering a private posting layout
control and one later combined root validation. All original limits remain.

### Compact document-index candidate (2026-10-15)

`superiority_v3/compact_document_index_v1/` completes two independently bounded
private memory mechanisms without schema/range/I/O/security/publication changes.
Root247files/sevenRust changes; all six final gates423/642+oneignored, rebuilt
Node4/release Node4/oracle8 and21complete transaction bytes pass. Pinned39JSON+
bincode/12parser cases and7native slot ranges pass. Requested native heap
~33.7→24.7→22.1MiB; posting reservation10,771,648→8,078,736bytes. Packing costs
remain24allocation/5peak/15retained-delta losses, +48box/+16peak long-key cases.
Wrong-dependency fixture/root-lock update and exact restoration remain documented;
only actual1.0.151controls count; product source/native runs use correct frozen locks.
Six common-client source/SQL-full cases check15,552calls with exact full/historical
results. Current/before QPS1.0097, static/serving-max RSS
0.8632/0.9148; current/SQL-full QPS8.4079, serving-max RSS
1.1057;0/6load gates,2/11rounds. All losses/quality/IDF differences stay.
Original24cells retain115,200offers,115200accepted,0drops,
3840ops,310920restart/310920cross-arm pairs;
8/24load gates,numerical=False,strict=False.
Pair losses:{"append_p99": 7, "compaction_ns": 8, "final_maintenance_lease_ns": 3, "producer_lag_p99": 6, "queue_delay_p99": 9, "reader_exclusion_p99": 1, "scheduled_response_p99": 6, "service_p99": 2, "writer_lease_wait_p99": 4, "writes_p99": 5}. All p50/max/queue/
producer/reservation/static-serving RSS losses and historical artifacts remain.
No unconditional recovery/adoption/competitive superiority; next inspect remaining
losses before new code or a prospective11round public control with bounded quiet
scheduling and unchanged gates. No authorization/data/compute/publication changes.

### Eleven-round public document decision checkpoint (2026-10-15)

`superiority_v3/document_decision_checkpoint_v1/` uses the sealed compact-index
source/native without edits/builds. Eleven prospectively alternating three-arm
rounds complete in117.82seconds including60.04seconds bounded quiet wait; all33
original load<3 gates pass, with no filtered/replaced/retried cells. All85,536calls
pass historical full-result checks:57,024HMS/28,512SQL-full,3,564before/current
first-pass pairs and74,844repeats. Existing423/642Rust/Node4/oracle8/byte gates
remain valid on identical source. Earlier0/6 and8/24load results stay unchanged.

Seven-member conservative Bonferroni paired-bootstrap diagnostic family:
current/before p95=.98487,p99=1.01079,serving-max RSS=.91433, allbounds<=1.10.
Current/SQL-full QPS=8.43366[8.40145,8.50574]; serving-max RSS=
1.10469[1.10327,1.12279] FAILS<=1.10. Query-paired nDCG/recall lowerbounds
-.003699/-.003150 pass only diagnostic noninferiority-.005. No relevance win.
Current loses open/p99 in8/11source pairs and API mean/p95/QPS in1/11each.
It loses open/static-max+median/serving-max+median againstSQL in11/11each.
Diagnostic family=false; primary contract/competitive superiority=false. SQL is
not required aligned Lucene/fastest eligible primary; absent modes/concurrency/
filter/cold/build/hardware/ANN/QA cells stay absent. Holm primary correction is
unestablished; diagnostic intervals do not replace it or earlier grid criteria.

Decision: isolate checked narrower private forward-term offsets before new root
changes; current454,312usize offsets request3,634,496bytes on64-bit. Savings/RSS/
tail effects remain hypotheses; preserve arbitrary input ranges with fallback,
full bytes/parser/errors, snapshots, retry, durability/security and publication.
Keep all historical losses/artifacts and original authority limits. This narrows
the next decision without making a superiority timeline promise.

### Checked term offsets, bounded validation complete (2026-10-15)

`superiority_v3/compact_term_offsets_v1/` changes one private module; checked u32
starts/full-range Tree fallback retain bytes/parser/ranges/publication/security/
durability. Final six gates424/643+oneignored, rebuilt Node4/release Node4/oracle8
and21complete transactions/72JSON+bincode/12parser checks pass. A32bit test-only
expectation correction retains initial4passed gates/interrupted5; final runtime
code before cfg(test) equals the measured prototype. No32bit runtime claim.
Six native controls pass15,552calls/10,368loaded exact;1,817,133fewer requested
loaded-minus-client bytes inall4pairs; no post-warmup growth.12shape allocation
request losses remain; no new measured peak/realloc/retention losses. Initial
fixture/type failure and5completed native cells/final-empty SIGABRT remain;
const TLS fixes only a global-delta self-check race, not product allocation.
All33prospective timing cells/11rounds passload<3/85,536exact calls. Current/SQL
QPS8.46425; servingRSS1.09196[1.08950,1.09683] fits1.10. Source-p99
1.00916[.97132,1.11122] FAILSupper1.10; diagnostic family=false. All source
andSQL p95/p99/QPS/open/static/serving losses remain. SQLite is not required
aligned Lucene/fastest eligible primary; other primary cells/Holm remain unknown.
Original24cells:{"accepted": 115200, "completed_wholly_during_append": 5631, "dropped": 0, "exact_restart_pairs": 310920, "offered": 115200, "operations_checked": 3840, "whole_bulk_queries": 3600};
cross-arm=310920; load=0/24,
numerical=False,strict=False.
Pair losses:{"append_p99": 5, "compaction_ns": 7, "final_maintenance_lease_ns": 5, "producer_lag_p99": 7, "queue_delay_p99": 7, "reader_exclusion_p99": 8, "sampled_serving_max_rss": 4, "scheduled_response_p99": 5, "service_p99": 6, "writer_lease_wait_p99": 7, "writes_p99": 6}; full p50/max/
queue/producer/reservation/static-serving RSS losses stay. No unconditional
recovery or superiority. Required Lucene9.12.3 core/JDK17 are verified cached;
availability receipt establishes no analyzer/scoring/tuning eligibility. Next
inspect remaining tails and bounded aligned lexical control; keep ANN/QA parked,
every prior loss and all original authority limits.

## Memory-only priority and architectural byte audit (2026-10-15, active)

User requires superiority in each category before moving on; memory is active.
Current same-output SQLite serving-RSS ratio1.091961 is a9.2% loss, despite
passing the prior1.10 allowance. No memory superiority or Lucene memory result.
Stop Lucene alignment work; preparation remains retained with its stop receipt.
Audit current frozen source in superiority_v3/document_memory_ledger_v1 before
another optimization: exact Rust allocation ownership/rounding, JavaScript
result retention/heap, and OS mapping/residency accounting, with every residual
explicit. No product change or new competitive claim is authorized by a diagnostic
measurement alone. Earlier sealed evidence and all losses remain historical.

## Architectural memory census (2026-10-15, bounded audit; memory remains active)

Evidence: `benchmarks/results/superiority_v3/document_memory_ledger_v1/`.
Unchanged production247files/native; no repeat of valid424/643/sixRust gates.
Isolated inventory/trace releaseClippy and Node4 pass.15,552native calls including
10,368loaded historical exact; reference21,373,582requested/23,005,392usable,
1,631,810rounding. Field-owned21,275,702;97,880reference runtime bytes explicitly
unattributed. Separate symbolized process resolves253residual blocks/98,080bytes
by actual allocation stacks; cross-process addresses/byte totals are not mixed.
648traced exact calls allocate223,105,606bytes cumulatively,640,058maxextra live.
One368byte empty-query delta remains; logged/address refinement648allbaseline
is a perturbed non-reproduction, not retroactive attribution or erased loss.
Eight V8snapshots completely sum bytype. Retained result graph12,430,368HMS vs
7,356,672SQL,44,172vs44,156hits;5,073,696gap. Duplicated IDs, result property
storage and larger metadata objects are measured; absent Option fields are absent
ownproperties. Initial missed hidden SQL numeric nodes overstated gap; preserved
and corrected. Whole pure malloc class sums27,280,736HMS vs7,057,824SQL.
Post spare capacity2,626,992bytes; one-key metadata-tree unused/bookkeeping
2,092,608. Initial multi-phase snapshots perturb later RSS; clean final-only
children/OS stop-resume measurements retain that distinction. Tooling failures,
all prior losses/native predecessors and historical evidence remain retained.
PhysicalRSS is not a unique byte-to-field partition; shared/partly-free pages,
current368trace delta and unmeasured write/recovery/compaction/other-feature peaks
prevent a universal every-byte claim. The audit establishes bounded allocation
and graph ownership, not memory superiority. Current accepted1.091961SQLiteRSS
loss remains; memory is the active category. Next isolate a single memory
mechanism only after this census/actual ownership paths, preserving exact bytes,
fullresults/ownproperties, retry/snapshots/recovery/durability/security and the
publication improvements. No broader control, category advancement or claim.

## Precision closure follow-up (2026-10-15, active; no product change)

User requires no unlabelled remaining bytes, including small buffers. Current
`document_memory_precision_v1` stopped loaded census labels123,944live malloc
blocks/27,510,640usable bytes with0unlabelled.108,223blocks22,903,408bytes join
actual Rust field allocation bases;15,700blocks4,603,248bytes have heap type/
allocation labels;21blocks3,984bytes have exact live reference-root witnesses.
Roots identify ObjC runtime maps/data lists, dyld thread state, libc basename
and profiling-worker storage. Counts do not mix other processes' addresses.
Source samplerClippy/Node4 pass; Root247/native unchanged; no fullgate repetition.
All2592query windows end atbaseline;48actual368byte allocations have captured
backtraces to captured posting buffers and are all released. Do not label the
original net368window as a368byte block. Historical trace reconciliation now
shows all648baselines21373786 and firstbaseline=lastfinal: original+368after
empty query is freed before nextquery(-368), not a retained whole-run increase.
Its historical source address was not recorded and cannot be back-assigned from
new unrelated368byte allocations. Original receipts/data remain unchanged.
`document_runtime_memory_stacks_v1` all-by-size yields complete same-process
malloc sums and43,360unlogged bytes/637blocks perLiteprocess; full logging/type/
reference-root analysis closes current live-heap labels. Instrumentation affects
resource use; no new competitive claim. Addenda not sealed until independent
verification. VM/Untagged mapping/source purpose and exact physical accounting
still being checked. Memory category remains active; no product optimization or
category advancement while the requested audit remains open.

## Current memory accounting closure and remaining architecture audit (2026-10-15)

Precision and runtime-stack addenda are sealed and independently reverified:
`document_memory_precision_v1`529files;`document_runtime_memory_stacks_v1`69files.
Root247/native unchanged. Three malloc checkpoints label all123944/123861/123910
blocks (27510640/27502432/27485888usable bytes); identification basis is preserved
per address.7776full historical queries exact; all windows/whole traces return
baseline;48actual368Bposting allocations/cell all freed. Two OS censuses resolve
all16Untaggedregions from latest live VM allocation stacks. The1.25MiB GraphCycles
arena reuses an address of an earlier freed1MiBprofiling mapping; those owners
must not be confused. Footprint category sums=198132656/196395616bytes, residual0;
self kernel RSS=184909824internal+37781504external+32768reusable=222724096exact.
Different planes/timestamps overlap and are not added as field-exclusive RSS.
Historical368source remains unobserved, but raw original trace proves all648
baselines21373786 and full net0: +368atqueryend is released before nextcall.
No retrospective owner transfer to original637unlogged Lite blocks or368window.
No memory/global superiority: accepted production/SQLRSS1.091961 remains a loss.
The requested architecture-wide audit continues in
`document_memory_architecture_layout_v1`: unchanged production clone/full stable
compiler DWARF for inline fields/header/padding, followed by lifecycle peaks for
write/recovery/compaction/concurrent snapshots. No product optimization or
category advancement before that audit closes; all prior limits/receipts stay.

## Inline layout addendum (2026-10-15; memory audit still active)

`document_memory_architecture_layout_v1` freezes unchanged247productionfiles,
retains25,055compiler type layouts and independently cross-checks30named structs'
complete fields/padding against DWARF. StoredChunk136pad7; CapturedPosting16pad4;
HmsCore3328align16pad3 contains1120inlineencryption/424signingOptionbytes even
when disabled. One-key metadata each640usable=56live+560spare+12bookkeeping+
4padding+8rounding;3633objects exactly2325120usable. No optimization or RSS-win
projection. Full-debug rebuiltNode4 passes; production gates need no repetition.
Initial missing-object archive, metadata-writer refusal and parser failures stay.
DWARF has49debug-type/storage conflicts; compiler diagnostic resolves actual
storage views without inventing disjoint async/zero-sized-function payloads.
Method amendment uses scoped sameRust1.96 print-type-sizes, no unstable language
features or product/source/config changes. Read report/analysis/cross-checked
layouts/verifier. Next ingest/plain-encrypted recovery/compaction/concurrent-
snapshot peaks. Plain encrypt already borrows; only plaintext recovery still
copies an owned frame. No product optimization until requested audit closes;
no memory or overall superiority, category remains memory.

## Lifecycle allocation census in progress (2026-10-15)

`document_memory_lifecycle_v1` privateV3/Root247 unchanged: TLS allocation-birth
phase request/usable peaks partition exactly at allocator boundaries; stage
traffic conserves all-thread deltas. ReleaseClippy/Node4 and explicit4095/4096/
4097buffer content/realloc/free self-check pass. V2+32globaldelta/SIGABRT retained;
V3 marked identities avoid assuming other live allocations stay fixed. Fresh
self-check0/0/0 does not assign old32source. Initial smoke receipt-envfailure
preserved; fresh10plain/encrypted smoke cells80exactcalls pass full transaction,
held-two-version snapshot/tail and same-candidate append-failure/retry proofs.
Prospective18cells46656historicalcalls running: unchanged production baseline
E19package-v2/privateV3, plain/encrypted fullbulk ingest/recovery/compaction/
snapshot+tail/retry/recovery-after-tail, complete frame hashes/full retry memcmp.
No competitive inference or category advancement. Scope labels are observed
allocation provenance, not exclusive field or physical-page ownership. Analyze/
independently verify/seal raw results before closure. Memory/global superiority
remain false; accepted servingRSS1.091961SQLite loss remains.

## Finer lifecycle byte ownership (2026-10-15; verification/seal pending)

Parent lifecycle census sealed678files/reverified:18cells46656exactcalls,
21804framepairs/36peaktraces/576rows0residual; all finaldelta/cipher losses stay.
Refinement `document_memory_lifecycle_precision_v1` adds36function/allocation
scopes, Clippy/Node4/markedselfchecks pass, six cells15552historicalcalls and
14538completeparentframepairs exact.16peaktraces/576rows sum0residual. Encoder
traffic10011400708/10207014912bulkbytes: majoritybundle3655922672,
bind2786673776,permute1359853304,deterministic1332890768,collections876060188.
All encoder-born requests retired bytraceend; frees byexecutingphase must not be
mistaken for retirement bybirth. Directnewscopes witness RNG336, cipher/output
5534896, rawfallback5534896, failedcompress5556603, Argon2workspace19922944bytes.
Metadata/source/layout/V8/OS seals stay; no cross-process pointer attribution.
Next verify/seal refined ledger then freeze one memory candidate; no product edit
or category advancement yet. Baseline serializerholds9043968cap/5534868payload
through publication; spare3509100. Incompressiblecopy and ownedplainrecoverycopy
are additional unimplemented lifetimes. No numerical/RSS projection or memory/
overall superiority; SQLite servingRSS1.091961loss remains accepted state.


## Finer lifecycle closure and static feature coverage (2026-10-15)

Refined lifecycle addendum `document_memory_lifecycle_precision_v1` SEALED358files
and independently reverified: sixcells15552completecalls/14538framepairs,16peak
traces/576rows0residual. All10011400708encoder-requestedbytes retire bytraceend;
executing-scopefree andbirth-retirement remain distinct. RNG/cipher/rawfallback/
LZ4/Argon2allocation scopes are directly witnessed in these fresh processes.
No old pointer attribution, product change or competitive claim. Root247/native
unchanged. Static feature coverage continues in `document_memory_feature_layout_v1`: 
unchanged source, default/all-feature compiler layouts with cached offline
dependencies; no model execution/download or broader benchmarking. This closes
a coverage question, not memory superiority; SQLite servingRSS1.091961loss stays.


## Static feature and external-root byte ownership (2026-10-15)

`document_memory_feature_layout_v1` SEALED310files: unchangedRoot247/native,
default/allfeature45252newlayoutrecords plusparent25055; all30HmsCorerootfields
maptosource/activation/ownership, sizes1784/3328/4208with3paddingeach,0unlabelled
inlinebytes. NineHTTPnamedstructs/6189records havecompletepartitions; successful
object-onlycompile startsno service/models. TwoPythonhost-symbol101linkfailures
remain, includingfailedadditionalemit; wrapperreplacesonlyselectedcrate'semit,
original/transformedargvchecked. Pretty-namecollisions9/79retained byrawrecord
identity; initialuniquenessassertionandnamespaceclarificationretained. Other
Node/Python/HTTP/private-search/caller-owned/runtime roots explicitlycatalogued.
Coverage.json distinguishes measured0residual fromunexecutedruntimecoverage and
threeirrecoverablehistoricalsitegaps; nouniversalruntimeclosureorcompetitive
claim. AcceptedSQLiteRSS1.091961lossandunmeasuredLucenememoryremain. NoRootedit,
categoryadvancementorunchangedproductiongatererun.


## Private commit-payload lifetime probe (2026-10-15; active)

`document_commit_buffer_lifetime_v1` freezes one private drop(payload) between
successfulappend andtransaction.publish, comparedto sealedE21source/package.
Existingwriterlease remainsheld; appendErrpath, JSONbytes, crypto, arenaformat,
publicationArcs andallalgorithms remain. ReleaseClippy/Node4/markedselfcheckpass;
Root247/native unchanged. Prospective16pairedcells/41472historicalcalls running:
plain/cipher ingest/recoveryaftertail/heldversions+tail/retry, completeframebyte
hashes/whole retrymemcmp/f64+Arcchecks. Diagnosticrequested/usablepeak partitions
only; no superiority/projection orRootintegration. Allcoverage/historylimits stay.


## Paired commit lifetime result (2026-10-15; private only)

`document_commit_buffer_lifetime_v1` sealed468files:16cells41472fullcalls,29084
parent/14542crossarmcompleteframepairs,4retrymemcmp/4heldversions-f64proofs,
44Rustpeaktraces/1584rows0residual. Plainbulkrequestedpeak -5562620bytes,
usable -6087184; ciphertext/snapshot/retry/recovery baseline-adjustedpeaks and
steadyretention unchanged. NewglobalpeakisearlierLZ4, so9MBpayloadcapacityis
notclaimed aswholesaving. Bulktrafficunchanged. Everyrawincreaseincluding
initial+4128,firstflush+2848/+2992,64bytepeakdifferences andplainretryfirstflush
+4128retainedincrement remains explicit. Noinventedruntime-site attribution.
PrivateClippy/Node4/markedselfchecks pass; Root247/nativeunchanged. Purefocused/
21shape/final-six-gate/Node/serving-resourcevalidation requiredbeforeintegration;
no memory/global superiority orcategoryadvancement.

## Fresh-session checkpoint and retained-memory priority (2026-10-16)

The E24 private `document_commit_lifetime_validation_v1/validation.json` now
records six successful Rust gates, 424 default/643 all-feature tests (one ignored
each), mutation11/snapshots2/legacy1 and rebuilt Node4 in both pure arms. Full
compile-input manifests contain 249 files: the original 247 plus two existing
COSE fixtures (658 B). Initial missing-fixture and package-export failures are
retained. Root production source/native remain unchanged by the lifetime
candidate. E24 is unsealed; its prepared serialization/serving scripts have not
run, and integration is not accepted. The plaintext transient saving does not
close the accepted 1.091961-times-SQLite serving-RSS loss; Lucene memory remains
unmeasured and memory/global superiority remain false.

User explicitly requests multiple subagents and calculations on both the local
machine and existing writerslogic-02; this supersedes the earlier delegation/
remote restriction. No agents or remote work have started at this checkpoint.
The next work must target retained postings, metadata, or compatible JS result
storage, with disjoint source/build ownership and exclusive paired measurements
per host. No further audit expansion or transient-only campaign is the priority.
Latest authoritative handoff is the leading 2026-10-16 section of
`docs/CONTINUATION.md`; older running-job and authorization statements are stale.


## Retained-memory implementation campaign (2026-10-16, active)

`retained_memory_candidates_v1/` freezes all249 current dirty compile inputs,
byte-identical to E24 origin-v2, plus current native/package before new edits.
Three disjoint private source/build agents implement adaptive retained postings,
compact private metadata and compatible JS result storage; production untouched.
Remote writerslogic-02 is WSL2 x86_64,22CPUs,16GiB RAM, Rust1.96 cached.
Its unrelated22-worker calculation currently saturates CPUs: use remote for
correctness calculations, no competitive measurements during contention.
User additionally requested keeping the host awake: dedicated Windows
ES_CONTINUOUS|ES_SYSTEM_REQUIRED thread is active (WindowsPID1916), verified
by powercfg SYSTEM request; no persistent power settings changed. Scripts and
receipts are in remote-awake/, with durable remote process/log paths in setup.json.
Current acceptedSQLiteRSS1.091961 loss remains; no new acceptance or superiority.


### Retained candidates: first implementation and screen (2026-10-16)

E25 `retained_memory_candidates_v1/`: safe adaptive4/6/8/native12B postings
passes28focused document tests and rebuiltNode4; standalone5boundary/state
tests pass ARM64 and writerslogic-02 x86_64. Private compact metadata passes
documents28/mutation11/Node4; remote4tests preserve parser/domain/f64/JSON/
bincode semantics and measure16B wrapper vsValue32B. Combined native changes
pass focused mutation11/snapshot2/legacy1 plain/encrypted checks. No integration.

`screening-v2` (directory spelled `screening-v2/`) retains all10cells/25,920full
operations with historical/cross-source equality. Two-round paired servingRSS
medians vs baseline: postings0.965390, metadata0.987532, JSv2 0.957565; vs full
SQLite1.052197/1.076331/1.043669 respectively. EVERY host-load gate fails;
results are descriptive, no accepted timing/superiority. JSv2 sourcep95/p99
ratios1.138237/1.136847 andQPS0.867309 retain an explicit query-speed loss;
native callback transfer is being improved before acceptance. JSv1 graph
7,510,208B vs sealedbaseline12,430,368B excludes cache/closure overhead; v2
removes151,168B extra result-array capacity, pending fresh graph measurement.
Rejected extra-promise JS wrapper and all predecessor sources/native/logs stay.

Initial screen-v1 immutable-progress checkpoint failure after2cells is retained;
corrected v2 reruns the entire protocol, no cell filtering/replacement. Cache
clones were stopped to avoid unrelated dSYM copying; original artifacts intact.
Next native retained candidate is adaptive interleaved packed-term offsets/
frequencies (existing arrays total3,634,496requestedB); savings remain hypothesis.
Final combined RSS/whole-byte/lifecycle/publication/sixgate acceptance pending.
Accepted productionSQLite servingRSS1.091961 loss, unmeasuredLucene memory and
no memory/global superiority remain unchanged; work stays on memory.


### Native terms candidate and JS transfer revisions (2026-10-16)

E25 pure terms candidate uses interleaved3/4/5/6/8B offset/frequency records
with fullu32 fallback, checked allocation sizes and unchanged serde maps.
Local27focused/Node4, remote6boundary/UTF8 checks,72JSON+72bincode pairs and
12parser controls pass. All4 shape allocation-loss counts0; raw deltas retained.
Measured layouts PackedTerms/TermData48→40B, StoredTerms8unchanged; an initial
56B assumption/layout assertion failure is preserved. Combined native posting/
metadata/terms passes documents35/mutation11/snapshot2/legacy1; finalgates pending.

Second screening-v3 retains12cells/31,104calls, all host-load gates fail.
TermsRSS/baseline0.977978; JSflatV3 0.932150, direct-per-hitV4 0.969789,
direct+metadatafastpathV5 0.976990. All remain aboveSQL individually.
Timing is highly variable (baselineQPS6327/5351); all timing results/losses
remain descriptive. Per-hitNAPI calls do not provide the best observedRSS.
V6 combines flat transfer with the tested safe metadata assignment fastpath,
exact output-array capacity and checked transfer allocation; no extra-promise
settlement, runtime/Node/NAPI requirement change, unsafe or dependency addition.
V6 and final combined serving remain unmeasured. Old JS variants/packages,
rejectedwrapper and every measured loss stay intact. Production remains frozen.

Qualified Lucene memory control is prepared/compiled using cached9.12.3/JDK17
and canonical actualHMS tokens; no index/serving measurement yet. Its fixed
16MiB initial/64MiB max heap and1activeprocessor precede results. Control
accounts simultaneous Node+JVMRSS; native f32 scoring/encodednorm alignment
and primary eligibility remain unestablished. No memory/global superiority.


### Combined retained candidate: RSS result and acceptance gaps (2026-10-16)

E25 combinedV6 passes all six Rust gates (435default/654allfeature, oneignored
per arm), Rust1.89 default-library and rebuiltNode7. Full combined-serving-v1
retains11paired rounds/33cells/85,536exact calls. ServingRSS/baseline median
0.878684; RSS/SQLite0.958282 with fourmetric familywise95 interval
[0.956131,0.961729], and every candidate/SQLite pair below1. Sourcep95/p99
ratios1.128783/1.171976 andQPS0.864084 are measured losses; every host-load
gate fails, so timing and final superiority acceptance remain unestablished.
V6 retained V8 result graph7,359,040B excludes cache/closure overhead.

JSv7 uses8rather than11 transfer fields when no optional source/text/semantic
fields exist; optional results retain the full path and golden cases. First
combined-v7 build was INVALID: copied source mtimes allowed stale pureJSv7
binary reuse (0.32s build). screening-v7 completed2rounds/8cells/20,736calls
with passing load gates but wrong candidate identity; all results, including
RSS and timing losses, remain retained and explicitly invalidated in
screening-v7/INVALIDATION.json. Forced byte-identical source recompile and a
fresh campaign are required; no result is silently replaced. Earlier V6
combined package built cold67s with distinct SHA and is unaffected.

Downstream serde_json/preserve_order feature unification may invalidate sorted
metadata-slice lookup assumptions. A separate metadata-v2 repro/fix is active;
this is a compatibility hypothesis until reproduced. No production integration.
Publication baseline probe compiled/frozen; candidate and24cell run remain
pending. Qualified Lucene fixed64MiB index now verifies3633documents/454312
postings/892967words and all DF/TTF/norm values; serving memory remains unmeasured.


### Corrected transfer screen and metadata defect (2026-10-16)

Corrected combinedV7 recompiled48.47s, native8f80039662a92a0dfbe413d5b594904d
b4d4c51803666606c895ceb22ffd1a42, with251unchanged source-content hashes and
Node7pass. screening-v7-corrected retains8cells/20,736complete calls: V7
RSS/baseline0.897179, RSS/SQLite0.978127 (pairs0.959772/0.996481), sourcep95
1.065776,p99 1.002455,QPS0.946500. V6 in the same campaign retains p95
1.149831,p99 1.188343,QPS0.858540. All load gates fail; these are descriptive
screens, not accepted timing. V7 is selected for final validation with metadata-v2.
The wrong-binary original screening-v7 remains invalid and fully retained.

The metadata-v1 downstream preserve_order defect is reproduced offline: exact
JSON/bincode order is retained, but binary-search lookup misses4/5insertion-ordered
keys, including u64/i64/f64 boundaries. Failing exit1/source/observations remain
in metadata-v2-evidence/v1-repro.json. Proposed correction retains sorted packed
objects but falls back to native map storage/iteration for unsorted objects;
common wrapper remains16B. Final compatibility checks and integration pending.


### Compatibility-fixed native source and final validation (2026-10-16)

Metadata-v2 preserves unsorted downstream maps through native fallback while
keeping sorted objects packed; default metadata5/documents29/mutation11 and
preserve_order metadata5/documents29 pass. Production-module proofs pass on
ARM64 and writerslogic-02 x86_64 with exact JSON/bincode/f64/order/full-domain
lookup. Wrapper remains16B; preserve_order Value measures72B on remote.
A receipt-copy collision lost local elapsed-time receipts only; unchanged six
successful complete logs support reconstructed command/status, with loss noted
in metadata-v2-evidence/receipt-collision.json. No source/test log was lost.

CombinedV8 freezes251inputs and passes all six Rust gates436default/655allfeature
(oneignored each), Rust1.89 defaultlib, and rebuiltNode7. V9 changes only the
node-api-cfg rawJS helper constant and Node regression tests: captured builtins
preserve searches after host Object/Array/Map/iterator methods are replaced.
Baseline succeeds but V7 rejects the concrete Object.keys replacement case;
that failure remains in javascript-v9-evidence/object-keys-observed.json.
Captures retain live Object.prototype descriptors and original result semantics.
Already-poisoned globals before construction are outside this mitigation; no
new cross-trust security claim. V9 finalmanifest2ff73c4c2281f81a92329429736baf6e
c6e5ff1ff28b191133eda58cc311ae4f, source251, is frozen; finalNode/Rust gates and
paired serving remain pending. acceptance-plan-v9.json prospectively requires
RSS/SQLite upper sevenmetric-family interval<1.0, source speed/memory10% guards,
full exactness, transaction/lifecycle/publication checks and retention of losses.

PublicationV8 default-Rust candidate compiles with real package rebuild and
Clippy; baseline binary reused by hash. V9 differs only in cfg-excluded JS,
so explicit equivalent-default-input receipts will bind this probe rather than
claim it was built from V9. Serialization uses the same explicit equivalence;
retainedNode diagnostics rebuild V9. OldV6/V8sources/probes stay unchanged.
Production249/native remain identical to the frozen baseline; no integration.


### Final V9 gates and prospective serving run (2026-10-16)

Final251-input V9 passes all sixRust gates436default/655allfeature, oneignored
each, Rust1.89 defaultlib; separate actual59.52s Node rebuild passesNode8 and
baseline4goldens. NativeSHA836db384443604191d75c7f0b07f25706ecd50d81a414fb4233
f8d307ecfb971 is independently copied/verified in combined-v9-evidence/package.
DiagnosticV9 retainedClippy/build/Node8 and4095/4096/4097 allocator boundaries
pass, with baseline/cfg-equivalent serialization reuse explicitly receipted.
All builds/probes stop before final uninstrumented serving. No production edit.

combined-v9-serving-v1 begins11prospectively alternating rounds/33cells with
quietwait bounded180s, no filtering/replacement/retry. A premeasurement amendment
counts sourceQPS as an eighth conservative family member (three other source
metrics,twoSQLmetrics,two unchanged historical quality checks), using bootstrap
orderindices15/4984 out of5000; prior sevenmember plan remains retained. RSS
upper must be<1.0 versusSQL, source latency/RSSupper<=1.10, sourceQPSlower>=1/1.10
andSQLQPSlower>=1.20. Every failed load gate and all metric losses remain.


### V9 measured loss and next allocation target (2026-10-16)

combined-v9-serving-v1 completes11rounds/33cells/85,536full exact calls after
180s bounded quietwait. All load gates fail. RSS/SQLite median0.978411,
eightfamily95[0.952794,1.001447] FAILS strictupper<1.0; sourcep95 1.075144
[1.026058,1.091648]passes10%guard, p99 1.164733[0.790852,1.222289]FAILS,
QPS0.927770[0.907508,0.981162]justFAILSlower1/1.10. Every loss remains;
no integration. Photos mediaanalysisd under currentUID consumes~1.2CPU for6days,
while other builds/measurements are stopped; no causal attribution of query/GC
loss follows from this observation. Host-resource intervention will be explicit
and prospective if used, with automatic restoration and no cell filtering.

V10 hypothesis targets redundant one-key metadata conversion: current helper
allocates a native metadata object, Object.keys array and fresh compact object.
A guarded scalar-only key/value transfer could remove the first two while
retaining fresh independent metadata. Complex values/prototype-accessor cases
must use legacyconversion to preserve sideeffect ordering. Sourceimplementation
and baseline semantics probes are authorized; no Cargo during diagnostics.
V9 serialization now confirms21complete transaction pairs byte-identical, all
allocation-counter losses retained; full24cell native lifecycle is running.


### Native retained reduction and durability confirmed (2026-10-16)

E25 diagnostics-v9 completes21transaction payload pairs and24plain/encrypted
lifecycle cells/62,208complete query calls; complete JSON/f64/frame sequences,
heldtwo-version snapshots and injected append-failure/retry proofs all pass.
Root independently compares all21payload files and snapshot/retry proofs;
root-verification.json records checks. All serialization allocation counters
are equal. All24native requested/usable pairs improve; no retained losses.
Plain recovered baseline21,374,138requested/23,005,840usable becomes
11,762,351/12,946,672 (requested−9,611,787 or44.97%, usable−10,059,168).
Acrossallmodes requested deltas−9,611,915…−9,609,234, usable−10,059,296…−10,059,168;
before/afterquerytotals stayequal. These are instrumented Rust System totals,
not C/V8/physicalRSS or a superiority claim. Rawcounts/proofs/losses remain.

V10 JavaScript transfer remains private source work. Native/default-code proof
may be reused only with explicit cfg-exclusion equivalence; later Node paths
were not exercised by V9 diagnostics. No production integration or category move.


### Guarded V10 metadata transfer built (2026-10-16)

combined-v10 freezes251inputs, manifest16b503a9d682e050aba22aab3b1a4b9c25ea8495
063ec5c7a879a0182eb2a9f2. AllsixRust gates436/655(oneignored), Rust1.89 defaultlib,
rebuiltNode9 andbaseline5goldens pass. NativeSHA74bfdbbd4d1bf277bb52fc1418ffdeed
3785ef9f5ba6f1f71087fa0b41f98245 is independently frozen/verified. Only guarded
Nodecode/test differsV9; native/default-source equivalence is explicit.

Scalar one-key metadata transfers key/value directly, creates fresh compact
objects, and avoids native metadata object+Object.keys allocation. Eligibility
requires allrows single primitive/null metadata/nooptional fields, exact u32
transport-count bounds and prototype preflight; everything else returns through
the ORIGINAL native Vec/DocumentResult converter. A real temporal setter probe
returns2native versus4naivebulk; fallback/goldens preserve2, readonly/__proto__/
numeric-prototype behavior and full f64/Unicode/NUL/mutation semantics. Initial
wrong id-setter expectation is retained; native result macro defines fields,
and the conservative eligibility guard still falls back for altered descriptors.
Already-poisoned globals before construction remain an explicit runtime limit.

acceptance-plan-v10.json retains strict eightmember-family guards. The complete
new11round campaign prospectively temporarily pauses only currentUID501 Photos
mediaanalysisd afterPID/path/start validation, restores with SIGCONT in finally
and detached720smaximum watchdog. quiet_host.py and source hashes/receipts make
this resource intervention explicit; allarms share it, no cell filtering/retry.
OldV9 noisy losses stay. FinalV10RSS/speed unmeasured at campaign start; allother
builds/probes stop. Fresh V10retained diagnosticClippy/Node9/allocatorbounds pass;
final24Node lifecycle and publication/Lucene serving await root release.


### V10 serving loss retained; bounded next implementations (2026-10-16)

combined-v10-serving-v1 completes11rounds/33cells/85,536exact calls. RSS/source
median0.927106, eightfamily[0.910138,0.930746]; RSS/SQLite1.013133
[1.009800,1.023003] FAILS strict superiority. Sourcep95 1.034142 upper1.047441,
p99 1.021764 upper1.089469, QPS0.957878 lower0.945434 fit numeric10%guards.
All-cell load qualification stillFAILS; decision.json records each guard and
eligiblecellcount, no acceptance. Photos resumes with verified sameidentity
SIGCONT/statR receipt; every prior noisy/intervened loss remains. Production
249inputs/native unchanged; accepted ratio stays1.091961. V10final24Node
lifecycle remains unrun because this candidate losesRSS; rebuilt diagnostic
Clippy/Node9/allocatorbounds evidence remains, no unnecessary rerun.

Four natural-GC diagnostic cells (10,368exact calls) compare frozen baseline/V10
without observe/forcedGC/heapcaps. Each has one Incremental Mark-Compact; baseline
has15Scavenges/cell versusV10 17, summedScavengepauses19.18/19.59ms versus
12.93/12.80ms, maximalpostScavengeheap29.6 versus22.4printedMB. Wholeprocess
trace is not servingRSS/latency qualification or proof of pretenuring/causality.
Initial majorcount parser omitted Incrementalprefix; unchanged rawlogs reparsed,
initialJSON/parser-revision retained, no runtime rerun.

Constructor-only V11 changes rawJS result allocation, preserving eight ownfields,
actualObject.prototype, eligibility/nativefallback/cache. Combined251 manifest
7ef40d0daa10577814ed4909d1e9110f9a89514aed71c2c6f6d17d8a28906b5a; onlylib.rs
differsV10, outsideNode-gatedrawJS everybyte isidentical. Source-only frozen;
no measuredsaving promised. New nativeimmutable-string pool targets allocations
before NAPI stringcreation, with bounded8192entries/1,048,576UTF16units, long
stringbypass, samequerypinning and abortedbatch reset. It is sourcework only;
header/map/cache/transition overhead must be measured. Lucene first2round
qualified memorycontrol is released exclusively, not a category claim.


### First qualified Lucene memory control measured (2026-10-16)

E25 lucene-memory/v10-screening-v1 completes2reversedrounds/4cells/10,368calls.
Same-sample servingaggregate peaks: V10HMS150,355,968/152,649,728B versus
LuceneNode+JVM236,863,488/238,010,368B; matchedratios0.634779/0.641357.
Lucenecomponents atthoseexactpeaks141,099,008+95,764,480 and
143,130,624+94,879,744B. Sharedphysicalpages canbedoublecounted; no subtracting
Nodeinputbaseline or summingseparatecomponentmaxima. Fixed16–64MiBheap, cached
Lucene9.12.3/JDK17, noforceGC/tuning/Photosintervention/filtering. Allfourload
gatesFAIL; allrawinputs/outputs/errors remain. HMScompletehistoricalJSON/f64
passes; Luceneeightownfields/Object.prototype/independentmetadata/repeatcontract
passes. Nativef32BM25/encodednorm alignment andstrongestprimaryeligibility are
UNESTABLISHED; qualifiedcontrolmeasurement establishesnocategorysuperiority.

Mutable serving.py addsV11/V12armnames and corrects the now-stale Luceneunmeasured
scope text only; exactpredecessordriver archivedbySHA under driver-history/.
No measurement parameters, bootstraps, adapters oracceptancebounds change.
An unrelated localPythonresearch job isrunning in /Volumes/A/researchpapers/
13-rule30/experiments/rule30/finite-seed-newton; preserveduntouched. Former
PhotosPID35969 hasexited naturally afterverifiedresume. Futuremeasurements must
recheckactualprocessidentity; do notreusePIDblindly orpauseunrelatedresearch.


### Constructor V11 rejected; native string pool boundary proof (2026-10-16)

V11actualNodefreshcompile nativeSHA0f27028ecaf020aeb59c7e6d1a475a890bb84f3
fe6df0ab7d6b9b5295b9e8932, rebuiltNode9/baseline5goldens pass. Source251 frozen,
root independently copies/verifies completepackage. screening-v11 two rounds/
sixcells/15,552exact calls reportsRSS/SQLite1.171390 (pairs1.166559/1.176221),
RSS/source0.915891, sourcep95upper1.119493 andQPSlower0.901631. Allloadgates
fail. Constructorhypothesis REJECTED, allsixrawcells/native/source/lossesstay;
no allsixRustgates wasted on losingJS-onlyscreen. V12 startsV10literal again.

NativeStringPool module sourceSHAf97dc0d583825221ed64a30c00bb96bf36182106b
1287690dc3229316c32117d passes8standaloneRust1.96tests onwriterslogic-02.
Remoteheader224B/Entry32B are measured representations, notcacheallocation
totals orpredictedRSSsaving. Tests coverUTF16/string/entry/budgetboundaries,
NUL/Unicode/full-literal fallback, samequerypinning and dirty/stale/rollover
recovery. JSintegration is sourcework only, no newruntimeacceptance.

Originalnativepublication24cell campaign isfinallyreleased using unchanged
90sboundedquietwait andfrozenbaseline/candidateprobes, explicitV10defaultcfg
equivalence; run-v10.log containsprogress, noPhotosintervention. Otherlocal
measurement/builds held; remote string calculationscontinue. Rootproduction
249inputs/native unchanged.


### Remote saved-output Lucene quality and frozen V12 integration (2026-10-16)

OnlysavedfourLucene/HMSoutputs+frozenqrels/existingqualityandbootstrapfunctions
werecalculated withuvPython3.13.9stdlib onwriterslogic-02; no retrieval/alignment
rerun. LucenenDCG@10 0.2902645573/recall@100 0.2268985959 versus
HMS0.2915822628/0.2274250051; meanLucene−HMS−0.0013177055/−0.0005264091,
existingdiagnosticsevenfamilylower−0.0037831126/−0.0013855998 pass−0.005.
ID/order/fullf64crossengineequalityfails descriptively; it isNOT auniversal
eligibilityrequirement (SQLite differsalso). Non-scorefields matchcommonIDs.
Qualitylosses29nDCG/9recallqueries andall324perquerydifferencesretained. Root
independently verifiesremoteanalysis/perqueryhashes; ready.json and
eligibility-clarification.json citeconcreteprospectivelimits:2vs11rounds,all4
loadgatesfail, notuningfrontier/equaltuning/primaryHolmfamilyqualification.
No primary/categorypromotionfromthisdiagnostic.

CombinedV12source252 frozen manifest661f9bd1774cdb8422ca41a9400708da72ef3dd
1780bd74989862cf627261813. NativepoolfinalSPDXsourcebdf2d36e6ef067ed0817b138
94f42486f8616a5cdf43f09f4890de98001eaa9c differspriorf97onlybylicenseheader;
bothexactremote8testsuccesslogs andoldsource remain. Nodeintegrationscopes
Mutexonlypreparation/commit, releasesbeforeJS; callbackappliesreset/clears/slot
updates thenresolvesimmutableprimitivesintosameeightordinaryownfields. Same
V10eligibility/nativefallback/resultliteral/Promisecompletion; no dependency/
NAPIlevelincrease/unsafe. NewNode10source addsoversizedpinnedbatch/full-literal
metadata/u64/retry/independentenginechecks; notyetbuiltormeasured. Unintended
default-testMuteximportcaughtandcorrectedbeforefreeze; initialdiffrejectedstays.

SourceproofusedPythonSequenceMatcher forestimated10–20sCPUduringpublication;
exactinterval/cellIDswerenotrecordedandarenotinvented. Thisisadditionalpotential
timingcontamination, separatelynoted inpublication-static-cpu-overlap.json;
allloadfailalreadydisqualifiesclean timingacceptance. Stoppedexpensiveproof,
usedlinearequality; no automaticrerun/filteringorlossdeletion.


### Publication campaign completes with preserved losses (2026-10-16)

E25 publication/final-v8 full24cell campaign/analysis/adversarialoracle8 exit0;
115,200 offered/accepted, zerodrops,3,840operations,310,920restart and310,920
crossarm exactresultpairs pass. Root verifiescomplete measurements/analysis
hashes andcounts. All24loadgatesFAIL, 90squietboundexhausted; estimated10–20s
staticCPUoverlap hasunknowncells/times andisexplicitlyretained. No clean timing
preservationclaim. Exclusion/service/scheduledresponse groupmedianp99 ratios:
5182batch2=1.1432/1.2811/0.8035;5182batch64=1.3535/1.2209/1.1386;
20728batch2=0.8169/0.8562/0.9911;20728batch64=0.8714/1.1767/1.4924.
NumericalandallpairpreservationguardsFAIL. Everyp50/p99/max/queue/wait/drop/
maintenanceandRSSloss remains. ServingmedianRSSratios0.5574/1.0304/2.1680/
1.0868 (fiveoftwelvepairedmediansregress); staticmedian0.8097/0.9574/1.0336/
1.0726. Corpusnativeallocationreduction isnotuniversalservingRSScoverage.
Source/probe guardsrecheck; lockreleased; productionintegration remainsblocked
byunmetacceptance, notpermission.

V12Nodeactualbuildstartedafterpublicationlockrelease. FirstnewNode10test fails
on unsupportedBigIntinput; baselineproof/testingcorrectioninprogress, oldtest/
package/failure retained. Rustsource remainsunchanged; do not rerunfullcompile
fortestonlycorrection. V13nativecompact-queryprojection isprivate sourcework
only, withdefault/publicRustAPI andinstalledmodelpath preserved; anycoreinput
change requiresfreshpublication/serializationacceptance, notV10cfgequivalence.


### Native string-pool V12 built and measured loss retained (2026-10-16)

FinalV12manifest252 SHA5769ef2de08022dc49630bab05ca2ab8a183c28ebfd212a2
baea1e91e952d8a2; Rustlib/module unchangedfromactual29.42s compile. NativeSHA
19ca44e7403974df620c2a7dd422c96e135f16690064c80c611228a1377248b3. Node10,
baseline6goldens, exactpool8localRust1.96tests allPASS. InitialnewBigIntinput
assertionfailedidenticallyonbaseline; attempt1source/package/logremain, corrected
testassertsoriginalinputrejectionthenMAX_SAFE_INTEGER/u64maxstring. Test-only
compile-reusereceipt provesallRustinputsidentical; rootverifiespackage+manifest.

screening-v12 completes2reversedrounds/6cells/15,552exactcalls; RSS/SQLite
median1.024842 (pairs1.025201/1.024483), RSS/source0.947884. Sourcep95
median1.072984 range0.964266–1.181703, p99 1.311848 range0.962063–1.661633,
QPS0.989932 range0.921908–1.057955. AllloadgatesFAIL. Noacceptance/full6gates/
24lifecyclecampaign for losingRSSscreen; everyloss remains. Boundedpoolactual
capacity/plateau diagnostic source preparedbutunrun.

Newhypothesis only: allnumericflattransportmaycreateDOUBLEelements andbox
integeroffsets/cachetokens; one-nullbatchsentinelcouldkeepelementsgeneric.
ActualNode--trace-elements-transitions onint/stringmetadataqueries shows only
JSslotcacheHOLEY_SMI→HOLEY_ELEMENTS, noDOUBLEtrace; hypothesisUNCONFIRMED, no
savingclaim orsentinelimplementationyet. DirectInspector/intrinsic bounded
diagnosticwillinspectactualcallbackfields beforeedit. V13compactqueryprojection
andMap::new+insert insteadofcollectscratchareprivate sourcealternatives.

### V12 array hypothesis disproved; additional private implementations measured (2026-10-16)

E25 `transport_array_kind_v12_v1/proof.json` inspects the actual callback fields
at an Inspector breakpoint, with intrinsic golden controls. Both integer and
string metadata batches have object=true/double=false/smi=false, length9;
complete output equals the uninstrumented result. The DOUBLE-elements hypothesis
is DISPROVED. No sentinel implementation or saving claim follows. The earlier
trace-only diagnostic and unconfirmed hypothesis remain as historical evidence.

`metadata-insert-v13/` changes only the compact one-key to_value path to
Map::new+insert, avoiding collect's scratch allocation; other domains are
unchanged. Manifest252 SHA48a3e69fe4a43d66e5579533be09f63bb28b26185beab455052b29de024b4b8e;
actual native7b29989eb94ba88fc5c1ed5b6343a875cefae39c3b947dcc2c85ae21c7242e0d.
Focused metadata6/documents30 default and preserve_order, Node/security Clippy,
Node10/baseline6 pass. Remote exact module22 cases each default/preserve_order
cover JSON, both bincode endiannesses, recursive f64, order and mutation.
`screening-metadata-insert-v13` has6 cells/15,552 exact calls; RSS/SQLite
1.019722 (pairs0.975108/1.064337), RSS/source0.880537, sourcep99 upper1.168298.
Every load gate fails. Parent V12 was not measured in this run, so no isolated
one-key speed or RSS attribution. No acceptance or full six-gate campaign.

`combined-v13/` implements compact final-ranked result projection, avoiding a
full serde metadata Map in the native Node path. Ranking/capture are unchanged;
the public Rust and installed-model paths preserve the original results. Exact
fallback covers the entire metadata domain. Native source authority
`native-projection-v13/`, remote actual crate4 tests cover JSON/bincode/f64,
dense128 rows, held snapshots/removals and durable restart. Final253 manifest
98a02c04c8a128f2f77187f0591c0c08e7ad1259847d9441874b90e96027f7d8;
actual native00f141c69aa44ae515842e50e07f61f8ca932731e104070b41de70a444e2b4e3,
Node10/baseline6 pass. `screening-v13`8 cells/20,736 exact calls includes V12:
V13 RSS/SQLite1.027090, RSS/source0.940094, sourcep95/p99/QPS
0.943254/0.922266/1.094415. V13/V12 paired RSS1.003838/0.996887 and
QPS1.027430/1.022188, descriptive two-pair improvement in throughput, no RSS
gain. All load gates fail. Core changes require fresh publication/serialization
acceptance; old default-cfg equivalence cannot substitute.

`combined-v15/` removes V13's native immutable-string pool and restores the
byte-identical V10 JS string Map protocol, retaining compact projection. The
pool source remains in the bounded manifest but is undeclared/uncompiled.
Manifest253 SHA4712ad5a7f0550c172496436660787ba4bb4253b6d2f72f48460b6b53ec9a349;
native3258cf4db6ed848b421c47cae541dd841eef1a83650ce972cdab75f02a919b76,
Node10/baseline6 pass. `screening-v15`8 cells/20,736 exact calls includes V13:
V15 RSS/SQLite1.087574 versus V13 1.103010 in this same run. V15/V13 paired
RSS0.985279/0.986707, QPS0.902873/0.898439, p99 1.146010/1.113580: small
RSS reduction costs throughput and tail latency. All load gates fail; retain
every loss, no integration or full six gates for this losing screen.

`combined-v14/` instead holds Arc<StoredDocument> plus chunk index/scores, borrows
IDs/version/metadata, and interns borrowed strings without native cache-hit
String allocation. Rows/Arcs are consumed before NAPI writes/JS calls. Whole
old-document Arcs nevertheless live through blocking return to callback prep;
mutation/retry/snapshot peaks may increase and remain unmeasured for V14.
Public fallback reuses the original materializer. Manifest253
c92252560e9fbe44a23be23b733c05210e3f9de3929d2ecab713b8d1fd9a221b;
native8c8c5a92f2de1fccf30bdbcc51764d5c422bc0f2a79396632517f622df0aeeed.
Actual remote integrated core4 and borrowed-pool10 tests pass; Node10/baseline6
pass after fresh local compile. `screening-v14`8 cells/20,736 exact calls:
V14 RSS/SQLite0.982891 (0.941595/1.024186), sourceRSS0.897763,
sourcep95/p99/QPS0.996438/1.310203/1.047671. V14/V13 paired RSS
1.067895/1.014452, QPS1.124633/0.992681 and p99 0.731726/1.462208.
V13 RSS/SQLite0.943625 in this run versus1.103010 in the previous run;
cross-campaign variance is material, not evidence that source changed.
All load gates fail. None of these two-round screens establishes superiority.

### V16 secure posting-map implementation built; V17 transport reuse frozen (2026-10-16)

`native-posting-map-v16/` starts from V13 and replaces the two retained posting
BTreeMaps with RandomState HashMaps<Box<str>,PostingList>. Randomized hashing
preserves collision resistance; query/document traversal and sorted posting
payloads remain ordered separately. Existing-key inserts avoid key allocation;
mutation-boundary shrink uses checked quarter-load arithmetic and preserves
hash builders. Rehash/high-water/header/bucket/key costs require measurement.
Manifest253 SHA4861f7ed325f26373066874647e9cbf339567a40ea8be84f8d6995b3161197c7;
native8f79692a6e96cc28ac11d7dd0b9abcb33731e75e10af79f6b9d5684da7d9f5c1.
Actual default/preserve_order documents38 each, default all-target Clippy,
Node/security Clippy, Node10/baseline6 pass. Remote extracted actual helper7
tests cover Unicode/NUL,4097 keys, full usize/u32 and quarter-load boundaries;
these extracted checks do not replace full crate gates. Root independently
verifies all253 source and9 package hashes. No serving result/full six-gate or
final lifecycle acceptance yet. Three-arm allocator diagnostic is SOURCE ONLY.

`combined-v17/` freezes254 inputs, manifest
3651db163fe0bf68cd5a111377ba8aec05b753d021ea919dbd2a7063d51dd366, lib
d1aa23eb42ed5d65e7577276f9562bc9e899f05c9a7a198b2ad1c019ae776cf5.
Core/pool equal V14; only Node transport and tests change. A private ordinary
JSArray reuses at most9000 primitive transport fields; all active fields clear
after success/JS throw, or on next acquisition after NAPI failure. Literal/string
or unsafe-integer metadata retains the exact original Vec converter. Pool lock
and held snapshot Arcs end before NAPI/JS. No unsafe/NAPI/runtime requirement
increase. Buffer backing, boxed numbers, high-water and helper overhead remain
unmeasured; no footprint claim. Root reviews diff and verifies254 source hashes;
exclusive local build/Node11/baseline7/factory7/kind proof now released to JS
agent. Numeric-domain Rust boundary regression is new and pending. V18 numeric
transfer module is separate SOURCE ONLY; safe pinned typed-array API viability
is still being checked before integration. No concurrent local serving campaign.

Root production249 inputs/native remain unchanged; no candidate is accepted or
integrated. Accepted same-output serving RSS/SQLite remains1.091961. Native
~45% recovered requested reduction is measured, serving and publication
preservation guards remain unmet. Memory/global superiority remain unestablished.

### V17 checks pass; V16 deletion-retention defect preserved (2026-10-16)

V17 actual native d23ced344493c89b30c7ce11d751cf6841d609ce115a3cf2bdb428fb9a32ad82.
Node11/baseline7/factory7 and new actual Rust numeric-domain boundary test1 pass.
Actual Inspector/intrinsic check sees generic object elements, double=false,
reused=true for numeric metadata; String fallback generic/reused=false. Complete
outputs equal prior oracle. Root verifies package hashes and freezes a separate
serving copy in `combined-v17-evidence/package/`. Full six gates remain separate.

Remote clustered-key probe establishes a real V16 retention defect: a table
formerly capacity56 reports public capacity1 with one live entry, then0 empty;
the quarter-load threshold skips shrink. Subsequent reserve(1) revives capacity56
versus3 in a fresh map. Tombstones conceal retained backing. Exact raw source/log
are `native-posting-map-v16-evidence/remote-clustered-deletion*`; no private std
bucket-layout inference. Frozen V16 is ineligible for integration; preserve its
implementation and every result. Separate SOURCE ONLY V19 will track an upper
capacity bound and rebuild/shrink with the existing cloned RandomState, including
empty reset, deletion/reinsert and threshold boundaries. Extra header and
transient table/rehash/publication costs remain to be measured.

Prospective `acceptance-plan-v16-v17-screen.json` freezes two reversed rounds,
12 cells/31,104 calls, baseline/V13/V14/V16/V17/SQLite, bounded300s quiet wait and
unchanged load<3/metric/exactness guards. V16 fixture has no deletion, so its
loaded-query measurements are useful but do not erase its known defect. No
concurrent local build/measurement. V18/V19 source-only work proceeds separately.
Do not filter/retry failed load cells or claim superiority from a two-round
screen. writerslogic-02 sleep inhibitor PID1916 and active SYSTEM request
reverified again in `remote-awake/recheck-v17.log`; leave it running.

### V17 closes the measured RSS gap; acceptance remains incomplete (2026-10-16)

`screening-v16-v17` completes12 cells/31,104 exact calls after a146.75s quiet
wait found one-minute load2.94385. Only6/12 subsequent cell load gates pass.
V13/V14/V16 RSS/SQLite medians1.130316/1.129843/1.130606. V16/V13 paired
RSS1.001512/0.999095, QPS1.006953/1.033286; no retained-dictionary RSS gain.
V17 RSS/SQLite0.934688 and RSS/source0.782651; sourceQPS ratios
1.147525/1.163650, p95 interval0.893990–0.895938, p99 0.882529–0.990296.
V17/V14 RSS0.825483/0.828934 and QPS1.066521/1.042270. Independent root
source/package hashes and completed output checks saved in root-verification.json.
The reusable primitive buffer is the first new strong serving-memory candidate
in this implementation sequence; the two-round screen alone is not acceptance.

Prospective `acceptance-plan-v17-confirmation.json` fixes unchanged guards,
baseline/V17/SQLite,11 reversed rounds/33 cells/85,536 complete exact calls,
eight-metric intervals, bounded300s quiet wait, no source/package/workload changes
and no concurrent local builds/measurements. Completed `combined-v17-serving-v1/`:
RSS/SQLite median0.938679, family8 interval[0.848729,0.982004], strictly below1;
RSS/source0.788392, family8 upper0.839945. Source p95 median0.885075 upper0.951179,
QPS median1.191207 lower1.119166 and QPS/SQLite9.788773 lower9.413390 pass
numeric guards. Source p99 median0.942150 but upper1.206919 FAILS1.10; only2/33
load cells pass. Thus the numerical RSS deficit is closed in this bounded run,
but source tail preservation/host qualification remain unmet. No integration,
accepted/category/global superiority. Every paired loss stays in analysis.json.
Root decision.json hashes measurements434a259689b8f5b32b682ae8d9d3edcaed598bfdba58df4df2ba21614756de32
and analysis132fad78adb1bd3a59baf661ff3b36685edae76e4f3954e0b6df015b9d8b19f4.

Selected strongest V17 copied byte-exact254 inputs to `combined-v17-final/`;
focused mutation/snapshot/legacy/documents checks then six Rust gates now run
exclusively in `combined-v17-final-evidence/`, driver logs at E25 root. Node11,
baseline7, factory7, numeric-domain1 already passed on its actual fresh native.
Fresh V17 publication sources/protocol are prepared in `publication/final-v17/`,
not built/run yet; baseline249 source/probe hashes reused exactly, candidate
default-core source compiled afresh before measurement. Prior final-v8 binary/
losses remain. Quiet wait300s is prospective; all acceptance guards unchanged.
V17 allocation/lifecycle diagnostics source preparation is delegated separately.
V18 typed-transfer backup and V19 posting-map shrink correction remain SOURCE
ONLY; no local builds during the selected V17 gates or future timing window.

Separate V19 shrink correction is frozen SOURCE ONLY at
`native-posting-map-v19-evidence/source-ready.json`,253-input manifest
d438334e84d601d4572d45378197453f5b97ec31fea8bb0dcb06044af76fcef0,
indexe43f8e330405caf1a8cd32c4d042d398b6a6071e5553ce5e8578228e29d28e44.
Only index.rs differs V16. Wrapper tracks a conservative capacity bound and
rebuilds using the existing cloned RandomState; empty reset drops old backing.
Remote9 extracted actual helper tests pass, including clustered tombstones and
deletion/reinsert quarter-load boundaries. Header/old+new table overlap/rehash
and publication performance remain unmeasured; no serving superiority claim.

V18 typed-transfer backup frozen256 inputs, manifest
caa11b306864adefe6a1b88e4190b9e19f16e48f75b1cbc838bfab519f13fda5.
Exact core/pool equal V17; only Node transfer/module/tests change. Remote std
module6+pool10 tests pass; new actual-factory8 tests are source-only/unrun.
No integrated Cargo/Node/serving run. V20 SOURCE ONLY fallback now prepares V13
owned compact results with the byte-identical V17 primitive-buffer factory.
This avoids whole-document Arc retention through pending Node callbacks, whose
V17 resource cost is still a hypothesis pending the new lifecycle diagnostic.
No assumption that the fallback is faster or smaller; measure before adoption.

V17 final six gates now PASS: fmt, default/all-feature all-target Clippy with
warnings denied, locked default436/all-feature670 tests (one ignored each),
Rust1.89 default-library MSRV. Focused documents/mutation/snapshots/legacy gates
also pass. `combined-v17-final-evidence/root-verification.json` independently
checks six exits/counts, all254 final inputs, original root249/native unchanged,
and binds existing rebuilt Node11/baseline7/factory7/numeric-domain1 evidence.
No code edits after gates; no acceptance while sourcep99/load and new lifecycle/
publication resource qualification remain incomplete. JS agent now has exclusive
short local V20 build/check window; V17 diagnostics source preparation continues,
then its build window. V18 remains unbuilt; publication V17 wrappers prepared
but builds/measurements held until selected-source/pending-Arc decision.

V20 owned-buffer fallback now actual rebuilt24.83s, native
75ac181180d0463dee16a660aab9942ad508e8c030c99d4ce8dae96c36357be8. Node11,
baseline7, actual-factory7 and numeric-domain Rust1 pass. Frozen254 inputs and
complete package hashes independently verified; source5f148317aee4e6f9438ae347d33e24acc50c66cfc2a1700ec06a7ccdd140e14b.
Initial post-test reporter expected TAP while Node26 emitted spec output; parser
failure/initial driver/logs retained. Receipt corrected from unchanged complete
passing logs/source/native, no redundant build/test rerun. Exact owned V13 core4
and pool8 predecessor proof is separate from V17 borrowed-pool10. No V20 six-gate
or publication acceptance. Prospective `acceptance-plan-owned-buffer-v20-screen.json`
freezes baseline/V17/V20/SQLite,2 rounds/8 cells/20,736 calls, unchanged guards,
300s bounded quiet wait and all losses; root runs `screening-owned-buffer-v20`
while matched pending-ownership diagnostic source preparation finishes. No
concurrent local builds. V18 remains unbuilt/unrun.

V20 ownership screen completes8 cells/20,736 exact calls,6/8 load gates pass.
V17 RSS/SQLite0.935908, sourceRSS0.769768/QPS1.145796; V20 RSS/SQLite0.969007,
sourceRSS0.797167/QPS1.124698. V20 sourcep95 interval0.905619–0.954207,
p99 0.981033–0.991352. V20/V17 paired RSS1.048519/1.022353 (both losses),
QPS1.006322/0.956113, p99 0.976874/1.031447. Thus owned results avoid the
whole-document lifetime extension at a measured2.2–4.9% RSS cost in this screen;
both candidates remain numerically below SQLite, no two-round acceptance.
Root verification retains complete measurements/analysis hashes and matched
parent ratios. Next active job diagnostics-v17 build-only session25315; fresh
retained baseline, candidate and candidate serializer, unchanged exact baseline
serializer reuse. No lifecycle or publication runtime until all builds stop.

diagnostics-v17 build session25315 completes0; three fresh Clippy/builds,
retained Node controls and allocator4095/4096/4097 boundaries pass. Exact
baseline serializer reuse receipt retained. `builds-ready.json` binds retained
baselinea15ee4a567ac4003357aaea39ba744734ced43f755a5fe30bdb8d57d84df7e6c,
candidateef90c7c338859f1b2f446823b4f625b65fe1654441c884b4e836c500da08d60e,
candidate serializer ec5b39fd3b34e95cdb37931470aedf6db3249d61783bfd06aabb7ea1fa663b68.
Source-ready manifest preserves baseline249/V17254 authority and full helper
hashes. Metadata agent now holds sole local diagnostic runtime:21 complete-byte
pairs then lifecycle-v1/24 cells with matched pending ownership and124,800 query
calls. No concurrent builds/publication or automatic retry; inspect actual
completion before claiming any new lifecycle/resource acceptance.

### Selected V17 durability/resource proof completes; final fixed-spacing policy (2026-10-16)

diagnostics-v17 serialization21 complete-byte pairs and lifecycle24 cells/
124,800 calls finish0, no failures/retries. Full JSON/f64, plain/encrypted
recovery, entire decrypted frames, append failure/retry, two held versions,
metadata independence and compaction pass. Root independently compares all21
raw .bin pairs, complete lengths and hashes, and verifies analysis/receipt hashes
in root-verification.json. All24 native requested/usable pairs improve:
requested savings8,879,311–9,611,136B, usable9,264,128–10,058,992B. After-query
pool overhead reduces the earlier ~45% cold native reduction; it is measured,
not omitted. Warmup adds729,144 requested/794,672 usableB; twelve after-query
pairs have3,544 additional live blocks despite lower bytes. Actual native pool,
headers/map/slots/Arc counts and cleared JS transport plateau at passes1/8/16;
wrapper/pending-owner drop proofs pass.

Plain/encrypted matched100-hit pending snapshots both expose a component loss:
baseline pending results release86,092 requested/92,480 usableB; candidate whole
old documents release134,071/151,408B after all other old owners drop, extra
47,979 requested/58,928 usableB and400 blocks. Both preserve exact results and
reclaim correctly. This is a measured one-chunk corpus case, not a universal
bound for multi-chunk documents or unbounded queued callbacks. The owned V20
alternative avoids whole-document retention at the separately measured2.2–4.9%
serving RSS cost. All losses remain; select V17 for bounded final publication
and serving qualification, no universal peak claim or integration yet.

Final measurements prospectively add a fixed20s pause BEFORE EVERY CELL,
including first. Per-cell workload, inputs, query/arrival counts, API metric
timing, fresh seeds, exactness, all metric/drop/load<3 guards are unchanged.
Reduce own preceding-cell/seeding carryover is a hypothesis, not established
causality or promised qualification. No background jobs paused, failed cells
filtered/replaced/retried, or old losses removed. serving.py archives every
prior driver byte-for-byte; acceptance-plan-v17-spaced-confirmation.json fixes
11 rounds/33 pauses/85,536 calls, not run yet. Publication final-v17 preserves
all five prior unrun scheduling artifacts/hashes in scheduling-before-fixed-pause/;
new protocolSHAf7e41325704935b7071c884458465568d53b71ff6e0ee5b8e5ef06e730a69202,
campaign42e96d343639ef48502ea5177dc31187bc7788dd28bd1d6dba4f9276c3ba87d8,
24 pauses recorded/verified. Root releases exclusive BUILD ONLY, session59237;
no measurements while compiling. Separate MEASURE release follows verified
fresh candidate default-core probe. V21 six-line one-key Map materializer is
SOURCE ONLY backup fbd253fa2d83f58e8e0363c55fc4e719bd597789944a830f387525a117e7989d;
all253 other V17 inputs identical. Do not claim a V21 Node RSS gain; it targets
default Rust to_value overhead only if publication evidence warrants use.

Publication freshV17 default-core BUILD exits0 in39.66s, Clippy-Dwarnings pass;
candidateprobef43d0a6a1d865afd869543d05baec0b6c3274a42db44b0539e599c6d2cc68c70
is distinct from retainedV8, baselineprobeb3cdf07e01a7763321454a15030471b390d3d7c52f3f6b9f272185384f772ad6 unchanged.
Root verifies binaries and build-completionSHA9a08d0dfc1566ba890e300a2a57283ab3de4f45d6d99a32bf209612c0f105a0d,
writes separate MEASURE release. Postings agent now owns exclusive local24cell/
24fixed20s-pause publication campaign; final V17 serving spaced confirmation
waits until it finishes. No concurrent builds or competing measurements.

Publication fixed-pause run39245 FAILS after first baseline5182/batch2 cell,
whose oracle passes and load fails. Initial quiet300.001s never finds load<3
(minimum3.63086). Logging error: existing compaction_updates.write uses open('x');
second aggregate pause write raises FileExistsError BEFORE candidate measurement.
Original driver/protocol/campaign/releases/log/firstpause20.010s/progress and14
completed-baseline files remain exact in logging-failure-before-resume/;
logging-failure.json records hashes. Interrupted candidate's extra pre-pause
duration was not saved and remains explicitly unrecorded, never inferred.

Root reviews separate immutable resume_driver.py/resume_campaign.py/protocol:
carry original baseline once, verify/reuse original5182seeds, run remaining23
cells in original order once, create20728seeds once; unique exclusive per-cell
pause files, aggregate only after completion. Preserve original failedquiet,
no retries/filtering/replacement, full_campaign_timing_qualified:false for the
interruption and failed first load gate. New resume protocolSHA
e58dfc77c9d0f7d625aff0ab221fc99bca8fc1795befc712cd0f2354abb4c964,
driverf13b8cf75e8c635e8215a131800ea5051fe8dca2f3988c55749316b514faac61,
campaign5db5aaab96a2ce26f90a4582211c5701d09808c9b0eb9cae06b514becd4eb071.
Root writes separate resume release; postings agent runtime10314 now resumes23
under exclusive global lock, run-resume-driver.log. Rust/inputs/metric/drop/load
guards unchanged, no new builds. Complete numerical/oracle evidence still useful;
this interrupted campaign cannot establish full timing preservation.

Root fixes the analogous UNRUN serving pause logger before its first execution,
after reading actual exclusive write helper: unique per-cell pause filenames,
one final aggregate. Old source/plan bytes archived by hash and prospective
acceptance-plan-v17-spaced-confirmation updated; syntax checked. No serving cells
were lost or rerun. Lucene driver source-only adds the same immutable receipt
policy; planSHAf80fafef93bacf478e11f5f49d8175a65dff0ec62c9015038982c1e5da432ffd,
measure.py76ca4d8d1697b561804bad3f7f4986b6a644c29771aac642bc402552a1c60221,
11round/22fixed20spauses. Cached settings/score/workload/guards unchanged,
quality reuse conditional on exact saved V17 JSON/f64; primary/tuning/Holm gaps
remain explicit. No Lucene runtime while publication is active.

Additional publication timing limitation: root fixture-inspection command assumed
dict but input was a list, printed list(a) (tool reports3,095,902 original output
tokens/11,335,030 omitted bytes), then failed get(). Reported command wall0.4s
rounded; exact absolute interval/cell overlap was not captured. Event order lies
between20728seeding-start andfirst20728baseline-active agent reports; do not infer
that it was outside measured cells. source-inspection-overlap.json preserves
this additional potential CPU/IO/terminal contamination. Campaign was already
load-failed/interrupted and remains unqualified; no losses/cells removed. Further
fixture inspection deferred until runtime finishes; print only schema-checked
bounded fields. Prior full V17 serving/durability measurements predate this event.

### Final V17 publication losses retained; spaced serving active (2026-10-16)

Publication final-v17 finishes24 cells,115,200/115,200 accepted exact queries,
zero drops,3,840 mutation operations,310,920 restart and310,920 cross-arm exact
pairs; adversarial oracle8 pass. All24 load gates fail. C/B p99 exclusion/service/
scheduled-response ratios:5182/b2 .906343/.856717/6.751590;5182/b64
1.050700/1.203513/.313849;20728/b2 .918811/.865741/4.297566;20728/b64
1.491164/1.181277/3.152092. Numerical and paired preservation guards fail.
Every latency/RSS loss remains; serving median RSS ratios include1.325861 at
20728/b2. Twenty-four known pauses total480.193761s; interrupted extra pause
remains unrecorded. Logging interruption and root source-inspection overlap
prevent clean timing attribution. No acceptance or causal claim follows.
Root verifies completion-with-limitations SHA5734c8baa4689d277fa14b4a8d815745763c14b8008f09588faf24a2dc92faff
and summary-final SHA05a9fd279102dc98f100f51078293f7aee453f2fd702d71fba89eb4aca6d6e85;
measurements3f64c4a25de94ab12677a5184d7b6309376b4788cb231990c924ebdef2f8a393,
analysis90a6fa2d2fa3e224b95033c5f65b0a1cbacc3a7e2a1f92c0b5ea2c58956861b7.

After publication processes stop, root starts the already prospective
combined-v17-serving-spaced-v1 full11-round/33-cell confirmation, session2502,
combined-v17-serving-spaced-driver.log, sole local runtime under timing.lock.
Fixed20s before every cell and bounded300s quiet; workload/metric/load/eight
guards unchanged, no filtering/retry. Metadata agent now implements V23 small
map materialization insertion with remote exact/allocation proof; JS agent owns
PostingList hot-path implementation. Local builds remain held. Proposed V22
borrowed serialization is already in V17, so no edits/runtime were performed;
combined-v22-evidence/premise-check.json records the rejected premise.

V23 private source metadata-small-insert-v23 now freezes254 inputs,
manifestcd9cd2b70f28a3043f00f0f5738e2d270fab5155a4958001f648dcf4938894fa;
only metadata.rs differs V17 (e54c07200a80824614258f6c23ecd7ed67d57e6962b613b3ca65442c8312ac40).
For packed maps with<=11 keys, to_value uses ordered insertion with capacity;
larger maps/Other and serialization stay unchanged. Remote exact production
module proof passes31 cases each default/preserve_order: full JSON, both bincode
endiannesses, f64, order and independent mutations. Default four-key primitive
and nested cases remove one allocation/224 requested+peak scratch bytes, retained
bytes equal; ten/eleven keys remove560/616 scratch bytes,>=12 fallback equal.
Preserve_order allocation counters are equal. Raw logs and initial saved-log
parser failure remain; extraction repaired without rerunning tests/builds.
source-ready.json and remote-allocation-analysis.json bind the results. The actual
publication reader service includes metadata.to_value for64 four-key hits before
JSON output serialization, so this optimization hits the measured path; latency
benefit remains unmeasured. V24 index-only candidate remote checks continue.

Fresh remote inspection finds writerslogic-02 idle (load~.09),22x86_64 CPUs,
~14.8GB available,855GB free; prior unrelated SAT contention has ended. No jobs
were paused. Windows PID1916 still has an active SYSTEM sleep inhibitor, no
powercfg overrides (remote-awake/recheck-serving-spaced.log). Node24.21.0 cached
but exact26.8.1 absent from bounded caches; official Node26.8.1 Linux archive
was retrieved and SHA3e301118d7df53d563b7e96c1617545f26e2f76f9724be668d6cab65c15dda5d
matched nodejs.org SHASUMS256.txt. Python3.13.9 already cached. Postings agent
builds complete baseline249/V17254 Linux node-api/security capsules, session3777,
jobs2, absolute guarded manifests/disjoint targets, actual fresh compile required.
No remote competitive measurement until all remote builds stop and root releases
the prospective selected arm (possibly V25 if ready). Local serving2502 remains
exclusive on its host. Same-host exact JSON/f64 and exact runtime SQLite version
are mandatory before treating remote paired resource results as eligible.

V24 remote actual-module proof completes posting6/default and documents41/node-api,
production index SHA5d211ba36ab1e61087ff82e629b69ce4dbf61b6ead4b287b20bde94b599c0818.
Test-only actual-helper counters:20728 Small frequency updates avoid20728 packed
entry reconstructions; ONE shrink boundary capacity32768>4*8191 avoids8191
width-scan decodes. This is logical operation counting, not time/allocation gain;
ordinary removals below the shrink threshold still perform search/Vec movement.
Full sorted outputs, domains and capacities match. Initial missing-helper counter
harness compilation failure retained; repaired only derived harness, no repeated
passed production-module tests. All remote V24 jobs stop.

Root composes private combined-v25/254 exact inputs from V17, replacing ONLY
metadata.rs/index.rs;252 other inputs verify unchanged. Final manifest
168493d18cf19972cb1ba358c6371fc84f37113aa31b4b102b219c910b584388 and source-ready.json
bind both parent manifests. No production changes or measured V25 speed claim.
Node driver combined-v25-node-evidence/build.py and focused+six gate driver
combined-v25-validation/validate.py --stage focused-gates are SOURCE ONLY,
awaiting local2502 completion. Prior unrun combined gate/Node driver preserved
when ownership was split; no duplicate checks executed. JS owns Node, metadata
owns gates. Local publication screen plan is two reversed rounds, sizes5182/
20728 batch64,8 cells comparing matched V17 parent/V25, not acceptance.

Remote baseline249/V17254 builds/checks complete: actual compile75.42/34.99s,
Node4 / Node11+factory7 pass, frozen Linux natives16cba26c042fcd6d098de149ac3495207734deab9f04fec39827ab332ea1ba5c
and66ffb9d6db18533f59b6b7db1951f07d8b8f037a93e80ded00f3d4781242f7ad.
Node26.8.1 runtime SQLite3.53.4 confirmed. build-completion.json SHA
5a9f8d4155469c621dea0720964d7d87419574047e9a37892fe6cd14ee42a4d6.
Postings agent now fresh-builds V25 in separate Linux source/target/package,
session46715; same features/versions, absolute identity/source guards. Portable
exact checker/assets preparation is disjoint. No remote measurement yet; full
eleven-round selected V25 prospective campaign still requires root release after
all builds stop and same-host exactness/version/protocol checks.

### Spaced V17 serving completes; V25 builds and acceptance checks begin (2026-10-16)

Local combined-v17-serving-spaced-v1 completes0:11 rounds/33 cells/85,536 exact
calls,33 immutable20s pauses verified. All six numerical serving guards pass:
RSS/SQLite median.9243097922 family8 upper.9736814838; sourceRSS.8319088
upper.8718686; sourcep95 upper.9245360; sourcep99 median.8822581 upper.9884163;
sourceQPS1.1903019 lower1.1454026, SQLiteQPS9.9461561 lower9.5510195. All33 load
gates fail, preserving UNQUALIFIED status; fixed spacing did not cure host load.
Every pair/loss remains, no acceptance/integration or superiority claim.
decision.json binds measurements6ddcd1f73926a0986b4185d07d45482e01cd16a0712f3d32674d511515a8334e,
analysis4a179953c6d5e975d91a9930b7e9cbb25f75427cda6c02e63a86fd27d637051e,
protocolee25bdf5d752a404474518b1f73c4c7dc31a4767aff6a6d62926c08779f76411.

After serving stops, root releases local V25 Node, focused+six gates and fresh
default Rust publication-screen BUILD only; disjoint sources/targets. Node startup
failed before Cargo because source capsule excludes rule files; failed partial
capsule/log retained, corrected driver copies rules from root without changing
254 frozen inputs. New session56604 active. Gate run-v1 got EAGAIN on global
timing.lock held by publication build77027 before any Cargo/source copy; failure
retained, run-v2 waits for lock release. No measurement during any builds/tests.
Original unrun gate driver/plan also retained when its pre-build no-other-builds
assertion was adjusted to record root-authorized disjoint concurrent builders.

Remote V25 fresh build/checks complete0, all remote jobs stop: Linux native
75afd07588a746cea8eda8e427900a4164a2b2a3e005f90b4ac9376b76a101da,
Node11/factory7/documents42 pass including new metadata domain test; actualfresh
compile and254 manifest168493...4388 verified. v25-build-completion.json SHA
ca7e6ee2f37219dd6fb5b4368d0d711ffd3ace28c77ed580dfadcb2f6d351f9c.
Portable source assets19 unchanged files/9.08MB freeze manifest
d125f3beaa9b69523cde020bdb3010219167435c815fc552a24de7cde4010d7c;
no checker execution/compression/large JSON parse at preparation. Remote full11
selected V25 driver/protocol still pending review/root release; no measurements.


### V25 local checks advance; first remote serving run stops on SQLite f64 witness (2026-10-16)

Local V25 Node56604 completes0, actual fresh37.49s compile: native
7091225f1aa28f698e7205f64969b0df373e07035dba4fd2c5b1cd05ba3d18e7,
Node11/base7/factory7/numericRust1/index6/metadata6 pass. Root independently
verifies254 inputs and complete package hashes, copies serving package into
combined-v25-evidence/package. root-node-verification.json binds ready/package
and source168493...4388. Source serving.py adds ONLY V25 whitelist entry; old
3ad759...4648 preserved byte-exact in driver-history, new hash
971458c4720ea26ea0666f1a0582ab92971e0bc65a663b4bd08455fdcb2cf867.
No old serving receipts rewritten. Private Node startup failure retained.

Local pubscreen freshBUILD77027 completes0: parent f43...68c70 unchanged,
candidate297b79b4c1b5425f027581a13a3abbe916b2ac0603a1e97b9c89d0bf534277b7;
Clippy/actual compile and instrumented source guards pass. Global lock released;
metadata gate run-v2 then acquires it, session34056. Focused38/11/2/1, fmt,
both all-target Clippy-Dwarnings and default438 tests(oneignored) pass so far;
all-feature tests/MSRV remain. No local measurements until all builders stop.
Root owns next8-cell publication screen runtime; source/prepare/driver immutable.

Remote root releases reviewed protocolc1def24a8c49241981d80ce6c0e01ae6c5b4c5602a341f3e389d421891befc5c
and driverc4173e5afbb37168732cf67284e792c674e01e79c94ccbb3d1ad62a64586a3f2;
fresh process guard finds no builds/Node and load.227. Full eleven-round run
52425 starts once, then STOPS1 after first3 cells under original exact checker.
HMS baseline and V25 both pass historical completeJSON/f64 and same-host all324
first-pass rows. Both load guards pass. SQLite raw third cell completes, but
historical lexicalScore bits fail at output-contract run.py:35:5479/22078
mismatches (4604x1ULP,842x2ULP,33x3ULP). ID/order/count/metadata/byte ranges and
score-vs-lexical relationships have ZERO mismatches. Cause remains unassigned;
no claim that libm caused it. All3 raw load gates pass. Descriptive servingRSS
baseline177328128,V25147771392,SQLite158912512B, V25/SQL.929891; first-pair QPS
3640.03/3639.80. No accepted full campaign or superiority; failed raw cells,
receipts, log and all mismatch losses remain, no retries/filtering/precision waiver.

All remote runtime stops; postings agent now independently checks every SQLite
score bit using cached nativeSQLiteCLI3.53.4 on the completed read-only database,
exact original expressions/order/limit. No competitive cells rerun. Primary API
for exact binary extraction: https://www.sqlite.org/floatingpoint.html section2.1.3
(ieee754_to_blob, big-endian8-byte f64; included CLI extension). This is a proposed
native-reference validation, not permission to relax HMS exactness or use numeric
tolerances. Any subsequent host-native SQLite witness and new prospective
protocol require root review; rejected run remains rejected. JS agent prepares
V25 diagnostics SOURCE ONLY from established V17 matched cases/baselines; no
build or resource runtime yet.


### V25 six gates pass; independent native SQLite witness and publication screen (2026-10-16)

All V25 Rust gates complete0: fmt, default/all-feature all-target Clippy-Dwarnings,
locked default438/0/1 and all-feature672/0/1 tests, Rust1.89 default-library check;
focused documents38/mutation11/snapshots2/legacy1 pass. Root verifies all10 exits,
complete254 selected source and249 current dirty root inputs/native unchanged.
run-result SHA3a50e09bc190447980898103a1a43446a2baec3d0d56f75e4f094d0de26abfec,
completed.json e6016f646f461b814655931f686079553363f5429c290b898c53bd2b1166c471;
root-gates-verification.json/root-gate-exits.json bind source and rebuilt Node.
All local builders stop before root releases exclusive8cell pubscreen26554,
publication/screen-v25/run-measure.log. Parent V17/probe f43...68c70 versus V25
297...277b7, matched fresh seeds, unchanged guards; fixed20s before each cell.
Initial5182/b64 pairs exact/zero drops, servicep99 C/parent.999424/.922566,
exclusion1.002120/1.037491, scheduled-response.265143/.437769, all4 load fail.
Partial medians are descriptive, not proof or causal attribution;20728 remains.

Independent Linux SQLiteCLI3.53.4 calculation on retained failed-v1 read-only DB
matches ALL324queries/22,078 NodeSQLite result IDs/order/score bits EXACTLY, zero
ULP tolerance and no decimal rendering. ieee754_to_blob(1.0) probe has8 bytes,
3FF0000000000000. CLI9dad0ffbbf5207363f367e4ecec6f128958dd70a498a9a7cf0a823f758bdbc80,
DB0319e2180eaf10ae5b38c4af16d9cc1c01ff69b4baadeab5f82870493fb4f61f,
queryfunction12d9b03f2b87e1d8946848f5c723567d1b8764a2e616ed9a9fbfbdf4b9d2b1d2,
proof receipt6cfcc40721000994896d278df7fbc36da65f075eb7b4bbc55d588084d9f18b9d;
completionf1c824d7d1ed253fb2f068b9f9c63082a07d090a7d16170404139424f88e9290.
Raw failed-v1 cells/receipts and all5479 ARM historical discrepancies retained
under failed-full11-receipts/ and JSONL896009510a633c4f01804c1695a1f52e4a6870cedf6117be331974edc084462f.
Cause remains unassigned; no claim of identified libm mechanism.

NEW prospective v2 uses independent CLI exact-bit witness for SQLite and keeps
original historical HMS fullJSON/f64 plus all same-host HMS comparisons. All
historical SQLite IDs/order/count/richmetadata/ranges stay exact; both score fields
must bit-match CLI witness at every query/rank/ID. Original nineteen assets,
adapters, workload/clocks/quality/8passes/load<3/20s pauses/statistics unchanged.
Checker validation14/14 accepts correct preserved artifacts, rejects oneULP in
either score, old historical-score substitution and metadata/range/ID/version/
extra-field/rank/type forgeries; original historical SQLite checker STILL rejects
failedv1. Initial missing0/1 bucket selector failure retained, only derived test
fixture changed to int->equal-valued float; no production checker/witness edit.
Validation receipt7b08f4ca6ed0b0c9cdd2397610bdd615ee79d19238855092f23b1527dab33abd.

Root reviews protocol616f7be6a8bf618f73802deb97db40b51ec166b0b3102b895db50eba9c99069c,
driver50cf9180eeef65d5660828755095ad12c64ca68cc3970845b0879904c58aa81a,
checkera1f2af9edcd1de60291321d5f39cb7ccb465192dd8af86888742cd318ab6c87e,
witness1bdd728b2c39b6b0382891e22d4d2c758d207ab788937387dab278c43057f5a3.
Fresh12:00:13Z remote load.02197, no compilers/Java/Node; explicit root MEASURE
release55413f8e644b86344388e827825b1c3e3da83de8a0b58b187608ced9866dc8f5.
Postings agent owns remote64630/PID137577, serve-v2-all.log and
serving-v25-native-full11-v2/ under serving-v17-linux-v1 remote root. Eleven
rounds/33 cells once, no competing remote jobs. No failedv1 cells resumed,
filtered or relabeled accepted; new protocol has independent platform witness.

Diagnostics-v25 source-only preparation ready: unchanged21 complete transaction
pairs/24 lifecycle cells, exact baseline allocator/serializer reuse, fresh
candidate builds required later. Metadata prepares source-only Lucene correction:
Integer.MAX_VALUE totalHitsThreshold selects ScoreMode.COMPLETE; finite threshold
is needed for contracted top-score/BM25 WAND control. Existing exhaustive results
remain. Linux JDK/jar absent in bounded known-cache check; no Java/download/build
while either current host measurement runs. No memory/category acceptance yet.

### V25 qualified Linux serving win; complete diagnostics and next controls (2026-10-16)

New native-witness serving v2 completes0 on writerslogic-02 Linux x86_64:
11 rounds/33 cells/85,536 exact calls, all33 load<3 guards and33 fixed20s pauses
pass. ALL six serving numerical guards pass. V25/SQLite servingRSS paired median
.90930941797, family-eight95% interval[.8961169814,.9498739048], strictly below1;
sourceRSS .834776, sourceQPS1.1019245291 interval[1.0813925428,1.1486570511],
sourcep95 upper.960360 and sourcep99 .8973727801 upper1.0075359146. SQLiteQPS
4.677909 lower4.522510. Two sourcep99 paired losses remain in analysis; no
paired RSS/p95/QPS losses. This is a qualified private serving endpoint, not
integrated production, completed MEMORY category or overall superiority.
The accepted production1.091961 RSS loss remains the production position.

Root independently checks33 records/operations/load/pauses/hashes and six guards
in retained_memory_candidates_v1/remote-serving-v17-evidence/v2-completed/
root-decision.json. Analysis216b28b53d9f3e2882234e0f0189b16d04460e7a96ef496ebeebc455d7530d6b,
completion6c716e33c2565740609767cd1ae5f4508b1e617d08482041e066ff9e250b0d37,
measurementsc97477392c6ee752611d9c0e8420e9aa3395a4b9adcfef82e0b8370ac9793724.
Failed historical-SQLite-witness v1 and every1–3ULP discrepancy remain failed;
v2 exact nativeCLI witness introduces no tolerance. Complete278-file raw archive
is retained separately; no archive parsing occurred during local lifecycle runs.

Local V25/V17-parent publication screen completes8cells/103,640 completeJSON/f64
cross-pairs, oracle8 and zero drops. All8 load guards fail, so timing is
descriptive. Group exclusion/service/scheduled-response p99 ratios:5182/b64
1.0195635/.960979/.353372;20728/b64 .999826/1.033639/.748447. Group numerical
guards pass but paired point guards fail:20728 round1 scheduled-response1.4757904,
queue1.12762/writes1.35507/lease1.54849, all retained. No publication-preservation
inference and no multiplication with another campaign's ratios.
publication/screen-v25 measurements054e1acc69d8c2ec6dc2fea8bc2edfbd6ac6301b610d8896545d850f2eacba5d,
analysis4a65054f55911cbb6699f9c9a8f20eec8944e846a33b90fbfbd1110c3fc573d5.

Fresh V25 diagnostics complete0: serialization21 entire transaction-byte pairs
are identical, with no measured serializer counter losses; lifecycle24 cells/
124,800 exact calls cover plain/encrypted ingest/recovery/compaction/snapshot/
append-failure+retry/recovered snapshots. Two held versions, frame payloads,
independent result mutations and pool plateaus/drop checks pass. Root separately
compares all21 raw binary pairs and all24 complete result JSON/f64/frame pairs;
diagnostics-v25/root-completion-verification.json binds the raw manifest and
confirms root249/selected254/native unchanged. Agent acceptance-ready.json
SHA dc0eee0c5ba5532d9348c68eccafc2db69fd7231414a429586f8e3b7f50c36bc,
raw-verification-manifest613395d26c926084292882650c2b6933759dfe572c09e20a10e0708e0dc55fc0.
Losses remain:12 after-query block comparisons +3544 (plain retry+3546),12
warmup deltas +7172blocks/+729144requested/+794672usable; both pending-content
comparisons +400blocks/+47979requested/+58928usable. Native requested/usable
snapshots exclude C/V8/RSS and are not universal peaks. Initial agent-verifier
extra-pass assertion and root-verifier native-filename errors are retained;
only verifiers were corrected, no campaign/source/binary rerun or change.

Local Lucene WAND exact proof completes0: fresh exhaustive output and finite
totalHitsThreshold100/1000 both match saved exhaustive complete JSON/IDs/order/
f64 for324queries/eightpasses. proofSHA
d2bc3c126b277f5c0ca6e93da384b1fe72455ed4249c7aae616bc60210209bff.
Original Integer.MAX_VALUE collector selects COMPLETE and is not contracted
WAND; old control/results remain unchanged. No speed/RSS claim follows proof.
Linux private Temurin17.0.20.1+1 SHA3808d1d15e3ec6bd5b84057fb5d84c33d8a1536a258146bcea2e603fc726e08e
installs under authorized existing-host software scope. Both finite class builds,
ELF/version/Lucene9.12.3/ScoreMode selfchecks pass; zero queries during BUILD.
Linux fresh output/quality proof and optimized resource-frontier confirmation
remain required; no cross-host timing pooling or Windows-Java/Linux-RSS mix.

Both remote shared BUILD stages finish0 and stop before any measurement: original
baseline249/V25254 fresh default publication probes8be17bc8f713e58c93c556233676b060a10d3cdf4a529bafbc463531081895df /
fb90e51e37a79e28220151e266dbc0fc41895d7784674fadc4b5d163920380b5,
both actual source recompiles and Clippy-Dwarnings pass. Root independently
checks complete source/probe/toolchain hashes and no competing commands,
load.1006. Per-arm binary.json mislabeled rustc field contained cargo version;
append-only actual-toolchain-verification.json records actual Rust1.96.0/LLVM22
without rewriting old receipts. Full24 original-baseline publication runtime
remains unreleased pending short Linux Java proof and prospective900s-per-lock
budget enforcement; unrun prior1800s-cap driver/protocol are preserved.

### Linux optimized output proof passes; original publication confirmation runs (2026-10-16)

Root unpacks only67 selected raw files from sealed278-file serving-v2 archive
SHA4d8e8c69e4454e277b33fa723bb763a329ce85542da71c1b848bfbeaec15d820
after local lifecycle stops. All33 original complete-output checks,22 full HMS
JSON/f64/type/member-order comparisons,11 independent nativeSQLite-witness
checks and33 raw load guards pass again (85,536operations). Root-selected manifest
SHA1fbc6150962071ed4815fe89f4ce2922cb393fd06bae23b0cfc88a2807fdcf07;
root-raw-completion-verification.json records this separate verification. No
retrieval rerun, source change, tolerance or failed-v1 relabeling.

New Linux Temurin17.0.20.1 exact proof97741 completes0 in6.70s, all three
Node/JVM owned groups verified stopped before canonicalEX release. Fresh
exhaustive and finite thresholds100/1000 have ZERO complete JSON/ID/order/f64
changes versus saved same-engine output across324queries/eightpasses each.
Existing public-dev diagnostic quality reuse is justified by output identity;
no resource/frontier/primary acceptance follows. Proof42031838d51ae6015c78d9ee3b252c0af10964c82bc433d5c58acec415b4f60b,
completion1fbeb3bbe50cfcf626b79e1a2eb5bd2c4dd97e031ecefea3c556beab6c4c438a;
root-receipt-verification.json also independently binds saved Linux HMS raw output.
Reviewed proof driver29fe377e is preserved; executed85e1c8dd adds only required
owned-process-group cleanup. Original reviewed plan remains alongside new pins.

Root reviews/releases NEW publication v2 driverfd5d53d3064703f57df9c47fd0c412a65cf5f180009d09d37b32b7481d71f0b4,
protocolf7e895fe440f3467c1b23674dab5b7d9f3883bc361a8b44594d36e73b2d56036,
analyzer1fdd781b82e7bd1bd201ff4d199a6ee184883f41278afbb4ed0218d015b34129.
Four predetermined size/batch groups of6cells have separate canonicalEX ownership,
900s maximum/895s watchdog and cleanup receipts; original24cell order/workload/
fixed20s pauses/oneinitialquiet300/all exact/load/drop/latency guards unchanged.
Original execute() already kills/reaps its own serving process group on errors.
Unrun v1 is retained; v2 also supplies missing per-arm CARGO_TARGET_DIR required
only by the original execution receipt, found before any v1 seed or cell.
Release95f8d9657570f19e9fa81b69ff1ac12dbdc654ddd0f6bb31f2302d17b1012bb8
binds source/probe build09e70c2d plus completed Java proof/cleanup. Postings owns
remote session1673/wrapper147906, publication-v25-linux-v1/measure-all-v2.log;
fresh preflight has no competing jobs, initial load.0542. First2cells exact,
both load guards pass, each4800arrivals/zero drops. No other remote jobs are
authorized between groups. Full publication outcome remains pending.

V25 diagnostics paired native requested savings range8,877,736–9,611,135 bytes,
usable9,260,000–10,058,992, across the recorded cold/after-query snapshots; block
deltas range−3628 to+3546. These are measured Rust-System live snapshots, not
physical RSS, universal peaks or unmeasured queued/multichunk guarantees.
Source-only integration plan binds8 changed+5 new inputs and dry-applies exactly
to frozen dirty root; patch93db9c6b7ea0ce61053225f40ee5d71d9825d675fd82f48cdef11cfbee1c792a.
No production integration. Review exposes a known pre-construction replaced/
poisoned-builtin helper-init compatibility limitation; JS agent now reproduces
and repairs concrete baseline/V25 differences in a NEW private variant while
all V25 measurement inputs remain frozen. This is an acceptance concern, not
silently waived exact behavior or permission to alter completed V25 artifacts.

### Concrete helper-init failures retained; bounded comparator adjudicated (2026-10-16)

Existing baseline nativeb8da5c6e and V25native7091225f are compared on five
before-construction builtin changes with fresh explicit stores. Baseline succeeds
all five with zero replacement calls. V25 Object=null/throwingArray reject its
constructor; replacedJSON.parse/Object.getPrototypeOf reject a later query even
after globals restore; replacementArray adds three observable calls (baseline0).
combined-v26-evidence/before-construction-receipt.json retains full outputs/errors;
the first omitted-storage probe is invalidation-marked and preserved. V25 cannot
be integrated as exact-behavior preserving with these concrete regressions.
Its qualified serving-memory measurements remain measurements of that frozen
private revision, not acceptance of this compatibility loss. JS agent prepares
a NEW private optional-helper bootstrap/fallback using intrinsic literals and a
checked length-framed key preflight; source-only/unbuilt at this checkpoint.
Module-init global-cache alternative was rejected after before-import side
effects were identified; no environment registry/NAPI requirement is added.

Root bounded-lexical-adjudication-v1.json verifies all eight existing authority
hashes and accepts finite100/1000 native Lucene9.12.3 BM25/WAND as a bounded
shared-prepared-token serving-memory comparator conditional on joint-family
public-dev quality noninferiority. Exact token/DF/TTF/full-wordcount and native
encoded norms, matched BM25 parameters and fresh Linux same-engine output proof
support that scope; nativef32/norm differences remain explicit. Cross-engine
f64/ID/order equality is not a prerequisite. Analyzer computation is excluded;
no existing primary flags, global-fastest claims, ANN or unrelated mode gaps
are silently promoted or imposed as new bounded-implementation prerequisites.

Prospective source-only finite frontier is frozen611cfd9f6175870c04bf5942c2f6d78e92bb869287b93be6860d52355a09d1f0,
plan31235aa9461b737e13c4670066b23b8270d6f585e2e3bfb3c0f3075d32cb027a:
threshold100/1000×heap32/64/128MiB, minimum16/CPU1,24 fixed20s-paused cells,
two reversed paired rounds/configuration, simultaneous Node+JVM RSS components
and all complete same-engine outputs/load/losses.895s watchdog/5s cleanup enforce
900s canonicalEX; unrun initial/integrity drafts remain. LowestRSS and fastest
controls selected separately and BOTH confirmed if different. Final confirmation
requires final accepted Node variant; no runtime release yet while publication
session1673 remains active. Fixed10/12-family saved-data analysis is prepared
locally, preserving old SQLite8/quality7 results and every query/round loss.

Publication first14/24cells pass all14 load guards and exact cell checks with
zero drops; both5182 groups and six cross-arm pairs finish. First two EX
ownerships200.333/195.508s are below900. First group wall elapsed differs from
monotonic ownership by about4.42s; both clocks retained, cause unassigned, no
step location inferred. After runtime stops, saved group/sample/event coverage
will be assessed without rerunning cells. Full20728 groups/analysis remain.

### V25 full Linux publication is a measured loss; V26 compatibility and native repairs (2026-10-16)

Full24 publication session1673 completes0, all24 load<3 guards pass:115,200
offered/accepted, zero drops,310,920 complete cross-arm JSON/f64 pairs and
310,920 restart pairs,3,840 checked mutations,3,600 whole-bulk queries,6,165
completed wholly during append; original8 adversarial oracle tests/endguards
pass. Numerical and paired preservation guards FAIL. Group C/baseline
exclusion/service/scheduled-response p99 ratios:5182/b2
1.116858/.906700/1.123901;5182/b64 1.258088/1.286803/1.203227;20728/b2
1.031371/.980244/.906950;20728/b64 1.149703/1.135035/1.135962.10/12 paired
guards fail; servicep50 loses12/12, exclusionp50 loses11/12. Every bulk pair
servicep99 loses:5182 +17.84–31.03%,20728 +11.70–17.10%. No paired RSS loss.
These are actual same-host/load-qualified publication deficits, not a timing
waiver. Do not integrate V25 or describe publication gains as preserved.
completiondd2552105bafab42f130c4026259b9018d90e97ca89b2075114a54a8be4b9b71,
analysis7d89003c30761115c84c1bdfb403ed0df096dff877f3e791c4036d070f56c8a3,
measurements04d568ca5ed8302e09ba8e074aa83f915bd7ce5798939b3c8ee1a7a9ed08fdfc.
All losses/raw records remain; no cells rerun. Four EX ownerships200.333/
195.508/271.031/235.857s each remain below900.

Saved clock calculation1bc2b6fc1f39699ca7ca19551360d8dc5083971a609845f5d5ced8ce69bfba78
finds group wall-minus-monotonic +4.420469/+4.693872/+7.960221/+6.385642s.
All24 have ≥8 static samples, every phase sampled, strictly increasing event
epochs/nonnegative nonoverlapping ps brackets;8 boundary-straddling samples
were excluded. Cause and step locations are unassigned; old epoch fields cannot
establish or reconstruct monotonic phase alignment. Separate saved serving-v2
pause check finds up to.598443s wall-minus-monotonic; two SQLite end-to-next-pause
gaps +.507718/.541220s, the other30 gaps under4µs. Neither gap
locates a step inside retrieval. Existing6 numerical/33 load guard passes remain
measured under their frozen protocol, with this new metrology limit retained.
Final fresh measurements will use prospective exact monotonic phase stamps and
ps brackets; old raw/analyses are untouched. Monotonic retrieval latency clocks
and observed publication losses are not excused by the epoch discrepancy.

V26 fixes concrete builtin-init regressions without module/global environment
registry, dependency, NAPI/Node requirement, extra promise or public type change.
Optional helper derives intrinsic constructors from literals and validates native
helper source; unavailable helper uses original public core search/native Vec.
Checked UTF16-length-framed key preflight removes captured JSON.parse dependency.
Only lib.rs/two existing Node tests differ from V25; other251 inputs/default-cfg
tokens are identical. Source254230aa933379c4353ec5f8f69ebdda0e77426a599b26a482b962d6b7af7eee0b8,
lib88fcbc39d1aab94013eed2e2da906a6f570d175904b37d8106410f2f007e53d7.
Fresh local native43c464553a2681d3b5a3d360cd392fa65f882648c33778c75abc1d1c5b13228b
passes Node12/baseline8/factory8/Rust numeric+framing2/posting6/metadata6. All
eight cases×before-import/construction have zero replacement calls and exact
results; ten valid UTF16/NUL/colon/9-10-11-length keys exercise actual native
fastpath and inherited-setter fallback separately. Initial omitted-store and
lone-surrogate expectation failures remain; pinned baseline drops lone-surrogate
keys, that behavior is preserved. readySHAec6618b30075b083a8b5e4a6677cac2780525556e9ac9cde74cf812a92e0ec4a.
V26 is private/unmeasured; six gates deferred until native publication fixes
compose, because Node-only compatibility repair does not close that deficit.

Two disjoint native repairs are now IMPLEMENTED/unmeasured, not promised savings.
Root native-small-term-lookup-v27 changes only packed_terms.rs: common3-byte
records decode fixed u16 offsets/u8 frequency and compare byte slices directly;
generic wider/full-domain path, allocation/construction/layout/serialization are
unchanged. Existing65534/65535/65536 offsets and frequency domains plus a new
empty/Unicode/NUL/neighbor-miss regression cover behavior. Manifest
0b9cfbd047d905a009af5ae6682a5fd049d4718fec0f17f664e92b1888944e05,
file9665d1ce83f16ab271cb15eaca21e493d4cf375fa8a292100ac5d7e33b88c4cd.
JS owns isolated local focused validation15322, globalEX BUILD/tests only.
Postings native-sorted-merge-v27 changes only index.rs: two linear sorted term
traversals replace repeated logarithmic compressed-term lookups, preserving the
entire original old-update/remove phase before new insertion and differing-stem
semantics. Manifest1e6afdf98ec72456267c18fc2f55aeebc4e770417114ae4ea9cb39a073463797,
file4315c4a07ad8bf52a39f78e93e393b7f6213cb42bc32f86fbbeafada074d0569;
three new differential/invariant tests, focused remote shared BUILD/tests next.
No added retained representation bytes by source inspection; serving/runtime
effect remains unmeasured. No competitive runtime overlaps these builds.

Saved-data joint10 AND12-family calculation passes existing SQLite/source guards
and quality guards without retrieval rerun. SQLiteRSS interval[.896117,.949874],
sourcep99upper1.007536/QPSlower1.081393; family12 Lucene-minus-HMS quality nDCG
[-.003851305,.000947545],recall[-.001430499,.000288944], both directions pass
−.005. Old quality7/serving8 bounds reproduce exactly and remain unchanged;
all66 round ratios/324 query losses retained. joint-family-saved-v1/results-v2/
analysis442c9d60755b051a94e8cf2a794c0bcb2f46907dacd02b1a43fb2e3ee0cbbc6d,
completionc4f38cf46b6ea2e2283bc75bf8d9b1faa1c20f8438d2bb3da23895738f5517bf.
Initial overbroad unrelated idle-Node guard failure is retained; no processes
were stopped. Lucene memory/QPS confirmation and final variant remain pending.

### V28 native repairs composed and fresh builds verified; publication screen started (2026-10-16)

Both native repair candidates pass focused validation: fixed3B term lookup
39 document tests including8 packed-term cases/defaultreleaseClippy; sorted
two-phase posting merge9 index tests and45 node/security document tests/Clippy.
The latter includes5120 legacy-differential interleavings and partial-failure
ordering. No candidate timing result is inferred from those tests.

Root composes V26 compatibility plus both V27 native repairs in combined-v28/,
254 inputs, final-manifest.json SHA64a65ed75bfb524bd8d911e2e8113c21ea7b46ca719dde223aa571b6b0942012.
Five files differ V25; other249 inputs exact. Source-ready.json binds component
receipts. Fresh local Node483fca28 passes12 checks and real Unicode/NUL key
framing; baseline8/factory8 are exact guardedreuse. Fresh Linux Node0b93c777
and default publication probe d1610482 pass actual compilations, publication
Clippy, Node12/baseline8/factory8 and ten-key framing. Remote sealed completion
1cad851ce16256f51efaab058e8fd3c8486351ffaf9cf49878bbcb0edb05bfc8 retains two
reporter-only failures (Node26 spec-prefix assertion; str passed to Path-only
metadata sha helper). No compilation/test rerun; Unicode executed once. Root
independently verifies all254/256/9 selected source/package hashes, seven actual
zero-exit phase logs/receipts, both freshcompiles and binaries. No runtime/build
was active before release; remote load.08.

Local six-gate validation retains initial harness-only wrong document-count
assertion: actual42 default document tests pass, exactly38 previous+4 new. Resume
31249 reuses that result without rerun; mutation11/snapshots2/legacy1/fmt/both
alltargetClippy gates pass, locked tests and1.89libMSRV still active. Separate
diagnostic builds stop0: candidateallocator a20ba8cc/serializer c3df0c7d, exact
baseline reuse. Root independently verifies allfour binaries/formattedmanifests
and full origin selections; allsix case/counterdrivers byte-exactV25. The21
transaction-byte and24 lifecycle cases remain unrun pending all local gates stop.

Root reviews/releases NEW original-baseline/V28 publication screen once:
remote-v28-publication-screen-v1/{driver.py,protocol.json,release.json}, driver
d72dd1608b0d412330df991caa7aa4e90e822b3cd4b48944b9ba96958777be6d, protocol
6f304a4f8647fb9d2c4a7c2c16a555abddf7383ea2470bbb96a14e0728dae8fb. Session61058
owns sole remote runtime: two sizes5182/20728,b64,two reversed pairs,eightcells,
fixed20s pauses/two bounded canonicalEX groups/all original exactness/drop/load
and paired preservation guards. RustInstant publication timings are primary;
epoch-aligned RSS stays diagnostic. Every loss remains, no retries/filtering;
full24 follows if strong. V28 performance is unmeasured at release. Production
249/native remain unchanged; production RSS1.091961 SQLite deficit and overall
unestablished status remain. Windows sleep inhibitor1916 stays running.

V28 local gate resume31249 now completes0:442 default/677 allfeature tests,
zero failures/one ignored each; fmt/bothalltargetClippy-Dwarnings andRust1.89
defaultlib pass. Focused42/11/2/1 allpass, initial42-count harness failure kept.
Root verifies allten actual log hashes/counts,254 private+source inputs and249
production inputs/native unchanged, no compiler processes. Root releases local
21 serialization then24 lifecycle once; diagnostics measurement launch pending.
Remote screen first5182/b64 pair has descriptive C/originalbaseline exclusion/
service/responsep99 .792763/.706512/.734672, exact/zero drops/bothloadpass.
Five ofeight cells now checked/allfive loadpass; campaign continues unchanged.
No final publication acceptance is inferred from a pair.

Root prepares NEW V28 source-only integration patch, eight changed+five new
paths/241 unchanged, no deletions. integration-plan/prepare.py verifies complete
dirty249 andV28254 selections/currentbranch, checks against the real dirtyroot,
applies ONLY thirteen private dry-apply files and verifies candidate hashes.
Patch00d31623bccd7aa791726650e5b390d6ce021de80bad3e9edc5c35daba417173;
production/native unchanged, not integration or performance acceptance. Full24
publication wrapper also prepared SOURCEONLY, reusing exactoldfull24main and
analyzer1fdd781b with only paths/sourceguard/runtime_env rebindings. Driver
44a2cb65e79cccace166663743e7b176956e0627d6eada06f64b25eb1b51b07d, protocol
b5ba00feae0b467caa84263abe7e12f54b14fef9436d9a0121dc4a55c4317b69; unrun,
unreleased, freshseeds/all24cells required ifscreen strong.

### V28 publication screen closes the observed bulk p99 deficit; full24 underway (2026-10-16)

Screen8 completes0/sole remote timing stopped: eight loadguards and four paired
1.10 preservation pointguards pass,38,400 offered/accepted/zero drops,103,640
complete JSON/f64 cross-arm and103,640 restart pairs,512 checked operations,
2,400 whole-bulk queries, original8 adversarial oracle tests pass. Group
C/originalbaseline exclusion/service/scheduledresponse p99 ratios are5182/b64
.742385/.679456/.693559 and20728/b64 .752238/.743693/.775843. Paired ranges
exclusion.693513–.813178,service.655532–.752505,response.658615–.792919. Thus
the tested bulk-tail deficit is reversed in this screen; full24 preservation
is not inferred. All losses stay recorded, including servicep50, queue/write/
append/lease tails and20728 compaction. EXownership143.050/212.067s<900;
fixed20spauses pass. Root independently recomputes group/paired primary ratios,
counts/pauses/ownership and checks actual oracle8 log and frozen result hashes.
Completion2359d2accc8c707a17ea2d05fddeda70f1d728df113fa6a5caaf0ec438ca0881,
analysise5c6d47d10087868182594e0517a801ded7a6c1d681f6fdb71fcfe8f900aa469,
measurements09cb0dd6b000d30980687adf156ff345d82392e352e8d956fcb176ca9e032ce8.

Root releases prospective full24 with originalbaseline249/V28254, NEW separate
arm seeds/all cells, no screen/V25 observations reused, unchanged original
measure_v2.main and analyzer, only guarded paths/source identities/runtime_env
rebound. Driver44a2cb65/protocolb5ba00fe/analyzer1fdd781b, release4897d902;
session90674 now runs on writerslogic-02/v28-build-v1/publication-full-v1/.
Four fixed size/batch groups have canonicalEX900s/895watchdog/5cleanup; sole
remote owner postings/no intervening builds, clock or Lucene runtime. Every
loss retained, no retries/filtering. RustInstant publication timings primary;
RSS epoch alignment remains diagnostic. Full24 results pending.

Local serialization21 completes0 and all complete transaction bytes match.
Full24 lifecycle session27538/lifecycle-v1 now18/24exact, encrypted candidate
cells remaining; sole local EXownerJS/no other local builds or measurements.
The fixed counters/cases are unchanged; final retention/lifecycle result pending.

V28 lifecycle27538 now completes0/all24 rawcells/124,800 exact calls, no rerun.
Root independently verifies all21 raw complete transaction payload byte pairs,
all24 complete JSON/memberorder/types/f64 result pairs and decrypted-frame pairs,
raw manifest74651bdf6bdfe9de0ade642e783b456bcd311ce299fcf08d09ce558c272ef675,
agentreadya42e509f30bd03dde3f08b85397bf8eb0b75c6287d7cb0b3ede5c69a90a59a59,
254candidate+249production inputs/native unchanged. No serialization counter
loss. All12 after-query native requested reductions are8,878,232–8,882,424B,
usable9,260,064–9,264,320B; those same12pairs retain3544–3546 MORE blocks.
Each candidate warmup retains7172blocks/+729144requested/+794672usable; both
100-onechunk pending Arc pairs retain400blocks/+47979requested/+58928usable.
Bounded poolplateau/drop checks pass; no universal pending/multichunk bound or
wholeprocess RSS superiority is inferred. Full counter/peak/retention losses
remain in original analyses and acceptance-ready.json. Alllocal runtime stops.

Prospective final measurement scope is made explicit in L/root-memory-scope-v28.json:
bounded prepared-token MEMORY only. Preserve the exact six inherited serving
guards from combined-v17-serving-spaced-v1/decision.json: SQLiteRSS corrected
upper<1, sourceRSS/p95/p99 upper<=1.10, sourceQPS lower>=1/1.10, SQLiteQPS
lower>=1.20. Lucene lowestRSS and fastest controls each require strict adjusted
RSS upper<1; LuceneQPS is measured with intervals/alllosses, no invented numeric
gate or primary efficiency win. Existing broad1.20 efficiency contract/oldflags
remain unchanged. New finalV28 observations replace no historical results.
Joint5000/seed24301 servingmedian bootstrap extends existing conservative
indices to family10[12,4987]/family12[10,4989]; quality uses existing mean/linear
quantile at.025/family and1-.025/family. Include all reportedQPS metrics even
without a LuceneQPS gate; one selected control uses10/two distinct uses12.
Final observers prospectively assign left<=begin<right,end<=right to exclude
zero-length boundary doublecount, with old source versions retained and focused
pure boundary tests required before runtime. No new monotonic runtime yet.

### V28 full24 retains two small-batch losses; three targeted query repairs implemented (2026-10-16)

Full24 session90674 completes0/all24 rawloadguards pass,115200accepted/zero
drops,310920complete cross-arm JSON/f64 and310920restartpairs,3840operations,
3600wholebulk/6120duringappend,originaloracle8/endguards pass. Group C/original
baseline exclusion/service/responsep99:5182b2 .856845/1.129067/.943741 FAIL,
5182b64 .735132/.659663/.685657 PASS,20728b2 .894898/.731884/.860161 PASS,
20728b64 .803230/.758998/.761173 PASS.10/12 paired guards pass; failedpairs
5182b2r0service1.727252 andr2response2.011764 (r2service1.011416). Servicep50
loses11/12; everyothercounter/latency/RSSloss retained. V28 is NOT ACCEPTED or
integrated. Completion990c80de1401e7b1a8a6ee0f953fc59b82ad90de1bd8e2537ee8a16b924d6d12,
analysis41f4e6ba54a6de20c2add2dd9faa4860e4f850de6711a97e6124139eb050e34d,
measurementsdfe9aa8e252c94c1ac1e60cd8d56aad68e0cdc28c8416751d181084d2d258d8b.
Root independently verifies24actual rawloadreceipts/allprimary group+paired
ratios/24pauses/fourEXbudgets/oracle8 inroot_verify.py/root-completion-decision.json.
Initial root inline verifier used the wrong pause/log folder afterload/ratio
checks; corrected read-only verifier does not rerun measurements. FourEXtimes
199.513/194.833/270.906/235.236s<900; wall−mono+6.1631/+6.4035/+8.9733/+7.6025s
retainmetrologylimits, noassignedcause/step location/RSSprecisionclaim.

Savedtiming-only19,200request analysis identifies actual measured query:
two explicit documentIDs+embedding force the scan path becauseposting_work+
slot_count>=5182 while2chunks×2queryforms=4. Postingarray decoder changes would
not directly affect it and are not pursued. r0candidate49service-tail queries
include22publication overlaps and26starting afterlastcommit; afterALLmutations
andcompaction Cservicep99=59087ns versusB29812, withoutCPU/wait separation.
r2response-tail49 have medianAPI10929ns/producerlag587625ns/offered→dequeue89865ns,
18compaction/1publication overlap. Arrival/queuecomponents explain the recorded
response decomposition but not their cause; no CPU/noise waiver. Initialgzip
fallback extractor-pathfailure is preserved; no query rerun.

Root implements native-idf-vector-v29/search.rs ONLY: collect IDF intoVec from
the identical sorted queryterm iterator ratherthan a tree used only foriteration;
allf64arithmetic/order/retainedrepresentations/serialization untouched. New48case
independent fullscan/fullJSON/exactscorebits test covers0/1/10/11/12/32terms,
mixedlegacy/stemmed/zero+absentterms,lexical/hybrid,filters,blend/RRF. Manifest
24f24e594d02114fef0b5671d94692f385e58f530d4c45b8b32652be00cfa223,
search78d373cd2dbd2ad5cb71bc54feeb4705f30e99ffaa30d85998ab67336f32a91d.
JSfocusedLOCAL64089 hasfmt/defaultreleasealltargetClippyPASS; securitydocs43
pending. Performance remains unmeasured, no retainedmemorysaving inference.
Postings implements bounded<=11 explicit-ID borrowedVec sort/dedup, >11 original
tree fallback, onlyindex9b31553b/manifest27f8ffe5, formatted/uncompiled withthree
meaningful boundary/fullresult regressions. Metadata implements pack0/1key and
existingOther nativeMap clone for2+ keys, onlymetadata26f1a77d initially; complete
semantics tests added, no mutablecache. Four-key retained+440requestedB/object
is explicitly INFERRED until pairedallocationproof; commonone-key target stays
packed bysourceinspection. Allthree sourcecapsules preserve V28 and all losses.
Next composeV30,focused/sixgates/freshbothhostNode+pubbuilds,then targetedmeasures.

FinalV28monotonicserving harness remainsSTAGEDONLY/STOPPED/unmeasured: driver
7052ea22/protocole8586562/source-ready2887840c/15puretestsPASS. Adds actual
900/895/5 canonicalEX bound includingseed/handshake/quiet, robustancestor-excluding
processguard and source/native/release/outputbindings. OfficialNodeexe19235a9b
matches the150425704B bin/node entry of verifiedarchive3e301118 andcurrentexe;
proofdd76a236 obtained afterpublicationSTOP without Node/clock/workload invocation.
Initialwronguvlauncherpath failed127 beforePython and is retained. Lucenefrontier
guardrepair adds actualreleasefile/SHA+all254sourcebytes+sameNodepin; allsixpure
tests pass, unstaged/unrun. Olddrafts retained. Finalcandidatebindings willchange
afterV30builds; no V28competitive measurement/publicationrerun is authorized.

### V30 composes three query fixes; metadata tradeoff actually measured (2026-10-16)

V29 rootIDF candidate focused64089 completes0: fmt/defaultreleasealltargetClippy
withwarningsdenied/securitydocuments43 pass, including new48case scan/JSON/f64
regression. Actualfreshcompile/full254source/private/root249 unchanged guard;
ready9ee1e90fdaa1663ae5bfd27021f305af425a92fcf6a4891977a522b927bace9d. Posting
explicit-ID candidate Rust1.96format leavesindex9b31553b/source27f8ffe5 exact;
standalonefocuseddriver remainsunrun toavoidduplicating combinedvalidation.

Metadata formattedsource dc12b14f3ccaf4053ebe34eeb26f995047c30f54ddbe0cb73a68c085f97be17d,
manifest91a3a3d260fc8ea2be3bdc214cf0a96c94f8b9bfc6db58d8e554fffea64c8d31.
Remote31480 actualfresh module builds and62cases EACHdefault/preserve_order
pass124 total: completeJSON/both-endianbincode/f64/order/parser/mutation exact.
AfterallbuildersSTOP freshEX allocationproof MEASURES default4-keyretained
+440requestedB/+1liveallocation perobject versusV28; heldStoredMetadata clone
same+440/+1. Query requestedallocation counts/bytes remainIDENTICALdefault,
so proposed CPU benefit is nativeMap nodeclone versusreinsertion, unmeasured.
Preserve_order sorted4-key retained+504B/result+312B vsV28 (resultmatchesbaseline);
reverse/nativefallback losses also retained. Construction/query/heldclone0/1key
deltas EXACT0 bothfeatures, enum16B unchanged. Fullraw/24selectedreceipts/all
losses preserved; analysis6a2d9359c211c696fdf7f587e3bad5abe7d09452636fb99761fe1b14572cd7b8.
BothremoteSH/EX released; no serving/publication/competitivequery run occurred.

Root verifiesallthree254-file component selections, only respectiveownedfile
differsV28, and composes combined-v30/254inputs includingbothCOSE. Finalmanifest
4fffac0779615b24d4310d28b7ead48d48a5fa989297d1a48f8c70fc4d015195;
onlyindex/metadata/search differV28, other251 exact. source-ready.json bindsall
componentfile/manifests; root249/native remain unchanged. V30performance is
UNMEASURED/unaccepted. Metadata nowownsLOCALfocused+sixgates85057/run-v1,
SHjobs2/isolatedretained-v30-validation-20261016; expectedsecuritydocs47 andall
five newregression names mandatory, focusedmutation11/snapshots2/legacy1 then
all sixgates. JSauthorizedparallel disjoint LOCALNode+diagnosticBUILDONLY,
sameSH/jobs2; no localmeasurementuntilALLbuildersSTOP/separaterelease. Postings
authorized freshREMOTEdefaultpublication+NodeBUILDONLY, isolatedv30sources/targets,
samefullguards/actualfreshcompiles/correctreporterfromstart. No competitive
runtimeyet. Prospective targeted8screen5182b2threepairs then5182b64onepair
willtest the remainingdeficit andbulksanity, preservealllosses/20spauses/oracles/
load andunchanged1.10pairedguards; rootMEASUREreview/release required, full24ifstrong.

### V30 gates pass; publication tails remain; fresh serving comparison launched (2026-10-16)

All six V30 gates and four focused checks complete without failure or retry:
focused documents47/mutation11/snapshots2/legacy1; fmt; default/all-feature
all-target Clippy with warnings denied; locked447 default/682 all-feature tests
(one ignored in each); Rust1.89 default-library. Completion726465f9 and
run-result64d4d4fb in combined-v30-validation/ bind the full254 manifest4fffac07.
Root independently hashes all10 actual logs, checks package identities and
absolute manifests, recomputes test totals, rehashes both254-file source capsules
and the original dirty249 selection/native: root-verification.json dfa51622.
Production source/native remain unchanged. One read-only root verifier schema
mistake (expecting full metadata instead of reduced identity receipt) is corrected;
no Cargo or measured workload was repeated.

Fresh Linux V30 build64389 completes0; build513cd6bc, default publication native
ad053dd0 and Node-api/security native6d426df6851977ff4b80231c9f73c20caf47e7b1f6c20ce9a8e007ded80e72af.
Root independently verifies254/256/9 source/package selections, actual fresh
compiles and Node12/baseline8/factory8. Fresh LOCAL build43983 completes0;
pure Node18f4b18e/Node12/Unicode10 pass. Four LOCAL diagnostic binaries and
formatted origin selections independently verified; builds-ready cff8739d,
candidate allocator3c1d2894/serializercccdd557. Cases/counters/adapters remain
byte-identical to V28; 21 serialization and24 lifecycle cases are not yet run.

Targeted publication8 campaign98322 completes0 with all8 raw load guards passing,
38400 accepted/zero drops,41456 complete cross-arm plus41456 restart JSON/f64
pairs,1664 publication operations,600 whole-bulk hits,2149 during-append hits,
oracle8/endguards PASS. Aggregate b2 exclusion/service/response ratios are
.898130/.961823/.983803; b64 .868876/.663191/.653525. However b2r0 response
1.218520 and b2r1 service1.379986 FAIL unchanged1.10 paired limits;2/4 pairs pass.
All losses remain in remote-v30-publication-target8-v1/. Group improvements do
not establish preserved publication performance. No full24 rerun or waiver is
released. Next source-only diagnostic separates thread CPU, gate/capture waits
and elapsed time; unassigned non-CPU residual will not be called noise.

Root releases fresh private V30 MEMORY measurement independently of pending
publication acceptance: remote-serving-v30-monotonic-v1/serving-release-v30.json
cf931beb719b54b0687bf57bff14027067c8ee23c0303ff01c842f7ff40c3ed4.
Source-ready45490f07/driver51224d01/protocol4e62100a bind final4fffac07/native6d426df6/
build513cd6bc/packagea9a33650/official Node19235a9b/current scopea250a1ab.
Read-only stage/preflight verify all40 inputs, full baseline249/candidate254 and
packages, no competing computation. Fixed11-round/33-cell/85536-call campaign
launches once as remote uv PID189315, canonicalEX900/watchdog895/cleanup5; all
other remote builds/calculations/measurements are held. New monotonic phase/RSS
brackets are prospective only; no old epoch timestamp repair or performance reuse.
Final Lucene V30 frontier/confirmation sources are ready1b56f39a, unstaged/unrun.
No integration/category acceptance; production serving RSS remains1.091961 SQLite.

V30 LOCAL diagnostics26746 complete0/STOP: all21 raw transaction-byte pairs exact;
24 lifecycle cells/124800 calls preserve complete JSON/member order/types/f64 and
decrypted transaction frames across plain/encrypted ingest/recovery/compaction/
snapshots/retry/recovery-with-snapshot. Agent readyc721ada8/rawmanifest03558808;
root independently verifies every selected raw hash and every complete byte,
result and frame pair, plus original249/native/final254 guards:
diagnostics-v30/root-completion-verification.json. All losses are preserved:
12 after-query pairs save8,879,871–8,882,496 requested bytes but use3541–3544 more
blocks;12 candidate warmups add7172 blocks/729144 requested/794672 usable bytes.
Encrypted-baseline-retry also has a distinct warmup loss3 blocks/136 requested/
144 usable (unassigned cause). Both one-chunk100-item pending Arc cases retain
400 more blocks/47979 requested/58928 usable bytes. These counters establish
neither universal peak improvement nor whole-process serving RSS superiority.

Before first Lucene runtime, source review finds classes/assets checked against
mutable manifests without pinning those manifests themselves. An accidentally
substituted exhaustive control plus coherent manifest could preserve exact
scores while losing WAND; output equality alone is insufficient. Metadata now
prepares a new preserved prospective v2 with immutable original Linux build
receipt pins for asset, compiled100/1000 and qualification authorities, both
start/end checks and meaningful coherent-substitution tamper regressions.
V1 remains unrun. Remote33 remains isolated; next remote work after STOP is
the fixed matched CPU/lock diagnostic, then the reviewed optimized Lucene grid.

Fresh V30 monotonic serving campaign finishes all33 cells once. Agent-observed
analysis reports33/33 loadPASS/85536 exact calls, serving RSS/SQLite pairedmedian
.9132755686213833 and joint12 CI[.9054959265485581,.9470615671641791]; all six
source/SQLite numerical guards PASS under joint10/12. Raw sealing/transfer98125
is active; root independent raw/clock/interval verification is pending, so this
is not yet an accepted result. Publication remains held after target8 losses;
Lucene remains unmeasured. Original production249/native and1.091961 loss persist.

Lucene prospective v2 is source-ready98bca529, unstaged/unrun. It pins eight
original Linux asset/compiled/qualification/javac/build authorities, validates
finite100/1000 source hashes/collector argument and every actual class byte,
and chains exact successful javac commands to original7-command completion at
both start/end. Pinned asset manifest anchors indexmanifest and all5 actual
index files, already checked before/after and eachcell. Three meaningful pure
tamper regressions PASS including coherent exhaustive-control substitution;
initial /var versus/private/var tempfile fixture failure and source are retained,
test path canonicalized, guard unchanged. Root verifies all frozen inputs,
actual3testlog and exact runtime AST equivalence to v1 after removing only the
new guard import/start/end calls. No settings, workload, schedule or thresholds
change; no control runtime yet. Durable root-source-verification.json in v2.

### V30 fresh SQLite/source serving result independently verified (2026-10-16)

Remote33 and sealed transfer98125 STOP0; all owned workload process groups gone,
SH/EX released. Complete262-member raw archive cab317ebe2dbf55f098b7a50404a4728e03554bf6a28860ba3b188768c0ce5bb,
28,755,136 bytes, all payload hashes verified. Canonical local copies in
remote-serving-v30-monotonic-v1/completed/ include all33 complete result files,
observer receipts/logs/seed proofs/pauses, source controller, exact clock proof,
analysis/measurements/completion and canonical lock receipt. Root independent
verifier extracts66 selected raw files from the sealed archive, repeats original
complete historical HMS and independent native SQLite bit oracles, compares all
22 HMS complete row sets/member order/types/f64, and recomputes EVERY monotonic
sample bracket/RSS phase/load guard/static8 barrier, all33 pauses, all8 paired
vectors, family8/10/12 intervals and six numerical guards. All PASS;85536 calls.
Durable root-completion-verification.json SHA1b3bf62d02791c3027587bad9589c8116bff560f6104ae83bfbf86ba190c95c4.

Private finalV30 serving RSS/SQLite pairedmedian .9132755686213833, joint12
CI[.9054959265485581,.9470615671641791], strict upper<1. SourceRSS .8199152542
CI[.8086329938,.8754041819]; sourceQPS1.1224514481 CI[1.0946165668,1.1646703629];
sourcep95 .8950460994 CI[.8485799256,.9234636535]; sourcep99 .8588124514
CI[.8227284015,.9621542931]. SQLiteQPS4.7619773417 CI[4.6727425369,4.9175331936].
All six serving guards pass jointly under10 AND12. One unadjusted sourcep99
round8 loss remains in all receipts/analysis; no cell filtering/retry. This is
a qualified private steady serving-memory comparison, not a transient ingest
claim. Full MEMORY acceptance remains unestablished: optimized Lucene has not
yet been measured and V30 publication target8 still has two paired tail losses.
Production249/native remain unchanged and accepted production ratio1.091961
remains worse. No integration/category/overall superiority claim.

After all remote measurement/archive work STOP, root independently rechecks
WindowsPID1916 alive and powercfg SYSTEM inhibition active; durable
remote-awake/recheck-v30-serving.json. No display/permanent policy change;
keep the dedicated inhibitor running. Next work uses the already authorized
machines: source-only Lucenev2 staging, matched CPU/lock diagnostic ABI/build/
fixed4 cells, then optimized Lucene24 while publication evidence guides repairs.

Matched CPU diagnostic v1 actual C ABI proof passes (clockID3/timespec16/align8/
resolution1ns/getcpu valid), then first fresh baseline Clippy rejects only a
manual modulo-is-zero assertion; no Rust FFI tests or measurement run. All v1
sources/logs/receipts remain. Disjoint query_cpu_lock_diagnostic_v2 changes only
that assertion to equivalent is_multiple_of; full252/257 manifests refreshed.
V2 ABI/default Clippy/both fresh probes/helper2+TLS1 tests PASS; build0068bd16,
natives66124b6f/c6fd0730. Root independently checks8actual logs/6metadata guards,
two fresh compiles/three tests/CABI proof, SSH-rehashes all509 source inputs and
actual natives underSH. Counter code is confined to private diagnostic capsules;
original finalV30 254-source candidate remains unchanged/no production unsafe.

Root releases exactly4 diagnostic cells once: query_cpu_lock_diagnostic_v2/
measure-release.json10757f1e; measurea8a66b93/protocol5e5c365b. B/C,C/B,
5182/b2/4800decisions percell/8s/four readers/20s pauses, canonicalEX900/895/5.
Original complete publication/result/restart oracles and eight adversarial tests
remain; supplemental counter checker/analyzer pass six pure tests. Five CPU
reads and two virtual-CPU endpoint reads per query are bracketed with Instant;
fixed1200 initialized frames/reader (883200 requested bytes) precede serving,
no hook allocation or added fileIO during serving, sidecars after all joins.
Signed residual bounds/negative values/all bracket widths/calibration remain;
no overhead subtraction or assignment of all nonCPU elapsed to scheduling.
Endpoint IDs cannot identify physical hybrid core type or establish causality.
Runtime starts uv195252/supervisor195251; all other remote work held through
sealed transfer STOP. These instrumented timings/RSS never establish acceptance.

Reviewed Lucenev2 staged/preflight STOP0 with all25 source files,55 original
assets,9 package files,254 actual source files, JDK/officialNode and finite
source/class/javac authority chains verified. No clock/JVM/Node/query runtime.
Root fresh V30 serving evidence object f8c852ab freezes five actual Linux
performance/clock/protocol file paths+hashes for later confirmation, with root
independent verification and complete raw archive provenance; no old performance.

Fresh V30 integration plan is reviewable but UNAPPLIED: ready6bb6e5c7, patch
d2948a0cc6d7fa0fbe89923af1f35001699a5d47ac1a794a3e7e7374e20e5198,
191248 bytes,241 unchanged/8 changed/5 new (13 paths), no removals/mode changes
to existing files. Private full249 baseline plus patch yields exact final254
and8 public package/testfiles; realroot git apply --check only passes, root249/
index/native unchanged. Publication paired tails and optimized Lucene proof
still prevent integration. Rejected V28 patch and all prior losses stay intact.

### Four CPU diagnostic cells complete; optimized Lucene grid underway (2026-10-16)

All4 fixed diagnostic cells complete:19200 complete5CPU+2virtualID frames,
20728 cross-arm complete JSON/f64 pairs, original per-cell oracles/load4 PASS,
zero drops. Driver THEN exits1 because its final torn-whole-bulk adversarial test
requires64-hit query absent from deliberately batch2 fixture (StopIteration
before mutation/check); seven tests pass, one errors. Root's measure review
missed this fixture mismatch. No measurement/test rerun or false pass; original
exit/log/sources retained. Separate read-only post-stop analyzer produces
analysis34002cff; sealed raw c32eb125859fb7eba5d13983beaa8dbcc277b30411abffde0faa540194380c9b,
54,670,298 bytes,70 members/89,872,239 raw bytes. Root rehashes every sealed
small receipt, extracts bounded raw data, revalidates ALL19200 frames and
recomputes COMPLETE analyzer output exactly: root-counter-verification.json.
Four original observer receipts/input fixture were not in the archive; originals
remain remote, requested five-file fetch only after Lucene timing STOP, then
root repeats original complete result/load oracles. No lost data/reconstruction.

Measured diagnostic C/B servicep99 .6734211298/.3564101957; CPUendpointp99
.7241/.3523 (candidate32.480/28.650 microseconds versus44.858/81.322 baseline).
CPUmedians candidate16.029/15.564 versus16.264/16.364 microseconds. Responsep99
1.0845335762/.9617347898; producerlagp99 LOSSES1.1744831906/1.0780376563,
firstpair maxservice LOSS3.9123491685; every other loss stays in full analysis.
Within-cell service-tail49 elapsed-minusCPU lies chiefly in mutation-read
acquisition intervals; these include gate implementation/other waiting, not
proof of one scheduling cause. CPU-read p50 brackets .368–1.033 microseconds;
getcpu .070–.175; many lower and some upper residual bounds negative, retained
without clipping/subtraction or CPUaccuracy claim. One endpoint-ID change among
19200 does not exclude intervening migrations/physical-host effects. These two
instrumented pairs demonstrate no queryCPU deficit; they neither waive old
uninstrumented losses nor establish publication acceptance.

After all diagnostic/postprocess/transfer STOP, optimized Lucene grid launches
ONCE as session13372, releaseaefa3d80, soleREMOTE EX900/895/5; fixed sixconfigs/
24cells, no method/config/retry changes. Latest report finalfour cells, all prior
load/exactness checks pass; no final selection yet. Root authorizes next original
uninstrumented V30 full24 staging only AFTER grid+sealedtransfer+five-fetch STOP,
then separate root measure release. This full24 has NEVER runV30; it completes
the preplanned fourgroups/threepairs using original all12 pointguards, preserving
all mixed target8 losses. All8 adversarial tests use genuine5182-64-2-candidate
bulk fixture, eliminating the diagnostic's mismatch without modifying oracles.
No new production microedit is justified by the current CPU evidence. Final
Lucene11 confirmation is prepared locally during full24 and held until its
outcome/root decision; no old performance reuse or category acceptance yet.

Mutable local resource check finds load4.852/4.134/3.860 on10-core arm64/32GiB;
unrelated Python7794(100%,8.8GB), checker10465(98%) and mediaanalysisd88062(106%)
are preserved. No unqualified local competitive campaign is launched; meaningful
builds/gates/allocation/exactness calculations have already used both machines.

### V30 optimized Lucene screen qualified; original full24 underway (2026-10-16)

Lucene grid24, sealed transfer and five-file CPU-original fetch ALLSTOP0.
All24 load/exact PASS; root independently verifies all223 payload hashes,
complete same-engine JSON/member order/types/f64, allraw loads/metrics,
monotonic brackets/static8/serving same-ps Node+JVM RSS sums, owned cleanup,
24fixed20s pauses/budget and all6configurations/12pair ratios/selections:
frontier-v30-monotonic-v2/root-grid-verification.json SHAec87499ecbac9c67a00c2b2bcd031a18d9794c33dadf304088e779c77020b73d.
All12 HMS/Lucene RSS screen ratios .5832–.7255. Prospectively selected lowest
RSS1000/128MiB median239613952B and fastest100/32MiB median1878.838388QPS
are distinct; both need fresh11rounds/joint12, no inferential screen acceptance.
Raw archivea4c2f19d (20705363B), manifest1509a016 (223 payloads). Initial root
read-only verifier assumed HMS phaseevents hadPID and pause start field;
actual node-process receipts/begin fields used instead, no data/runtime rerun.

Original uninstrumented V30 full24 runs ONCE as uv200263, supervisor200262,
publication-full-v1/all.log. Latest16/24 exact/loadPASS,76800offered/0drops;
postings owns entire24/fourEXgroups+sealedtransfer. Rootrelease1f9753c8,
wrapper29de9d34/protocoleea77936; all12 original paired1.10 guards required,
all8 adversarial tests use true5182-64-2candidate. All prior target8/fullV28
losses stay explicit, no cause reassignment/filter/waiver or automatic retry.
JS prepares LOCAL independent fullraw verifier while timing runs exclusively.

Final Lucene33 planf2dc27e1 is frozen LOCALLY/UNSTAGED/UNRUN. Unchanged
confirm6cdf2c58/jointdd6e5c5a, selectedcontrols1000/128 then100/32,11rounds/
33fixed20s pauses,EX900/895/5,joint12. Root reviews complete source/driver/
analysis/schedule/17sourcepins/freshfiveV30sourceSQLite authorities; receipt
root-confirmation-source-review.json. Runtime held through full24+sealSTOP
and independent rootpublication decision. No V25/V28performance reuse.
CPU missing5 originalfiles now present; root-original-verification.json reruns
all4 original full-result/load oracles/19200queries and20728 complete crosspairs
PASS. Original diagnostic7tests/onefixtureStopIteration/exit1 and every measured
loss remain unchanged; no measurement or test rerun. Production249/native
unchanged, integration patch read-only checkPASS/UNAPPLIED; MEMORY only.

### Original V30 full24 completes; one paired publication limit fails (2026-10-16)

First original V30 full24 driver STOP0; sealed raw transfer still active. All24
load/exact PASS,115200 accepted/zero drops,310920 complete crossarm JSON/f64
pairs, original8 adversarial tests PASS. Four group exclusion/service/response
medians PASS:5182b2 .865167/.838213/.749046;5182b64 .760592/.709366/.716557;
20728b2 .734724/.884587/.830937;20728b64 .956594/.768661/.780411. Only11/12
original paired point guards PASS:20728b2r1 scheduledresponse1.131893 FAIL,
though service.487959/exclusion.705615; producerlag1.278877/write1.193436/
append1.407028 losses remain. Publication NOT ACCEPTED; no waiver/filter/noise
cause assignment or automatic rerun. completionc419a0a4/analysis8a5713c7 in
remote-v30-publication-full-v1/completed-receipts. Root independent fullraw
verification and failedpair rawtimeline analysis follow sealedtransferSTOP.
FinalLucene33 protocol/source ready but staging/runtime HELD; sourcepatch and
native install also HELD. Current dirty249/rootnative unchanged. Need actual
write/arrival-path evidence and a compatible implementation improvement before
further final performance validation; memory gains alone do not accept V30.

### V30 fullraw verification rejects integration; V31 implementation begins (2026-10-16)

Full24+postanalysis+seal/transfer ALLSTOP. Archive da37f10201799295574879beda166b4e38805fe6f24db327c890fcdfdc85626d,
651306477B;390members/366files/727095565B extracted after bounded path/type/size
checks. Root reviews frozenverifierd29a5e5b and independently produced receipt
6783fb6468ac84350039b060af71a17e5e5d65bfe1f5b5685cc8392603847199:
all24 original raworacles/resources/loads/24pauses/4EX,310920complete typed/
member-order/f64crosspairs, originalfullanalyzer/everyloss,8original plus8
independentgenuineadversarial tests PASS. The one20728b2r1 responseguard remains
1.1318933233566977 FAIL. root-completion-decision.json publication_acceptedfalse;
integration-plan/publication-rejection.json keeps patch/native/LC33 unapplied/
unrun. JS now analyzes actualfailedpair rawtimeline, no unmeasuredcausality.
Root rechecks awake afterSTOP: WindowsPID1916 live/SYSTEM request active,
remote-awake/recheck-v30-publication-full.json, permanentpolicy unchanged.

Private V31 source implementation: postings fixed-stride3-byte TermIter selected
atconstruction, genericfallback/storage unchanged,254source74fd60e5; root reviews
fullsource and authorizes focusedREMOTE SHbuild only. Metadata implements skip
of proven-unused SCANposition map and smallfusion sortedVec at combined list
bound<=11; original largeBTreefallback, +0thenidentical+=/f64order retained.
Both require meaningful boundary/fullwire tests and actualallocation/paired
component evidence before finalperformance. They do not explain historical
append/producerlag losses by assumption. No rootsource/native changed.
V30grid24/controlselections stay historical; future finalvariant must freshly
measure sourceSQLite andbothfixedLCcontrols, keeping family12/all guards and
no oldHMSperformance reuse. No calibration/source authority silently relabelled.

### V31 focused correctness passes; marginal request accounting guides component test (2026-10-16)

Root verifies actual focusedlogs/source guards/absoluteCargo identities:
termiterator REMOTE freshbaseline48/candidate49 documents tests andcandidate
lib/testsClippyPASS; allfivelayoutsizes equal (TermIter/TermIterData72,
StoredTerms8,TermData/PackedTerms40), zeroheader/stackdelta. Completioncbc938b7.
Small-query v1 failedcompile usedobsoletebincode1 test API despitepinned2;
allv1source/logs preserved. V2 corrects testAPI/import ONLY, productionbytes
identical. Full254manifest7d445e86/search7cc098b1, LOCAL49default+49security
plusalltargetClippyPASS, completion1aa775f5. Rootreceipt
v31-components-focused-root-verification.json binds actuallogs/254sources.
No CPU/RSS timing accepted; full-core allocation probe is in preparation.

Exactfailedpair fixedthreshold sensitivity from all4800rawrows: baselinep99
272637ns gives11/10limit299900.7ns.50candidate rows exceed (max48allowed), so
atleast2 needreach<=299900ns withallotherobservationsfixed.43alreadyexceed
beforeAPI,7have arithmeticservice room; this is not anengine/hostcause claim.
Marginal ranks48/49/50 seq815/257/782 have309440/308596/301633ns response,
262425/226949/273866ns service; need9540/8696/1733ns reduction andstartinside
writerpublications242/71/231, respectively. OriginalFAILunchanged. Durable
failed-pair-threshold-accounting-v1 analysis4ca616d6; all4800componentrows,
preciseoverlaps/lateproducertails remain underfailed-pair-timeline-v1.

Prospective termcomponent protocol includes ALL3 contexts71/231/242, real
20728canonicalstore/revisions/actualreplace_postings, separate original
Mutation+magicJSON, no reimplemented merge. Source/prep only; fixedwarmups/
iterations/reversed3pairs/load<3/EXbudget/allraw andfullposting+bytesoracles
mustfreeze before root runtime release. Noappendinterval/wholep99 prediction,
no observedrow alteration. Rootproduction249/native remain unchanged.

### V31 implemented and all six gates pass; component calculation released (2026-10-16)

Private combined-v31 full254 manifest19ce3802acc17ae75ae6924ba0faaab08a59346ed25fa1252e4f17310ed7bd44
combines the fixed-stride termiterator with conservative SCAN-position omission
and smallfusion sortedVec. Only packed_terms.rs/search.rs differfromV30; root
sourceverification checks all254 plus originaldirty249/native unchanged.
ALL SIX gates STOP/PASS: focused50documents/11mutation/2snapshot/1legacy,
fmt,bothdefault/allfeaturealltargetClippy-Dwarnings,locked450default/685allfeature
(oneignoredeach),Rust1.89defaultlibrary. completed.json6b8ef62c andindependent
root-verification.jsonfe8bb520 undercombined-v31-validation bindactual10logs/
Cargo identities/completeinputs. No rerun needed withoutsourcechange/gap.

FreshLinuxpublication/defaultandNode/security builds STOP0, completionda94a810
underremote-v31-build-evidence. ActualfreshnativeSHAaf18bca6(pub)/ff1d1950(Node),
254pure/256instrumented/9packageinputs,closedcache/touch205/freshcompiles,
bothClippy/runtimeNode12/baseline8/factory8/Unicode10 andactualCargoidentities
rootverified. This is build/correctness evidence only. FreshLOCALNodefb873836
alsoSTOP0with12/8/8/10PASS; independentrootreviewpending. LOCAL diagnostics-v31
fresh256-inputbinaries nowbuilding forcomplete21transactionbytes/24lifecycle
comparisons; no localcompetitiveRSS/timing accepted underunrelatedhostload.

Actual small-query fullcore allocation comparison360releasecases completes:
252reduce allocationcount/requestedtotal(up to5allocs/2148B),108equal;
retainedrequested identicalALL360. LOSSES:36scopedrequestedpeaksincrease(max80B),
12improve(max76B),312equal. FullJSON/bincodeLE/BE equalALL360; rootrecomputes
everydelta/raworacle/probeand510sourceinputs in
native-small-query-v31-v2-evidence/allocation-v1/root-verification.json.
MeasuredFusedScores type32BvsBTreeMap24B adds8B; not callerstack/RSSclaim.
Preserve_order dependencyfeature9searchtestsPASS. Allinitialv1bincodeAPIcompile
failure/probes/losses kept; requestedallocatoraccountingisnotuniversalcoverage.

Termcomponentv1buildFAIL(twoprivateClippylints) kept. V2 onlyequivalenthelper
placement/enumerate correction; bothfreshbinaries/fouractualCargoidentities/
bothClippySTOP0, completion45976ff5. Rootreviewf4ac1ad5 checksactualsource,
all3realcontexts71/231/242/completepostingandframeoracles/fixed32warmup+4096
iterations/3reversedpairs/20sarm-pauses/3EX900-895-5/load<3/allraw. Prospective
method6c0da3a0/protocolec35f0b9/controller2c5f9ca9 authorizedONCE byrootrelease
033f713e in native-term-iterator-v31/evidence/component-v2/measure-release.json.
Postings owns soleREMOTE throughcalculation+sealtransferSTOP; allotherremote
workheld. Results pending; no historicalappendinterval/wholep99prediction.

FirstV31full24publication preparedSOURCEONLY inremote-v31-publication-full-v2,
driver6ab39d3b/protocol2a00066f. ExternalreleaseSHA andallsharedsources verified
beforeprojectimports; original24schedule/all12guards/oracles/numerics unchanged.
Frozenunrunv1 preserved. RuntimehelduntilcomponentSTOP/rootdecision. Future
V31freshsourceSQLite33andbothfixedLucenecontrols33 mustusefreshHMS evidence,
historicalV30grid/controlselections unchanged,family12/allguardssame. V30full24
remainsREJECTED; rootproduction1.091961RSS/SQLiteunchanged andMEMORYunestablished.

### V31 durability proof completes; actual iterator work decreases (2026-10-16)

FreshLOCALNode31 independentlyverified f1caee8b (nativefb873836): actual254both
sourcecapsules/9package/Cargoidentity/freshcompile/12Node+8baseline+8factory+
10Unicode-framing/root249/native exact. Descriptive inherited concurrentgates
fieldtrue wasincorrect; append-onlyreporter-correction.json preservesoldready
andrecordsactualfalse; no source/native/results changed. D31 freshbuild81592,
runtime54688andbothfullrawchecks ALLSTOP0,completed4ab4f022. All21complete
transactionbytepairs and24lifecycle/124800calls exactJSON/memberorder/types/
f64/decryptedframes PASS,includingplain/encryptedrecovery/retry/heldsnapshots.
Root independently runscomplete rawverificationadb30f31 thenchecksactualall4
source/binaryselections/twofreshcompiles/bothClippy/baselinereuse andallnumeric
counterlosses inroot-diagnostic-review.jsonb3f82365. Serializationall21counters
equal; postqueryrequested saves8,879,871–8,886,487B, butall12pairedblockcounts
LOSE3542–3544. All12candidatewarmupsLOSE7172blocks/+729144requested/+794672usable;
both100pendingArccontextsLOSE400blocks/+47979requested/+58928usable. Preserve
alllosses, no physicalRSS/universalpeak/timing claim. AllLOCALruntime STOP.

REMOTE actualtermcomponent6arms STOP0: all3contexts/fullpostingandtransaction
frameoracles,6loadguards,3EXbudgetsPASS. MedianC/B innermeans replacement
rev71/231/242 .760453/.743233/.726481, serialization.728112/.686919/.664693;
all9meanpairseachcomponent improve. Rev71serializationinnermaxmedian1.238019
LOSS andallraw/statlosses retained. analysis31871c85/measurementsb016eec0.
Compression+sealedtransferSTILLACTIVE underpostingssoleownership; rootactual
rawverification pending. Descriptive3-pair component evidence only; neither
historicalappendintervalcausality nor wholepublicationp99improvement established.

ProspectiveV31 source-onlyintegrationpatch123e9690across8changed/5newpaths
privatelyappliesdirty249→exact254,241unchanged; actualrootread-onlyapplycheck
PASS/index/native/sourceunchanged. Nativepreservation+atomicinstallplan only,
no rootapply/nativearchive/replacement yet. Finalpublication/fullfreshserving/
bothLuceneconfirmations remainrequired; oldV30rejectedpatch prohibited.

### V31 component independently verified; first original full24 runs (2026-10-16)

Component+sealing+transfer ALLSTOP0. Archive21bfb8d9f66d9ebe5e54087e9ffbc99a63d650851ed3091e1fd87ddabad25d3a,
252580524B/190members/876421577Braw; rootchecksall190payloadSHAs/tarregularunique
relativeheaders/sizes/seala0d52d4e/transferSTOP76686fff. Root independently
recomputes147456rawintervals/all18component-context-pairs/means/medians/p99/max/
outermeans/all9pairedratios,completeframesand2,409,776-entryposting snapshots
eachrevision,small2+1shapes/12loads/6fixedpauses/3EXbudgets in
component-v2/root-measurement-verification.jsonc54607c2. Max1minload.91504;
actualEXUnixspans158.395/160.038/159.150sec <900. Bothcomponents improveall9mean
pairs; eightpointLOSSES kept,includingrev71serializationr1p991.480696/max11.302293
andmedianmax1.238019. Three-pairdescriptiveCPUmechanismonly,noappend/wholep99claim.

Rootrechecksawake/nojobsonidleREMOTE: WindowsPID1916/SYSTEMrequestlive;
remote-awake/recheck-v31-components.json. FirstV31full24source/fullrawverifier
reviewed; verifier72680ccc derivedfromactuald29a5e5b withunchangednumerical/
oraclelogic andexternalrelease/protocol/archivepins. Initialrootstaticreview
typedunverifiedfulloriginSHA; assertionrejectedguess,primaryfilehashcomparison
correctedandfailurekept; nosource/method/measurementchange. StagingSTOP5afb60b3,
all21files/249+251+254+256sources/bothprobes/sharedassetsPASS/nojobs/load.0093.

Rootpositivefirstfull24releasea73ad144a0adc7cf5543b8f19bec83c2d07eedb688b3445ce86ed5f7d0afad89
binds12actualproofs/D31/gates/Node/componentalllosses/STOP/staging. Original
V31full24 RUNNING ONCE uv219170/supervisor219169, publication-full-v2/all.log;
postings ownssoleREMOTEthroughpostanalysis+sealedtransfer. Originalall24/all12
pairedpointguards/zeroadditionaldrops/8genuinebulkfixtures/freshseeds/fixedpauses/
4EX unchanged. Conveniencepreflight firstmatchedownPythoncommandliterals;
launcher-onlyfailure preserved,ownPIDexcludedbeforefirstseed/cell. No measured
retry/driver/protocol/sourcechange. Publicationacceptanceremainspending.

NewS31sourceSQLite33 andbothLC33bridges frozenSOURCEONLY/UNSTAGED/UNRUN,
source-readye586cf0a/489b5aa1. Allnumericmethodsbodies unchanged,preimport
externalSHA/sourceguardbootstrap,pure24PASS; initialfixturealiasfailurekept.
ActualpositiveV31scope/protocol/plan/releases remainABSENT untilpublication
accepted; LCadditionallyrequiresfreshqualifiedV31sourceSQLite evidence. Both
historicalgridcontrols1000/128and100/32remainfixed,family12/quality/sixnumeric
guardssame; neitheroldHMSperformance noroldsourceauthority relabelled. Root
production249/nativeb8unchanged,1.091961RSS/SQLite remainsacceptedproduction.

Component-clock clarification: all reported iterator component intervals are
std::time::Instant elapsed times, not thread-CPU counter observations. The
reduced decoding work is inferred from the code; all wall-time losses remain.

### First V31 full24 has a paired latency failure; integration stays held (2026-10-16)

First originalV31full24 driverSTOP0:all24exact/load,310920completeJSON/f64
crosspairs,originaloracle8andall4EXbudgetsPASS. Allgroupnumericguardspassbut
paired_point_guards_metFALSE; publication NOT ACCEPTED. Failedpairdetails,
sealedrawtransferandindependentrootverification pending; postingsownssoleREMOTE
throughtransferSTOP. No freshS31/LC staging/runtime orsource/nativeintegration;
no measuredretry/filter/waiver/schedulingcauseclaim. Actualtermcomponentelapsed
improvements remain measured but do notestablishend-to-endpublicationbehavior.

Source-onlyinstallerprepared separatelyataccepted-v31-integration-prep,
source-ready536ec0d3:existing13-pathpatch/nativeidentity,externallypinnedfinal
acceptance,original+candidatearchives/fsync/atomicnative/rootNodechecks/recovery.
Privatefixture262a56dc passesactual249→254/native,automatic/manualrollback,
unexpected-selected-edit refusal andunrelateddirtyworkpreservation; actualroot
249/native/indexunchanged,actualaccepted-v31-integration-v1absent. Installerand
allS31/LCdisabledtemplates stayHELD; this rejectedV31 hasnointegrationacceptance.

### V31 fullraw rejection is final; V32 targeted publication check begins (2026-10-16)

All V31 full24 runtime, postanalysis, archive transfer and root verification are
stopped successfully. The independent root proof57486c74 verifies445 payloads
(archive447a3103,648691543 bytes), all24 exact/resource/load outcomes,310920
complete typed/member-order/f64 crosspairs, original+independent8 adversarial,
24 pauses/4EX and all numeric losses. Only9/12 paired guards pass:5182/b2/r2
service1.1265744006 and scheduled response1.5532074182;20728/b2/r2 response
1.7156220409;20728/b64/r0 exclusion1.1755872101. All four group medians passing
does not waive those losses. E/remote-v31-publication-full-v2/root-completion-
decision.json7e627dfd and combined-v31-evidence/integration-plan/publication-
rejection.json37e0e777 prohibit V31 source/native installation and fresh serving/
Lucene campaigns; production249/nativeb8 and its1.091961 RSS/SQLite remain unchanged.

Saved-trace accounting finds two separate problems. All28,800 query rows and
1,152 commits from the three failed pairs are preserved in failed-pair-timelines-
v1; findingsdcf805ec, ready24565c01. Root independently verifies58 input/authority
files,27 outputs, all raw query timings and commit bounds. In5182/b2/r2,62 responses
exceed270351.4ns and57 pre-API intervals alone exceed it; even zero API service
leaves p99=353411ns. In20728/b2/r2,69 exceed345018.3ns and64 pre-API intervals alone
exceed it; zero-service p99=524491ns. At least9/16 pre-API violations respectively
must change with the rest fixed. These are arithmetic limits, not causal or
scheduler claims. The small-cell service failure has51 requests above37117.3ns
(48 allowed),23 with exclusion overlap and28 without; no rank/cause inference.

Bulk failing revision51 exclusion7935477ns comprises publication work7882012ns
and release53465ns; baseline p99=6750224ns and fixed limit7425246.4ns. Exactly1/64
candidate commits exceeds that limit. It has no whole-compaction/final-maintenance
overlap. The separate revision14 writer wait13158729ns overlaps final maintenance
for13150257ns; candidate final lease48963211ns vs baseline26627759ns. Absolute
intervals and all64 phase deltas are retained, without assigning CPU/block causes.

The revised workflow screens one implementation after focused correctness before
another full campaign. V32 source254 manifest3239def3 changes only mutation.rs
88ca6ebd: a one-chunk ID equality branch avoids constructing a BTreeSet during
publication; other counts use the exact prior path. Two prepared tests cover
full UTF8 domains/byte results, prior chunks/held snapshots, append failure/retry,
and plain/encrypted recovery. Performance benefit is still a hypothesis. Focused
REMOTE SH/jobs2 build uv231654/supervisor231651 is running under R/single-chunk-
publication-v32/focused-v1; default mutation13 passed, remaining checks pending.
The current host has22 unrelated rule30 CPU workers/load22.22: preserve them;
no competitive timing while occupied. A fixed six-run paired V32 screen is being
prepared SOURCE ONLY; no timing/full gates/Node/serving/Lucene campaign released.

V32 focused validation update: default/security each13mutation+2snapshot+1legacy,
fmt and both release lib/testsClippy-Dwarnings PASS; all jobs/sealed transfer STOP0.
Completionf1fd4521/root-verification3f65598b verifies67 receipts, both fresh actual
compiles/eight package identities and254 source inputs; production249/native
unchanged. New awake proofdfb216de confirms Windows1916/SYSTEM sleep request at
remote load22.37. Six-run screen ready6d618f9c remains source-only/unbound/unrun,
fixed B/C,C/B,B/C, no timing release while occupied. Actual allocation-event proof
is being implemented/run locally, without timing/RSS claims. No full campaign.

Source-backed next hypothesis, not implemented or measured: mutation validation
calls normalize_embedding and discards its Vec (combined-v31 mutation.rs157).
The normalizer in documents.rs285 checks length/finite values/positive scale,
then computes norm and allocates a normalized vector. Normal commit validates
before append and apply_mutations validates again during publication; for each
embedded chunk this predicts two discarded normalized vectors in those validation
calls, separate from required preparation/query normalization. A later candidate
could share the exact fallible checks without materializing validation output;
all normalization bytes and invalid-input errors must remain identical. This is
not assigned as a cause of the recorded failures and does not replace the fixed
V32 screen. Do not start another broad validation campaign on this hypothesis.

Default-only publication probe build was separately root-reviewed/released:
root-source-review1fa9c3e5/release3a7c02ee, driver619cde3a/protocol34278e9e;
all254 pure+256 instrumented inputs/archives and original assets verified.
Running uv235236/supervisor235235 at R/single-chunk-publication-v32/default-
publication-build-v1,SH/jobs2; launch load22.04/13.5GB available, unrelated work
preserved. This is compilation/Clippy only, no performance/Node/full campaign.
Local actual-allocation proof initial V1 compile failed fixture sum type inference;
V2 stopped before Cargo on a correctly rejected unformatted-helper identity.
Both iterations and full errors are retained; V3 freezes the formatted helper
before capsule creation. No V1/V2 counter or semantic measurement results exist.

### Highest-value publication implementation priority (2026-10-16)

User reports rule30 expects roughly45min more and requests continued implementation,
prioritizing highest value toward general superiority. Treat that as availability
guidance, not measured idle state or a completion promise; retain MEMORY first.
Two independent V31-based SOURCEONLY candidates now target recurring publication
work: V33 shares the exact embedding validation/scale checks while avoiding unused
normalized Vec construction; V34 retains unchanged secondary metadata/source
memberships during replacement rather than rebuilding every filter entry.
No claimed timing/retained-memory gain or failure causality; each needs source
review, focused correctness and prospective paired screening before full gates.

V32 probe build is complete: native4f23af1a/completion486eff86; root proof01a048c6
verifies all35 receipt members, actual ELF/fresh compile/package identities,
254/256 source inputs, fmt/Clippy and original baseline reuse guards. All remote
jobs/sealing stopped. Keep V32screen unbound/unrun as an archived fallback.
Its lower-value allocation probe was stopped: V1 fixture sum inference, V2 helper
format guard and V3 fixture k4096 (APImax1000) errors all preserved; V3 sentinel
passed but the24-case campaign did not. V4 had just started before root steering
and was stopped. Do not claim any paired allocation savings from this probe.

V33 source freezeffcd966c now exists:254inputs, twochangedfiles,252exactV31;
root-source-review4ca2c59f confirms exact normalizer arithmetic/error checks.
Two new regressions prepared for49embedding domain/bit/error cases and mutation
validation/rawMAX/-0/retry/plain+encrypted recovery; not yet measured savings.
LOCAL focused runnerV1 stopped before source extraction/Cargo on7 unrelated idle
editor Node processes; no source/test failure, all artifacts retained. Root-reviewed
versioned controllerV2 eac380d2 removes generic Node rejection for correctness-only
SH builds; private manifests/targets/fresh compiles and benchmark exclusion remain.
V33plan5be640f3/release4b8b7e0d authorized JS execution; no full gates or performance.
V34 metadata filter source and prospective disabled V35 six-run screen are being
prepared independently; V35 composition will require both focused validations.

V33 focused default checks now pass51documents/12mutation/2snapshot/1legacy,
including both new regressions and lib/testsClippy-Dwarnings; security pending.
V34 firstsource32cdf605 is preserved but not released for build: root inspection
found four scalar_key canonical String constructions for each unchanged field
versus two in legacy, plus retained old key cloning. This is a source-operation
finding, not measured regression. Requested a smaller-allocation exact primitive/
integer comparison backed by existing serde_json equality/display semantics;
retain both-phase ordering and missing-index invariants. V34 revision is in progress,
no performance/functional acceptance inferred from the firstsource.

V33 focused default/security ALLSTOP/PASS132 (each51/12/2/1), fmt and bothClippy;
completion47c63625/rootdd497801 independently verifiesactualtests, four absolute
Cargo+targetidentities, freshbinaries,254inputs and production249/nativeunchanged.
V34V2e5905d21/index5db26748 retains exact invariants with allocation-free scalar
comparison. Rootb6428a96 verifies all254/253unchanged and pinned serde_json1.0.151
NumberEq/Display source6d6f4475: guardedinteger equality implies identical original
index key under normal and arbitrary_precision representation. Staticcanonical
formatcounts nowunchanged1/changed2; not measured allocations or latency.
REMOTEfocusedPID237548/supervisor237547 plan1992bcaa/release007578b0 is running
54/11/2/1 eachfeature,SH/jobs2. Initial local deployment selector failedbefore
remote writes and is retained; fixeddeploymentselectsreleasedrunnerexactly.
LOCAL combined preserve_order+arbitrary_precision check separately authorized,
with no source/lock edit, Node/fullgates/performance/counterwork. ProspectiveV35
screenready a56c6864 staysdisabled/unbound; defaultprobe templates being prepared.

V34V2 focusedcompileSTOP1 before runtime/security on untypedfixture revision
range at is_multiple_of (E0689). Full273failurepayload archiveb757c47c/STOPd743396c
preserved. V3cf41841f changes only the test range to1usize..=5; production exactV2,
rootreview2821dcab independently verifies254 inputs/single-token delta. REMOTE V3
PID238365/supervisor238364 is running under newsharedV3paths, plan29920363/release
4de4350f,samecontroller eac380d2,SH/jobs2;fmtPASS, launchload22.02/13.62GB. JS
combined-feature check was stopped before V2Cargo and is authorized onlyV3 locally.
No V34performance or functional acceptance inferred; no duplication of known error.


### V35 composition and quiet-window publication screen (2026-10-16)

V33 focused132 and V34V3 focused136 tests passed under both default/security,
with fmt/Clippy and all jobs stopped. Root proofs dd497801/c0b779d1 independently
verify actual source, fresh binaries, package/target identities and test names.
The V34 combined serde_json preserve_order+arbitrary_precision check also passed
all4 filter/7 metadata regressions; root5cd22731 verifies actual254source/fresh
binary, exact dependency features, identities and logs. Initial root verification
used an absent timeout receipt field; its read-only failure is preserved, followed
by schema inspection and successful actual-field verification.

V35 pure25473f1b401 includes only V33 documents.rs/mutation.rs and V34V3 index.rs
relative to V31. Composition receiptcca1a99a binds both focused root proofs;
V32 is excluded. Default-only remote publication probe build completed7fc01782,
nativebb84fa07, instrumented256manifestd62c206e. Root4fcae3cc independently
verified all43 sealed receipts, actual selected source hashes, native bytes,
fresh Cargo compile/identities/targets and fmt/Clippy. Build/transfer stopped.

After user reported writerslogic-02 idle, actual preflight load was0.0693.
One prospective six-run paired screen is now root-released: source-ready a56c6864,
driver2766b661, actualprotocol9a6a4068, releasef94d1c74, source reviewaa61bedb.
At this checkpoint staging/runtime are pending, owned solely by postings;
all other remote work remains held. Exact original workload/oracles/load/exclusive
rules and all publication limits remain unchanged. No full24, Node, serving,
Lucene or integration release. No latency/allocation/RSS saving inferred from
source-operation reductions. Production baseline/native remain unchanged and
accepted serving RSS remains1.091961 SQLite. MEMORY/overall not established.


V35 screen actually launched: supervisor241270, stage33files verified,
load0.1504/free15.25GB and no competing runtime/build. Canonical exclusive lock
and all fixed guards remain active. No result/acceptance yet. Root also prepared
fresh V35 source.patchaf1e3220 against actual dirty249baseline, privately applied
and verified every254selectedinput (root0ba9ca58). This does not modify production
or authorize integration; old V31 installer/patch/native remain prohibited.


V35 six-run screen completed and sealed ALLSTOP0: six load gates and three
paired primary point guards pass, original8/root8 adversarial tests pass,
28800 accepted/0 drops,93276 complete crossarm result pairs exact including
member order/type/f64 bits. Archive36adda31 contains158 verified regular files;
completionc3075e64/analysis32810e9a. Independent root12b25111 recomputed every
raw cell oracle/resource/load/ratio/loss, fixed pauses and exclusive lock budget.
C/B p99(exclusion,service,response):5182b2(.844398,.940977,1.069663),
20728b2(.775012,.507322,1.049199),20728b64(.605319,.620468,.623391).
ALL losses remain, including firstpairappendp991.537322, allthree producerlag
p991.487295/1.426238/1.666664, smallpair response+6.97%/+4.92% and every
p50/max/compaction/lease loss. This is one pair/group, no final preservation or
memory superiority and no V33/V34 isolated causal inference. Promising screen
permits preparing fresh fullsix/Node/D35/full24; newruntime release still pending.


V35 acceptance execution now released after independent screen proof12b25111.
REMOTE originalfull24 supervisor244581 at publication-cost-v35-full-v1 launched
load0.0083/free15.23GB after25stagedfilesverified; source-readyffe55ab3,
rootbedca3d0, protocoldab381c6/releaseef4fda37. Original shared24main/analyzer/
11assets/numericlimits byte-exact; strong default-build/source/native/composition
guards reused from screen; no pointless Node-build dependency. Postings soleREMOTE.
LOCAL JS runs Rustfourfocused+sixgates72457 and freshNode28421, SH/jobs2each.
Validation1bae7f8c/protocol556d67a2/rootreleases ef88517a/857c0292. Initial Node
prep archived, then four lines explicitly isolated Cargo intermediate outputs
and guard actualmetadata build_directory. No source change. Results pending;
no integration/serving release. D35source review5c2dd4ba verifies15exacthelpers
and minimal build/binding changes; needs actualfreshNode and two diagnosticbuilds.


LOCAL V35 Node28421 STOP0/PASS: nativee62830d6, ready8faf519a, packagee6146235.
Root56f15db7 independently verifies254inputs in bothcapsules/package, actual
Cargo package/target/build_directory and freshcompile, actual Node12/baseline8/
factory8/Unicode10 plus builtin-poison regression. No performance inference.
ROOT D35diagnosticbuild62391 now runs under release13af52f1,SH/jobs2, own private
sources/outputs and log diagnostics-v35/build-launch-v1.log. Runtime remainsheld
until bothactualbuilds/controlproof and allLOCALbuildersSTOP. Rust72457active;
REMOTEfull24244581active (last7/24 exact/loadPASS0drops). FutureS35/LC35sourceonly.
V35 rootfullrawverifier8a915063 preserves original72680ccc verificationbody,
changing only source/build/defaultnative/composition/screen identities/history;
derivation1b3003c1 reviewed. It remainsUNRUN until actualsealed full24STOP.


### User-requested competitor comparison (2026-10-16; MEMORY remains first)

M = measured within the named experiment; E = qualitative estimate, low confidence;
V = vendor-native result under unmatched conditions. This covers the tracked
competitor set, not every product on the market. Private candidates are not accepted
production. V35 passed its cheap publication screen but its full24 publication campaign
was REJECTED8/12; fresh serving comparisons remain unrun. No other-category evaluation started.

| Category | HMS | Competitor/control | Position and confidence |
|---|---|---|---|
| Accepted serving memory | M RSS1.091961× SQLite | SQLite1.000× | HMS9.2% worse; superiority unestablished. |
| Private serving memory | M V30 paired RSS0.913276× SQLite | SQLite1.000× | Historical private8.7% saving; publication rejected, not integrated, not V35 evidence. |
| Private serving memory | M V30 RSS roughly0.60–0.65× selected Lucene | Lucene9.12.3, two fixed controls | Screening35–40% saving; final33-pair confirmations unrun. |
| Prepared-token lexical speed | M V30 QPS4.76198× SQLite | SQLite3.53.4 | Historical private win; current candidate unmeasured. |
| Prepared-token lexical speed | M V30 QPS2.15–2.24× selected Lucene | Two selected Lucene controls | Screening win; current candidate and confirmations unmeasured. |
| Lexical relevance, NFCorpus324 | M nDCG@10 .291582 | SQLite .291410; Lucene .290265 | Essentially tied; Lucene quality-equivalence interval passes ±.005, not a relevance-superiority claim. |
| Learned sparse relevance | Unmeasured matched HMS result | SPLADEv3 | E likely stronger competitor relevance with added model/inference cost; no defensible numeric gap. |
| Structured/associative memory | Explicit symbolic chains; experimental holographic compression loses development accuracy to flat controls | Flat/exact controls | M compression tradeoff, no matched product advantage; E retain exact controls as accuracy leaders. |
| Long-term QA | M reused development87/100 | V Zep451/500; V Mem0managedv3 top50 474/500 | Different models/judges/data; not comparable accuracy. E HMS may trail, low confidence; matched margin unknown. |
| Durable updates/publication | V35 screen3/3 passed; full24 REJECTED8/12 | Exact dirty baseline; no matched SQLite/Lucene maintenance comparison | No accepted comparative superiority; V31 failed3/12 original pairs. |
| Production maturity/operations | Local transactions, snapshots/recovery; bounded regression evidence | SQLite/Lucene ecosystems | E HMS behind maturity/coverage; no numeric matched operational score. |
| Security/privacy | Local processing and scoped encryption-at-rest/signing support | Comparative security not evaluated | No superiority estimate justified. No encrypted search, PoSME or RATS verification feature claim. |

Historical ANN below is the separate October11–12 virtualized GCE C4 full-GloVe
diagnostic,1,183,514×100 vectors,10,000queries, five query repetitions but one build.
These are fastest sampled points at recall≥95%, with differing achieved recall;
not a current V35, bare-metal or accepted original-campaign result. Index MiB is
serialized storage, not serving RSS. Peak RSS includes runtime/inputs/build history
and omits separate CLI-child RSS; do not compare it to prepared-token serving RSS.

| ANN method | Recall | QPS | Index MiB | Build seconds | Peak RSS MiB | HMS comparison |
|---|---:|---:|---:|---:|---:|---|
| HMS | .9575 | 1023 | 392.29 | 1435.41 | 4316.50 | Reference, M. |
| FAISS HNSW | .9532 | 965 | 758.64 | 585.61 | 1293.79 | HMS smaller index; slower build/higher peak. |
| hnswlib | .9539 | 984 | 763.28 | 665.58 | 1377.54 | HMS smaller index; slower build/higher peak. |
| RaBitQ IVF | .9540 | 1699 | 190.55 | 86.35 | 724.29 | HMS loses speed/index/build/peak. |
| ScaNN | .9579 | 1288 | 513.24 | 74.07 | 1523.14 | HMS smaller index, slower queries/build/higher peak. |
| SymphonyQG | .9607 | 1240 | 1462.78 | 374.43 | 5629.27 | HMS smaller index/lower peak, slower queries/build. |
| DiskANN Vamana | .9555 | 534 | 1051.19 | 721.82 | 1287.12 | HMS faster queries/smaller index, slower build/higher peak. |
| Glass | .9610 | 1157 | Unknown | 489.11 | 1338.04 | HMS slower queries/build/higher peak; storage unknown. |
| FAISS IVF-PQm25 | Target missed | — | 37.74 | 46.90 | 639.32 | Does not qualify at95%; no matched-speed conclusion. |
| FAISS IVF-PQm100 | Invalid config | — | — | — | — | Excluded; no performance conclusion. |
| FAISS Flat | Untested | E slower exact scan | Calculated451.5 rawf32 payload before index overhead | Unknown | Unknown | Raw payload is arithmetic, not measured index/RSS. |
| Lucene HNSW | Timeout | — | — | — | — | No accepted conclusion. |
| NGT ONNG | Timeout | — | — | — | — | No accepted conclusion. |

The tighter95.0–95.5% recall band gives HMS877, FAISSHNSW965, hnswlib984,
RaBitQ1699QPS. HMS has no sampled99% point; the other three achieve235/270/530QPS.
Thus even historical ANN does not establish HMS superiority over all competitors.
Sources: existing `ann_extended_diagnostic_v1/report.md`, frozen
`docs/COMPETITORS.md` vendor/source pins, V30 serving root1b3bf62d and optimized
Lucene screening rootec87499e. Estimates are explicit, with no invented numeric
results for unmatched cells. MEMORY implementations and acceptance remain priority.


### V35 full24 rejection verified; query service implementation begins (2026-10-16)

Full24+seal/transfer ALLSTOP0. Raw449archivee79b7d1c, completionbad4b75d;
independent rootb23f8db6 rehashesall449actualpayloads/recomputesall24raworacles,
resource/load/metrics/losses,310920 completeJSON/memberorder/type/f64pairs,
24pauses/fourEXbudgets and original8+root8adversarial checks. All24load/exactPASS,
115200accepted0drops. Publication REJECTED8/12paired and2/4group guards:
5182b2r0 service1.337609,response1.160677;r2 service1.391617;
20728b2r0 service1.506756;r1 service1.122456. Smallgroup service medians1.197267/
1.209573; all exclusions/bulkpairsPASS. Every metric/p50/max/append/queue/lag/
lease/compaction/RSS loss remains in frozenanalysis and numeric-summary b99d4cf1.
Rootdecisionfce44ea8 prohibits V35patch/nativeintegration, LinuxNode35,
S35/LC35 runtime, outcome retries/waivers/filtering. Accepted productionunchanged,
RSS1.091961SQLite; MEMORY/overall unestablished. Cheap screen success did not
predict full24smallservice preservation and is not relabeled as acceptance.

V35 correctness nevertheless passed: completed8ef3943f/root101fad6b verifies
alltenactualRustcommands/logs/identities/source254,456default/691allfeatures
(oneignored each), fmt/bothClippy/MSRV1.89 and required newregressions. PureNode
root56f15db7 remainsPASS. D35builds alsoSTOP0/readya8992e2c/roota88bd053 verifies
twoactualfreshcandidatebinaries a2df88c0/27403aaf, exactbaseline reuse,256/251
instrumentedinputs,Clippy/Node12/allocator4095/4096/4097 controls. Serializer21/
lifecycle24 were NOT RUN afterpublicationrejection; no D35allocation/RSS gain.

New highvalue source-only implementations target heap work in smallquery service:
V36(metadata) shares exact existing tokenizer with a query-only sorted keyset,
inlineempty/oneString and immediateBTreeSetpromotionatsecondDISTINCTkey. Stored
termfrequency increments unchanged; repeatedtokens nevergrowanintermediateVec.
V37(JS) removes scores/rank/fusion heap scratch together for<=11chunks, retaining
exact arithmetic/selection/order and whole >11fallback. Each needs new differential
fullJSON/bincode/f64/domain/held tests, rootreview/focusedchecks, and prospective
small12screen beforefullgates. No savings or latency causality inferred yet.
Postings prepares minimal originalsmallgroups screen with allpoint/load/work
limits unchanged; seven applicable originaladversarialtests, bulk-only eighth
explicitly deferredto requiredfinalfull24, nofakefixture. Futurecompleteinput
selection includesnewmodules and2COSE, exactcount notyetfrozen. AllREMOTE work
stopped and awake1916 preserved; no productionedits/commits/push/publication.


V36 source255ef6abcfb/206Rust frozen, rootec8778f0 verifiesallselectedinputs,
fullsharedtokenizer/frequency/orderedkeyiterator and four legacy fullwire/held/
dense/8MiB regressions. V37 pure254c6bf7891/205Rust frozen, root60eade4f verifies
onlysearch.rs and entireoriginal pipelinebyte-exact apartfrom <=11dispatch/new
helper/three substantive boundarytests. Sum identities/order remain literal.
No actual allocation/RSS/latency savings claimed, stack/header costs unmeasured.
REMOTEfocused V36supervisor257727 and LOCALfocused V37session16482 nowrun,
SH/jobs2/disjoint private sources/targets, exactsharedrunnercd55fe24 onlyallowlist
changefromclosedeac380d2. Plans/releases526aa16f/80231250 and2a291dcc/bff032f0;
requireddefault+security docs59/58,mutation12,snapshot2,legacy1+fmt/libtestsClippy.
V36 outergenericPythonworkerpreflight and missingprivateparenttransfer failures
were preserved beforeCargo; unchangedreleasedrunner actuallylaunchedonce. The22
unrelatedworkers had exited byactualremote preflightload0.2368/free15.23GB;
no cause assigned, awake1916 preserved. LOCALcachepreparer0ac759a8 fixesonly
cacheoriginselectiontoactualclosedV33 targets, preservinggenericoriginal.

Prospective small12template4b313073/root776e09b6 is source-reviewed, unchanged
originaltimedgroupbody/arithmetic/work/limits with two selectedb2groups/3pairs.
Original7applicableadversarialnames explicit; originalbulk-only eighth remains
mandatoryonfinalfull24. No fakebulkfixture/weakening/finalacceptance. Future
V38 likely255/206pure and257/208instrumented, not earlier256/207guess; actual
manifest/composition/defaultpubbuild/native stillunbound. Metadata prepares a
reused wholequery360case allocationproof sourceonly, V35control/newcandidate
pinsdisclosed, no counterrun/CPU/RSSclaim. No failedV35 serving/LC/Node rerun.


### V38 composed after focused correctness; new publication probe built (2026-10-16)

V36 focused148/root64d1c878 and V37 focused146/root199667fc ALLSTOP/PASS;
root independently verifiesactualtests/identities/freshbinaries/source/manifests,
including321V36archivepayloads. One roothelper artifact-basename guess failed
read-only; originalhelper/failure preserved, resolution usesuniqueverifiedarchive
payloadSHA. No source/testchange. Root V38 pure255/2068f770233 composed byexact
V36copy +exactV37searchpatch; onlythreefiles changeV35, receipt9b89d5f4 binds
bothrootfocusedproofs. LOCALcomposedfocused70865 nowtests bothmechanismstogether,
62docs12mut2snap1legacy/all7newnames perarm, defaultPASS/securitypending; no Node/
fullsix/counter/performance. Actualplansf2763349/release4d5ee455 kept.

Default REMOTEV38pubprobe ALLSTOP/PASS, completionf0f93157/native6c3003ab,
pub257manifest976e28a6. Rootd4f658d8 independently rehashes42sealedreceipts,
actualpure255/pub257sources, freshCargo/package/isolatedtarget+builddir,
fmt/Clippy andSTOP. Rootsourcea97168ef reviewedfull dynamiccounts derivation;
actualprotocol da8abb7f/release386d8460/source-readybe81ffa0. All206/208Rust
inputs touched; baseline249/251 andclosedB35cache readonly, no baselinecompile.
Small12 method/root776e09b6 reviewed, but actualprotocol/release/staging/runtime
remainABSENT untilcomposedfocusedproof and freshidleeligibility. No saving claimed.

A meaningful concurrent-state boundary appeared: finalcomposition root249guard
foundnew unrelated qgraph/mod.rs andvertex.rs edits. Composition itself succeeded
privately; failedfinalassertion and exactold/newhashes arepreserved in
combined-v38-evidence/concurrent-root-change.json. Those rootfiles are preserved,
notourmemoryedits, originalnativeb8 unchanged. Never recapsule/contaminate frozen
comparisonarms or claimcurrentroot249unchanged. Finalproductionintegration must
preserveandvalidate/rebuildactualrootwiththosechanges; nonecurrentlyauthorized.
Oldroot249unchangedproofs remainhistoricaltruewhenverified. Source-onlyallocation
method nowprospectivelycoversoriginal360 +84additionalpairs atactual0/1/2/10/11/
12/48chunks ×one/sixnormalizedkeys×blend/RRF×3, allcases/losses retained; no
counter/CPU/RSSmeasurement yet. Itsrootguard mustrecordactualboundedlaunchsnapshot
andconcurrentdelta withoutskippingfiles. MEMORY andoverallremainunestablished.

### V38 composed correctness passed; prospective small12 released (2026-10-16)

Composed V38 passed 154 focused tests: default and security each ran 62
document, 12 mutation, two snapshot and one legacy test. All seven new test
names ran in both features; fmt and both library/tests Clippy checks passed.
All owned local jobs stopped. Completion `4e31181d` and root independent proof
`9eb240a2` bind the actual 255-input/206-Rust source, four Cargo identities and
fresh binaries. Full six gates and fresh Node validation remain pending.

Root bound the fixed small12 protocol `5cd9fb8a` and positive release `7c8a65aa`
to actual V38 publication completion `f0f93157`, native `6c3003ab`, composition
`9b89d5f4`, build proof `d4f658d8` and composed correctness proof `9eb240a2`.
Postings is the sole remote owner for staging and one prospective run at
`R/query-service-repair-v38-small12-v1`. Eligibility must be checked at launch.
Two fixed batch2 groups, three reversed pairs per group, original workloads,
EX900 budgets, load threshold and all 1.10 preservation guards remain intact.
This screen cannot establish final publication or memory acceptance; original
full24 remains mandatory. The seven applicable original adversarial methods
also passed a root method check against genuine saved V35 batch2 data; this
is method verification, not a new candidate measurement.

The independent whole-query allocation proof is source-only: original 360
cases plus 84 prospectively fixed chunk/key boundary cases. Root review found
the prepared driver used SH for both stages; measurement must use EX before
release. No counter values or new CPU/RSS savings are claimed. Root production
memory sources remain untouched; unrelated qgraph edits are preserved.

V38 small12 actually launched once, remote supervisor `265749`, with no
competing jobs, EX availability and load 0.2266. An earlier staging guard
preserved a transient unrelated Cargo/rustc failure before any cells; those
PIDs exited before bounded ownership inspection, so ownership is unknown.
No unrelated process was touched and no measured cells were retried.

Local whole-query allocation build launched under SH/jobs2, metadata session
`78951`, using reviewed revision2 source-ready `83ad11e3`, root review
`ddeec837`, plan `12f8c3a1` and positive build release `87eb309c`. The original
source-only revision is preserved verbatim; measurement now requires EX and
a separate release binding actual fresh binaries. An unrelated local stable
Cargo build remains preserved. No allocation counter run has occurred.

Allocation builds completed with both fresh private compiles exiting zero:
build completion `0f45986d`, baseline probe `9c6a53b0`, V38 probe `8a65754c`.
The controller exited one only at its final shared-root preservation assertion:
other work changed `src/core/engine/mod.rs`, `src/core/qgraph/kernels.rs` and
`src/core/qgraph/vertex.rs` during the build. Exact before/after hashes, original
failure and all edits are preserved; original native `b8da5c6e` is unchanged.
Root independently verified the complete immutable private source maps, helper,
actual fresh Cargo artifacts, package identities and isolated target/build
directories. That proof explicitly records shared-root inequality; it does not
reinterpret the failed guard as passed. The private probes remain valid without
a build retry. Seal `c60c3ff8` binds 546 selected payloads, STOP `95312978`.
Unrelated local stable Cargo builds continue, so the EX counter run is held.

V38 small12 reached nine of twelve cells: all exact with zero drops, first four
load-qualified and later five load-ineligible. No cause is assigned. The fixed
remaining cells continue and every loss remains retained; no qualifying timing
acceptance or follow-on runtime release is possible from the partial results.
Future fullsix/Node (`validation-v38/source-ready.json`, `de7427a6`) and original
full24 (`remote-publication-cost-v38-full-v1/source-ready.json`, `f2812d0a`) are
source-only preparations, with no positive runtime release.

### V38 rejected after complete raw verification; allocation gains and losses (2026-10-16)

Small12 is ALLSTOP, with all 253 sealed raw payloads independently rehashed
and every original oracle, metric, resource sample, pause and EX budget
recomputed by root proof `729a27d8`. All 155,460 complete cross-arm result
pairs match. Five of six paired guards and both group guards pass, while only
four of twelve load gates pass. The decisive qualified loss is 5182/batch2/
round1: response p99 ratio **1.373891569**, with both load gates passing;
exclusion and API service ratios are 0.753471 and 0.762074. Group median-of-p99
service ratios are 0.813713 and 0.686176. Every other metric loss is retained.
Root decision `62863e74` forbids V38 integration and further V38 full24,
six-gate/Node/serving/Lucene campaigns or retries/filtering/waivers to pass it.

Current failed-pair algebra (`root-failed-pair-bounds.json`, `376e2b8f`) gives
baseline response p99 234,453 ns and candidate pre-API p99 290,389 ns. Even
setting API duration to zero with the recorded start times fixed leaves that
receipt above the 257,898.3 ns response guard. This does not assign a cause or
predict another workload. Further API-only optimization cannot repair this
saved failure. The new raw temporal decomposition is held separately under
`small12/diagnostics-v1`; old V31 investigations are not reconstructed.

The 444 whole-query allocation pairs all match complete JSON and both bincode
endian encodings. Root proof `e0d45e87` rehashes all four actual probe logs and
recomputes every delta: requested bytes improve in 372 pairs and regress in 72
(maximum +32 B); peak bytes improve in 360, regress in 42 (maximum +32 B), and
tie in 42. Retention ties in all 444. Maximum requested/peak savings are
1,176/1,096 B. Allocations improve in 330 and tie in 114, with up to five fewer;
all 84 supplementary cases improve requested/peak bytes. Actual current
aarch64 type sizes show query forms 48→64 B, PreparedSearch 312→328 B and
CapturedQuery 472→488 B; declared arrays total 715 B, actual stack frames are
unmeasured. These are process-local requested-layout diagnostics, not serving
RSS or retained-memory superiority.

All four probes exited zero, but the controller exited one at its final root
guard because concurrent work changed engine/mod.rs; that failure and both
snapshots remain intact. Original native b8da5c6e is unchanged. Launch had no
builders, while two unrelated Cargo/rustc jobs appeared by the post-stop
observation; their start/overlap interval is unknown, so whole-run absence of
competing builds is unproven. Completion `1a42d3a3`, measure seal `3da8ecd9`
and STOP `954952b1` are retained without retry. Remote campaign activity also
stopped (`cc91150a`); bounded post-stop loads were 0.1201/0.1104/0.1011, and
the earlier high load remains unexplained. Awake inhibitor 1916 was preserved.

Next implementation work is source-only minimal retained-subset V39 prep from
the original frozen dirty 249-input baseline. Metadata and JavaScript agents
will use existing component receipts to isolate the strongest memory savings
with the fewest native publication changes. This is a new composition strategy,
not a rerun of rejected V38. No new savings are promised and no runtime release
exists. Accepted serving RSS remains 1.091961× SQLite; memory and overall
superiority remain unestablished.

### Next composition: isolate JavaScript retained storage first (2026-10-16)

JavaScript-only V39 preparation reuses the validated compatible V26 literal
factory, immutable-ID reuse and primitive buffer, while returning to the
original frozen249 `core.search_documents` owned-result path. All native
core/query/publication code stays at baseline. Complete inputs, exact source
hashes and callback adaptation are not yet frozen; no build/runtime release
exists. The relevant old evidence is mixed: selected V8 graph saving
5,071,328 B is not a total-RSS bound; an older JS-only screen remained
load-ineligible, with SQLite RSS ratio 1.043669 and about 13% QPS loss. Later
buffer-only V14→V17 comparisons on an optimized native base reduced serving
max RSS by approximately 28 MB (first pair load-qualified, second not).
Standalone behavior and serving RSS on original249 remain unmeasured.

The optional native fallback is frozen at `retained-subset-v39-prep/source`:
250 inputs/203 Rust, both COSE fixtures, manifest `5c1c7c02`, ready `7d019029`.
It combines adaptive postings with compact zero/one-key metadata, preserving
original forward terms and Node code. Existing representation/layout evidence
suggests a conditional 7.31 MB requested-byte target including map/header
costs; no isolated new allocator or serving-RSS measurement establishes it.
JavaScript-only is first; the native subset is held unless needed.

The new V38 failed-pair diagnosis is complete (`diagnostics-v1/completed.json`,
`2dda4a0e`; conclusion `77eeb8b8`). The candidate p99 response request happens
after all writes and compaction, and its predecessor API had finished before
the offer. Unclocked result rendering, file writes and receiver waits prevent
causal attribution. Saved wall/monotonic pause differences also leave precise
query-to-epoch mapping unestablished; original RustInstant durations stand.

A single prospective identical-binary A/A diagnostic is being prepared:
original249 baseline/default publication probe in both arms, original5182/
batch2 group, three reversed pairs, unchanged work, pauses, EX900, load and
1.10 numeric guards. It can assess repeatability but cannot qualify a memory
candidate or waive any failure. Root initially suggested the obsolete legacy
reader-lock checker label; actual saved baseline traces showed it already has
append-before-publication behavior. The label was corrected before any run;
the ordinary unchanged checker branch applies to both arms. A separate
post-stop four-clock diagnostic is source-only. No remote activity is running.

### JavaScript-only V39 passes fresh Node checks; Linux build active (2026-10-16)

The actual capsule has 252 inputs/203 Rust, both COSE fixtures, manifest
`899f77ba`. Only lib.rs changes from original249; 248 other inputs are exact,
plus the previously validated strings module and two Node test inputs. Root
review `06213ee2` covers the full lib/callback adaptation and cfg-only changes.
Fresh local Node checks passed API12, baseline8, factory8, Unicode10 and
builtin16. Native `1e69af73`, ready `49f4e9bf`, completion `70e9a945` and root
proof `3af5e922` bind both full source maps, actual package/absolute manifest/
target/build_directory, fresh compile, package bytes and actual test logs.
Shared-root changes from concurrent work were recorded and preserved; original
native b8da5c6e was observed unchanged. These are correctness results, not RSS
or full-six-gate acceptance.

The Linux Node build launched as sole remote supervisor `290015` at
`R/js-only-v39-linux-build-v1`, using revision3 ready `2583611f`, driver
`a6b88398`, protocol `b044dca6`, positive build release `6ae852c9` and root
source review `db9a7d99`. All 19 staged assets were verified; launch load was
0.11865 with 15.25 GB available and no competing jobs. The driver verifies the
252-input source separately from the closed 254-input donor, preserves cache
mtimes, touches all 203 candidate Rust files, and requires an actual fresh ELF
plus Node12/base8/factory8/Unicode10 checks. Actual Linux artifact pins remain
unknown until completion. No competitive measurement is running.

The next product measurement is a prospective three-round/nine-cell original
source/candidate/SQLite serving screen, preserving the existing workload,
complete output checks and monotonic phase alignment. Source-ready `0f8ef1bd`
is frozen; actual protocol/native/build pins and positive release remain absent.
Family6 screen intervals use the fixed 5,000/seed24301 paired-round method and
endpoints [20,4979]; all eight metric vectors and losses remain retained. This
cannot replace final family12 endpoints [10,4989], 33 source/SQLite rounds,
both fixed Lucene arms, quality gates or original full publication validation.

The A/A diagnostic is held and entirely unstarted, despite its earlier positive
release: product RSS has priority. A separate remote publication controller
under `R/publication-cost-v36-full-v1` and a public-bench process were observed
and preserved. They had exited by eligibility receipt `2a9f469d`; no ownership
or cause is inferred for earlier load failures.

A concrete local build-loop issue was observed: `cp -cR` reset copied
dependency/fingerprint mtimes; sample values and the roughly 294-second clone
are saved in `js-only-v39-prep/evidence/cache-mtime-observation.json`. Future
local copies should preserve metadata with `cp -cpR`, while touching all
candidate Rust sources for the required fresh HMS build. No speedup or sole
rebuild cause has been measured, and the completed V1 build was not rerun.

Concurrency note: another session replaced the leading CONTINUATION authority
with ANN/QA/Rust1.98/push instructions. Its text is preserved under its own
heading. This task keeps the user's direct MEMORY-first/Rust1.96/no-commit/
no-push/no-broad-ANN constraints and the frozen comparison sources. No category
change or authorization is inferred from the concurrent file edit.

## Checkpoint 1 of the superiority session (2026-10-16, 15:50)

Status and all numbers: benchmarks/results/superiority_ledger.json. Gap = HMS over the next competitor.

| Category | HMS | Competitor/control | Standing | Gap to next |
|---|---|---|---|---|
| Vector search, QPS at recall 0.95 (glove, GCE C4) | M 877 (accepted) | RaBitQ IVF 1,699 | behind; two levers measured paired this session, both lost: neighbour-block fast-scan 0.62x/0.64x (nytimes 0.90/0.95), sorted pool 0.97x/0.96x; x86 kernels fixed (AVX2 dot, prefetch: estimate 196 -> 53 ns on writerslogic-02), GCE confirmation blocked on gcloud auth | 0.52x |
| Vector search, index bytes (glove) | M 359.6 MB accepted; 266.6 MB untimed with varint adjacency and 4-bit residual (-0.01 recall) | RaBitQ IVF 190.6 MB | behind; measured floor about 176 B/vector = 1.04x | 0.53x (0.71x untimed) |
| Vector search, build s / peak RSS | M 1435 s / 4317 MiB | RaBitQ IVF 86 s / 724 MiB | not started | 0.06x / 0.17x |
| Long-term QA, dev 100 | M 87/100 (v1), v2b 87/100, delta 0.00 CI [-0.04, 0.04]; coverage 81 -> 83/84 | V Zep 451/500, Mem0 474/500; matched cells unrun | unestablished; obligation sub-queries lower recall at any weight | n/a |
| Learned sparse relevance | unmeasured | SPLADE v3 | not started | n/a |
| Accepted serving memory | M RSS 1.0920x SQLite | SQLite 1.000x | V35 rejected 1/12, V36 rejected 11/12 (scheduled_response p99 1.388 in one batch-2 cell); mechanism closed; compact postings and mmap text in flight | 0.92x |
| Lexical speed / relevance | M private 4.76x SQLite, 2.15-2.24x Lucene / tied 0.2916 | SQLite, Lucene | not started this session | n/a |
| Durability, security | unmeasured | SQLite, Lucene | not started | n/a |

Units of progress this period: two (both negative): benchmarks/results/qgraph_block_paired_m4_nytimes_v1.json and qgraph_pool_paired_m4_nytimes_v1.json. Wall clock 12:20 to 15:50: src 5.4 h, harness and documents 1.5 h (including the forks' time). In flight: x86 paired runs of the sorted pool and patience (writerslogic-02, x86ny1 and x86gl1), the glove paired run of the compact encodings (M4, fsgl1), the 2A compact-postings fork, the 1B abstention-calibration fork. No best accepted number improved yet. Open: the GCE confirmation of the x86 kernel fix needs an interactive gcloud auth login; no push yet (user asked to reconcile with main; branch commits f3539a9 and the next are verified in the worktree).
