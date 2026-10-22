# Direct IP serving and independent result retention

Baseline: `4d5168f7ce8253b15aefe2f7ce24f0e248de4deb`. Worktree:
`/Volumes/A/.hms-wt-direct-ip`, branch `feat/direct-ip-reservoir`.
The renewed request opens these two interventions independently; v1–v3 evidence
and rejected experiments remain frozen. No commit or push is authorized.

Keep lifted construction, training queries, adjacency, entry points, upper layers,
IDs, code precision (8/8), alignment and traversal policy fixed. Direct serving
omits the construction coordinate and has an explicit negative-IP metric.
Legacy serving remains the default. Independent retention is also opt-in, with
capacity 0 preserving the final-beam behavior. Offer every row score, including
descent scores, before beam rejection; deduplicate without changing visited marks.
Do not mix coarse and fine score units. Keep the existing `ef >= k` rule.

Development data: only the existing 256 reserved validation queries and 32768
construction queries for each full llama/yi dataset, using existing graph caches.
Compare lifted/direct serving crossed with reservoir 0/400 at ef 100/256/512,
rerank 400, patience 0, k 100, knobs OFF. Build/encode each representation once
and reuse it across search settings. Report evaluated, final-beam, reservoir,
rerank-set and returned coverage; original-float oracle rerank every set separately.
Record actual refinement counts, expansions, unique evaluated IDs, storage geometry
and the centered artificial coordinate's quantization-scale dominance.

Timing must exclude diagnostics/oracles, run serially on writerslogic-02 under
the existing `~/hms-timing.lock`, with one worker and disclosed affinity. Retain
three alternating blocks, five passes each, and report the saved fifth pass.
Native timing, if used, must be identified separately from Python/VIBE throughput.
No new official performance claim: no untouched final set is certified.

Acceptance: explicit IP zero-query semantics, original magnitudes and IDs,
deterministic ties, safe persistence discriminator and round trips; original
angular/Euclidean behavior; same evaluated IDs, beam and expansion counts when
only retention changes; a candidate rejected/evicted from the beam can survive
for reranking, including rerank > ef; capacities bounded by n; meaningful regression
tests plus the repository's relevant Rust/Python checks. No topology change until
full-data discovery/retention/ranking diagnostics justify a separate experiment.

Implementation: `VMetric` explicitly identifies Angular/Euclidean/InnerProduct.
Serving files retain HMSVGR01 with metric word 2 for direct IP; older readers
reject that discriminator. Construction caches remain lifted HMSGRF04. The new
`from_lifted_ip_graph` encodes the original coordinate prefix without copying
float rows. Python opts in with `direct_ip=True` and `reservoir=capacity`.
The reservoir includes upper-layer scores and uses distinct coarse-score keys;
screens/neighbor blocks with independent retention are explicitly rejected.

Completed initial validation: Rust 1.89 check of experimental/Python library and
diagnostic binary; graph suite 86 passed, 3 ignored; Linux native Python suite
52 passed. Regression cases include beam rejection, upper-layer retention without
duplicates across epoch wrap, unchanged traversal with patience/frontier/batched
scoring, direct IP magnitudes/ties/zero queries, mapped/heap persistence, and the
128-dimensional storage cliff. The final source adds 127/129 boundary coverage.

Strict Clippy 1.89 is blocked by two unchanged baseline errors:
`src/core/documents/postings.rs:488` (`ptr_arg`) and
`src/core/triple_store.rs:25` (`new_without_default`). Neither file is edited,
and no lint has been suppressed. Full log is `clippy.log`.

Both full-data diagnoses completed on writerslogic-02 under the timing lock.
All recorded native files passed their frozen manifest SHA256 checks. Trained
graph hashes match v2: llama 4314091726643580909, yi 3182353186893347835.
For every query/setting/representation, enabling retention preserves the exact
evaluated-ID set, ordered beam and expansion count.

At ef100, lifted recall increases 0.915234375→0.954921875 (llama) and
0.9055078125→0.91796875 (yi). Original-float reranking all evaluated IDs gives
the same respective ceilings. At ef256/512, loss after discovery is very small;
the larger reservoir does not materially improve high-recall results. Direct
serving has mixed small recall changes, so no precision benefit is assumed.

Direct live encoded bytes: llama 206536668→168512296, yi
150906736→123105908 (18.4105%/18.4225% savings). Centered lift magnitude dominates 20.4016%/6.8387% of
rows, with mean primary scale ratios 1.08038/1.01490. These are measurements of
representation effects, not throughput or isolated RSS.

Final rebuild/all-features Linux gate and native Python gate are queued before
the VIBE comparisons. The timing adapter builds two representations once and
shares each across reservoir settings and all ef/repetition values. The driver
requires exact default-ID parity with all frozen v2 blocks, then exact four-arm
VIBE/native-diagnostic parity before accepting a report. Source/binary receipts
distinguish the initial diagnostic build from the final tested timing build.

Final local MSRV library gate: `cargo +1.89.0 test --locked --features
experimental --lib` passed 603 tests, 4 ignored, including the 127/128/129
storage boundaries. Touched Rust formatting and Python syntax checks pass.
The Linux all-features test is using 1.96.0 because the inherited local-model
dependencies require a newer compiler than the core library's 1.89 MSRV.

The all-features gate initially found complete VSearchParams literals in the
benchmark driver and prefetch integration test, plus the driver's metric type
comparison. All callers were checked and updated; all test targets compiled
before the final run. Initial failure logs are preserved. Final Linux 1.96 gate:
991 passed, 6 ignored, including doctests; final native Python gate: 52 passed.
Caller hashes extend the final source receipt. No library scoring/search code
changed after the final production build.

The VIBE driver also independently verified ordered default IDs against every
frozen v2 repetition, not only aggregate recall. Four-arm timing is now running
under the shared lock with the final tested native extension. Main remains
untouched; all changes and evidence belong to this isolated worktree.

Measurement is complete. Both VIBE runs passed all default-ID, native-ID,
encoded-size, input-immutability, thread and affinity checks. Each used CPU0,
12 configurations, three alternating repetitions and five passes per repetition.
Recovered raw/report hashes and all 13 final source/caller hashes pass local
verification (`recovery.json`). Compiled artifacts remain in
`/home/davidcondrey/sol-qag/v4/`; their SHA256 receipts are retained locally.

Direct-only median gains over lifted-only at ef100/256/512 are
41.3801%/47.9524%/38.4421% on llama and
27.1235%/29.9697%/29.9100% on yi. Every one of the 18 corresponding paired
blocks improves and all six observed QPS ranges are separated. These compare
equal effort; recall changes slightly, and no untouched final evaluation is claimed.
At the sampled 99% validation points, direct-only QPS is 3488.74 (llama,
recall 0.9905859375) and 2295.34 (yi, recall 0.990859375).

The reservoir is a recall/cost control, not a universal speed improvement. At
ef100, direct+reservoir reaches llama recall 0.9553125 at 4820.97 QPS; its added
retention and refinement costs reduce QPS at fixed ef. On yi, that beam still
has only 0.9179296875 recall. At high effort, direct-only is faster and reservoir
coverage adds little. Retain both controls independently, default off; do not
promote the combination as universally optimal. Topology and selective refinement
remain separate untested interventions, not hidden changes in this experiment.

`README.md` and `summary.json` were generated from the recovered diagnoses and
paired reports by `generate_report.py`. The report explicitly limits claims to
reserved validation, three sampled ef values, the tested 8/8 precision, and this
single host. Old official and stopped/inconclusive regression evidence remains
frozen. No new official, competitor-superiority, hardware-counter, isolated-RSS,
or in-distribution performance claim is established. No commit or push was made.

The user subsequently authorized PR submission. The implementation was rebased
onto main at 1cfb43f850fbdf11e823f9217010b22a364106f6, incorporating its streaming
consolidation changes. Encoding now reuses main's VertexLayout validation and
GraphView BFS interface; the streaming encoder and its byte-parity tests remain
intact. No benchmark timings were rerun or relabeled after this integration.

The measured implementation is recoverable by applying measured-source.patch
to 4d5168f7ce8253b15aefe2f7ce24f0e248de4deb. Reconstruction in a separate worktree
matched all 13 final-source/final-callers hashes. That verification worktree was
then removed. Large diagnostic JSONs and raw HDF5 files now use Git LFS, following
the existing results policy; their bytes and recorded hashes are unchanged.
The compact reports, receipts, logs, harness snapshots and source patch remain
ordinary Git content. Authored reproduction scripts carry the repository's
license header; captured harness snapshots remain verbatim.

Fresh integration gates passed 994 Linux all-features tests (6 ignored),
52 native Python tests, and 606 Rust 1.89 library tests (4 ignored). The archived
test source and the signed-off feature commit have identical source/test trees;
pr-source.sha256 also matches all 13 remote/local files. pr-integration.json
records the source identity, commands and separate lint result. These functional
checks validate the integration without replacing the frozen performance evidence.
