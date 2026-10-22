# Query-aware graph experiment ledger

## Revisions and reproducibility

- Baseline `08d37778a66b58709461456e5b9a7ab6a05b399f`, fetched2026-10-22; branch `feat/query-aware-graph`.
- Worktree `/Volumes/A/.sol-wt-qag`; Linux task root `/home/davidcondrey/sol-qag`; unrelated original-tree changes preserved.
- Projection/API `85b0af081c26e65f6a5643722386137ca332e587`; VIBE integration `3f285cebcbebffe8f2f32c430e738f0eb79ae9cc`.
- Patience API `f73305b624305836211c8157975c4f1c37f10df9`; all three commits have valid signatures.
- Exact fingerprints, execution receipts, scripts and test logs remain in `provenance.json`, `sources/`, `configs/`, and `logs/`.
- Initial diagnostic binary/source hashes are retained; exact initial uncommitted source snapshots are unavailable. Current diagnostic source is committed.

## Data, protocol and acceptance

VIBE source `7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c`; RoarGraph paper: <https://arxiv.org/abs/2408.08933>.
Llama base/learn=256921/255921; yi=187843/186843; official test=1000 each. Source hashes and split semantics are in `provenance.json`.
Seed24301: reserve `default_rng(seed).permutation(len(learn))[:256]`; construction uses the following32768 rows. Official test/truth never build or tune.
Prefix=first200000 base rows. Recompute top100 by float64 original inner product, descending score then base-ID ties; no normalization.
Scalar-product and asymmetric-MIPS spot checks matched top100 on four queries; maximum f32 augmentation identity error5.6871e-5.
Architecture: two-pass Vamana-style base graph, scalar4-bit(default)/8-bit codes,8-bit residual refinement; no production FWHT or exact-f32 rerank.
MIPS appends `sqrt(max_norm²-row_norm²)` to base rows and zero to search/training queries; the graph uses Euclidean coordinates.

Required: full llama/yi recall≥0.99; QPS at0.90/0.95≥90% of best comparable CPU competitor; improve0.99 QPS.
Regression IDs: `sift-128-euclidean`, `glove-100-angular`, `yahoo-minilm-384-normalized`; ≤3% matched-recall QPS loss.
Timings only on writerslogic-02 under `flock ~/hms-timing.lock`, using `~/vibe-hms/bench.sh ... nolock` and harness affinity; never taskset.
OMP/MKL/OPENBLAS/NUMEXPR/RAYON threads=1. Alternate three paired blocks; use saved fifth-pass totals after four warm passes, ignoring best-of-five attrs.
Report median[min–max], actual minimum strict recall across repetitions, and fastest measured eligible config; no interpolation or rounding to targets.
All arms co-resident: process peak RSS and VIBE USS delta are aggregate; encoded-byte attributes are not isolated RSS or persisted-file size.
Budget: ≤3 interventions,≤6 configurations each. All three six-configuration screens are exhausted; no further intervention or blind grid.
Supplied standing maximum0.9585 and HMS/Roar QPS701/5175 at0.90,290/2947 at0.95,unreached/520 at0.99 are sourced best-of-five targets, not paired medians.

## Diagnosis before intervention

Fixed graph degree64/build_ef500/seed24301, ef2000/rr1000, k100,4-bit routing,8-bit residuals, knobs OFF.
B=normal/coded; O=true nearest original base ID/coded; E=normal/unquantized routing; OE=oracle/unquantized. E also uses unquantized descent; rerank unchanged.
`llama-prefix-diagnosis.json` was saved before selecting a fix; all values below are strict-ID fractions.

| Control | Recall@100 | Evaluated coverage V | Retained coverage | Prerank coverage |
|---|---:|---:|---:|---:|
|B|0.9655078125|0.99203125|0.9862890625|0.9659375|
|O|0.96578125|0.9923046875|0.9865625|0.9662109375|
|E|0.991484375|0.9931640625|0.9931640625|0.9931640625|
|OE|0.99171875|0.9934375|0.9934375|0.9934375|

B evaluates35934.39 unique vertices/query. Exact reranking its1000 candidates gives0.9659375; all2000 retained gives0.9862890625.
Decision: entry adds little; coded ordering/retention dominates this prefix's loss. Base-only navigation is not supported as the primary cause here; full-index behavior still needs checking.

## Intervention 1: precision and retained capacity — six configurations

Prediction: more rerank capacity recovers retained coverage;8-bit routing closes most of the routing gap. Reject if full llama/yi remain below0.99 or low-ef throughput fails.
Same prefix/graph/seed; all rows are untimed accuracy screens, with8-bit residuals.

| Routing bits | ef | Rerank | Strict recall@100 |
|---:|---:|---:|---:|
|4|2000|2000|0.9851953125|
|8|2000|1000|0.9933203125|
|8|2000|2000|0.9933203125|
|8|200|400|0.823359375|
|8|512|400|0.9394921875|
|8|1000|1000|0.9778515625|

Full-base fixed ef2000/rr1000 validation: llama4/8 recall0.95484375/0.9916796875; yi4/8=0.9802734375/0.9840234375, yi exact routing0.98421875.
Decision: capacity alone rejected; precision insufficient for both required datasets. Promote optional8-bit encoding, retaining default4-bit behavior.
Evidence: `llama-prefix-b*.json`, `llama-full-b*.json`, `yi-full-b*.json`.

## Intervention 2: query-aware projected edges — six configurations

Prediction: independent construction-query bridge edges reduce search work at matched recall. Reject if ef512 cannot reach0.95 or yi high-recall coverage cannot improve.
All old base edges retained; explicit forward/reverse quotas. Original-IP top100 construction neighborhoods recomputed with float64 block products.
FAISS absence stopped the first preparation before output; NumPy fallback completed without a new dependency. No external RoarGraph code copied.

| Construction queries | Quota | ef | Rerank | Strict recall@100 | Mean evaluated | Encoded bytes |
|---:|---:|---:|---:|---:|---:|---:|
|8192|16|512|400|0.992890625|14806.93359375|141572548|
|32768|16|512|400|0.995703125|15158.73046875|141572548|
|32768|32|512|400|0.9980859375|15803.78515625|160772548|
|32768|32|100|100|0.9159375|5219.10546875|160772548|
|32768|32|160|200|0.9789453125|7213.984375|160772548|
|32768|32|256|400|0.9906640625|9947.44921875|160772548|

Decision: promote32768 queries/quota32. Four quota32 runs share graph7022208822022038126;80157 edges added to12797114, mean degree64.386355, capacity128.
Unprojected8-bit encoding=122372548 bytes. Prefix improvement supports promotion, not full-dataset or competitor parity. Evidence: six `llama-prefix-proj-*.json`.

## First paired prefix validation

36 groups, three alternating blocks; production projection reproduces diagnostic recall. QPS below is median[min–max].

| Method/config | Strict recall@100 | QPS |
|---|---:|---:|
|Baseline4 ef1000/rr1000|0.948515625|970.36[966.77–976.62]|
|Baseline4 ef2000/rr1000|0.9655078125|543.34[543.25–555.25]|
|Projected8 ef100/rr100|0.9159375|6071.76[5643.25–6077.62]|
|Projected8 ef160/rr200|0.9789453125|4185.90[3672.75–4219.46]|
|Projected8 ef256/rr400|0.9906640625|2778.25[2205.93–2885.32]|
|Roar96 L120|0.9561328125|6086.98[4522.09–6977.56]|

Prefix median-throughput0.90 passes;0.95 fails. Roar48 L100<0.90, Roar96 L800<0.99; measured MLANN settings<0.90. Full results remain unmeasured.
`projection-validation-summary.json` retains an erroneous official-test flag; its original source marker and frozen receipt explicitly establish validation-only scope. A separate v2 is scheduled.
Timed adapter SHA30e519088a56ced4d2dce0150d23955cdd5c8ac58f6bb9bc77bd0ef6661f1afc matches3f285ce and `sources/qag_pair_projection_validation.py`.
Raw/receipt/summary remain canonical; do not replace the frozen original when correcting metadata.

## Intervention 3: existing stale-top100 stop — six configurations

Prediction: ef160 overachieves0.95, allowing late expansions to stop earlier. Reject if no setting maintains0.95 with less evaluated work, or paired QPS<90% of eligible competitor.
Same projected graph, bits8, ef160/rr200, seed24301 and256 reserved queries. Diagnostics are untimed; patience0 control recall0.9789453125/evaluated7213.984375.

| Patience | Strict recall@100 | Mean evaluated |
|---:|---:|---:|
|4|0.8493359375|2711.1484375|
|8|0.9056640625|3612.03515625|
|16|0.9461328125|4805.4375|
|20|0.9550390625|5277.60546875|
|24|0.9625|5729.046875|
|32|0.970078125|6323.8515625|

Decision: promote patience8/20/24 at ef160/rr200; retain prior patience0 ef100/rr100,ef256/rr400,ef512/rr400 controls. Patience20 reduces evaluated work26.84%; this is not QPS.
Initial driver selected mutable stable1.99.1 and stopped at a missing smoke-helper path before timing. Those diagnostics are untimed only; no1.99-built candidate timing accepted.
Corrected Python build: explicit1.96.0, `-C target-cpu=native`, release/python/lib;46.99s. Extension SHA31988dc57c8998e12f2daabb8e6b77086775d8532d60f475ac12095d7d2d0a26.

## API, compatibility and completed checks

Optional Rust row-major training input validates all rows and uses at most first32768; `None` directly delegates to legacy construction.
Python `VectorIndex(...,code_bits=4)` accepts4/8; `fit(X,training_queries=None)` transforms original MIPS queries with a zero coordinate.
`set_query_arguments(ef,rerank=64,patience=0)` preserves prior defaults; zero resets positive patience. Invalid training inputs preserve a fitted index.
Persisted layouts remain `HMSGRF04`/`HMSVGR01`; Rust byte-compatibility/projection assertions are added but unrun. Python parity does not prove all serialized-byte contracts.
Native five-arm smoke:128-row/128-dim synthetic data, strict recall1.0,16 default-parity probes; candidate training count0, so this is ABI/default correctness only.
Python suites:17 precision,32 trained,38 patience passed; preparation/export43, reporter28, latest combined preparation/export/report/freeze52 passed. Logs retained.
Training tests cover omitted/None, Fortran coordinates, invalid shape/finiteness/angular-zero and asymmetric-MIPS equivalence. No throughput claim from these tests.
MLANN uses aligned32-byte storage and a synchronized query copy included in timing; RF10/depth14 forest seed uncontrolled, one forest reused. Roar final degree may reach3*M_pjbp.

## Durable commands, frozen batch and outstanding gates

Preparation/reproduction sources: `sources/qag_prepare_initial.py`, `sources/qag_training.py`, training `*-meta.json`, and frozen exporter/reporter snapshots.
Diagnostic/corrected-build commands: `configs/patience-promotion.sh`, `configs/patience-screen-v2.sh`; run with the shared outer timing flock.
Full batch: `flock ~/hms-timing.lock bash ~/sol-qag/full-evaluation-v1.sh`; exact script is `sources/full-evaluation-v1.sh`.
42-group config=`sources/final-candidates-config-v1.yml`:14 settings×3 blocks, middle block reversed; identical32768-row construction pool for HMS/Roar/MLANN.
`llama-full-validation-v1-frozen.json` and `patience-execution-source-receipt.json` preserve exact sources/extensions/harness/splits; snapshots remain under `sources/`.
Batch order: full llama/yi validation; official llama/yi with validation-report freeze guard; three default4-bit regressions; corrected prefix v2 report. Batch running; no full success asserted.
Integration patch retains legacy grids and an explicit IP-only query-aware definition; pins and final definition updates are tracked separately from frozen timing sources.
Remaining: full accuracy/QPS and regression decisions, isolated per-index peak RSS/persisted-file size, corrected prefix v2, and final Rust gate.
No Cargo check/test/Clippy loops or feature sweeps. Only experiment release builds ran. Final `cargo fmt --check`, then exactly one Linux `cargo +1.89.0 test --all-features`, remain unrun.

Comparator completeness, declared before official evaluation: RF10/depth14
MLANN saturates below0.90 on the prefix across all3 fixed voting thresholds.
Also measure the existing reference RF80/depth13 at those same3 thresholds,
using identical32k construction queries, full base/test, CPU/threads/harness
and saved-fifth-pass protocol. This adds competitor evidence, no new HMS
intervention/configuration. A separate MLANN-only adapter avoids rebuilding
HMS; preserve all first-batch snapshots and do not tune from official results.

Full validation reports are canonical.

## Frozen official recovery, 2026-10-22

Resumed signed HEAD b6d82ef3195d7045afb46f2aa607a1d6ec8455da; clean worktree.
Origin has no query-aware branch or matching PR. All four main frozen runs
completed successfully. Verified official raw and freeze SHA256 against report
receipts. Raw HDF5 results and the VIBE source snapshots are not committed: each
`frozen.json` records their SHA256 and the VIBE commit they came from.
No successful run was restarted. Official source metrics remain unnormalized IP.

Official strict recall maximum: llama0.99687 (ef512/rr400/p0); yi0.98931
(ef512/rr400/p0). Yi does not qualify for0.99. At0.95 llama4800.27 QPS
versus Roar96/L1206363.46:24.57% deficit. At0.90 llama6399.41 versus6363.46
(+0.56%), with overlapping observed ranges: superiority is inconclusive.
Yi beats measured RoarGraph at0.90/0.95, pending the predeclared RF80 comparator.
Neither frozen competitor qualifies at0.99; no matched high-recall victory claimed.

The user's new objective is superiority at all three thresholds, replacing the
old90%-of-competitor progress gate. The report now requires separated observed
ranges for a positive superiority status; selection still uses actual recall
and measured medians. Full points/frontiers are in pareto.json/pareto.csv.
The original three-intervention budget remains closed. Additional work is in
../query_aware_graph_v2/ledger.md, validation only, without reusing official data.

Default4 regression is still running under the original outer lock; RF80 remains
queued. The sole Rust1.89 all-features gate is unconsumed.
