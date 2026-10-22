# Validation-only search-cost optimization

Started 2026-10-22 from signed HEAD b6d82ef3195d7045afb46f2aa607a1d6ec8455da.
The v1 three-intervention budget is exhausted and remains closed. The user's
renewed request authorizes profiling and a distinct storage-layout experiment.

This phase uses only the existing 256 reserved learn queries and 32768
construction queries, with their v1 checksums. Official queries and their
outcomes cannot select settings, edges, precision, or termination rules.
The v1 official reports are historical evaluations of the frozen v1 binary.
No untouched replacement official test is currently identified.

Budget: one compact-adjacency intervention, two layouts, no graph/search
parameter sweep; at most one release build per required artifact unless a
compiler error requires correction. Reuse the three existing full/prefix base
graph caches for diagnostics. Allow at most two full production builds per
OOD dataset for the paired VIBE comparison. Retain three paired blocks and
five passes, reporting the saved fifth pass. No new timing overlaps the
running v1 regression/RF80 batches; use the shared flock and bench.sh.
The sole Linux Rust 1.89 all-features test remains reserved until code is final.

Hypothesis: exact-degree packed adjacency reduces unused prefetch traffic and
the adjacency footprint without changing any neighbor, neighbor order, code,
entry, beam, or result. Offsets may cost more than the saved traffic; compaction
is rejected as a speed optimization if paired end-to-end measurements do not
support a repeatable gain. Preserve rejected evidence.

Before layout timing, profile actual production traversal with counters and
stage timers in a separate diagnostic invocation. Ordinary search compiles
without timers/counters. Record adjacency/visited/prefetch and scoring/pool
as combined stages when their costs cannot be independently identified.
Do not infer cache misses, cycles, instructions or allocations from these
timers. No perf, gdb, valgrind or py-spy executable was found on the timing host;
perf_event_paranoid is 2. Hardware-counter availability remains unverified.

Production invariants: Rust 1.89, optional training, default legacy bytes,
HMSVGR01/HMSGRF04, original metric semantics, no dependency additions.
Compaction must preserve save/open semantics and exact returned-ID parity.

Execution: profile-v2.sh is queued under flock (PID412272, local tool
session3751); layout-v2.sh follows (PID414352, session81893). Both use the
isolated remote source /home/davidcondrey/sol-qag/repo-v2. Rust sources were
formatted with Rust1.89 rustfmt and uploaded before queuing. No release build,
profile, layout timing or new Rust test has executed yet. The layout driver
requires identical diagnostic IDs/persisted bytes and at least10% encoded-byte
saving on both datasets before building its Python extension. It reuses only
three earlier patience0 settings:100/100,256/400,512/400, two layouts, three
alternating blocks, five passes. Exact source/binary hashes are recorded remotely.

Frozen-v1 reports and Pareto export are signed commit
4e01d0f5f470a167ebf457003c5162ade6b0bace, pushed to origin/feat/query-aware-graph.
Draft PR: https://github.com/writerslogic/holographic-memory/pull/204 .
The GitHub connector denied creation (403); authenticated gh created the draft.
The only final all-features Cargo gate remains unrun.

Read-only research also identified two exact-work-reduction candidates:
the CPU advertises avx_vnni, while the current dispatcher requires AVX-512
for VNNI; Rust's AVX-VNNI dpbusd intrinsic is stable since1.89. MIPS query
augmentation appends zero, but128-dimensional IP queries still score144
8-bit padded coordinates, including an entirely zero16-coordinate block.
These are source/hardware observations, not measured QPS improvements.

On resume, disassembly of the frozen v1 native extension's dot_i8_avx2_checked
(SHA31988dc57c8998e12f2daabb8e6b77086775d8532d60f475ac12095d7d2d0a26)
shows vpmovsxbw widening and vpmaddwd, with some compiler-generated VEX
vpdpwssd already present. See logs/v1-dot-i8-avx2.txt. An explicit byte-lane
dpbusd path would differ, but merely detecting AVX-VNNI does not establish a
benefit. Rust primary docs confirm _mm256_dpbusd_avx_epi32 is stable since1.89
and requires avxvnni. No kernel intervention has been declared or measured.

At18:41UTC the user stopped the remaining v1 regression builds to prioritize
RF80/profiling/layout. SIFT's inconclusive result and cancellation evidence remain
in v1. This does not relax exact-result/persisted-byte layout acceptance or certify
the unmeasured default-path throughput. No further long regression build is queued.

Both profiles completed. All256 returned-ID lists, persisted bytes and per-query
work counts match across fixed/compact layouts. Live encoded savings are23.2639%
for llama(206536668→158488097) and23.2400%for yi(150906736→115835997).
Persisted files remain206622512/150982476bytes respectively, identical across
layouts. Fixed-layout instrumented fractions: llama adjacency/visited/prefetch
47.35%,scoring/pool37.69%,rerank/sort11.08%; yi45.48%/36.37%/13.45%.
These do not isolate dot-product arithmetic or supply production QPS. Profile
source/binary/compiler receipts and per-query counters are recovered; derived
summary.json/README.md use qag_cost_report.py (two data-integrity tests passed).

The one Python release build succeeded, then the native test bootstrap stopped
with ModuleNotFoundError:pytest before any test or layout timing. Original log
is preserved. resume-layout-v2.sh installs the missing test dependency and
continues from the existing extension without a rebuild, then chains the
separately declared v3 experiment. Queued flock423852, log
~/sol-qag/v2/logs/resume-layout-driver.txt. At18:54UTC an unrelated sol-lex
release-second.py process414739 owns the shared lock; do not stop or alter it.
Corrected-prefix reporting/persisted hashes are queued ahead as flock421598.

The plain production counters imply91.412/81.400neighbor checks per expansion
for llama/yi. With one initial pool estimate excluded,55.72%/71.13%of neighbor
checks find an already visited vertex. This follows from the measured counters
and the plain loop's admission contract; it is not a cache-miss or branch counter.

Ranked next experiments, not opened by this ledger or included in the current
v2/v3 budget:

1. Reduce repeated adjacency/visited work using one bounded construction-only
   pruning/connectivity intervention, preserving the original-metric validation
   gate. The measured loop is the largest stage and repeated checks are common.
   [NGFix](https://arxiv.org/html/2510.22316v1) repairs neighborhood defects using
   historical queries; this motivates a connectivity study, not an assumed HMS
   gain. Its [upstream prerequisites](https://github.com/yuhuifishash/NGFix#prerequisites)
   require AVX-512, so no unchanged-host comparison is established on the185H.
2. Test selective score refinement before bounded-pool admission at the existing
   low-beam point. The v1 prefix's evaluated-versus-retained coverage and exact
   routing ablation motivate this; the larger rerank/expansion tradeoff must be
   measured. [PAG's design](https://github.com/KejingLu-810/PAG/blob/main/docs/design.md)
   places projection rejection before full scoring and packs edge metadata, but
   its AVX-512 layout is not a measured185H comparator or an HMS implementation.
3. Test exact block-aligned zero-tail scoring, separately from VNNI and layout.
   Source inspection shows a16-coordinate all-zero8-bit MIPS query tail after
  128→129augmentation/padding144. Preserve4-bit32-coordinate interleave and
   all fine/batch/screen paths. No timing or net gain is established; arithmetic
   is only part of the measured scoring/pool stage.

All architecture changes need validation-only selection and a genuinely untouched
final test before a new official claim. Historical all-learn construction means
remaining learn rows must not be relabeled fresh without a source-derived audit.

At19:02UTC the unrelated holder finished normally. The report-only job completed:
corrected-prefix metadata says validation, with unchanged recall/QPS; both saved
files per dataset also have identical SHA256 (persisted.sha256). The resume job
installed pytest9.1.1;44 native Python contracts passed in1.40s and the real-native
wrapper smoke passed, including legacy/default/baseline IDs and rejection-state
preservation. Llama layout comparison started19:02:30UTC as worker425480;
yi then v3 follow under the same flock. No layout QPS has been recovered yet.

Llama completed and recovered at19:15UTC. Exact IDs match at all settings/blocks.
Compact median deltas at100/100,256/400,512/400 are-3.7348%,-5.5158%,-5.3625%.
All three256/400paired blocks are slower(-0.3072%,-4.9320%,-5.6145%); the other
points have mixed paired signs and are inconclusive. Reject a llama speedup claim;
retain the explicit memory-saving option, default off. One-build times350.092s
fixed/345.635scompact; process peak1216140KiB includes both arms and input/runtime.
Yi is active; no extrapolation from llama or the unpinned profile timings.

Yi completed at19:21UTC and both raw/receipt/report/source-snapshot sets passed
recovery checks (recovery.json). Compact median deltas are-3.2829%,+2.6812%,
-1.3007%at100/100,256/400,512/400. Only100/100has all negative paired signs;
the remaining points are inconclusive. No tested layout point demonstrates a
repeatable speedup with separated observed ranges. Yi's build observations are
176.030s/176.085s, co-resident peak956880KiB. This closes the v2 timing budget;
retain compact adjacency as an explicit footprint tradeoff, default off.

Fresh-test audit: the pinned VIBE generator trims Q/K[1:-2047] before its seeded
learn/test split. The original Q_12_15/Q_31_13.npy files are absent from the
installed VIBE data directory. The discarded endpoints have not been audited for
padding, duplicates, validity or historical use and are not certified fresh data.
No new final split is established by this observation.
