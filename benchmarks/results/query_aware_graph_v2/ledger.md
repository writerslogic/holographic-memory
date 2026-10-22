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
