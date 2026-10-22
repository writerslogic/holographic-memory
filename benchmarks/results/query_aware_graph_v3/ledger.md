# Validation-only exact integer scoring

Declared2026-10-22 after both v2 production profiles, before any v3 build/timing.
The frozen v1 three-intervention budget remains closed; v2's sole adjacency
intervention remains separate and its immutable source stays in repo-v2 remotely.

Measured motivation: v2 high-recall plain production traversal spends37.69%
(llama) and36.37%(yi) of instrumented total in combined scoring/pool. Adjacency,
visited checks and prefetch are larger,47.35%/45.48%. These unpinned instrumented
fractions include timer overhead and do not isolate dot-product compute. Existing
native disassembly already contains some16-bit VNNI, but still widens i8 inputs.

Hypothesis: an exact signed byte-dot implementation using AVX-VNNI dpbusd with
bias/sum correction reduces scoring cost on the185H. Neither the source feature
gap nor the profile proves a speedup. Zero-tail scoring stays unimplemented and
unmeasured, preserving attribution. No changes to graph, recall settings, codes,
residuals, adjacency layout, metric, stopping or persisted bytes in this phase.

Bound: one integer8-bit kernel intervention, one release Python build unless a
compiler error requires correction, one full paired experiment per OOD dataset
(two production builds each), the same earlier patience0 settings100/100,
256/400,512/400; three alternating blocks, five passes, saved fifth timing pass.
Compare fixed adjacency in both arms: frozen v2 extension versus v3 extension.
Exact ordered returned IDs are mandatory at every setting/repetition. Reject
as a speed claim unless all observed paired blocks improve with separated QPS
ranges. Preserve inconclusive/negative evidence; no retuning or repeated timing.

Use only the existing256 reserved learn queries and32768 construction rows,
their original hashes and full llama/yi bases. Official v1 results cannot select
settings and are not rerun. No untouched official split exists for new source.
The user cancelled long Glove/Yahoo regression builds; do not restart them.
Run native synthetic API/default parity checks and report that full default-path
throughput preservation remains unverified. The4-bit kernel algorithm stays the
existing AVX2 implementation on this CPU.

Remote source repo-v3 and sources under v3 are isolated from frozen v2. Builds
and timings hold ~/hms-timing.lock; VIBE chooses affinity, no taskset wrapper;
all five thread variables1; Rust1.96.0/native matches v1/v2 timing builds.
After all intended Rust edits, reserve exactly one Linux cargo+1.89.0 test
--all-features plus cargo fmt --check. No Cargo test/check/clippy before that.

Primary API: https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_dpbusd_avx_epi32.html
(stable since1.89, requires avxvnni); runtime detection uses Rust's documented
is_x86_feature_detected!("avxvnni").

Source frozen locally as kernel.patch/source-local.sha256 and uploaded to the
isolated full repo-v3. Layout/revision helpers use a separate qag_cost_v3 module;
the original queued v2 helpers remain unchanged. Kernel source SHA256 is
827c117aa6477442fb3443b4d4e43360db87c89e49a741fdbfe487e688acbcc8.

The continuation flock423852 holds the lock across resumed v2 checks/layout
timings and then kernel-v3.sh. A final-gate.sh job is queued separately as
flock426458, log v3/logs/final-gate-driver.txt. It requires v3 completion, checks
all six predeclared validation points against the rejection criterion, and saves
decision.json. It selects unchanged v3 kernels only if every point passes;
otherwise it restores the frozen v2 kernel into a separate full repo-final-gate.
This automatic rejection rule was fixed before v3 timing. No official result
participates. It then runs Rust1.89 formatting and the single all-features gate,
using the repository CI's dev/test opt-level1/debug0, two build jobs and a separate
target-final-189 directory. No retry or additional Cargo suite is scheduled.
The sole gate remains unconsumed until that queued job actually reaches Cargo.

At19:21UTC v2 completed and the chained v3 release build succeeded (one attempt).
The native Python44 tests passed in1.34s and the wrapper smoke passed. Llama
comparison began19:22UTC as worker428360; yi follows. V3 SO SHA256 is
75a805129c40ecfce5469ff8f56c890862e8c419c82cbd6719575aeb5d3bb8df.
The initial declaration uploaded before the build is preserved as declaration.md;
source-local.sha256 now names that relocated immutable declaration. Later ledger
observations do not overwrite the frozen declaration.

Llama completed with identical IDs. Median byte-VNNI deltas are+2.7513%,
-0.1585%,-5.7332%at100/100,256/400,512/400. First two points have mixed paired
signs; all three512/400blocks are slower(-6.0204%,-12.0326%,-4.3937%). This
already fails the predeclared all-six-point rule. Local shipping kernels.rs was
restored to signed2d4967e before the final Cargo gate; the exact experimental
source remains kernel.patch and remote repo-v3. Yi still completes its frozen
comparison; no rerun, retuning or additional intervention. The automatic final
decision will also select the original v2 kernel regardless of yi's outcome.
