# Holography implementation review, 2026-10-24

Baseline: `12c80dec` on `main`. User-authorized implementation with three parallel
agents in isolated worktrees; the user subsequently authorized merging the verified
work into `main`. No changes to official LongMemEval data, spending
records, external services, or persisted storage formats are part of this work.

## Decisions and acceptance checks

- Phase arithmetic: preserve public `Vec<u16>`-backed access and exact quantized
  results. Cache bounded trigonometric tables, remove unnecessary modular division,
  and measure cold setup separately from warm throughput. Test all phase differences
  at representative resolutions and quantization boundaries.
- Resonator: preserve update order, recovered factor indices, convergence, and
  numerical scores. Optimize allocations and repeated transforms; compare an
  explicitly packed codebook against expanded phasors. Count retained index state
  and query scratch, not just the theoretical bits per phase. Packing is not
  automatically the latency default.
- Conversation recall: preserve immediate candidate admission first, then recover
  complete dependency closures of admitted episode anchors using the remaining
  session/version, byte, source, and round budgets. A long-session regression showed
  that enriching first could displace other useful anchors; the two-pass design
  preserves that breadth. Original source records remain authoritative. Measure any phase episode sidecar
  against exact posting/intersection controls at matched budgets before promotion.
- Phase graph: separate subject and object roles. Baseline regressions already
  demonstrate false reverse assertions and wrong incoming-edge retrieval. Preserve
  deterministic event replay and commitments; test repeated-relation paths.
- Capacity: audit the existing spatially coupled superposition experiment with
  decoder/codebook memory included. A compressed signal alone is not a complete
  memory budget. Any new capacity proposal must beat a strong separately addressed
  baseline, not just an overloaded global sum.

Focused regression tests precede integration. Parent reviews delegated diffs,
reruns integrated checks, and runs latency comparisons serially after concurrent
builds finish. Timings under concurrent load are exploratory. Required repository
formatting, lint, test, and MSRV checks are recorded with failures distinguished
from changes introduced here.

## Evidence boundaries from the supplied papers

The useful transfers are role-sensitive composition, separately addressed capacity,
whole-episode cues, quantized codebooks, and partitioned computation. Physical
optical/quantum gains are not CPU benchmarks. Dense optical associative memory
implements attention already represented in HMS's cleanup paths. The main search
path uses lexical and dense retrieval; experimental phase improvements do not
establish a measured production search or LongMemEval gain.

The knowledge-manifold paper (`2509.10518`) was withdrawn for significant errors;
its numerical claims are excluded. Prior failed phase-router and holographic
late-interaction experiments remain negative evidence. Do not rename those
mechanisms or discard their controls to claim a gain.

## Work areas

- Root: phase graph, capacity accounting, paper review, integration, final gates.
- Phase substrate: `src/core/phase_hvec.rs`, dedicated kernel harness/results.
- Phase resonator: `src/core/phase_resonator.rs`, dedicated harness/results,
  `docs/DETERMINISTIC-RESONATOR.md`.
- Context recall: conversation `context.rs`, its tests and documentation, bounded
  context experiment/results.

Per-area evidence is retained under `benchmarks/results/holography_phase_graph_v1`,
`holography_phase_substrate_v1`, `holography_resonator_v1`, and
`holographic_context_v1`. Final implementation and limitations belong in
`docs/RESEARCH.md` and each area's result README.
