# Worktree audit, 2026-10-20

Sixteen detached worktrees and one uncommitted-work patch were compared with `origin/main` at
`b35f171` (PR #81) by content and patch-id, not by commit hash. Every commit and every dirty or
untracked file was checked. All worktrees were removed with `git worktree remove --force` and
pruned. The worktrees were detached, so no local branch existed to delete. After removal their
HEAD commits are unreachable: `9fa7893..648ac99` (research chain), `f3539a9`, `7cbf2d3`, `18a0eec`,
`80f438e`, `72b3ea2`, `573ee71`, `865f399`.

Method:

- The four perf commits `f3539a9`, `7cbf2d3`, `18a0eec`, `80f438e` have the same `git patch-id --stable`
  as `f115e7e`, `a80ebf1`, `16f4c86`, `1d686ed` on main.
- The 203 files changed by `9fa7893..648ac99`: 189 are byte-identical in main; 14 are older versions
  that main has since rewritten (a lint pass, the local-reader support in `check_longmemeval_answers.py`,
  `gates_2026-10.json` checkpoint 9 replacing checkpoint 8, pyo3 0.30).
- Dirty files: each tracked change was reverse-applied to main, and every worktree-only line was
  searched across all of main. Remaining lines are the pre-rewrite forms of code main reworked
  later (`documents/{index,search}.rs` compact postings, `MutationGate` writer queue, `StagedArena::new(path, format)`).
  Every worktree file is dated 2026-10-13 or earlier; `fd5adf3` is 2026-10-20.
- Evidence files were compared with the main tree copies and are archived in the `evidence-v1` release.

| Worktree | HEAD | Verdict | What was merged or why discarded |
| --- | --- | --- | --- |
| `.hms-wt-1c` | `7cbf2d3` | already in main | Both perf commits match main by patch-id; research chain in main. `src/core/mod.rs` is older than main; `sparse-bench.rs` and `sparse/mod.rs` are identical to main. |
| `.hms-wt-3a` | `18a0eec` | already in main | Perf commit matches `16f4c86`. Untracked `durability-driver.rs` lacks main's `title_weight` field. |
| `.hms-wt-7cbf` | `7cbf2d3` | already in main | Clean; same commits as `.hms-wt-1c`. |
| `.hms-wt-build` | `80f438e` | already in main | Matches `1d686ed`; main also has `ccf2c18` on top. Modified `public-bench.rs` lacks main's `neighbour_cap` and `title_weight`. |
| `.hms-wt-fastscan` | `b35f171` | already in main | Clean checkout of PR #81. |
| `hms-dense-document-next` | `648ac99` | superseded | Draft of the document store, dense index and staged compaction. Main has the later `documents/{index,search,dense}.rs` (compact postings), fair-queue `MutationGate`, `PreparedCompaction`; all draft test names exist in main. Its `run_integration_gates.py` is an older variant of the main-tree evidence copy. |
| `hms-document-competition-isolated` | `648ac99` | superseded | Same draft lineage. `engine-caller-root.patch` passes `text_encoder` to `documents::prepare`; main's `document_api.rs` already does. Scratch `*.py.lock` files discarded. |
| `hms-gate-checkpoint5-default` | `72b3ea2` | superseded | Older `Cargo.toml` (pyo3 0.29, serde without `rc`), `qgraph/{mod,kernels}.rs` before the build-code screen and runtime dotprod detection. `algebraic-codebook-bench.rs` is identical to main. |
| `hms-ivf-routing-next` | `648ac99` | superseded | Draft `ivf.rs` and document-store files; main's `ivf.rs` has the `as_chunks` decoding and later work. |
| `hms-metal-sdpa` | `573ee71` | already in main | The Metal SDPA draft (`qwen3.rs` change, three scripts, preregistration and addendum) is archived byte-for-byte in `benchmarks/results/metal_sdpa_draft_2026-10-09/`. No SDPA quality or speed result exists, so the code change is not merged. `target/sdpa-validation` file set matches the archive. |
| `hms-native-harness` | `865f399` | already in main | Harness and bridge sources identical to main. `native_harness_v1` results identical to the main-tree evidence copy (only `archive_manifest_v2.json` differs, by absolute worktree paths); the directory is in `evidence-v1`. |
| `hms-server-concurrency-next` | `648ac99` | superseded | `server_concurrency_v1` evidence identical to the main-tree copy. Main has the multi-tenant `hms-server` (`TenantSlot`, tenant-root escape check) and `hms-server/concurrency_tests.rs`. |
| `hms-wt-1b` | `648ac99` | superseded | Only change is the older `b"/-".contains` spelling in `operand_retriever.rs`. |
| `holographic-memory-memory-competition` | `648ac99` | valuable-unmerged, merged | `benchmarks/results/memory_competition_v1/supplemental.json`, the verification record of the already merged `supplemental.patch`, was missing from main and is added. The rest (`Cargo.toml`, `CONTINUATION.md`, `production-readiness.md`, document store) is older than main. |
| `holographic-memory-model-integrity` | `648ac99` | superseded | Draft document and qgraph files older than main; only scratch `*.py.lock` otherwise. |
| `benchmarks/results/superiority_v3/append_stage_recovery_v1/validation-turnaround/worktree` | `648ac99` | discarded | A validation checkout with 707 tracked files deleted and one 45-byte probe (`lookup_probe.rs`, an identity function). Removed before `superiority_v3` was archived so the archive holds no worktree link file. |
| `main-tree-uncommitted-2026-10-20.patch` (not a worktree) | n/a | already in main | 36 of 51 file diffs reverse-apply to main (already present). The forward-applying ones (`check_learned_sparse.py`, `learned_sparse_*.py`, `durability_matrix.py`, `security.rs`, `CHANGELOG.md`) are older forms main has since lint-fixed. The three that apply neither way split code into modules (`mutation.rs` tests now in `document_snapshot_tests.rs` and `document_stage_tests.rs`, tenant code in `hms-server/`); all 21 function names it removes from `mutation.rs` exist in main. The patch file stays outside the repo. |

Also removed in this pass: the stray `Volumes/` tree at the main-tree root. It held one file,
`generated-index.d.ts`, written by `run_checks-interrupted.py` in
`superiority_v3/direct_commit_serialization_v1` when `npm run build:debug -- --dts <absolute path>`
joined the absolute path under the working directory. The file was moved to the path the driver
meant to write, and `/Volumes/` is now ignored.
