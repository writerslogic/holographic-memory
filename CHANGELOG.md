# Changelog

All notable changes to this project are generated from the commit history.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) +
[Conventional Commits](https://www.conventionalcommits.org/).
## [Unreleased]

### Added
- ConnectionGraph -- plastic event-sourced relation store, wired into engine
- Trust-anchored provenance verification and sharded ANN persistence
- Runnable cross-verification of the cogmem C2PA sample

### Changed
- Gate experimental VSA modules behind experimental feature
- Migrate HMS agent identity from did.cose to CAWG ICA

### Documentation
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Pre-register non-self-inverse binding + nonlinear readout experiment
- Update changelog [skip ci]
- Update changelog [skip ci]
- Replace logo with continuous-rotation animation
- Update changelog [skip ci]
- Update changelog [skip ci]
- Restructure README with collapsible sections
- Update changelog [skip ci]
- Update changelog [skip ci]
- Rewrite README — fix logo for npmjs, restructure with quick start first, improve clarity
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Add agent-provenance stack cross-reference to README
- Update changelog [skip ci]

### Fixed
- Close remaining provenance trust-anchor gaps (cawg, vc-only records)
- Use sort_by_key to satisfy clippy 1.96 unnecessary_sort_by
- Repair --all-features build (port SCITT to ureq 3 API, drop dead CliffordVec bench, gate provenance example)
- Allow CDLA-Permissive-2.0 license from webpki-roots via ureq

### Security
- Harden signing-key file perms, zeroize copies, bound JUMBF recursion

### Research
- Path-plasticity -- holographic generalization a cache can't do
- Living-connection-graph slice -- plasticity beats saturation, verifiably
- #2 hardening RETRACTS the sparse-wins claim (AUC artifact)
- #2 strong baselines HRR + MAP vs sparse permutation
- #2 involution control disconfirms the self-inverse framing
- #2 density-matched control rules out the confound
- #2 step-1 binding discriminator harness + result

### Style
- Apply rustfmt to experiment binaries
## [0.6.0] - 2026-06-21

### Added
- Add provenance system with COSE, VCs, C2PA, JUMBF, Sigstore, KERI, CAWG
- Add research-grade benchmarks, scaling analysis, and visualizations
- Add sparse Clifford algebra multivector type (CliffordVec)
- Define HolographicAlgebra trait boundary for future geometric algebra
- Add Hopfield-Fenchel-Young energy-based associative retrieval
- Multi-scale encoding, morphological decomposer, fault-tolerant federation
- Improve SOTA for analogy, planner, and concept synthesis
- Add cost/total_cost fields to planner N-API types
- Evidence-weighted multi-hop confidence, ranked results
- Expose cleanup_vector N-API method for standalone Hopfield denoising
- Add agency layer (GoalStore, Planner, QuestionGenerator, SelfModifier)

### Documentation
- Update all documentation for v0.5.0 accuracy

### Fixed
- Replace bare unwrap() with expect() or total_cmp in production code
- Remove dead CliffordVec code and fix clippy warnings
- Perf and safety improvements from audit
- 5 bugs from audit (deadlock, double-decay, over-count, double-start, stale-used)
- Cargo fmt

### Performance
- Use posting-list candidate generation for synthesizeConcepts (>500 vectors)
## [0.5.0] - 2026-06-18

### Added
- Add cognition layer, distributional refiner, CI coverage

### Fixed
- Use sort_by_key for clippy on newer Rust
- Cargo fmt formatting
## [0.4.0] - 2026-06-17

### Added
- N-API bindings, persistence, load_from_log, compact for meaning memory (G6)
- Integrate meaning memory into HmsCore, wire write/delete paths (G5)
- Fuzzy_structural_query + multi_hop_query pipelines (G4)
- AtomMemory, CompositeMemory, TripleStore, RuleStore, Decomposer (G3)
- Shared utilities + IndexedMemory with Hopfield attractor (G1-G2)
- Rename to holographic-memory, fix npm logo, update all URLs
- Graph engine with multi-hop traversal, inference, temporal, federated queries
- Complete security integration, DP wiring, docs, LF readiness
- Add security features behind feature flag for LF Decentralized Trust readiness
- Migrate to writerslogic org, add npm publish pipeline
- Eliminate remaining gaps, expose train_nsg/train_ivf via N-API
- Expose diffusion config through HmsConfig and N-API
- Configurable thresholds, N-API config constructor, status getters
- Add query_sequence, elevate test coverage for weak features
- Add multi-shard support with auto-sharding
- Initial release of Holographic Memory System (HMS)

### Changed
- Deduplicate RNG pruning, unify patterns, add doc comments
- Eliminate dead code, add delete/compact with persistence
- Remove redb dependency, use in-memory FxHashMap with arena log persistence

### Documentation
- Add meaning memory architecture, structural queries, attractor cleanup
- Add meaning memory architecture, structural queries, attractor cleanup

### Fixed
- Benchmark crate rename, cargo fmt
- Resolve all clippy warnings for CI
- Quality pass on G1-G4 code
- Use absolute URL for logo on npmjs, add project.yaml for orchestrator
- Resolve 15 audit findings (3 critical, 8 high, 4 medium)
- Crash-safety, error propagation, and compaction correctness
- Propagate errors from shard insert/remove, guard ShardManager invariant
- Use .clamp() instead of .max().min() to satisfy clippy

