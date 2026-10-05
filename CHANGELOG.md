# Changelog

All notable changes to this project are generated from the commit history.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) +
[Conventional Commits](https://www.conventionalcommits.org/).
## [Unreleased]

### Added
- Add hms-server HTTP service for the Python SDK
- Add experimental query-private similarity scoring behind private-search
- Subset-union bundle experiment and capacity sweep

### Changed
- Drop rejected intersection kernel candidates

### Documentation
- Attribute semantic search to the document API
- Update changelog [skip ci]
- Record the outcomes of the algorithm work
- Give the SDK README the project header, badges and a quick start
- Use absolute logo URLs so the README renders on package registries
- Update changelog [skip ci]

### Fixed
- Decode benchmark data with as_chunks for the current clippy
- Resolve leftover merge markers in .bestpractices.json
- Serve SDK queries by exact cosine through the document API
- Return query results best-first on exact-scan, multi-shard and federated paths

### Performance
- Speed up from_dense with branchless term compaction and add encoder-eval
- Gallop before AVX2 on skewed sizes; add intersection kernel bench
- Route sparse queries to the exact inverted index

### Bench
- Add public BEIR and ann-benchmarks comparisons
- Report encoder candidate recall after exact rerank
## [0.6.1] - 2026-10-05

### Documentation
- Update changelog [skip ci]
## [py-v0.6.1] - 2026-10-05

### Added
- Introduce IdentityRegistry enforcing W3C DIDs, Verifiable Credentials, and POSME Receipts for queries
- Add AsyncHolographicClient, batch uploading, and metadata filtering
- Add Pydantic AI and SmolAgents tool wrappers
- Add Phidata integration and fix broken Rust examples
- Add Semantic Kernel and DSPy integrations to Holographic SDK
- Add LlamaIndex SDK adapter and CI pipeline
- Pivot to Zero-Trust Edge Architecture (FHE-Lite, AGPL-3.0, SIMD)
- Python bindings + PyPI packaging for holographic-vsa (maturin/pyo3)
- PhaseResonator reusable-index API + clean re-export + doctest (§20 capability)
- PhaseHVec core type + real Frady/Kent resonator on quantized-phase substrate
- Counting membership store with Poisson z-score readout (4x Bloom capacity)
- Real SHA-256 commitment for verifiable phasor mutation (lane 3)
- Multi-hop reasoning over the phasor memory (retrieve_path)
- PhaseGraph -- phasor relational memory with relation algebra + retrieval
- Presence-gate the observer effect (no confabulation-on-query)
- Persist ConnectionGraph via its event log; survives restart
- ConnectionGraph -- plastic event-sourced relation store, wired into engine
- Trust-anchored provenance verification and sharded ANN persistence
- Runnable cross-verification of the cogmem C2PA sample
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
- Add cognition layer, distributional refiner, CI coverage
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
- Centralize SDK HTTP client and add CrewAI + Embedchain platforms
- Consolidate LangChain, LlamaIndex, and Haystack adapters into unified holographic-sdk
- Remove dead self-inverse resonator; move ResonatorConfig to phase_resonator
- Extract shared wire codec (write/read_lp_str, write/read_deltas)
- Extract meaning_ctx() helper shared by structural_query and multi_hop
- Gate experimental VSA modules behind experimental feature
- Migrate HMS agent identity from did.cose to CAWG ICA
- Deduplicate RNG pruning, unify patterns, add doc comments
- Eliminate dead code, add delete/compact with persistence
- Remove redb dependency, use in-memory FxHashMap with arena log persistence

### Documentation
- State the AGPL-3.0-or-later license in the security policy
- Update changelog [skip ci]
- Record algorithm findings and private-search threat models
- Rewrite README to highlight Zero-Trust Enterprise Security and Python SDK
- State the testing policy explicitly in CONTRIBUTING.md (#66)
- Sync changelog (#54)
- Align guidance and track deferred production work (#52)
- Update changelog [skip ci]
- Update changelog [skip ci]
- Standardize badges (add OpenSSF Scorecard; consistent order/format)
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- §29 confirmed on unseen seeds — 4-bit matches float, 8-16x footprint win; log floor-exploration findings
- Wire deterministic-resonator capability into RESEARCH.md (discoverability + correct citation)
- Overnight closeout — queue exhausted, 337 tests green, flag pre-existing §21 bin lint
- Correct resonator-factorize repro command — bin is std-only, no experimental feature
- Reproducibility header for resonator-factorize (fixed seeds, correct repro command)
- DETERMINISTIC-RESONATOR.md — §20 validation write-up from in-loop 24-seed run
- Block codes also floor (§24); capacity gap is retrieval-vs-information, open not impossible
- Retract 'info-theoretically impossible' framing; 0.1D Plate is a readout artifact, closing gap to ceiling is open (§22)
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
- Record why presence gate uses an absolute margin, not SNR
- Update changelog [skip ci]
- Update changelog [skip ci]
- Update changelog [skip ci]
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
- Update all documentation for v0.5.0 accuracy
- Add meaning memory architecture, structural queries, attractor cleanup
- Add meaning memory architecture, structural queries, attractor cleanup

### Fixed
- Remove unimplemented security claims and retired cloud code
- Add missing experimental feature gate to zero_trust_demo
- Add missing experimental feature gate to zero_trust_demo
- Resolve CodeQL alerts and README formatting
- Resolve clippy::chunks_exact_to_as_chunks lint regression on main (#61)
- Resolve workflow code scanning alerts (#53)
- Give relations a distinct magic byte (0xFA), accept legacy 0xFE on read
- Clippy needless_range_loop in resonator-bundle (§21 bin) — enumerate rewrite, outcome-neutral
- Close remaining provenance trust-anchor gaps (cawg, vc-only records)
- Use sort_by_key to satisfy clippy 1.96 unnecessary_sort_by
- Repair --all-features build (port SCITT to ureq 3 API, drop dead CliffordVec bench, gate provenance example)
- Allow CDLA-Permissive-2.0 license from webpki-roots via ureq
- Replace bare unwrap() with expect() or total_cmp in production code
- Remove dead CliffordVec code and fix clippy warnings
- Perf and safety improvements from audit
- 5 bugs from audit (deadlock, double-decay, over-count, double-start, stale-used)
- Cargo fmt
- Use sort_by_key for clippy on newer Rust
- Cargo fmt formatting
- Benchmark crate rename, cargo fmt
- Resolve all clippy warnings for CI
- Quality pass on G1-G4 code
- Use absolute URL for logo on npmjs, add project.yaml for orchestrator
- Resolve 15 audit findings (3 critical, 8 high, 4 medium)
- Crash-safety, error propagation, and compaction correctness
- Propagate errors from shard insert/remove, guard ShardManager invariant
- Use .clamp() instead of .max().min() to satisfy clippy

### Performance
- Optimize Zero-Trust FHE-Lite key generation and eliminate string cloning in hot loops
- Use posting-list candidate generation for synthesizeConcepts (>500 vectors)

### Security
- Remediate repository alerts and dependencies (#38)
- Harden signing-key file perms, zeroize copies, bound JUMBF recursion

### Assets
- Static transparent logo from first frame, text removed
- Simplify logo — strip floating decorative specks, keep the connected graph mark
- Replace logo with self-contrasting node-graph mark (animated gif + static svg), drop garbled-text variants

### Deps
- Land 10 Dependabot major bumps with API fixes (#26)

### Example
- Story_memory -- HMS reasoning memory serving a writer (scrivener-mcp)

### Research
- Fractional-power encoding gives the phasor memory continuous 'near'
- Phase memory retrieves under load where sparse collapses
- Rotation binding gives relation algebra the current substrate lacks
- Path-plasticity -- holographic generalization a cache can't do
- Living-connection-graph slice -- plasticity beats saturation, verifiably
- #2 hardening RETRACTS the sparse-wins claim (AUC artifact)
- #2 strong baselines HRR + MAP vs sparse permutation
- #2 involution control disconfirms the self-inverse framing
- #2 density-matched control rules out the confound
- #2 step-1 binding discriminator harness + result

### Style
- Apply rustfmt to experiment binaries

