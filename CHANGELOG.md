# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-06-17

### Added
- **Meaning Memory layer**: AtomMemory, CompositeMemory, TripleStore, RoleRegistry with cyclic-shift role binding, RuleStore for composition rules, Decomposer for text-to-triple extraction.
- Hopfield attractor cleanup with 96.9% erasure tolerance.
- `fuzzy_structural_query` with algebraic and materialized paths.
- `multi_hop_query` with rule rewriting.
- N-API methods: `memorize_meaning`, `structural_query`, `multi_hop_query`, `meaning_cleanup`, `declare_composition_rule`.
- `MeaningConfig` gated behind `meaning.enabled`.
- Persistence for atoms (0xFD), composites (0xFC), triples (0xFB) via arena magic bytes.

### Changed
- Renamed package from `@writerslogic/hms-native` to `holographic-memory` (unscoped npm).
- Renamed crate from `hms-native` to `holographic-memory`.
- Repository moved to `writerslogic/holographic-memory`.

## [0.3.0] - 2026-06-16

### Added
- **Graph engine**: Explicit relation storage with multi-hop BFS traversal, typed relations, and temporal filtering.
- **Typed relation inference**: Transitive (A->B->C implies A->C) and symmetric (A->B implies B->A) relation semantics.
- **Temporal relations**: `valid_from`/`valid_to` timestamps on relations with time-range queries.
- **Federated queries**: Query across multiple HMS instances in parallel without centralizing data.
- **N-API graph methods**: `addRelation`, `removeRelation`, `traverse`, `outgoingRelations`, `incomingRelations`, `declareRelationType`, `federatedQuery`.
- Relation persistence in arena log (survives restart and compaction).

### Changed
- Version bump from 0.2.0 to 0.3.0.

## [0.2.0] - 2026-06-16

### Added
- **Security features** (behind `security` feature flag):
  - Ed25519 signing for audit log non-repudiation.
  - AES-256-GCM encryption at rest with Argon2 key derivation.
  - Append-only audit trail with `audit_since()` query API.
- **Differential privacy**: `bundle_dp()` with configurable epsilon Laplace mechanism.
- `SecurityConfig` and `PrivacyConfig` in `HmsConfig`.
- N-API exposure: `auditSince`, `bundleTexts`, security/privacy config fields.
- Cross-platform npm publish workflow (linux-x64, darwin-arm64, darwin-x64, win32-x64).
- `cargo-deny` security auditing in CI.
- CODEOWNERS file for WritersLogic org.
- Documentation: `docs/SECURITY.md`, `docs/ARCHITECTURE.md`, `docs/PRIVACY.md`.
- Apache-2.0 SPDX license headers on all source files.

### Changed
- Migrated to `holographic-memory` npm scope.
- Updated all repository URLs to `github.com/writerslogic/hms`.
- CI workflow enhanced with caching, cargo-deny, and separate build stages.

## [0.1.0] - 2024-12-01

### Added
- Initial release: Binary Spatter Code (BSC) hyperdimensional computing engine.
- Sparse EntangledHVec with rho=1/256 sparsity.
- NSG (Navigable Small World) graph index for approximate nearest neighbors.
- IVF (Inverted File) with product quantization and Nystrom projection.
- Persistent mmap arena with LZ4 compression and CRC32 framing.
- N-API bindings for Node.js with async worker thread execution.
- Knowledge graph: triplet encoding, sequence memory, analogical reasoning.
- Concept synthesis via similarity-based clustering.
- Diffusion-based vector factorization.
- Text readability analysis.
- Auto-sharding at configurable thresholds.
