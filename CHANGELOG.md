# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-06-21

### Added
- **Provenance system** (behind `provenance` feature flag):
  - COSE Sign1 (RFC 9052) envelopes with Ed25519 for all provenance records.
  - W3C Verifiable Credentials (VC Data Model 2.0) with `eddsa-jcs-2022` Data Integrity
    proofs using two-step hash (RFC 8785 JCS canonicalization).
  - DID:key (Ed25519 multicodec `0xed01`, base58btc) and DID:web identifiers with
    DID document generation and key extraction.
  - C2PA Content Credentials manifests with assertion labels (CreativeWork, actions,
    hash.data, AI disclosure, store metadata) and ingredient chain support.
  - JUMBF (ISO 19566-5) binary box encoding/decoding for C2PA manifests with
    superbox/leaf-box hierarchy. Store manifests now include JUMBF binary.
  - CAWG Identity Assertion (Creator Assertions Working Group 1.1) binding DIDs to
    manifests with signed claims and HashedUri references.
  - Sigstore bundle v0.3 format with local Ed25519 keyful signing, digest verification,
    and bundle serialization roundtrip.
  - KERI (Key Event Receipt Infrastructure) Key Event Log with inception, rotation, and
    interaction events, self-addressing digests, chain verification, and disk persistence.
  - SCITT (draft-ietf-scitt-architecture) signed statements with CBOR payload
    (optional transparency log submission behind `provenance-scitt` feature).
  - BitstringStatusList credential revocation with per-credential status indexing.
  - Hash chain provenance log: append-only JSONL with SHA-256 chaining and COSE-signed
    head anchor for tamper detection.
  - Lamport logical clock for causal ordering, resilient across restarts and deletions.
  - Merkle tree batch signing for bulk import with per-record inclusion proofs.
  - Key rotation with chain-logged events and KERI integration.
  - HmsCore public API: `create_fact_provenance`, `create_triple_provenance`,
    `create_batch_provenance`, `create_self_manifest`, `verify_fact_provenance`,
    `verify_store_manifest`, `verify_provenance_log`, `revoke_credential`,
    `is_credential_revoked`, `create_sigstore_bundle`, `verify_sigstore_bundle`,
    `create_cawg_assertion`, `verify_cawg_assertion`.
- **Hopfield-Fenchel-Young**: Energy-based associative retrieval module.
- **Multi-scale encoding**: Morphological decomposer and fault-tolerant federation.
- **Agency layer**: GoalStore, Planner, QuestionGenerator, SelfModifier.
- **Research benchmarks**: Scaling analysis and visualization suite.

### Removed
- **CliffordVec**: Removed Clifford algebra multivector type and all associated code.
  Linear readout on additive bundles reads statistics not structure; unitary HRR matches
  or beats CliffordVec at O(D log D) on all axes at matched D.

### Fixed
- Replaced 20 bare `unwrap()` calls in production code with descriptive `expect()` or
  `total_cmp()` to prevent opaque panics.
- Fixed 4 silent `let _ =` error swallowing patterns in provenance log writes to use
  `tracing::warn`.
- Fixed logical clock sequence collision on restart after deletions (now uses max
  sequence + 1 instead of record count).
- Fixed eddsa-jcs-2022 signing to use two-step hash (proofOptionsHash + documentHash)
  per W3C Data Integrity spec.
- Fixed SCITT content type from envelope type to payload media type.

## [0.5.0] - 2026-06-18

### Added
- **Cognition layer**: Background discovery engine with 8 modules.
  - PatternScanner: relation co-occurrence analysis.
  - AbstractionEngine: prototype concept discovery via frequency counting.
  - GapDetector: epistemic gap detection against peer profiles.
  - HypothesisEngine: gap-filler proposals via Hopfield cleanup.
  - AnalogyDetector: structural isomorphism via connected components.
  - CognitionLoop: configurable background thread with read-only access.
  - MemoryGovernor: deduplication, forgetting, IDF refresh.
  - DistributionalRefiner: self-organizing atom vectors from relational context.
- `CognitionConfig` with all tuning parameters.
- N-API: `startCognition`, `stopCognition`, `runCognitionOnce`, `governMemory`.
- Coverage CI workflow (90.3% line coverage).
- MSRV set to 1.82.

### Fixed
- Multi-hop confidence decay: chained lookups now decay by 0.9 per hop.
- Empty-bundle in small-N abstraction: frequency counting fallback for < 10 members.
- `deny.toml` updated to cargo-deny v2 schema.
- Security advisories: updated lz4_flex and rand.
- Stale `index.d.ts` regenerated with full API surface.

### Changed
- AtomMemory, CompositeMemory, TripleStore wrapped in Arc for background thread sharing.

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
