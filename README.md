<p align="center">
  <img src="https://raw.githubusercontent.com/writerslogic/holographic-memory/main/assets/logo.png" width="200" alt="Holographic Memory System">
</p>

<h1 align="center">Holographic Memory System (HMS)</h1>

<p align="center">
  <strong>Privacy-preserving semantic search and associative memory — runs entirely on your machine.</strong>
</p>

<p align="center">
  <a href="https://github.com/writerslogic/holographic-memory/actions/workflows/ci.yml"><img src="https://github.com/writerslogic/holographic-memory/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/writerslogic/holographic-memory"><img src="https://api.securityscorecards.dev/projects/github.com/writerslogic/holographic-memory/badge" alt="OpenSSF Scorecard"></a>
  <a href="https://github.com/writerslogic/holographic-memory/actions/workflows/coverage.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/writerslogic/holographic-memory/main/.github/badges/coverage.json" alt="coverage"></a>
  <a href="https://www.npmjs.com/package/holographic-memory"><img src="https://img.shields.io/npm/v/holographic-memory.svg" alt="npm"></a>
  <a href="https://www.npmjs.com/package/holographic-memory"><img src="https://img.shields.io/npm/dm/holographic-memory.svg" alt="npm downloads"></a>
  <a href="https://crates.io/crates/holographic-memory"><img src="https://img.shields.io/crates/v/holographic-memory.svg" alt="crates.io"></a>
  <a href="https://crates.io/crates/holographic-memory"><img src="https://img.shields.io/crates/d/holographic-memory.svg" alt="crates.io downloads"></a>
  <a href="https://docs.rs/holographic-memory"><img src="https://docs.rs/holographic-memory/badge.svg" alt="docs.rs"></a>
  <a href="https://blog.rust-lang.org/2024/10/17/Rust-1.82.0.html"><img src="https://img.shields.io/badge/MSRV-1.82-blue.svg" alt="MSRV"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#installation">Install</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#why-hms">Why HMS?</a> &middot;
  <a href="#features">Features</a> &middot;
  <a href="#performance">Performance</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#provenance--content-credentials">Provenance</a>
</p>

---

HMS is a high-performance vector memory engine for Node.js, powered by Rust via N-API. It implements **Vector Symbolic Architectures (VSA)** using **Binary Spatter Code (BSC)** to deliver semantic search, analogical reasoning, relational knowledge graphs, and associative memory at scale — with no external API calls, no cloud dependencies, and no data leaving your device.

> Developed by [WritersLogic](https://github.com/writerslogic)

## Installation

```bash
npm install holographic-memory
```

```toml
# Rust
[dependencies]
holographic-memory = "0.6"
```

## Quick Start

### Semantic Search

```javascript
const { HolographicMemorySystem } = require('holographic-memory');

async function main() {
  const hms = new HolographicMemorySystem(10000, './hms_storage');

  await hms.memorizeText('paris', 'capital of france');
  await hms.memorizeText('berlin', 'capital of germany');

  const results = await hms.query('What is the capital of germany?', 1);
  console.log(results[0]); // { id: 'berlin', similarity: 0.85 }

  const analogy = await hms.findAnalogy('france', 'paris', 'germany');
  console.log(analogy[0].id); // 'berlin'
}

main().catch(console.error);
```

### Relational Knowledge (Meaning Memory)

```javascript
const hms = new HolographicMemorySystem(16384, './hms_storage', {
  meaningEnabled: true,
});

await hms.memorizeTriplet('t1', 'paris',  'capital_of', 'france');
await hms.memorizeTriplet('t2', 'berlin', 'capital_of', 'germany');
await hms.memorizeTriplet('t3', 'john',   'father',     'mark');
await hms.memorizeTriplet('t4', 'mark',   'father',     'bob');

// "What is the capital of France?"
const result = await hms.structuralQuery(['paris'], ['capital_of'], 'object');
console.log(result[0].entityId);   // 'france'
console.log(result[0].confidence); // 0.98

// "Who is John's grandfather?" (father → father)
const grandpa = await hms.multiHopQuery('john', ['father', 'father']);
console.log(grandpa[0].entityId);  // 'bob'
```

## Why HMS?

| Capability | HMS | Traditional vector DB |
|---|---|---|
| Runs locally | Yes | Usually cloud/daemon |
| External API calls | None | Often required |
| Analogical reasoning | Native | Not supported |
| Relational queries | Yes (role-filler algebra) | No |
| Multi-hop inference | Yes | No |
| Compression | Up to 4,096x | None |
| Language | Rust (N-API) | Python/Go |

<details>
<summary><strong>Features</strong> -- hybrid retrieval, symbolic operations, meaning memory, cognition engine</summary>

- **Hybrid Retrieval**: NSG (Navigable Small World) + IVF (Inverted File) + Sparse Inverted Index, routing dynamically by dataset statistics.
- **Symbolic Operations**: Binding (XOR), Bundling (Majority Rule), Permutation (Cyclic Shift) — native bitwise VSA operations.
- **Meaning Memory**: Structured relational layer with role-filler algebra, triple stores, multi-hop reasoning, and Hopfield attractor cleanup.
- **Cognition Engine**: Background discovery of patterns, abstractions, knowledge gaps, hypotheses, and cross-domain analogies from stored triples.
- **Graph Engine**: Typed relations with multi-hop traversal, transitive/symmetric inference, and temporal filtering.
- **Persistent Storage**: Custom `PersistentArena` with CRC32 integrity, LZ4 compression, and segmented mmap for crash-safe append-only persistence.
- **Federated Queries**: Query across multiple HMS instances in parallel without centralizing data.
- **Performance**: Zero-copy N-API, O(1) ID resolution, FxHash backend, O(N) selection via `select_nth_unstable`.

</details>

<details>
<summary><strong>Use Cases</strong> -- RAG, knowledge graphs, sequence matching, MCP tool servers</summary>

### Local RAG (Retrieval-Augmented Generation)
Store document chunks as hypervectors. Ingest external embeddings from any LLM (`Float32Array`) and use HMS as a local retrieval layer — no vector database infrastructure required.

### Semantic Knowledge Graphs
Encode `(Subject, Predicate, Object)` triples. Query: "What is the capital of France?" becomes `(France ⊗ Capital) ⊛ ?`. Solve analogies: `King : Man :: ? : Woman`.

### Sequence Pattern Matching
Use Cyclic Permutations to represent order. Query a sequence as fast as querying a single item — ideal for time-series, sentence structures, and behavior trajectories.

### MCP Tool Servers
HMS ships as the semantic memory backend for [scrivener-mcp](https://github.com/writerslogic/scrivener-mcp) and is designed for any Model Context Protocol integration that needs local semantic search.

</details>

<details>
<summary><strong>Performance</strong> -- compositional algebra, capacity scaling, noise tolerance benchmarks</summary>

All results use research-grade datasets: 120 real-world knowledge graph facts, 2,000 synthetic facts (Zipfian), 350 analogies across 7 relation types, sequences up to length 200.

### Compositional Algebra (D=16,384, density 1/256)

| Task | Accuracy | Dataset |
|------|----------|---------|
| Knowledge graph retrieval | **100%** | 2,120 facts, 114 entities, 7 relations |
| Analogy completion (A:B :: C:?) | **100%** | 350 analogies, 7 relation types |
| Sequence encoding & positional retrieval | **100%** | lengths 3–200, vocab 500, 10 trials each |
| Multi-hop inference (1–2 hops) | **100%** | 20 country chains |
| Binding fidelity (signal vs noise d') | **353.7** | 500 bind/unbind pairs |

### Capacity Scaling

| Dimension | Density | Hard Wall (95% recall) | Encode ops/s | Compression |
|-----------|---------|------------------------|--------------|-------------|
| 16,384 | 1/256 | 2,478 | 1,918,811 | 256x |
| 65,536 | 1/1024 | 9,800 | 1,888,303 | 1,024x |
| 262,144 | 1/4096 | 58,432 | 1,373,826 | 4,096x |

Scaling law: capacity wall ~ `density_denom × ln(dim)`.

### Noise Tolerance (Hopfield cleanup)

| Corruption | Jaccard NN | Hopfield cleanup |
|------------|-----------|------------------|
| 30% | 100% | 100% |
| 50% | 100% | 100% |
| 70% | 100% | 100% |

### Reproducing Benchmarks

```bash
# Compositional algebra, analogies, interference, sequences
cargo run --release --bin hms-research-bench -- --dim 16384 --density 256 --json

# Capacity walls, throughput, compression
cargo run --release --bin hms-scaling -- --dim 16384 --density 256 --json

# Full 8-section suite
cargo run --release --bin hms-benchmark-suite -- --dim 16384
```

</details>

<details>
<summary><strong>Architecture</strong> -- core retrieval, meaning memory, cognition engine, configuration</summary>

### Core Retrieval

HMS uses a hybrid index that routes each query based on dataset statistics:

- **NSG (Navigable Small World)**: Proximity graph for approximate nearest neighbors, high search efficiency and index compactness.
- **IVF (Inverted File)**: Coarse-grained quantization for large datasets.
- **Sparse Inverted Index**: Term-based retrieval for high-sparsity queries.

### Meaning Memory

A structured knowledge layer on top of the holographic vector space:

- **AtomMemory**: Stores individual concept vectors with deterministic seeding for reproducible embeddings.
- **CompositeMemory**: Encodes `(subject, relation, object)` triples as single composite vectors via role-shifted XOR binding.
- **TripleStore**: Symbolic FxHash index with four-way lookup (by subject, relation, object, composite ID).
- **Hopfield Cleanup**: After algebraic unbinding, uses sparse softmax attention to snap noisy residuals to the nearest stored atom.

### Cognition Engine

Background discovery thread (default 60s interval):

- **PatternScanner**: Surfaces structural regularities across triples.
- **AbstractionEngine**: Bundles atom vectors to create prototype categories when N entities share a relation pattern.
- **GapDetector**: Finds missing relations by comparing an entity's profile to its peers.
- **HypothesisEngine**: Proposes fillers for detected gaps using Hopfield cleanup.
- **AnalogyDetector**: Finds structurally isomorphic domains via bipartite relation mapping.

### Configuration

```javascript
const hms = new HolographicMemorySystem(16384, './storage', {
  meaningEnabled: true,
  meaningBeta: 24.0,         // Hopfield temperature
  meaningMaxFanout: 40,      // Algebraic vs materialized path threshold
  meaningMaxHopDepth: 10,    // Multi-hop chain limit
});
```

```rust
let mut config = HmsConfig::default();
config.meaning.enabled = true;
config.cognition.enabled = true;
config.cognition.interval_secs = 60;
config.meaning.beta = 24.0;
config.meaning.algebraic_max_fanout = 40;
```

</details>

<details>
<summary><strong>Provenance and Content Credentials</strong> -- COSE Sign1, W3C VC, C2PA, SCITT, KERI, Sigstore</summary>

HMS includes tamper-evident provenance built on open standards — entirely local, no external services.

```toml
[dependencies]
holographic-memory = { version = "0.6", features = ["provenance"] }
```

| Standard | Implementation |
|----------|----------------|
| COSE Sign1 (RFC 9052) | Ed25519 signature envelopes |
| W3C Verifiable Credentials 2.0 | `eddsa-jcs-2022` Data Integrity proofs |
| DID:key / DID:web | Ed25519 multicodec, domain-based identifiers |
| C2PA 2.1 | Content Credentials manifests |
| SCITT | Signed statements with optional transparency log |
| KERI | Persistent Key Event Log with rotation |
| Sigstore Bundle v0.3 | Local keyful signing |

```rust
use holographic_memory::HmsCore;

let hms = HmsCore::new(16384, Some("./storage".into()), None)?;

let record = hms.create_fact_provenance("fact-001", b"Paris is the capital of France", None)?;
assert!(record.cose_envelope.is_some());

let result = hms.verify_fact_provenance(&record)?;
assert!(result.valid);

let manifest = hms.create_self_manifest(Some("My Knowledge Store"))?;
assert!(manifest.jumbf_manifest.is_some());
```

**Verify it yourself:**
```bash
cargo run --features provenance --example verify_cogmem_sample
```
Re-verifies the exact COSE/SCITT statements from cogmem's public C2PA sample under this crate's independent implementation — identical bytes, different verifier.

</details>

## Part of the Agent-Provenance Stack

HMS is one component of the WritersLogic verifiable agent-provenance pipeline — agent identity, memory, reasoning, and signed output, cryptographically bound end to end.

| Project | Role |
|---|---|
| [cogmem](https://github.com/writerslogic/cogmem) | Agent identity (CAWG credential) + verifiable memory (COSE/SCITT) |
| [crosstalk](https://github.com/writerslogic/crosstalk) | Multi-model orchestrator; signs reasoning/orchestration audit |
| **holographic-memory** | Durable memory store; cross-verifies signed statements and agent identity |
| WritersProof | C2PA producer: binds identity + memory + reasoning to the signed asset |

All four share one substrate — COSE_Sign1 / SCITT (Ed25519) and W3C DID — specified in [UNIFIED-PROVENANCE.md](https://github.com/writerslogic/cogmem/blob/main/UNIFIED-PROVENANCE.md).

## Development

```bash
# Build (set local cargo dirs to avoid permission issues)
export CARGO_HOME=$(pwd)/.cargo_home
export CARGO_TARGET_DIR=/tmp/hms-target
npm run build

# Test
cargo test --lib
```

## Security

Hyperdimensional vectors are inherently lossy; original content cannot be reconstructed from stored vectors. For vulnerability reporting see [SECURITY.md](.github/SECURITY.md).

## License

Apache License, Version 2.0 — see [LICENSE](LICENSE).
