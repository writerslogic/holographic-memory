<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-white.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/logo-black.svg">
  <img src="./assets/logo-black.svg" width="120" alt="Holographic Memory System" align="left">
</picture>

<h3>Holographic Memory System (HMS)</h3>
<p><strong>Privacy-preserving semantic search and associative memory — runs entirely on your machine.</strong></p>

<br clear="left">

<!-- Badge palette: dynamic health; metadata #007ec6; standards #6a4c93; label #20232a; platform brand colors. -->

<p align="center">
  <a href="https://github.com/writerslogic/holographic-memory/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/writerslogic/holographic-memory/ci.yml?branch=main&amp;style=flat-square&amp;label=CI&amp;labelColor=20232a" alt="CI"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/writerslogic/holographic-memory"><img src="https://img.shields.io/ossf-scorecard/github.com/writerslogic/holographic-memory?style=flat-square&amp;labelColor=20232a" alt="OpenSSF Scorecard"></a>
  <a href="https://github.com/writerslogic/holographic-memory/actions/workflows/coverage.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/writerslogic/holographic-memory/main/.github/badges/coverage.json&amp;style=flat-square&amp;labelColor=20232a" alt="Coverage"></a>
  <a href="https://www.bestpractices.dev/projects/13977"><img src="https://www.bestpractices.dev/projects/13977/badge" alt="OpenSSF Best Practices"></a>
  <a href="https://www.npmjs.com/package/holographic-memory"><img src="https://img.shields.io/npm/v/holographic-memory?style=flat-square&amp;color=007ec6&amp;labelColor=20232a&amp;logo=npm" alt="npm version"></a>
  <a href="https://crates.io/crates/holographic-memory"><img src="https://img.shields.io/crates/v/holographic-memory?style=flat-square&amp;color=007ec6&amp;labelColor=20232a&amp;logo=rust" alt="crates.io version"></a>
  <a href="https://docs.rs/holographic-memory"><img src="https://img.shields.io/docsrs/holographic-memory?style=flat-square&amp;color=007ec6&amp;labelColor=20232a&amp;logo=docs.rs" alt="docs.rs"></a>
  <a href="https://blog.rust-lang.org/2025/08/07/Rust-1.89.0.html"><img src="https://img.shields.io/badge/MSRV-1.89-007ec6?style=flat-square&amp;labelColor=20232a&amp;logo=rust" alt="MSRV 1.89"></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/local--first-yes-007ec6?style=flat-square&amp;labelColor=20232a" alt="Local-first">
  <a href="https://github.com/sponsors/dcondrey"><img src="https://img.shields.io/badge/sponsor-dcondrey-EA4AAA?style=flat-square&amp;labelColor=20232a&amp;logo=githubsponsors&amp;logoColor=white" alt="Sponsor dcondrey"></a>
</p>

<p align="center">
  <a href="#installation">Install</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#why-hms">Why HMS?</a> &middot;
  <a href="#features">Features</a> &middot;
  <a href="#performance">Performance</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#provenance--content-credentials">Provenance</a> &middot;
  <a href="docs/production-readiness.md">Production Readiness</a>
</p>

---

HMS is a high-performance vector memory engine for Rust and Node.js. It implements **Vector Symbolic Architectures (VSA)** using **Binary Spatter Code (BSC)** to deliver semantic search, analogical reasoning, relational knowledge graphs, and associative memory — with no external API calls, no cloud dependencies, and no data leaving your device. Optional features add encrypted storage, signed audit logs, credential-gated agent access, and COSE/SCITT/C2PA provenance.

> Developed by [WritersLogic](https://github.com/writerslogic)

## Python SDK (experimental)

`holographic-sdk` (in `bindings/python/holographic-sdk`) is an HTTP client with adapters for LangChain, LlamaIndex, Haystack, Semantic Kernel, CrewAI, Phidata, PydanticAI, SmolAgents, DSPy, and Embedchain. It expects an HMS-backed HTTP service exposing `/api/v1/documents` and `/api/v1/query`; **this repository does not ship that service**, so the SDK is only useful against a server you provide.

## Rust & Node.js Core Installation

The core backend is implemented in pure Rust.

```bash
cargo add holographic-memory
# Or for Node:
npm install holographic-memory
```

```toml
# Rust
[dependencies]
holographic-memory = "0.6"
```

## Security Features

All of these are opt-in Cargo features; `default = []`. See [SECURITY.md](docs/SECURITY.md) and [PRIVACY.md](docs/PRIVACY.md) for the threat model and limits.

- **Encrypted storage and signed audit log** (`security`): AES-256-GCM over arena payloads and index caches with an Argon2id-derived key; Ed25519-signed audit entries.
- **Client-side vector masking** (`security`, `core::mask::VectorMask`): an Argon2id-keyed secret permutation applied before vectors leave the client, so a remote store can rank them without the original coordinates. This is obfuscation, **not encryption and not homomorphic encryption**: it preserves and therefore reveals all pairwise similarities, and known plaintext/masked pairs progressively reveal the key.
- **Credential-gated agent access** (`provenance`, `core::provenance::access`): agents are admitted by W3C Verifiable Credentials with an `eddsa-jcs-2022` proof from an issuer `did:key` the host explicitly trusts, with expiry and revocation. `HmsCore::query_as` enforces a read grant. The host must still authenticate that a caller controls the DID it presents.
- **Provenance** (`provenance`, `provenance-scitt`): COSE_Sign1 signed statements, SCITT registration, and C2PA manifests in JUMBF.

Not implemented: PoSME receipts, RATS attestation verification, `did:web` resolution for access credentials, and any form of search over encrypted data.

## Quick Start

### Document Search

```javascript
const { HolographicMemorySystem } = require('holographic-memory');
const { DocumentMemory } = require('holographic-memory/semantic');

async function main() {
  const hms = new HolographicMemorySystem(16384, './documents-v2');
  const memory = new DocumentMemory(hms);
  await memory.memorize({
    id: 'backups', text: 'Restore deleted documents from verified backups.',
    sourceUri: 'operations.md', version: '1', metadata: { project: 'alpha' },
  });
  const [hit] = await memory.search('restore backups', { filter: { project: 'alpha' } });
  console.log(hit.documentId, hit.sourceUri, hit.text);
  await memory.flush();
}
main().catch(console.error);
```

Writing the same document ID replaces its chunks atomically. Results include source byte
offsets; `memory.delete(id)` removes every chunk. See [resource limits and migration](docs/production-readiness.md).
**Existing stores without the new embedding schema require re-encoding from original sources.**

### Local Embeddings and Reranking

Install `@huggingface/transformers` and obtain compatible ONNX model files in local directories.
The factories below never download models. Use actual artifact revisions and configure the
store from the returned embedding space; it also fingerprints pooling, prefixes, and dtype.

```javascript
const { createLocalEmbedder, createLocalReranker, DocumentMemory } = require('holographic-memory/semantic');

const embedder = await createLocalEmbedder({
  modelPath: process.env.HMS_EMBEDDING_DIR,
  modelId: 'Xenova/all-MiniLM-L6-v2',
  revision: process.env.HMS_EMBEDDING_REVISION,
  dtype: 'q8',
});
const rerank = await createLocalReranker({ modelPath: process.env.HMS_RERANKER_DIR, dtype: 'q8' });
const hms = new HolographicMemorySystem(16384, './semantic-v2', {
  embeddingModel: embedder.space.model,
  embeddingRevision: embedder.space.revision,
  embeddingDimensions: embedder.space.dimensions,
});
const memory = new DocumentMemory(hms, { embedder, rerank });
await memory.memorize({ id: 'vehicle', text: 'The automobile needs a mechanic.', sourceUri: 'notes.md' });
console.log(await memory.search('Where can I get my car repaired?', { k: 3 }));
await memory.flush();
await rerank.dispose();
await embedder.dispose();
```

Run the complete [document example](examples/local-documents.mjs) or the reproducible
[quality and latency evaluation](docs/evaluation.md). Exact dense cosine and BM25 scan eligible
chunks before optional reranking; benchmark your corpus before assuming a deployment capacity.

### Relational Knowledge (Meaning Memory)

```javascript
const hms = new HolographicMemorySystem(16384, './knowledge-v2', {
  meaningEnabled: true,
});

await hms.memorizeTriplet('t1', 'paris',  'capital_of', 'france');
await hms.memorizeTriplet('t2', 'berlin', 'capital_of', 'germany');
await hms.memorizeTriplet('t3', 'john',   'father',     'mark');
await hms.memorizeTriplet('t4', 'mark',   'father',     'bob');

// "Paris is the capital of which country?"
const result = await hms.structuralQuery(['paris'], ['capital_of'], 'object');
console.log(result[0].entityId);   // 'france'
console.log(result[0].confidence); // confidence depends on the stored knowledge

// Follow two outgoing father relations: john → mark → bob.
const descendants = await hms.multiHopQuery('john', ['father', 'father']);
console.log(descendants[0].entityId);  // 'bob'
```

## Why HMS?

- Local lexical and semantic document retrieval with passages, versions, and metadata filters.
- Vector-symbolic composition and structured multi-hop queries in one native engine.
- Transactional mutations, verified compaction generations, and explicit index maintenance.
- Optional encryption, audit, and provenance features with runtime capability reporting.

<details>
<summary><strong>Features</strong> -- hybrid retrieval, symbolic operations, meaning memory, cognition engine</summary>

- **Documents**: BM25 + supplied dense cosine embeddings + optional local cross-encoder reranking.
- **Vector Retrieval**: NSG (Navigable Small World) + IVF (Inverted File) + Sparse Inverted Index, routing dynamically by dataset statistics.
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

The following historical results describe isolated algebra/research workloads, not the new document pipeline or current end-to-end latency. Their datasets include: 120 real-world knowledge graph facts, 2,000 synthetic facts (Zipfian), 350 analogies across 7 relation types, sequences up to length 200.

### Compositional Algebra (D=16,384, density 1/256)

| Task | Accuracy | Dataset |
|------|----------|---------|
| Knowledge graph retrieval | **100%** | 2,120 facts, 114 entities, 7 relations |
| Analogy completion (A:B :: C:?) | **100%** | 350 analogies, 7 relation types |
| Sequence encoding & positional retrieval | **100%** | lengths 3–200, vocab 500, 10 trials each |
| Multi-hop inference (1–2 hops) | **100%** | 20 country chains |
| Binding fidelity (signal vs noise d') | **353.7** | 500 bind/unbind pairs |

### Capacity Scaling

Items stored in one union (Bloom) bundle before members and non-members stop being separable, from `benchmarks/results/benchmark_scaling_results.json` (50 member and 50 non-member probes per point):

| Dimension | Density | Last N with a positive member/non-member gap | Last N with d' ≥ 2 | Encode ops/s | Compression |
|-----------|---------|----------------------------------------------|--------------------|--------------|-------------|
| 16,384 | 1/256 | 500 | 1,000 | 1,918,811 | 256x |
| 65,536 | 1/1024 | 2,800 | 4,000 | 1,888,303 | 1,024x |
| 262,144 | 1/4096 | 11,200 | 16,000 | 1,373,826 | 4,096x |

At D=16,384 the measured false-positive rate is 4% at N=500 and 32% at N=1,000. Earlier versions of this table reported 2,478 / 9,800 / 58,432 as a "hard wall (95% recall)"; that is the point where the bundle is fully saturated and the false-positive rate is 100%, not a usable capacity. The tested points are research observations; they do not establish an application capacity guarantee.

### Noise Tolerance (Hopfield cleanup)

| Corruption | Jaccard NN | Hopfield cleanup |
|------------|-----------|------------------|
| 30% | 100% | 100% |
| 50% | 100% | 100% |
| 70% | 100% | 100% |

### Reproducing Benchmarks

```bash
# Compositional algebra, analogies, interference, sequences
cargo run --release --features experimental --bin hms-research-bench -- --dim 16384 --density 256 --json

# Capacity walls, throughput, compression
cargo run --release --features experimental --bin hms-scaling -- --dim 16384 --density 256 --json

# Full 8-section suite
cargo run --release --features experimental --bin hms-benchmark-suite -- --dim 16384

# Machine-readable recall and latency regression report
cargo run --release --bin hms-eval -- \
  --vectors 1500 --queries 100 --dimensions 16384 --assert-min-recall 0.90
```

Synthetic results characterize controlled HMS workloads; they are not a substitute for
evaluation on your application data. See the
[production-readiness guide](docs/production-readiness.md) for capacity planning and selection
criteria versus conventional vector databases.

### Store Administration

```bash
# Read metadata or verify every frame
cargo run --release --bin hms-admin -- inspect ./store
cargo run --release --bin hms-admin -- verify ./store

# Create a locked, verified, atomically published copy
cargo run --release --bin hms-admin -- migrate ./store ./store-copy
```

Writable stores are protected by an exclusive process lock. The production-readiness guide
documents the exact locking and migration behavior.

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

Lossy vectors are not encryption or anonymization. Document ingestion stores source passages by default. Enable encryption when required, inspect `securityStatus()`, and read [PRIVACY.md](docs/PRIVACY.md) for precise guarantees and limitations. For vulnerability reporting see [SECURITY.md](.github/SECURITY.md).

## Licensing (Dual-License Model)

Holographic Memory System (HMS) uses a **Dual-Licensing** model:

1. **Open Source (AGPL-3.0):** The core engine is free to use and modify for open-source projects, personal use, or internal evaluation, provided you comply with the [GNU Affero General Public License v3.0](LICENSE). Note that using HMS as a backend for a proprietary service over a network requires you to open-source your service under AGPL, or purchase a commercial license.
2. **Commercial License (WritersLogic Enterprise):** For companies building closed-source, proprietary software, you must purchase a Commercial License. This bypasses the AGPL restrictions and provides production SLA support.

Contact licensing@writerslogic.com for enterprise inquiries.
