<p align="center">
  <img src="https://raw.githubusercontent.com/writerslogic/holographic-memory/main/assets/logo.svg" width="200" alt="HMS Logo">
</p>

# Holographic Memory System (HMS)

[![CI](https://github.com/writerslogic/holographic-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/writerslogic/holographic-memory/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/writerslogic/holographic-memory/main/.github/badges/coverage.json)](https://github.com/writerslogic/holographic-memory/actions/workflows/coverage.yml)
[![crates.io](https://img.shields.io/crates/v/holographic-memory.svg)](https://crates.io/crates/holographic-memory)
[![docs.rs](https://docs.rs/holographic-memory/badge.svg)](https://docs.rs/holographic-memory)
[![npm](https://img.shields.io/npm/v/holographic-memory.svg)](https://www.npmjs.com/package/holographic-memory)
[![npm downloads](https://img.shields.io/npm/dm/holographic-memory.svg)](https://www.npmjs.com/package/holographic-memory)
[![crates.io downloads](https://img.shields.io/crates/d/holographic-memory.svg)](https://crates.io/crates/holographic-memory)
[![MSRV](https://img.shields.io/badge/MSRV-1.82-blue.svg)](https://blog.rust-lang.org/2024/10/17/Rust-1.82.0.html)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Privacy-preserving semantic search using hyperdimensional computing.**

A high-performance **Holographic Memory System (HMS)** for Node.js, powered by Rust. This library implements **Vector Symbolic Architectures (VSA)** using **Binary Spatter Code (BSC)** and **Sparse Distributed Representations (SDR)** to enable semantic search, analogical reasoning, and associative memory at scale.

> Developed by [WritersLogic](https://github.com/writerslogic) -- local-first intelligence with no data leaving your machine.

## Features

- **Hybrid Retrieval Architecture**: 
  - **NSG (Navigable Small World)**: Fast proximity graph for approximate nearest neighbors.
  - **IVF (Inverted File)**: Coarse-grained quantization for large datasets.
  - **Sparse Inverted Index**: Term-based retrieval for high-sparsity queries.
- **Symbolic Operations**: Native bitwise **Binding (XOR)**, **Bundling (Majority Rule)**, and **Permutation (Cyclic Shift)**.
- **Performance Optimized**:
  - **O(1) Resolution**: Cached physical location lookups for instant ID retrieval.
  - **FxHash Backend**: Ultra-fast non-cryptographic hashing for all retrieval collections.
  - **O(N) Selection**: Linear-time candidate pruning using `select_nth_unstable`.
- **Persistent Storage**: Custom `PersistentArena` with CRC32 integrity, LZ4 compression, and segmented mmap for crash-safe append-only persistence.
- **Meaning Memory**: Structured knowledge layer with role-filler algebra, triple stores, multi-hop reasoning, and Hopfield attractor cleanup.
- **Cognition Engine**: Background discovery of patterns, abstractions, knowledge gaps, hypotheses, and cross-domain analogies from stored triples.
- **Graph Engine**: Explicit typed relations with multi-hop traversal, transitive/symmetric inference, and temporal filtering.
- **Federated Queries**: Query across multiple HMS instances in parallel without centralizing data.
- **Node.js Bindings**: High-efficiency N-API implementation with asynchronous worker thread execution.

## Integrations

HMS is available as both a Node.js package and a high-performance Rust crate.

### Node.js (N-API)
```bash
npm install holographic-memory
```

### Rust (Crates.io)
Add to your `Cargo.toml`:
```toml
[dependencies]
holographic-memory = "0.4"
```

## Core Architecture

HMS is designed for local-first intelligence, combining advanced research in Hyperdimensional Computing with efficient retrieval algorithms.

- **Advanced Search**: Implements the **NSG (Navigable Small World)** algorithm, offering high search efficiency and index compactness.
- **Adaptive Routing**: Employs a retrieval strategy that dynamically switches between graph-based, quantized, and inverted indexing based on dataset statistics.
- **Neuro-Symbolic VSA**: A robust implementation of **Binary Spatter Code (BSC)**, enabling relational logic $(A \otimes B)$ combined with the associative matching of high-dimensional vector spaces.
- **Efficient Data Path**: Engineered with a zero-copy N-API interface, $O(1)$ ID resolution, and hardware-aware optimizations for high single-node throughput.

## Use Cases

### 1. Semantic Search & Local RAG
Store text fragments or documents as hypervectors. While HMS uses high-speed **Deterministic 3-Gram Encoding** for lexical similarity, it also supports **LLM Integration**. Ingest SOTA embeddings from models like GPT-4 or Llama-3 (via `Float32Array`) and use HMS as your high-performance retrieval and reasoning layer.

### 2. Symbolic Knowledge Graphs (Holographic Graph)
Encode relational triplets `(Subject, Predicate, Object)` into a single hypervector. 
- **Querying**: "What is the capital of France?" becomes `(France ⊗ Capital) ⊛ ?`.
- **Analogies**: Solve `King : Man :: ? : Woman` via holographic vector arithmetic.

### 3. Real-Time Sequence Pattern Matching
Use **Cyclic Permutations** to represent order. Ideal for temporal data like time-series patterns, sentence structures, or user behavior trajectories. Querying for a sequence is as fast as querying for a single item.

### 4. Concept Synthesis & Abstraction
Use the `synthesizeConcepts` method to identify "abstractions" within your memory. HMS clusters similar hypervectors and generates a **centroid** that represents the common features of the cluster—essentially "dreaming" up generalized categories from raw data.

### 5. Explainable Vector Decomposition
Hypervectors in HMS are **Distributed Representations**. You can use `analyzeComponents` to decompose a complex bundled vector back into its constituent symbols, providing a "reasoning" trace for why a certain item was retrieved.

## Meaning Memory

Meaning Memory is a structured knowledge layer built on top of the holographic vector space. It enables HMS to store, query, and reason over relational knowledge using role-filler algebra rather than flat vector similarity alone.

### Architecture

The system is composed of five cooperating modules:

- **AtomMemory**: Stores individual concept vectors (atoms). Each concept (e.g., "paris", "capital_of") gets a unique high-dimensional vector. Supports deterministic seeding for reproducible embeddings.
- **CompositeMemory**: Stores composite vectors formed by binding atoms with role assignments. A triple like `(paris, capital_of, france)` is encoded as a single composite vector using role-shifted XOR binding.
- **TripleStore**: A symbolic index over `(subject, relation, object)` triples with four-way FxHash indexing (by subject, relation, object, and composite ID). Enables fast materialized lookups when algebraic decoding is too expensive.
- **RoleRegistry**: Assigns cyclic-shift values to named roles (subject=0, relation=1, object=3). The asymmetric shifts break XOR commutativity, ensuring `(A, r, B)` and `(B, r, A)` produce distinct composites. Custom roles can be registered for n-ary relations.
- **IndexedMemory**: The shared substrate beneath AtomMemory and CompositeMemory, providing posting-list overlap scanning, IDF weighting with proportional clipping, and tombstone-based soft deletion.

### Structural Queries

Structural queries answer relational questions like "What is the capital of France?" by operating on the composite vector space:

1. **Compose** the known bindings into a partial query vector using role algebra.
2. **Overlap scan** the composite memory for matching composites (IDF-weighted posting intersection).
3. **Admission gating** decides the decode path based on fan-out:
   - **Algebraic path** (fan-out <= threshold): XOR-unbind the query from each composite, inverse-permute by the target role's shift, then run Hopfield attractor cleanup to recover the missing atom.
   - **Materialized path** (fan-out > threshold): Fall back to TripleStore index lookups for exact symbolic matches.

### Multi-Hop Reasoning

Multi-hop queries traverse chains of relations (e.g., "Who is John's grandfather?" via `father -> father`):

- **Rule rewrite**: If a `CompositionRule` maps `[father, father] -> grandfather`, the query is rewritten as a single algebraic lookup against the derived relation.
- **Chained lookup**: Otherwise, the system iteratively walks the TripleStore, expanding entities hop-by-hop up to a configurable depth limit.
- **Single algebraic**: For single-relation queries, the system uses a direct structural query.

### Hopfield Attractor Cleanup

After algebraic unbinding, the result vector is noisy. HMS uses a modern Hopfield network with sparse softmax attention to "clean up" the residual vector and snap it to the nearest stored atom:

1. Overlap-scan the noisy vector against AtomMemory's posting lists.
2. Compute sparse softmax attention weights over the top candidates (temperature controlled by `beta`).
3. Reconstruct a new vector from the attention-weighted sum of stored atoms.
4. Iterate until convergence (similarity > 0.999) or max iterations reached.

### Configuration

Enable meaning memory in your config:

```rust
let mut config = HmsConfig::default();
config.meaning.enabled = true;
config.meaning.beta = 24.0;              // Hopfield temperature
config.meaning.idf_clip_factor = 3.0;    // IDF clipping (poisoning defense)
config.meaning.algebraic_max_fanout = 40; // Admission control threshold
config.meaning.max_hop_depth = 10;       // Multi-hop chain limit
```

```javascript
const hms = new HolographicMemorySystem(16384, './storage', {
  meaning: {
    enabled: true,
    beta: 24.0,
    idfClipFactor: 3.0,
    algebraicMaxFanout: 40,
    maxHopDepth: 10,
  },
});
```

## Cognition Layer

The Cognition layer is a background discovery engine that finds implicit knowledge from stored triples. All components operate with read-only access to meaning memory stores; discovered insights require explicit confirmation before becoming stored facts.

### Components

- **PatternScanner**: Groups triples by relation, counts co-occurring subject/object atoms, and surfaces recurring structural regularities.
- **AbstractionEngine**: When N entities share the same relation pattern, bundles their atom vectors to create a prototype/category concept.
- **GapDetector**: Compares an entity's relation profile against its peers to find missing relations (e.g., "most cities have a country, but city X doesn't").
- **HypothesisEngine**: Proposes fillers for detected gaps by bundling peer data and running Hopfield cleanup to find the nearest stored atom.
- **AnalogyDetector**: Finds structurally isomorphic domains via connected components and greedy bipartite mapping by relation overlap.
- **CognitionLoop**: Background thread that runs all components on a configurable interval (default 60s). Insights are collected in a separate buffer without write-locking the main stores.
- **MemoryGovernor**: Consolidates near-duplicate composites, forgets stale entries, and rebuilds IDF weights. Called explicitly, not automatically.

### Configuration

```rust
let mut config = HmsConfig::default();
config.meaning.enabled = true;
config.cognition.enabled = true;
config.cognition.interval_secs = 60;
config.cognition.min_pattern_freq = 3;
config.cognition.min_hypothesis_confidence = 0.3;
```

## Core Concepts

### Hyperdimensional Computing (HDC)
Traditional AI uses deep vectors (weights). HDC uses high-dimensional (e.g., 10,000+), sparse vectors where information is "holographically" distributed across every dimension. 

- **Binding (⊗)**: Combines two vectors into a new, orthogonal vector representing their relationship. Reversible.
- **Bundling (⊛)**: Combines multiple vectors into a single vector that retains similarity to all its components.
- **Permutation (Π)**: Represents sequence and structure by shifting bits.

## Quick Start

### Semantic Search

```javascript
const { HolographicMemorySystem } = require('holographic-memory');

async function main() {
  // Initialize with 10,000 dimensions
  const hms = new HolographicMemorySystem(10000, './hms_storage');

  // Memorize associations
  await hms.memorizeText('paris', 'capital of france');
  await hms.memorizeText('berlin', 'capital of germany');

  // Semantic Query
  const results = await hms.query('What is the capital of germany?', 1);
  console.log('Match:', results[0]); // { id: 'berlin', similarity: 0.85 }

  // Analogical Reasoning
  const analogy = await hms.findAnalogy('france', 'paris', 'germany');
  console.log('Result:', analogy[0].id); // 'berlin'
}

main().catch(console.error);
```

### Meaning Memory (Structural Queries)

```javascript
const { HolographicMemorySystem } = require('holographic-memory');

async function main() {
  const hms = new HolographicMemorySystem(16384, './hms_storage', {
    meaning: { enabled: true },
  });

  // Store relational triples
  await hms.memorizeTriple('paris', 'capital_of', 'france');
  await hms.memorizeTriple('berlin', 'capital_of', 'germany');
  await hms.memorizeTriple('john', 'father', 'mark');
  await hms.memorizeTriple('mark', 'father', 'bob');

  // Structural query: "What is the capital of France?"
  const result = await hms.structuralQuery(
    { subject: 'paris', relation: 'capital_of' },
    'object'
  );
  console.log(result[0].entityId);    // 'france'
  console.log(result[0].confidence);  // 0.98

  // Multi-hop: "Who is John's grandfather?" (father -> father)
  const grandpa = await hms.multiHopQuery('john', ['father', 'father']);
  console.log(grandpa[0].entityId);   // 'bob'
  console.log(grandpa[0].hops.length); // 2
}

main().catch(console.error);
```

## Benchmark Results

HMS implements **Entangled Hypervectors (EHV)**: sparse binary vectors with deterministic structure. The core representation uses Binary Spatter Code with configurable sparsity (density = 1/k), enabling both high capacity and extreme compression.

All results below use research-grade datasets: 120 real-world knowledge graph facts (countries, capitals, languages, continents, currencies, rivers, animals), 2000 synthetic facts with Zipfian distribution, 350 analogies across 7 relation types, and sequence encoding up to length 200 with vocabulary 500.

### Compositional Algebra (D=16,384, density 1/256)

| Task | Accuracy | Dataset |
|------|----------|---------|
| Knowledge graph retrieval (arg1, arg2, relation) | **100%** | 2,120 facts, 114 entities, 7 relations |
| Analogy completion (A:B :: C:?) | **100%** | 350 analogies, 7 relation types |
| Sequence encoding & positional retrieval | **100%** | lengths 3-200, vocab 500, 10 trials each |
| Multi-hop inference (1-hop, 2-hop, chained) | **100%** | 20 country chains (capital + continent) |
| Structured role retrieval (up to 100 roles) | **100%** | 100 unique permutation-based roles |
| Binding fidelity (signal vs noise d') | **353.7** | 500 bind/unbind pairs |

### Capacity Scaling (Modal cloud, 9 configurations)

| Dimension | Density | Active | Hard Wall (95% recall) | Encode ops/s | Compression |
|-----------|---------|--------|------------------------|--------------|-------------|
| 16,384 | 1/256 | 64 | 2,478 | 1,918,811 | 256x |
| 65,536 | 1/1024 | 64 | 9,800 | 1,888,303 | 1,024x |
| 131,072 | 1/1024 | 128 | 12,308 | 735,841 | 1,024x |
| 262,144 | 1/4096 | 64 | 58,432 | 1,373,826 | 4,096x |
| 524,288 | 1/4096 | 128 | 51,232 | 397,041 | 4,096x |

**Scaling law**: capacity wall ~ density_denom x ln(dim). Throughput depends on active index count, not total dimension.

### Noise Tolerance (Hopfield attractor cleanup)

| Corruption | Jaccard NN | Hopfield cleanup |
|------------|-----------|------------------|
| 30% | 100% | 100% |
| 50% | 100% | 100% |
| 70% | 100% | 100% |

### Interference: Individual vs Bundled Storage

Individual composition (each fact in its own vector) maintains 100% accuracy at all scales tested. Bundled Bloom storage (multiple facts OR'd into one vector) degrades due to union saturation:

| Facts bundled | Individual accuracy | Bundled accuracy | Bundle density |
|---------------|--------------------|-----------------:|----------------|
| 5 | 100% | 20.0% | 3.8% |
| 10 | 100% | 10.0% | 7.5% |
| 50 | 100% | 1.7% | 32.4% |
| 500 | 100% | 0.0% | 98.0% |

This is the known limitation of binary OR bundling: union degeneracy destroys per-fact identity as density approaches 1.0.

### Reproducing Benchmarks

```bash
# Research benchmark (compositional algebra, analogies, interference, sequences)
cargo run --release --bin hms-research-bench -- --dim 16384 --density 256 --json

# Scaling benchmark (capacity walls, throughput, compression)
cargo run --release --bin hms-scaling -- --dim 16384 --density 256 --json

# Full 10-section suite (binding, Hopfield, Clifford, HBM, encoding)
cargo run --release --bin hms-benchmark-suite -- --dim 16384

# Cloud-parallel scaling across 9 configs (requires Modal account)
modal run modal_benchmark.py
```

## Development

### Build Environment
To bypass global permission issues and optimize build performance, use the following configuration:

```bash
# Set local cargo home and target directory
export CARGO_HOME=$(pwd)/.cargo_home
export CARGO_TARGET_DIR=/Volumes/C/target

# Build
npm run build
```

### Testing
```bash
# Run the unit and integration tests
export CARGO_HOME=$(pwd)/.cargo_home
export CARGO_TARGET_DIR=/Volumes/C/target
cargo test --lib
```

## Ecosystem

### Built with HMS

- **[scrivener-mcp](https://github.com/writerslogic/scrivener-mcp)** -- MCP server for Scrivener writing projects, using HMS for semantic search across manuscripts and research notes.

### Where HMS fits

HMS is a general-purpose vector memory engine. Beyond writing tools, it's well-suited for:

- **Local RAG pipelines** -- privacy-preserving retrieval for LLM applications without sending data to external APIs
- **MCP tool servers** -- semantic memory backend for any Model Context Protocol integration
- **Knowledge management** -- personal knowledge bases with analogical reasoning (Obsidian plugins, Zettlekasten tools)
- **Edge/embedded AI** -- lightweight enough for single-node deployment; no vector database infrastructure needed
- **Research tools** -- academic paper similarity, citation graph exploration, concept mapping
- **Content moderation** -- near-duplicate detection using holographic similarity

## Security

HMS is designed with privacy as a core principle. Hyperdimensional vectors are inherently lossy representations; original content cannot be reconstructed from stored vectors.

For security policy and vulnerability reporting, see [SECURITY.md](.github/SECURITY.md).

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
