<p align="center">
  <img src="https://raw.githubusercontent.com/writerslogic/holographic-memory/main/assets/logo.svg" width="200" alt="HMS Logo">
</p>

# Holographic Memory System (HMS)

[![CI](https://github.com/writerslogic/holographic-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/writerslogic/holographic-memory/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![crates.io](https://img.shields.io/crates/v/holographic-memory.svg)](https://crates.io/crates/holographic-memory)
[![npm](https://img.shields.io/npm/v/holographic-memory.svg)](https://www.npmjs.com/package/holographic-memory)

**Privacy-preserving semantic search using hyperdimensional computing.**

A high-performance **Holographic Memory System (HMS)** for Node.js, powered by Rust. This library implements **Vector Symbolic Architectures (VSA)** using **Binary Spatter Code (BSC)** and **Sparse Distributed Representations (SDR)** to enable semantic search, analogical reasoning, and associative memory at scale.

> Developed by [WritersLogic](https://github.com/writerslogic) -- local-first intelligence with no data leaving your machine.

## 🚀 Features

- **Hybrid Retrieval Architecture**: 
  - **NSG (Navigable Small World)**: Fast proximity graph for approximate nearest neighbors.
  - **IVF (Inverted File)**: Coarse-grained quantization for large datasets.
  - **Sparse Inverted Index**: Term-based retrieval for high-sparsity queries.
- **Symbolic Operations**: Native bitwise **Binding (XOR)**, **Bundling (Majority Rule)**, and **Permutation (Cyclic Shift)**.
- **Performance Optimized**:
  - **O(1) Resolution**: Cached physical location lookups for instant ID retrieval.
  - **FxHash Backend**: Ultra-fast non-cryptographic hashing for all retrieval collections.
  - **O(N) Selection**: Linear-time candidate pruning using `select_nth_unstable`.
- **Persistent Storage**: Integrated `sled` (key-value) and custom `Arena` (binary) for ACID-compliant persistence.
- **Meaning Memory**: Structured knowledge layer with role-filler algebra, triple stores, multi-hop reasoning, and Hopfield attractor cleanup.
- **Graph Engine**: Explicit typed relations with multi-hop traversal, transitive/symmetric inference, and temporal filtering.
- **Federated Queries**: Query across multiple HMS instances in parallel without centralizing data.
- **Node.js Bindings**: High-efficiency N-API implementation with asynchronous worker thread execution.

## 🔌 Integrations

HMS is available as both a Node.js package and a high-performance Rust crate.

### Node.js (N-API)
```bash
npm install holographic-memory
```

### Rust (Crates.io)
Add to your `Cargo.toml`:
```toml
[dependencies]
holographic-memory = "0.2"
```

## 🏗 Core Architecture

HMS is designed for local-first intelligence, combining advanced research in Hyperdimensional Computing with efficient retrieval algorithms.

- **Advanced Search**: Implements the **NSG (Navigable Small World)** algorithm, offering high search efficiency and index compactness.
- **Adaptive Routing**: Employs a retrieval strategy that dynamically switches between graph-based, quantized, and inverted indexing based on dataset statistics.
- **Neuro-Symbolic VSA**: A robust implementation of **Binary Spatter Code (BSC)**, enabling relational logic $(A \otimes B)$ combined with the associative matching of high-dimensional vector spaces.
- **Efficient Data Path**: Engineered with a zero-copy N-API interface, $O(1)$ ID resolution, and hardware-aware optimizations for high single-node throughput.

## 🎯 Use Cases

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

## 🧩 Meaning Memory

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

## 🧠 Core Concepts

### Hyperdimensional Computing (HDC)
Traditional AI uses deep vectors (weights). HDC uses high-dimensional (e.g., 10,000+), sparse vectors where information is "holographically" distributed across every dimension. 

- **Binding (⊗)**: Combines two vectors into a new, orthogonal vector representing their relationship. Reversible.
- **Bundling (⊛)**: Combines multiple vectors into a single vector that retains similarity to all its components.
- **Permutation (Π)**: Represents sequence and structure by shifting bits.

## 🛠 Quick Start

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

## 🔧 Development

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
# Run the 92+ unit and integration tests
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
