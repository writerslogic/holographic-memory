# HMS Existing Codebase Reference

> 35 source files, 8798 lines, 133+ tests.
> This document catalogs every reusable function and struct that new
> Meaning Memory code should build on rather than duplicate.

---

## entangled.rs — Sparse Hypervector Primitives

The core data type. All meaning memory operations compose these.

### Struct

```rust
pub struct EntangledHVec {
    pub(crate) dim: usize,
    pub(crate) indices: Vec<u32>,  // sorted, deduplicated active bit indices
}
```

### Public Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `new_deterministic` | `(dim: usize, seed: u64) -> Self` | Create random sparse HVec. Entanglement groups + birthday backfill. |
| `from_indices` | `(indices: Vec<u32>, dim: usize) -> Self` | Create from pre-sorted, deduped indices. Debug-asserts sorted. |
| `from_dense` | `(dense: &[f32], target_dim: usize) -> Self` | Achlioptas ternary projection, top-k by magnitude. |
| `from_scalar` | `(value: f64, min: f64, max: f64, dim: usize) -> Self` | Sliding window + prime permutation. Monotonic similarity. |
| `indices` | `(&self) -> &[u32]` | Read sorted active indices. |
| `to_deltas` | `(&self) -> Vec<u32>` | Delta-encode indices for compact storage. First element is absolute. |
| `from_deltas` | `(deltas: &[u32], dim: usize) -> Self` | Restore from delta encoding. |
| `hamming` | `(&self, other: &Self) -> u32` | Hamming distance via sorted-merge intersection. |
| `similarity` | `(&self, other: &Self) -> f64` | Jaccard: \|A intersect B\| / \|A union B\|. |
| `bind` | `(&self, other: &Self) -> Self` | XOR = sorted symmetric difference of index sets. |
| `permute` | `(&self, shifts: usize) -> Self` | Cyclic shift all indices by `shifts` mod dim. Re-sorts. |
| `bundle` | `<V: Borrow<Self>>(vectors: &[V]) -> Self` | Majority-vote thresholding. Prunes to target_count = dim/256. |
| `bundle_dp` | `<V: Borrow<Self>>(vectors: &[V], epsilon: f64) -> Self` | Bundle with Laplace noise (epsilon-DP). |

### Crate-internal Functions

```rust
pub(crate) fn sorted_symmetric_difference(a: &[u32], b: &[u32]) -> Vec<u32>
pub(crate) fn sorted_symmetric_difference_count(a: &[u32], b: &[u32]) -> usize
```

### Key Constants

```rust
pub(crate) const DEFAULT_RHO_DENOM: usize = 256;  // sparsity: ~dim/256 active
const NUM_ENTANGLEMENTS: usize = 64;               // hash-seeded groups
```

---

## intersection.rs — Sparse Intersection

```rust
pub fn sparse_intersection_count(a: &[u32], b: &[u32]) -> usize
```

Adaptively selects galloping search (skewed sizes, `|small|*8 < |large|`)
or sorted merge (balanced). Both inputs must be sorted.

This is the overlap primitive used by Hopfield cleanup and structural query.

---

## storage.rs — PersistentArena

RwLock-guarded segmented mmap arena with LZ4 compression and CRC32 framing.

### Struct

```rust
pub struct PersistentArena {
    base_path: PathBuf,
    read_segments: RwLock<Vec<Arc<Mmap>>>,
    active_segment: Arc<RwLock<MmapMut>>,
    active_id: AtomicUsize,
    write_offset: AtomicUsize,
    version_counter: AtomicUsize,
}
```

### Public Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `new` | `<P: AsRef<Path>>(path: P) -> Result<Self>` | Open or create arena. Discovers existing frames via CRC walk. |
| `write_slice` | `(&self, data: &[u8]) -> Result<usize>` | Append CRC32+LZ4 frame. Returns global offset. Thread-safe (write lock). |
| `read_frame` | `(&self, global_offset: usize) -> Result<(Vec<u8>, u32)>` | Read and decompress frame. Returns (data, version). Verifies CRC. |
| `next_offset` | `(&self, global_offset: usize) -> Result<usize>` | Compute offset of next frame after the one at global_offset. |
| `read_slice` | `(&self, global_offset: usize, len: usize) -> Result<Vec<u8>>` | Raw byte read across segments. |
| `replace_with_compacted` | `(&self, temp_base: &Path) -> Result<()>` | Atomic swap of arena contents with compacted version. |

### Frame Format

```
[CRC32: u32][RawLen: u32][CompLen: u32][Version: u32][CompressedData: bytes]
```

Header size: 16 bytes. Max raw frame: 50 MB. Segment size: 1 GB.

### Magic-Byte Convention

The arena stores raw bytes. Callers prepend a magic byte to distinguish
frame types:

| Magic  | Type             | Owner           |
|--------|------------------|-----------------|
| `0xFD` | Atom vector      | AtomMemory      |
| `0xFC` | Composite vector | CompositeMemory |
| `0xFB` | Triple record    | TripleStore     |
| `0xFE` | Relation / Rule  | RelationStore / RuleStore |

On `load_from_log`, each frame's first byte is checked to dispatch
to the correct deserializer.

---

## graph.rs — RelationStore

In-memory graph index with FxHashMap indices.

### Key Pattern: Serialize/Deserialize with Magic Byte

```rust
const RELATION_MAGIC: u8 = 0xFE;

pub fn serialize_relation(rel: &Relation) -> Vec<u8> {
    let mut buf = Vec::with_capacity(len);
    buf.push(RELATION_MAGIC);                              // magic byte first
    buf.extend_from_slice(&(source.len() as u16).to_le_bytes());  // length-prefix
    buf.extend_from_slice(source);                         // string bytes
    // ... repeat for each field
    buf
}

pub fn deserialize_relation(data: &[u8]) -> Option<Relation> {
    if data.is_empty() || data[0] != RELATION_MAGIC {      // check magic first
        return None;
    }
    // ... parse length-prefixed fields
}
```

New persistence formats (atoms, composites, triples) must follow this same
pattern: magic byte, then length-prefixed fields, then `Option` return on
deserialize with magic check.

### Public Methods

| Method | Purpose |
|--------|---------|
| `new() -> Self` | Empty store |
| `declare_type(&self, rel_type: RelationType)` | Register type with inference semantics |
| `add(&self, rel: &Relation) -> usize` | Insert, update all indices |
| `remove(&self, source, type, target) -> bool` | Soft delete |
| `outgoing(&self, source, type?, time) -> Vec<Relation>` | Forward traversal |
| `incoming(&self, target, type?, time) -> Vec<Relation>` | Reverse traversal |
| `traverse(&self, start, type?, depth, time, sim_fn) -> Vec<GraphPath>` | BFS with symmetric/transitive inference |
| `serialize_relation(&Relation) -> Vec<u8>` | For arena persistence |
| `deserialize_relation(&[u8]) -> Option<Relation>` | From arena bytes |
| `count(&self) -> usize` | Live (non-deleted) count |
| `load_relation(&self, &Relation)` | Log replay |
| `snapshot(&self) -> Vec<Relation>` | For compaction |

---

## security.rs + audit.rs — Security Infrastructure

### security.rs

```rust
pub struct SigningManager { signing_key, verifying_key }
  -> new(key_path) -> sign(data) -> verify(data, sig)

pub struct EncryptionManager { cipher }
  -> new(passphrase) -> encrypt(data) -> decrypt(data)
```

Both feature-gated behind `#[cfg(feature = "security")]`.

### audit.rs

```rust
pub struct AuditLog { file: Mutex<File>, path: PathBuf }
  -> new(base_path) -> log_operation(op, id, sign_fn?) -> read_all() -> verify_all(verifying_key)

pub enum AuditOp { Memorize = 1, Delete = 2, Compact = 3 }
```

Fixed 105-byte entries: `[timestamp_ms: u64][op: u8][id_hash: 32][sig: 64]`.

**For meaning memory:** arena writes go through `arena.write_slice(data)`.
When encryption is enabled, `security.rs` wraps the data before writing.
Audit logging is optional and independent of meaning memory operations.

---

## config.rs — Configuration

```rust
#[derive(Clone, Debug, Default)]
pub struct HmsConfig {
    pub ivf: IVFConfig,
    pub nsg: NSGConfig,
    pub shard: ShardConfig,
    pub query: QueryConfig,
    pub concepts: ConceptsConfig,
    pub diffusion: DiffusionConfig,
    pub security: SecurityConfig,
    pub privacy: PrivacyConfig,
}
```

**Pattern for adding MeaningConfig:**

```rust
pub meaning: MeaningConfig,  // add to HmsConfig

#[derive(Clone, Debug)]
pub struct MeaningConfig {
    pub enabled: bool,         // default false
    pub dim: usize,            // default 16384
    pub hopfield_beta: f64,    // default: 1/sqrt(64)
    pub hopfield_k: usize,    // default: 64
    pub hopfield_max_iter: usize, // default: 3
    pub fanout_limit: usize,   // default: 40
    pub idf_clip_factor: f64,  // default: 3.0
}

impl Default for MeaningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            dim: 16384,
            hopfield_beta: 1.0 / (64.0_f64).sqrt(),
            hopfield_k: 64,
            hopfield_max_iter: 3,
            fanout_limit: 40,
            idf_clip_factor: 3.0,
        }
    }
}
```

All sub-configs implement `Default`. The N-API layer maps `Option<T>` fields
to config fields with `if let Some(v)` guards.

---

## engine/mod.rs — HmsCore

```rust
pub struct HmsCore {
    config: HmsConfig,
    arena: Arc<PersistentArena>,
    dimensions: usize,
    storage_path: PathBuf,
    shards: RwLock<ShardSet>,
    graph: RelationStore,
    audit: Option<AuditLog>,
    #[cfg(feature = "security")] signing: Option<SigningManager>,
    #[cfg(feature = "security")] encryption: Option<EncryptionManager>,
}
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `new(dimensions, storage_path?, config?) -> Result<Self>` | Create instance |
| `load_from_log(&self) -> Result<usize>` | Replay arena frames, dispatch by magic byte |
| `compact(&self) -> Result<()>` | Rewrite arena without deleted entries |
| `memorize(&self, id, text) -> Result<usize>` | Encode + store + persist |
| `delete(&self, id) -> Result<bool>` | Soft delete + persist tombstone |

**For meaning memory integration:** add optional meaning memory fields:

```rust
meaning_atoms: Option<AtomMemory>,
meaning_composites: Option<CompositeMemory>,
meaning_triples: Option<TripleStore>,
meaning_rules: Option<RuleStore>,
```

Initialized only when `config.meaning.enabled == true`.
`load_from_log` dispatches 0xFD/0xFC/0xFB frames to these stores.

---

## lib.rs — N-API Pattern

```rust
#[cfg(feature = "node-api")]
#[napi]
pub struct HolographicMemorySystem {
    core: Arc<HmsCore>,
}

// Async pattern: wrap blocking work in spawn_blocking
async fn run_async<T, F>(f: F) -> Result<T>
where T: Send + 'static, F: FnOnce() -> anyhow::Result<T> + Send + 'static
{
    tokio::task::spawn_blocking(f).await?.map_err(napi_err)
}
```

### HmsConfigJs Pattern

```rust
#[cfg(feature = "node-api")]
#[napi(object)]
pub struct HmsConfigJs {
    pub field_name: Option<Type>,  // all fields are Option<T>
    // ...
}
```

Every field is `Option<T>` to allow partial configuration from JavaScript.
The `into_config()` method applies only the `Some` values to a default config.

**For meaning memory:** add fields like:

```rust
pub meaning_enabled: Option<bool>,
pub meaning_dim: Option<u32>,
pub meaning_hopfield_beta: Option<f64>,
// etc.
```

### Derive Pattern for N-API Types

```rust
#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "node-api", derive(napi_derive::napi))]
#[cfg_attr(feature = "json-schema", derive(schemars::JsonSchema))]
```

New result types (StructuralResult, MultiHopResult) need these derives
for N-API exposure.

---

## encoding.rs — Text Encoding

```rust
pub fn encode_text_internal(text: &str, dim: usize) -> EntangledHVec
```

Character trigram (n=3) encoding. This is the ONLY text-to-HVec function.

**For meaning memory:** the decomposer works on raw text strings to extract
triples. Each atom string is then encoded to an HVec using
`EntangledHVec::new_deterministic(dim, hash(atom_str))`, NOT via
`encode_text_internal`. The atom HVec must be deterministic from the
string content so that the same atom always produces the same vector.

`encode_text_internal` is for the existing memorize path (whole-text
encoding for similarity search). Meaning memory atoms use seed-based
generation for exact reproducibility.

---

## Import Conventions

### Intra-module (within `core/`)

```rust
use super::entangled::EntangledHVec;
use super::storage::PersistentArena;
use super::types::{Relation, RelationType};
```

### Cross-module (from engine to core)

```rust
use crate::core::config::HmsConfig;
use crate::core::entangled::EntangledHVec;
```

### Feature-gated

```rust
#[cfg(feature = "security")]
use super::security::SigningManager;
```

Meaning memory code does NOT need feature gating. It is controlled by
runtime config (`config.meaning.enabled`), not compile-time features.
The types and structs exist unconditionally; only initialization and
execution are gated.

---

## Library Choices

| What | Crate | NOT |
|------|-------|-----|
| HashMap | `fxhash::FxHashMap` | `std::collections::HashMap` |
| RwLock | `parking_lot::RwLock` | `std::sync::RwLock` |
| Errors | `anyhow::Result` | custom error types (except `HmsError` for N-API) |
| Serialization | `serde` + manual binary | NOT bincode/postcard for arena frames |
| Compression | `lz4_flex` | built into PersistentArena |
| Integrity | `crc32fast` | built into PersistentArena |

---

## Codebase Stats

- 35 source files across `src/core/` and `src/lib.rs`
- 8798 total lines
- 133+ tests (unit + integration)
- Key modules by size: `lib.rs` (1907), `engine/mod.rs`, `graph.rs`, `storage.rs`, `entangled.rs`
