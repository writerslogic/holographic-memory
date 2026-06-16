# Architecture

## Module Graph

```
lib.rs                          N-API bindings (HolographicMemorySystem)
  |
  core/
  +-- engine/
  |   +-- mod.rs                HmsCore: main orchestrator
  |   +-- query.rs              Query routing and execution
  |   +-- router.rs             Adaptive retrieval strategy selection
  |   +-- shard.rs              ShardSet, ShardManager, Shard
  |   +-- concepts.rs           Concept synthesis (clustering + bundling)
  |   +-- knowledge.rs          Triplets, sequences, analogies
  |
  +-- entangled.rs              EntangledHVec: sparse hypervector type
  +-- encoding.rs               Text -> hypervector (character trigrams)
  +-- storage.rs                PersistentArena: mmap segmented log
  +-- config.rs                 HmsConfig and sub-configs
  +-- security.rs               SigningManager, EncryptionManager (feature-gated)
  +-- audit.rs                  AuditLog: append-only operation log
  +-- diffusion.rs              DiffusionFactorizer: score-based vector decomposition
  +-- text.rs                   TextProcessor: readability metrics
  +-- types.rs                  Shared types (RetrievalResult, ConceptCandidate, etc.)
  +-- error.rs                  HmsError enum
  +-- intersection.rs           Sparse sorted-merge intersection
  +-- index/
  |   +-- mod.rs                Index traits
  |   +-- inverted.rs           Sparse inverted index for high-sparsity queries
  +-- ivf/
  |   +-- mod.rs                IVFIndex: inverted file with product quantization
  |   +-- training.rs           IVF training pipeline
  |   +-- query.rs              IVF query execution
  |   +-- kmeans.rs             K-means clustering
  |   +-- pq.rs                 Product quantization
  |   +-- nystrom.rs            Nystrom dimensionality reduction
  |   +-- inverted_list.rs      Inverted list storage
  +-- nsg/
      +-- mod.rs                NSGIndex: navigable small-world graph
      +-- training.rs           NSG construction
      +-- search.rs             Greedy graph search
      +-- graph.rs              Graph operations (KNN, pruning, centroid)
```

## Lock Ordering

Strict ordering prevents deadlocks:

```
ShardSet (read/write) -> Shard.vectors -> Shard.ivf -> Shard.nsg
```

The Arena lock is independent (managed internally by `PersistentArena`). Arena writes acquire an exclusive write lock on `active_segment`; reads acquire a shared read lock.

The AuditLog uses its own `Mutex<File>` independent of all other locks.

## Data Flow

### Write Path (memorize)

```
memorize(id, vector)
  -> serialize_log_entry(id, vector)     // [id_len:u16][id][delta_count:u32][deltas:u32*]
  -> maybe_encrypt(entry)                // AES-256-GCM if enabled
  -> arena.write_slice(payload)          // LZ4 compress, CRC32, write to mmap
  -> audit.record(Memorize, id)          // append to audit.bin (optional)
  -> shards.insert(id, vector)           // update in-memory index
  -> auto-train check                    // IVF/NSG/auto-shard thresholds
```

### Read Path (query)

```
query(query_vec, k)
  -> router.plan(collection_stats)       // choose strategy: brute/NSG/IVF/inverted
  -> shards.for_each_shard(|shard| {
       execute_plan(shard, query_vec, k) // parallel per-shard query
     })
  -> merge and sort by similarity        // top-k across all shards
```

### Recovery Path (load_from_log)

```
HmsCore::new()
  -> PersistentArena::new()
     -> discover_offset()                // walk frames, validate CRC32, find write head
  -> load_from_log()
     -> for each frame:
        arena_read_frame(offset)         // read + maybe_decrypt
        parse_log_payload()              // extract id + vector (or tombstone)
        shards.insert/remove()           // rebuild in-memory state
  -> load_indices()                      // load NSG/IVF from disk (maybe_decrypt)
  -> rebuild_inverted_index()            // reconstruct sparse inverted index
```

## Persistence Guarantees

### Arena Log

- **Append-only**: New entries are appended to the active mmap segment. No in-place mutation.
- **CRC32 framing**: Each frame has a CRC32 checksum over decompressed data. Corrupt frames are rejected on recovery.
- **Crash safety**: `mmap::flush_range()` after each write. Partial writes are detected by CRC32 failure during `discover_offset()`.
- **Segment rotation**: When a segment fills (1 GB), a new segment is created. Old segments become read-only.

### Compaction

- **Snapshot isolation**: Acquires exclusive `shards` write lock, snapshots all live vectors.
- **Atomic swap**: Writes snapshot to temp directory, then atomically replaces arena files via `replace_with_compacted()`.
- **Index preservation**: NSG and IVF indices are re-saved after compaction.

### Delete Semantics

- **Tombstone-first**: Tombstone (delta_count = 0xFFFFFFFF) is written to arena before in-memory removal.
- **Crash recovery**: If crash occurs after tombstone write but before memory removal, `load_from_log` replays the tombstone and correctly removes the vector.

## Concurrency Model

- **Readers**: Multiple concurrent readers via `RwLock<ShardSet>` read lock.
- **Writers**: `memorize()` acquires shard read lock (concurrent with other memorizes). Arena writes are serialized by `PersistentArena`'s internal write lock.
- **Compact**: Acquires exclusive shard write lock, blocking all concurrent reads and writes.
- **Auto-train**: NSG/IVF training acquires shard read lock and index write lock within a single shard.

## Retrieval Strategies

The `router` module selects strategy based on collection statistics:

| Condition | Strategy | Complexity |
|-----------|----------|-----------|
| < 1000 vectors | Brute force | O(n) |
| NSG trained | Graph search (greedy) | O(log n) |
| IVF trained | Inverted file + PQ | O(n/k) per probe |
| High query sparsity | Sparse inverted index | O(active_indices) |

Strategy selection is automatic per-query based on the `QueryPlan` generated by `router::plan()`.
