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
  |   +-- structural.rs         Fuzzy structural queries (algebraic + materialized)
  |   +-- multi_hop.rs          Multi-hop reasoning (rule rewrite + chained lookup)
  |
  +-- cognition/
  |   +-- mod.rs                CognitionLoop: background discovery engine
  |   +-- patterns.rs           PatternScanner: relation co-occurrence analysis
  |   +-- abstraction.rs        AbstractionEngine: prototype concept discovery
  |   +-- gaps.rs               GapDetector: epistemic gap detection
  |   +-- hypothesis.rs         HypothesisEngine: gap-filler proposals
  |   +-- analogy.rs            AnalogyDetector: structural isomorphism
  |   +-- governor.rs           MemoryGovernor: dedup, forgetting, IDF refresh
  |   +-- refiner.rs            DistributionalRefiner: self-organizing atom vectors
  |   +-- loop.rs               Background thread lifecycle
  |
  +-- agency/
  |   +-- mod.rs                Goal-directed reasoning layer
  |   +-- goals.rs              Goal definition and lifecycle
  |   +-- planner.rs            Plan generation from goals
  |   +-- questions.rs          Question generation for knowledge gaps
  |   +-- self_modify.rs        Self-modification proposals
  |
  +-- entangled.rs              EntangledHVec: sparse binary hypervector type
  +-- ternary.rs                TernaryHVec: ternary {-1, 0, +1} hypervector type
  +-- algebra.rs                HolographicAlgebra trait (EntangledHVec, TernaryHVec)
  +-- encoding.rs               Text -> hypervector (character trigrams)
  +-- block_codes.rs            BlockCodeVec: structured block-code bundles
  +-- bloom_memory.rs           BloomMemory: Bloom-filter-based bundled storage
  +-- cls_memory.rs             CLSMemory: concept-level sparse memory
  +-- hopfield.rs               Modern Hopfield network with sparse softmax
  +-- resonator.rs              Resonator network for symbolic factorization
  +-- compose.rs                Vector composition utilities
  +-- decompose.rs              Decomposer: vector decomposition
  +-- sparse_autoencoder.rs     Sparse autoencoder for representation learning
  +-- graph.rs                  Graph engine: typed relations, multi-hop BFS, temporal
  +-- storage.rs                PersistentArena: mmap segmented log
  +-- config.rs                 HmsConfig, MeaningConfig, CognitionConfig, and sub-configs
  +-- security.rs               SigningManager, EncryptionManager (feature-gated)
  +-- audit.rs                  AuditLog: append-only operation log
  +-- diffusion.rs              DiffusionFactorizer: score-based vector decomposition
  +-- text.rs                   TextProcessor: readability metrics
  +-- types.rs                  Shared types (RetrievalResult, ConceptCandidate, etc.)
  +-- error.rs                  HmsError enum
  +-- intersection.rs           Sparse sorted-merge intersection
  +-- atom_memory.rs            AtomMemory: concept vector store (meaning memory)
  +-- composite_memory.rs       CompositeMemory: role-bound composite vectors
  +-- triple_store.rs           TripleStore: symbolic (S, R, O) index
  +-- role.rs                   RoleRegistry: role-shift algebra
  +-- rules.rs                  RuleStore: composition rule definitions
  +-- admission.rs              AdmissionControl: fan-out gating
  +-- indexed_memory.rs         IndexedMemory: posting + IDF substrate
  +-- posting.rs                PostingShard: per-dimension posting lists
  +-- idf.rs                    IdfWeights: IDF with proportional clipping
  +-- tombstone.rs              TombstoneMap: soft-delete tracking
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

## Meaning Memory Module Graph

```
HmsCore (engine/mod.rs)
  |
  +-- atom_memory.rs            AtomMemory: concept vector store
  |     +-- indexed_memory.rs   IndexedMemory: posting lists + IDF + tombstones
  |           +-- posting.rs    PostingShard: inverted posting lists per dimension
  |           +-- idf.rs        IdfWeights: IDF weighting with proportional clipping
  |           +-- tombstone.rs  TombstoneMap: soft-delete tracking
  |
  +-- composite_memory.rs       CompositeMemory: role-bound triple vectors
  |     +-- indexed_memory.rs   (shared substrate with AtomMemory)
  |
  +-- triple_store.rs           TripleStore: symbolic (S, R, O) index
  |                             Four-way FxHash index: by_subject, by_relation,
  |                             by_object, by_composite
  |
  +-- role.rs                   RoleRegistry: role -> cyclic-shift mapping
  |                             compose(), unbind(), compose_triple()
  |
  +-- rules.rs                  RuleStore: CompositionRule definitions
  |                             Maps relation chains to derived relations
  |
  +-- admission.rs              AdmissionControl: fan-out gating
  |                             Algebraic vs. MaterializedLookup decision
  |
  +-- decompose.rs              Decomposer: vector decomposition
  |
  +-- engine/
      +-- structural.rs         fuzzy_structural_query(): algebraic + materialized paths
      +-- multi_hop.rs          multi_hop_query(): rule rewrite + chained lookup
```

### Data Flow: Structural Query

```
fuzzy_structural_query(known_bindings, target_role)
  -> RoleRegistry.compose(known)          // build partial query vector
  -> CompositeMemory.overlap_scan(query)  // IDF-weighted posting intersection
  -> AdmissionControl.check(fan_out)      // gate on candidate count
     |
     +-- Algebraic path (fan_out <= limit):
     |   -> composite.bind(query)         // XOR-unbind known roles
     |   -> permute(dim - target_shift)   // inverse cyclic shift
     |   -> hopfield_cleanup(residual)    // attractor network recovery
     |      -> overlap_scan + sparse_softmax + iterate
     |   -> return (entity_id, confidence)
     |
     +-- Materialized path (fan_out > limit):
         -> TripleStore.by_composite_id() // symbolic index lookup
         -> extract target_role field
         -> return (entity_id, score)
```

### Data Flow: Multi-Hop Query

```
multi_hop_query(start, [rel1, rel2, ...], ctx, rule_store)
  -> if single relation:
       single_hop -> fuzzy_structural_query
  -> if two relations + matching CompositionRule:
       rule_rewrite -> fuzzy_structural_query(derived_relation)
  -> else:
       chained_lookup -> TripleStore walk, hop by hop
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
