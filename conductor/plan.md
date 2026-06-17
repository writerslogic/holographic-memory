# WritersEye: In-Memory HDC Engine Architecture

## Objective
Redesign the HMS architecture to perfectly match the WritersEye workload: a fast, in-memory Vector Symbolic Architecture (VSA) engine supporting 10K-1M vectors per user, with simplified asynchronous persistence, eliminating the synchronous disk I/O bottleneck.

## Workload Profile (WritersEye)
- **Scale:** 10K to 1M vectors per user (single-tenant per process).
- **Read/Write Ratio:** Heavily read-dominated (analysis/reasoning is the hot path).
- **Operations:** Compositional representations (bind/bundle), analogical reasoning, plot permutations. Native HDC/VSA support is mandatory.
- **Durability:** Generous (writing tool; losing seconds of data on a hard crash is acceptable).

## Architectural Simplification

### 1. Storage: In-Memory First, Append-Only Persistence
We will remove `redb` completely. It introduces B-Tree overhead and synchronous `fsync` requirements that are unnecessary for an in-memory workload.

- **Primary Store:** A standard Rust `HashMap<String, EntangledHVec>` serving as the absolute source of truth in RAM.
- **Persistence (Append-On-Write):**
  - We retain `PersistentArena` as an append-only log.
  - On every `memorize` call, we immediately serialize the vector and append it to the `PersistentArena`. We **do not** call `fsync` explicitly. We rely on the OS kernel's page cache to flush the data to the NVMe asynchronously. This gives us sub-millisecond append latency with strictly ordered persistence, avoiding the complexity of a separate snapshot worker thread.
  - **Clean Shutdown:** Because we rely on OS buffering, we will trap `SIGTERM` and graceful process exit paths to issue an explicit `fsync` (`PersistentArena::sync_all()`) before exiting, ensuring no data loss on normal shutdown.

### 2. Deduplication, Updates, and Deletions
With an append-only log, updates and deletions must be handled via tombstones.

- **Updates (Last-Write-Wins):** Appending a vector with an existing ID logically overwrites it. During startup recovery, we sequentially read the log. The last appearance of an ID in the log determines its final value in the in-memory `HashMap`.
- **Deletions (Tombstones):** A delete operation appends a special "Tombstone Frame" (e.g., zero-length vector or specific flag) to the `PersistentArena`. During startup recovery, encountering a tombstone removes the ID from the `HashMap`.
- **Compaction:** To prevent unbounded disk growth, a background task (or manual trigger) will periodically write a fresh `PersistentArena` containing only the current active state of the `HashMap`, swap the files, and delete the old log.

### 3. Memory Budget and Scale Ceiling
- **Vector Data:** 1M vectors at ~2KB (sparse indices) = ~2GB.
- **Graph & Indices:** The NSG graph edges and `HashMap` overhead will add approximately 1.5GB - 2GB.
- **Realistic Ceiling:** The per-process memory footprint for 1M vectors will be roughly **3.5GB to 4GB**. For the primary target (10K - 100K vectors per manuscript), memory usage will be trivial (< 400MB). This fits comfortably within modern desktop RAM limits for a single-tenant application.

### 4. Startup Recovery & NSG Rebuild
- **Base Data Recovery:** Scanning 1M vectors from the NVMe log to rebuild the `HashMap` is heavily I/O bound but sequentially fast (< 2 seconds).
- **Index Rebuild:** Rebuilding the NSG graph from scratch for 1M vectors would take unacceptable minutes on a single thread.
- **Solution (Lazy Build):** For the typical scale (10K - 100K), rebuilding the NSG graph takes 1-5 seconds, which is acceptable on application launch. For larger sets, we will implement **Persistent Graph Serialization**. The background compaction process (see point 2) will periodically serialize the entire NSG graph state into a separate `index.bin` file. On startup, we load the `index.bin` first, then only replay the `PersistentArena` log for vectors added *since* the last compaction.

## Implementation Steps (Plan)

1.  **Remove `redb` Dependency:**
    - Remove `redb` from `Cargo.toml`.
    - Strip out `REGISTRY_TABLE` and `DELETED_IDS_TABLE` logic from `HmsCore`.
2.  **Establish In-Memory Source of Truth:**
    - Replace the complex `registry` and `id_to_offset` tracking with a unified in-memory storage structure: `Arc<RwLock<FxHashMap<String, EntangledHVec>>>`.
3.  **Implement Append-On-Write Persistence:**
    - Modify `memorize` to append directly to `PersistentArena` on every call, returning immediately without explicit `fsync`.
    - Implement Tombstone frames for deletions.
4.  **Simplify Startup Recovery:**
    - Rewrite `HmsCore::new` to linearly scan the `PersistentArena` log, applying inserts and tombstones in Last-Write-Wins order to rebuild the `HashMap`.
5.  **Graph Persistence & Compaction:**
    - Implement a `compact()` method that writes the active `HashMap` to a new arena, serializes the NSG graph to `index.bin`, and swaps the files.

## Conclusion
This simplified architecture leverages RAM for speed and an OS-buffered append-only log for durability, perfectly aligning with WritersEye's scale and relaxed durability requirements while preserving the load-bearing HDC capabilities.