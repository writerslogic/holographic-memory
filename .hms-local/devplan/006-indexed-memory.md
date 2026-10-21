---
status: pending
group: G2
depends_on: ["001", "003", "004"]
files_to_modify:
  - src/core/indexed_memory.rs
  - src/core/mod.rs
test_command: cargo test --lib indexed_memory
---
Create src/core/indexed_memory.rs: IndexedMemory wrapping 64 PostingShards + IdfWeights + TombstoneMap. Implements insert, delete, get, overlap_scan (IDF-clipped, tombstone-filtered). Also: standalone hopfield_cleanup fn (softmax attention, weighted superposition, top-k projection loop) and sparse_softmax. These are the SHARED attractor functions. See devplan/MODULES.md and devplan/ARCHITECTURE.md for attractor algorithm.
