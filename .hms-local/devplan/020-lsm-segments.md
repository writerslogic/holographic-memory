---
status: pending
group: G7
depends_on: ["016"]
files_to_modify:
  - src/core/indexed_memory.rs
test_command: cargo test --lib indexed_memory
---
Add LSM write path to IndexedMemory: mutable delta PostingShard + immutable main + Arc-swap compaction. See devplan/ARCHITECTURE.md Production Safety section.
