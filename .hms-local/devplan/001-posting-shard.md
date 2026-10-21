---
status: pending
group: G1
depends_on: []
files_to_modify:
  - src/core/posting.rs
  - src/core/mod.rs
test_command: cargo test --lib posting
---
Create src/core/posting.rs: PostingShard (256 sorted posting lists per shard, RwLock-guarded) + sorted_merge_overlap (scan across 64 shards, return (atom_id, overlap_count) sorted desc) + global_to_shard helper. Register in mod.rs. See devplan/MODULES.md for API, devplan/ARCHITECTURE.md for invariants. Reuse galloping merge pattern from src/core/intersection.rs.
