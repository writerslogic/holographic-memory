---
status: pending
group: G1
depends_on: []
files_to_modify:
  - src/core/tombstone.rs
  - src/core/mod.rs
test_command: cargo test --lib tombstone
---
Create src/core/tombstone.rs: TombstoneMap with atomic CAS state transitions (Live->Tombstoned->Sweeping->Reclaimed). Monotonic alloc, never reuses IDs. See devplan/MODULES.md for API.
