---
status: pending
group: G5
depends_on: ["016"]
files_to_modify:
  - src/core/atom_memory.rs
  - src/core/composite_memory.rs
  - src/core/rules.rs
  - src/core/engine/mod.rs
test_command: cargo test --lib tests::test_basic_persistence
---
Add serialize/deserialize for atoms (0xFD), composites (0xFC), rules (0xFB) following the magic-byte pattern from graph.rs (see devplan/CONTEXT.md). Write to arena via HmsCore::arena_write. Test: store, restart, verify recovered.
