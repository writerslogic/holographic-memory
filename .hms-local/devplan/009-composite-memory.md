---
status: pending
group: G3
depends_on: ["006", "008"]
files_to_modify:
  - src/core/composite_memory.rs
  - src/core/mod.rs
test_command: cargo test --lib composite_memory
---
Create src/core/composite_memory.rs: CompositeMemory wrapping IndexedMemory + triple_refs. insert composes triple via RoleRegistry. scan uses overlap_scan. cascade_delete_atom tombstones all composites referencing a given atom. See devplan/MODULES.md for API.
