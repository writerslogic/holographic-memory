---
status: pending
group: G3
depends_on: ["006"]
files_to_modify:
  - src/core/atom_memory.rs
  - src/core/mod.rs
test_command: cargo test --lib atom_memory
---
Create src/core/atom_memory.rs: AtomMemory wrapping IndexedMemory + string_id to u32 mapping. Provides insert (error on dup), delete, cleanup (calls hopfield_cleanup), set_retrieve (single-pass multi-extract), get. See devplan/MODULES.md for API, devplan/ARCHITECTURE.md for Hopfield dynamics.
