---
status: pending
group: G3
depends_on: []
files_to_modify:
  - src/core/triple_store.rs
  - src/core/mod.rs
test_command: cargo test --lib triple_store
---
Create src/core/triple_store.rs: columnar (s_id, r_id, o_id) with forward/reverse FxHashMap indices. lookup_objects, lookup_subjects, fan_out, triples_for_atom. See devplan/MODULES.md for API.
