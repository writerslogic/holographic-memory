---
status: pending
group: G7
depends_on: ["007"]
files_to_modify:
  - src/core/entangled.rs
  - src/core/engine/mod.rs
test_command: cargo test --lib tests::test_from_dense
---
Enhance from_dense() with optional IDF-weighted projection. Add memorize_dense method to HmsCore. Optionally ship default English prior.
