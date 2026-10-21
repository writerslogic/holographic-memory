---
status: pending
group: G3
depends_on: []
files_to_modify:
  - src/core/decompose.rs
  - src/core/mod.rs
test_command: cargo test --lib decompose
---
Create src/core/decompose.rs: Decomposer with heuristic English NP/VP extraction. See devplan/ARCHITECTURE.md Text Decomposition Rules for preposition list, verb list, NP grammar, and extraction patterns. Precision over recall.
