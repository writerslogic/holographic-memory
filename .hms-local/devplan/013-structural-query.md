---
status: pending
group: G4
depends_on: ["007", "008", "009", "010"]
files_to_modify:
  - src/core/engine/structural.rs
test_command: cargo test --lib structural
---
Create src/core/engine/structural.rs: fuzzy_structural_query pipeline. Build query from known roles, overlap scan CompositeMemory, branch on fan-out (<=40 algebraic, >40 materialized). See devplan/ARCHITECTURE.md Data Flow: Structural Query.
