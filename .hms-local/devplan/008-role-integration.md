---
status: pending
group: G3
depends_on: ["002"]
files_to_modify:
  - src/core/engine/knowledge.rs
  - src/core/engine/mod.rs
test_command: cargo test --lib tests::test_triplet
---
Update engine/knowledge.rs to use RoleRegistry for triple encoding. memorize_triplet uses compose_triple (role-bound, not plain XOR). Add role_registry field to HmsCore, initialize in new(). Existing tests must still pass.
