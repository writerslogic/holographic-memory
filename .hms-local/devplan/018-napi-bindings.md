---
status: pending
group: G6
depends_on: ["016"]
files_to_modify:
  - src/lib.rs
  - src/core/types.rs
test_command: cargo check
---
Add N-API methods: structural_query, multi_hop, cleanup, declare_rule, register_role. Add all new types to types.rs (see devplan/MODULES.md New Types section). Add meaning config fields to HmsConfigJs.
