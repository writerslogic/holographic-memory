---
status: pending
group: G5
depends_on: ["013", "014", "015", "005", "012"]
files_to_modify:
  - src/core/engine/mod.rs
test_command: cargo test --lib
---
Add all new subsystems as fields on HmsCore (gated by config.meaning.enabled). Initialize in new(). Wire load_from_log to dispatch by magic byte. Wire compact(). Add public methods: structural_query, multi_hop, cleanup, declare_rule, register_role. ALL existing 133+ tests MUST pass unchanged.
