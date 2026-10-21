---
status: pending
group: G3
depends_on: ["007", "008"]
files_to_modify:
  - src/core/rules.rs
  - src/core/mod.rs
test_command: cargo test --lib rules
---
Create src/core/rules.rs: RuleStore with declare_rule (rejects cyclic rules via DFS), apply_rules (longest prefix match), list_rules. Max recursion depth 10. See devplan/MODULES.md for API.
