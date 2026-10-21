---
status: pending
group: G4
depends_on: ["013", "011"]
files_to_modify:
  - src/core/engine/multi_hop.rs
test_command: cargo test --lib multi_hop
---
Create src/core/engine/multi_hop.rs: multi_hop_query. Check RuleStore for matching rule, rewrite to single structural_query. No rule: chain TripleStore lookups. Max depth 10, multiplicative confidence. See devplan/ARCHITECTURE.md Data Flow: Multi-Hop Query.
