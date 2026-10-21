---
status: pending
group: G4
depends_on: []
files_to_modify:
  - src/core/config.rs
test_command: cargo check
---
Add MeaningConfig to config.rs: enabled (default false), beta (24.0), algebraic_max_fanout (40), auto_decompose (false), max_hop_depth (10), max_rule_depth (10). Add validate_config fn. Follow existing sub-config pattern.
