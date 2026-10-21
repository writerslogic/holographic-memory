---
status: pending
group: G8
depends_on: ["019", "021"]
files_to_modify:
  - Cargo.toml
  - package.json
  - CHANGELOG.md
test_command: cargo test --lib
---
Bump to v0.4.0. Update CHANGELOG with Phase 1 features. Full quality gate. Tag and push for CI release.
