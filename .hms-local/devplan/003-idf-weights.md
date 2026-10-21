---
status: pending
group: G1
depends_on: []
files_to_modify:
  - src/core/idf.rs
  - src/core/mod.rs
test_command: cargo test --lib idf
---
Create src/core/idf.rs: IdfWeights with smoothed formula log(1+N/(1+df)), proportional clipping at C*median weight for poisoning defense. See devplan/MODULES.md for API.
