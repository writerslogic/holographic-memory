---
status: pending
group: G1
depends_on: []
files_to_modify:
  - src/core/admission.rs
  - src/core/mod.rs
test_command: cargo test --lib admission
---
Create src/core/admission.rs: AdmissionControl with fan-out-charged try_admit, RAII AdmitGuard, hysteresis at p_max/2. See devplan/MODULES.md for API.
