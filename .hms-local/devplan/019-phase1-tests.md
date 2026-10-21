---
status: pending
group: G6
depends_on: ["016"]
files_to_modify:
  - src/lib.rs
test_command: cargo test --lib
---
Write 8 certification tests (see devplan/TASKS.md Group 6): basin certification, structural query fan-out sweep, role inversion, poisoning defense, concurrent correctness, multi-hop with rules, persistence roundtrip, backward compatibility.
