---
status: pending
group: G1
depends_on: []
files_to_modify:
  - src/core/role.rs
  - src/core/mod.rs
test_command: cargo test --lib role
---
Create src/core/role.rs: RoleRegistry mapping role names to odd cyclic-shift values coprime with D=16384. Default roles: subject=0, relation=1, object=3. compose_triple and unbind_role methods. Uses existing EntangledHVec::permute(). See devplan/MODULES.md for full API. Max 8192 roles.
