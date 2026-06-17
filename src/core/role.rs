// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use fxhash::FxHashMap;

use super::entangled::EntangledHVec;

/// Maps role names to cyclic-shift values for role-based binding.
/// Fixes XOR commutativity: without shifts, S XOR R XOR O = O XOR R XOR S.
pub struct RoleRegistry {
    shifts: FxHashMap<String, usize>,
    dim: usize,
}

impl RoleRegistry {
    pub fn new(dim: usize) -> Self {
        let mut reg = Self {
            shifts: FxHashMap::default(),
            dim,
        };
        reg.shifts.insert("subject".to_string(), 0);
        reg.shifts.insert("relation".to_string(), 1);
        reg.shifts.insert("object".to_string(), 3);
        reg
    }

    pub fn register(&mut self, role: &str, shift: usize) -> Result<()> {
        if let Some(&existing) = self.shifts.get(role) {
            if existing != shift {
                return Err(anyhow!(
                    "role '{}' already registered with shift {}, cannot re-register with {}",
                    role,
                    existing,
                    shift
                ));
            }
            return Ok(());
        }
        self.shifts.insert(role.to_string(), shift);
        Ok(())
    }

    pub fn shift_for(&self, role: &str) -> Option<usize> {
        self.shifts.get(role).copied()
    }

    /// Compose a vector from role-filler bindings.
    /// T = XOR_i(permute(filler_i, shift_i))
    pub fn compose(&self, bindings: &[(&str, &EntangledHVec)]) -> Result<EntangledHVec> {
        if bindings.is_empty() {
            return Err(anyhow!("compose requires at least one binding"));
        }
        let dim = bindings[0].1.dim;
        let mut result = EntangledHVec::from_indices(vec![], dim);
        for &(role, vec) in bindings {
            let shift = self
                .shifts
                .get(role)
                .ok_or_else(|| anyhow!("unknown role: {}", role))?;
            let shifted = if *shift == 0 {
                vec.clone()
            } else {
                vec.permute(*shift)
            };
            result = result.bind(&shifted);
        }
        Ok(result)
    }

    /// Unbind known role-fillers from a composite to isolate the unknown role's filler.
    /// result = composite XOR XOR_i(permute(known_i, shift_i))
    /// Then inverse-permute by the target role's shift.
    pub fn unbind(
        &self,
        composite: &EntangledHVec,
        known: &[(&str, &EntangledHVec)],
        target_role: &str,
    ) -> Result<EntangledHVec> {
        let target_shift = self
            .shifts
            .get(target_role)
            .ok_or_else(|| anyhow!("unknown target role: {}", target_role))?;

        let mut known_combined = EntangledHVec::from_indices(vec![], composite.dim);
        for &(role, vec) in known {
            let shift = self
                .shifts
                .get(role)
                .ok_or_else(|| anyhow!("unknown role: {}", role))?;
            let shifted = if *shift == 0 {
                vec.clone()
            } else {
                vec.permute(*shift)
            };
            known_combined = known_combined.bind(&shifted);
        }

        let residual = composite.bind(&known_combined);

        if *target_shift == 0 {
            Ok(residual)
        } else {
            Ok(residual.permute(self.dim - target_shift))
        }
    }

    /// Convenience: compose a standard (subject, relation, object) triple.
    pub fn compose_triple(
        &self,
        subject: &EntangledHVec,
        relation: &EntangledHVec,
        object: &EntangledHVec,
    ) -> EntangledHVec {
        self.compose(&[
            ("subject", subject),
            ("relation", relation),
            ("object", object),
        ])
        .expect("default roles always registered")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(dim: usize, seed: u64) -> EntangledHVec {
        EntangledHVec::new_deterministic(dim, seed)
    }

    #[test]
    fn test_compose_unbind_roundtrip() {
        let reg = RoleRegistry::new(16384);
        let s = make_vec(16384, 1);
        let r = make_vec(16384, 2);
        let o = make_vec(16384, 3);

        let composite = reg.compose_triple(&s, &r, &o);

        // Unbind object given subject + relation
        let recovered_o = reg
            .unbind(&composite, &[("subject", &s), ("relation", &r)], "object")
            .unwrap();
        assert!(
            (recovered_o.similarity(&o) - 1.0).abs() < 0.0001,
            "Object should be recovered exactly"
        );

        // Unbind subject given relation + object
        let recovered_s = reg
            .unbind(&composite, &[("relation", &r), ("object", &o)], "subject")
            .unwrap();
        assert!(
            (recovered_s.similarity(&s) - 1.0).abs() < 0.0001,
            "Subject should be recovered exactly"
        );

        // Unbind relation given subject + object
        let recovered_r = reg
            .unbind(&composite, &[("subject", &s), ("object", &o)], "relation")
            .unwrap();
        assert!(
            (recovered_r.similarity(&r) - 1.0).abs() < 0.0001,
            "Relation should be recovered exactly"
        );
    }

    #[test]
    fn test_commutativity_fix() {
        let reg = RoleRegistry::new(16384);
        let a = make_vec(16384, 10);
        let r = make_vec(16384, 20);
        let b = make_vec(16384, 30);

        let t1 = reg.compose_triple(&a, &r, &b); // "A r B"
        let t2 = reg.compose_triple(&b, &r, &a); // "B r A"

        // These MUST be different (commutativity fix)
        assert!(
            t1.similarity(&t2) < 0.5,
            "Swapped subject/object should produce different composites, got similarity {}",
            t1.similarity(&t2)
        );
    }

    #[test]
    fn test_register_duplicate_same_shift() {
        let mut reg = RoleRegistry::new(16384);
        assert!(reg.register("subject", 0).is_ok()); // same shift, ok
    }

    #[test]
    fn test_register_duplicate_different_shift() {
        let mut reg = RoleRegistry::new(16384);
        assert!(reg.register("subject", 5).is_err()); // different shift, error
    }

    #[test]
    fn test_unknown_role_error() {
        let reg = RoleRegistry::new(16384);
        let v = make_vec(16384, 1);
        assert!(reg.compose(&[("nonexistent", &v)]).is_err());
    }
}
