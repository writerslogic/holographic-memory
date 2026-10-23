// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

/// Fan-out gating for structural queries.
/// (Note: This is for Structural Query Routing, not Security Identity Admission. For W3C DIDs and VCs, see `src/core/security.rs`).
/// Decides whether to use the algebraic path (unbind + cleanup)
/// or the materialized path (TripleStore lookup).
pub struct AdmissionControl {
    fanout_limit: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDecision {
    Algebraic,
    MaterializedLookup,
}

impl AdmissionControl {
    pub fn new(fanout_limit: usize) -> Self {
        Self { fanout_limit }
    }

    pub fn check(&self, fan_out: usize) -> AdmissionDecision {
        if fan_out <= self.fanout_limit {
            AdmissionDecision::Algebraic
        } else {
            AdmissionDecision::MaterializedLookup
        }
    }

    #[allow(dead_code)]
    pub fn fanout_limit(&self) -> usize {
        self.fanout_limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fan_out_above_the_limit_uses_materialized_lookup() {
        use AdmissionDecision::{Algebraic, MaterializedLookup};
        for (limit, fan_out, expected) in [
            (40, 1, Algebraic),
            (40, 39, Algebraic),
            (40, 40, Algebraic),
            (40, 41, MaterializedLookup),
            (40, 1000, MaterializedLookup),
            (0, 0, Algebraic),
            (0, 1, MaterializedLookup),
        ] {
            assert_eq!(
                AdmissionControl::new(limit).check(fan_out),
                expected,
                "limit {limit}, fan-out {fan_out}"
            );
        }
    }
}
