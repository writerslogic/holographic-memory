// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

/// Fan-out gating for structural queries.
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
    fn test_admission_algebraic() {
        let ac = AdmissionControl::new(40);
        assert_eq!(ac.check(1), AdmissionDecision::Algebraic);
        assert_eq!(ac.check(40), AdmissionDecision::Algebraic);
    }

    #[test]
    fn test_admission_materialized() {
        let ac = AdmissionControl::new(40);
        assert_eq!(ac.check(41), AdmissionDecision::MaterializedLookup);
        assert_eq!(ac.check(1000), AdmissionDecision::MaterializedLookup);
    }

    #[test]
    fn test_admission_zero() {
        let ac = AdmissionControl::new(0);
        assert_eq!(ac.check(0), AdmissionDecision::Algebraic);
        assert_eq!(ac.check(1), AdmissionDecision::MaterializedLookup);
    }
}
