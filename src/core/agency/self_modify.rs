// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Self-modification proposals with mandatory approval.
//!
//! The system can propose changes to its own rules, thresholds, or
//! decomposition patterns. ALL proposals require explicit user approval
//! before taking effect. Every proposal and its outcome is logged.

use parking_lot::RwLock;

/// A proposed self-modification.
#[derive(Clone, Debug)]
pub struct Proposal {
    pub id: usize,
    pub kind: ProposalKind,
    pub reason: String,
    pub status: ProposalStatus,
}

/// What kind of modification is proposed.
#[derive(Clone, Debug)]
pub enum ProposalKind {
    /// Add a new composition rule.
    AddRule {
        name: String,
        input_relations: Vec<String>,
        output_relation: String,
    },
    /// Adjust a numeric threshold (e.g., fanout limit, IDF clip factor).
    AdjustThreshold {
        parameter: String,
        current_value: f64,
        proposed_value: f64,
    },
    /// Add a new decomposition pattern to the decomposer.
    AddPattern { description: String },
    /// Remove a stale or incorrect triple.
    RemoveTriple {
        subject: String,
        relation: String,
        object: String,
    },
}

/// Lifecycle of a proposal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProposalStatus {
    Pending,
    Approved,
    Rejected,
    Applied,
}

/// Manages self-modification proposals. Nothing is applied without approval.
pub struct SelfModifier {
    proposals: RwLock<Vec<Proposal>>,
    audit_log: RwLock<Vec<AuditEntry>>,
}

/// Record of a proposal event.
#[derive(Clone, Debug)]
pub struct AuditEntry {
    pub proposal_id: usize,
    pub action: String,
    pub timestamp_ms: u64,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl Default for SelfModifier {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfModifier {
    pub fn new() -> Self {
        Self {
            proposals: RwLock::new(Vec::new()),
            audit_log: RwLock::new(Vec::new()),
        }
    }

    /// Submit a proposal. Returns the proposal ID.
    pub fn propose(&self, kind: ProposalKind, reason: String) -> usize {
        let id = {
            let mut proposals = self.proposals.write();
            let id = proposals.len();
            proposals.push(Proposal {
                id,
                kind,
                reason,
                status: ProposalStatus::Pending,
            });
            id
        };
        self.log_action(id, "proposed");
        id
    }

    /// Approve a pending proposal. Returns true if it was pending.
    pub fn approve(&self, id: usize) -> bool {
        let changed = {
            let mut proposals = self.proposals.write();
            if let Some(p) = proposals.get_mut(id) {
                if p.status == ProposalStatus::Pending {
                    p.status = ProposalStatus::Approved;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        if changed {
            self.log_action(id, "approved");
        }
        changed
    }

    /// Reject a pending proposal. Returns true if it was pending.
    pub fn reject(&self, id: usize) -> bool {
        let changed = {
            let mut proposals = self.proposals.write();
            if let Some(p) = proposals.get_mut(id) {
                if p.status == ProposalStatus::Pending {
                    p.status = ProposalStatus::Rejected;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        if changed {
            self.log_action(id, "rejected");
        }
        changed
    }

    /// Mark an approved proposal as applied. Returns true if it was approved.
    pub fn mark_applied(&self, id: usize) -> bool {
        let changed = {
            let mut proposals = self.proposals.write();
            if let Some(p) = proposals.get_mut(id) {
                if p.status == ProposalStatus::Approved {
                    p.status = ProposalStatus::Applied;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        };
        if changed {
            self.log_action(id, "applied");
        }
        changed
    }

    fn log_action(&self, proposal_id: usize, action: &str) {
        self.audit_log.write().push(AuditEntry {
            proposal_id,
            action: action.to_string(),
            timestamp_ms: now_ms(),
        });
    }

    pub fn pending_proposals(&self) -> Vec<Proposal> {
        self.proposals
            .read()
            .iter()
            .filter(|p| p.status == ProposalStatus::Pending)
            .cloned()
            .collect()
    }

    pub fn get(&self, id: usize) -> Option<Proposal> {
        self.proposals.read().get(id).cloned()
    }

    pub fn proposal_count(&self) -> usize {
        self.proposals.read().len()
    }

    pub fn pending_count(&self) -> usize {
        self.proposals
            .read()
            .iter()
            .filter(|p| p.status == ProposalStatus::Pending)
            .count()
    }

    pub fn audit_entries(&self) -> Vec<AuditEntry> {
        self.audit_log.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_propose_approve_apply() {
        let sm = SelfModifier::new();
        let id = sm.propose(
            ProposalKind::AddRule {
                name: "uncle".to_string(),
                input_relations: vec!["parent".to_string(), "sibling".to_string()],
                output_relation: "uncle".to_string(),
            },
            "Discovered uncle pattern in triples".to_string(),
        );

        assert_eq!(sm.pending_count(), 1);
        assert!(sm.approve(id));
        assert_eq!(sm.pending_count(), 0);
        assert_eq!(sm.get(id).unwrap().status, ProposalStatus::Approved);

        assert!(sm.mark_applied(id));
        assert_eq!(sm.get(id).unwrap().status, ProposalStatus::Applied);
    }

    #[test]
    fn test_reject() {
        let sm = SelfModifier::new();
        let id = sm.propose(
            ProposalKind::AdjustThreshold {
                parameter: "fanout_limit".to_string(),
                current_value: 40.0,
                proposed_value: 60.0,
            },
            "High fan-out queries are slow".to_string(),
        );

        assert!(sm.reject(id));
        assert_eq!(sm.get(id).unwrap().status, ProposalStatus::Rejected);
        assert!(!sm.approve(id)); // can't approve a rejected proposal
    }

    #[test]
    fn test_cannot_apply_without_approval() {
        let sm = SelfModifier::new();
        let id = sm.propose(
            ProposalKind::RemoveTriple {
                subject: "wrong".to_string(),
                relation: "is".to_string(),
                object: "right".to_string(),
            },
            "Incorrect triple".to_string(),
        );

        assert!(!sm.mark_applied(id)); // still pending, not approved
    }

    #[test]
    fn test_audit_trail() {
        let sm = SelfModifier::new();
        let id = sm.propose(
            ProposalKind::AddPattern {
                description: "passive voice pattern".to_string(),
            },
            "Improve decomposer".to_string(),
        );
        sm.approve(id);
        sm.mark_applied(id);

        let entries = sm.audit_entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].action, "proposed");
        assert_eq!(entries[1].action, "approved");
        assert_eq!(entries[2].action, "applied");
    }

    #[test]
    fn test_multiple_proposals() {
        let sm = SelfModifier::new();
        sm.propose(
            ProposalKind::AddRule {
                name: "r1".to_string(),
                input_relations: vec![],
                output_relation: "r1".to_string(),
            },
            "reason 1".to_string(),
        );
        sm.propose(
            ProposalKind::AddRule {
                name: "r2".to_string(),
                input_relations: vec![],
                output_relation: "r2".to_string(),
            },
            "reason 2".to_string(),
        );

        assert_eq!(sm.proposal_count(), 2);
        assert_eq!(sm.pending_count(), 2);

        let pending = sm.pending_proposals();
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn test_empty() {
        let sm = SelfModifier::new();
        assert_eq!(sm.proposal_count(), 0);
        assert_eq!(sm.pending_count(), 0);
        assert!(sm.get(0).is_none());
        assert!(sm.audit_entries().is_empty());
    }
}
