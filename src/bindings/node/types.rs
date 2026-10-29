// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use napi_derive::napi;

#[napi(object)]
pub struct AuditEntryJs {
    pub timestamp_ms: f64,
    pub op: String,
    pub id_hash: String,
    pub signed: bool,
}

/// A stored document as `listDocuments` returns it: identity and provenance, not text.
#[napi(object)]
pub struct ListedDocument {
    pub id: String,
    pub version: String,
    pub source_uri: Option<String>,
    pub metadata: serde_json::Value,
    pub chunk_count: u32,
}

#[napi(object)]
pub struct StructuralResultJs {
    pub entity_id: String,
    pub confidence: f64,
    pub path: String,
}

#[napi(object)]
pub struct MultiHopResultJs {
    pub entity_id: String,
    pub confidence: f64,
    pub method: String,
}

#[napi(object)]
pub struct CleanupResultJs {
    pub id: String,
    pub confidence: f64,
}

#[napi(object)]
pub struct GovernanceReportJs {
    pub composites_merged: u32,
    pub composites_forgotten: u32,
    pub atoms_forgotten: u32,
    pub idf_refreshed: bool,
    pub atoms_refined: u32,
}

#[napi(object)]
pub struct GoalJs {
    pub name: String,
    pub utility: f64,
}

#[napi(object)]
pub struct PlannedActionJs {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub depth: u32,
    pub cost: f64,
}

#[napi(object)]
pub struct PlanJs {
    pub goal: String,
    pub actions: Vec<PlannedActionJs>,
    pub complete: bool,
    pub total_cost: f64,
}

#[napi(object)]
pub struct QuestionJs {
    pub text: String,
    pub goal_relevance: f64,
}
