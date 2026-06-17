// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "node-api")]
use napi_derive::napi;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[cfg_attr(feature = "node-api", napi(object))]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, TS, JsonSchema)]
#[ts(export, export_to = "bindings/")]
#[serde(rename_all = "camelCase")]
pub struct TextMetrics {
    pub word_count: u32,
    pub sentence_count: u32,
    pub syllable_count: u32,
    pub vowel_count: u32,
    pub consonant_count: u32,
    pub punctuation_count: u32,
}

#[cfg_attr(feature = "node-api", napi(object))]
#[derive(Clone, Serialize, Deserialize, Debug, TS, JsonSchema)]
#[ts(export, export_to = "bindings/")]
pub struct RetrievalResult {
    pub id: String,
    pub similarity: f64,
}

impl PartialEq for RetrievalResult {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && (self.similarity - other.similarity).abs() < f64::EPSILON
    }
}

impl Eq for RetrievalResult {}

impl PartialOrd for RetrievalResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RetrievalResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a MAX-heap. We want it to act like a MIN-heap for Top-K
        // (keeping the K LARGEST similarities, with the SMALLEST of those at the top).
        // So we want A < B (in similarity) to mean A is "Greater" in Ord.
        debug_assert!(
            !self.similarity.is_nan(),
            "NaN similarity for id={}",
            self.id
        );
        debug_assert!(
            !other.similarity.is_nan(),
            "NaN similarity for id={}",
            other.id
        );
        other
            .similarity
            .partial_cmp(&self.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            // If similarities are equal, we want a stable order.
            // If A.id < B.id, let's say A is "smaller" (top of heap) so it gets popped.
            // Actually for stability in the final result, we want consistent tie-breaking.
            .then_with(|| self.id.cmp(&other.id))
    }
}

#[cfg_attr(feature = "node-api", napi(object))]
#[derive(Clone, Serialize, Deserialize, TS, JsonSchema)]
#[ts(export, export_to = "bindings/")]
#[serde(rename_all = "camelCase")]
pub struct ConceptCandidate {
    pub centroid_id: String,
    pub member_count: u32,
    pub coherence: f64,
    pub member_ids: Vec<String>,
}

/// Input item for batch memorization — a single id/text pair.
#[cfg_attr(feature = "node-api", napi(object))]
#[derive(Clone, Serialize, Deserialize, TS, JsonSchema)]
#[ts(export, export_to = "bindings/")]
pub struct MemorizeBatchItem {
    pub id: String,
    pub text: String,
}

/// An explicit directed relation between two entities.
#[cfg_attr(feature = "node-api", napi(object))]
#[derive(Clone, Debug, Serialize, Deserialize, TS, JsonSchema)]
#[ts(export, export_to = "bindings/")]
#[serde(rename_all = "camelCase")]
pub struct Relation {
    pub source_id: String,
    pub relation_type: String,
    pub target_id: String,
    pub properties: Option<String>,
    /// Milliseconds since epoch. 0 = unbounded.
    pub valid_from: f64,
    /// Milliseconds since epoch. 0 = unbounded (still active).
    pub valid_to: f64,
}

/// Declares a relation type with inference semantics.
#[cfg_attr(feature = "node-api", napi(object))]
#[derive(Clone, Debug, Serialize, Deserialize, TS, JsonSchema)]
#[ts(export, export_to = "bindings/")]
#[serde(rename_all = "camelCase")]
pub struct RelationType {
    pub name: String,
    /// If true, A->B and B->C implies A->C.
    pub transitive: bool,
    /// If true, A->B implies B->A.
    pub symmetric: bool,
}

/// A single hop in a graph traversal path.
#[cfg_attr(feature = "node-api", napi(object))]
#[derive(Clone, Debug, Serialize, Deserialize, TS, JsonSchema)]
#[ts(export, export_to = "bindings/")]
#[serde(rename_all = "camelCase")]
pub struct PathHop {
    pub node_id: String,
    pub relation_type: String,
    pub similarity: f64,
}

/// Result of a graph traversal or path query.
#[cfg_attr(feature = "node-api", napi(object))]
#[derive(Clone, Debug, Serialize, Deserialize, TS, JsonSchema)]
#[ts(export, export_to = "bindings/")]
#[serde(rename_all = "camelCase")]
pub struct GraphPath {
    pub hops: Vec<PathHop>,
    /// Product of similarities along the path.
    pub score: f64,
}
