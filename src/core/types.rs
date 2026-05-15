#[cfg(feature = "node-api")]
use napi_derive::napi;
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "node-api", napi(object))]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TextMetrics {
    pub word_count: u32,
    pub sentence_count: u32,
    pub syllable_count: u32,
    pub vowel_count: u32,
    pub consonant_count: u32,
    pub punctuation_count: u32,
}

#[cfg_attr(feature = "node-api", napi(object))]
#[derive(Clone, Serialize, Deserialize, Debug)]
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
#[derive(Clone, Serialize, Deserialize)]
pub struct ConceptCandidate {
    pub centroid_id: String,
    pub member_count: u32,
    pub coherence: f64,
    pub member_ids: Vec<String>,
}
