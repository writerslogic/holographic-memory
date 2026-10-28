// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![deny(clippy::all)]

mod bindings;
pub mod core;

#[cfg(feature = "node-api")]
pub use bindings::node::*;

pub use crate::core::documents::{
    chunk_document, DocumentChunk, DocumentInput, DocumentRecord, DocumentResult, SearchOptions,
};
pub use crate::core::entangled::EntangledHVec;
pub use crate::core::error::HmsError;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::core::memory_api::{Memory, MemoryConfig};
pub use crate::core::types::{
    ConceptCandidate, HardwareCapabilities, IndexStatus, MemorizeBatchItem, QueryExplanation,
    RetrievalResult, StorageHealth, TextMetrics,
};
pub use crate::core::EmbeddingSpace;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::core::HmsCore;

#[cfg(feature = "provenance")]
pub use crate::core::provenance::status_list::StatusListCredential;
#[cfg(feature = "provenance")]
pub use crate::core::provenance::types::{
    DeletionRecord, ProvenanceRecord, StoreManifest, VerificationDetail, VerificationResult,
};
#[cfg(feature = "provenance")]
pub use crate::core::provenance::TripleProvenanceParams;

#[cfg(test)]
mod tests;
