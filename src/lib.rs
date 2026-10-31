// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![deny(clippy::all)]

mod bindings;
pub mod compatibility;
pub mod core;

#[cfg(feature = "node-api")]
pub use bindings::node::*;

#[cfg(not(target_arch = "wasm32"))]
pub use crate::core::audit::{AuditEntry, AuditOp};
pub use crate::core::documents::{
    chunk_document, DocumentChunk, DocumentInput, DocumentRecord, DocumentResult, SearchOptions,
};
pub use crate::core::entangled::EntangledHVec;
pub use crate::core::error::{CommittedMutationError, WarmError};
#[cfg(not(target_arch = "wasm32"))]
pub use crate::core::memory_api::{Memory, MemoryConfig};
pub use crate::core::types::{
    ConceptCandidate, HardwareCapabilities, IndexStatus, MemorizeBatchItem, QueryExplanation,
    RetrievalResult, StorageHealth, TextMetrics,
};
pub use crate::core::EmbeddingSpace;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::core::WarmCore;
#[cfg(not(target_arch = "wasm32"))]
pub use WarmCore as HmsCore;
pub use WarmError as HmsError;
#[cfg(feature = "node-api")]
pub use WarmMemorySystem as HolographicMemorySystem;

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
