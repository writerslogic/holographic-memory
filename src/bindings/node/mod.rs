// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use napi::bindgen_prelude::*;
use std::sync::Arc;
use tracing::info_span;

use crate::{
    chunk_document, ConceptCandidate, DocumentChunk, DocumentInput, DocumentResult, EntangledHVec,
    HardwareCapabilities, IndexStatus, MemorizeBatchItem, QueryExplanation, RetrievalResult,
    SearchOptions, StorageHealth, TextMetrics, WarmCore,
};

mod api;
pub use api::WarmMemorySystem;
mod config;
mod types;

#[cfg(feature = "local-embed")]
mod embed;

pub use config::WarmConfigJs;
pub use types::*;

fn napi_err(e: anyhow::Error) -> napi::Error {
    if e.is::<crate::CommittedMutationError>() {
        napi::Error::from_reason(format!("[mutation_committed] {e}"))
    } else {
        napi::Error::from_reason(e.to_string())
    }
}

/// Memory API errors carry their class in a `[code]` prefix; `memory.js` turns it into
/// `error.code`.
fn memory_err(e: anyhow::Error) -> napi::Error {
    use crate::core::memory_api::MemoryError;
    let code = match e.downcast_ref::<MemoryError>() {
        Some(MemoryError::Invalid(_)) => "invalid_request",
        Some(MemoryError::NotFound(_)) => "not_found",
        Some(MemoryError::Conflict(_)) => "conflict",
        Some(MemoryError::Concurrent(_)) => "concurrent_mutation",
        None if e.is::<crate::CommittedMutationError>() => "mutation_committed",
        None => "internal",
    };
    napi::Error::from_reason(format!("[{code}] {e}"))
}

async fn run_async<T: Send + 'static, F: FnOnce() -> anyhow::Result<T> + Send + 'static>(
    f: F,
) -> Result<T> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))?
        .map_err(napi_err)
}
