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
    } else if let Some(limit) = e.downcast_ref::<crate::core::query_limits::QueryLimitExceeded>() {
        napi::Error::from_reason(format!(
            "[{}] {limit}",
            crate::core::query_limits::QueryLimitExceeded::CODE
        ))
    } else {
        napi::Error::from_reason(e.to_string())
    }
}

/// Memory API errors carry their class in a `[code]` prefix; `memory.js` turns it into
/// `error.code`.
fn memory_err(e: anyhow::Error) -> napi::Error {
    use crate::core::memory_api::MemoryError;
    if e.is::<crate::CommittedMutationError>()
        || e.is::<crate::core::query_limits::QueryLimitExceeded>()
    {
        return napi_err(e);
    }
    let code = match e.downcast_ref::<MemoryError>() {
        Some(MemoryError::Invalid(_)) => "invalid_request",
        Some(MemoryError::NotFound(_)) => "not_found",
        Some(MemoryError::Conflict(_)) => "conflict",
        Some(MemoryError::Concurrent(_)) => "concurrent_mutation",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::query_limits::{QueryBudget, QueryLimits};

    #[test]
    fn query_exhaustion_keeps_its_code_through_both_node_paths() {
        for resource in ["work", "bytes", "frontier"] {
            for convert in [napi_err, memory_err] {
                let mut budget = QueryBudget::new(QueryLimits {
                    max_work: 1,
                    max_bytes: 1,
                    max_frontier: 1,
                });
                let error = match resource {
                    "work" => budget.work(2),
                    "bytes" => budget.reserve(2),
                    _ => budget.frontier(2),
                }
                .unwrap_err();
                let error = convert(anyhow::Error::new(error).context("private context"));
                assert_eq!(
                    error.reason,
                    format!("[resource_limit] query {resource} limit 1 exceeded")
                );
            }
        }
        assert_eq!(
            memory_err(anyhow::anyhow!("query work limit 1 exceeded")).reason,
            "[internal] query work limit 1 exceeded"
        );
    }
}
