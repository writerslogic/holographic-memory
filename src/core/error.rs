// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
#[error("data mutation is durable, but {effect} failed; reopen or reconcile before retrying")]
pub struct CommittedMutationError {
    pub effect: &'static str,
    #[source]
    pub source: anyhow::Error,
}

pub(crate) fn after_commit<T>(
    effect: &'static str,
    result: anyhow::Result<T>,
) -> anyhow::Result<T> {
    result.map_err(|source| CommittedMutationError { effect, source }.into())
}

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
#[cfg_attr(
    feature = "node-api",
    derive(ts_rs::TS),
    ts(export, export_to = "bindings/")
)]
#[cfg_attr(feature = "server", derive(schemars::JsonSchema))]
#[serde(tag = "type", content = "message")]
pub enum WarmError {
    #[error("Invalid parameter: {details}")]
    InvalidParam { details: String },

    #[error("Storage failure: {context}")]
    StorageFailure { context: String },

    #[error("Encoding failed: {context}")]
    EncodingFailure { context: String },

    #[error("Query failed: {context}")]
    QueryFailure { context: String },

    #[error("Index not trained: {index_type}")]
    IndexNotTrained { index_type: String },

    #[error("Capacity exceeded: {limit}")]
    CapacityExceeded { limit: String },

    #[error("Internal error (code {code}): {context}")]
    Internal { code: i32, context: String },
}
