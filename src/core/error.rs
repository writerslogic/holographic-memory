// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use ts_rs::TS;

#[derive(Debug, Error, Clone, Serialize, Deserialize, TS, JsonSchema)]
#[ts(export, export_to = "bindings/")]
#[serde(tag = "type", content = "message")]
pub enum HmsError {
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
