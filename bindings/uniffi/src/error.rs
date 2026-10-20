// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::io::ErrorKind;

/// Errors raised by the store. Every variant carries a details that is safe to log: none
/// contains key material or document text.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum HmsError {
    /// An argument failed validation (identifier, JSON, option range, embedding length).
    #[error("invalid input: {details}")]
    InvalidInput { details: String },
    /// Another engine, in this process or another, holds the store directory.
    #[error("store is locked: {details}")]
    StoreLocked { details: String },
    /// The encryption key is missing, wrong, or does not match the store.
    #[error("encryption error: {details}")]
    Encryption { details: String },
    /// The store was created with different dimensions, embedding space or encryption setting.
    #[error("incompatible store: {details}")]
    IncompatibleStore { details: String },
    /// A path that must exist does not.
    #[error("not found: {details}")]
    NotFound { details: String },
    /// A path that must be absent or empty is not.
    #[error("already exists: {details}")]
    AlreadyExists { details: String },
    /// A snapshot failed verification, or the store's files are damaged.
    #[error("corrupt data: {details}")]
    Corrupt { details: String },
    /// The operating system refused a file operation.
    #[error("i/o error: {details}")]
    Io { details: String },
    /// The store handle was closed.
    #[error("store is closed")]
    Closed,
    /// A bug or panic inside the engine.
    #[error("internal error: {details}")]
    Internal { details: String },
}

impl HmsError {
    pub(crate) fn invalid(details: impl Into<String>) -> Self {
        Self::InvalidInput {
            details: details.into(),
        }
    }

    pub(crate) fn corrupt(details: impl Into<String>) -> Self {
        Self::Corrupt {
            details: details.into(),
        }
    }

    /// Classifies an engine error. The engine reports validation failures as plain messages,
    /// so the match is on its stable phrases; I/O errors are classified by their kind.
    pub(crate) fn from_engine(error: &anyhow::Error) -> Self {
        let details = format!("{error:#}");
        let lower = details.to_ascii_lowercase();
        if lower.contains("already open by another engine") {
            return Self::StoreLocked { details };
        }
        if lower.contains("incompatible store schema") || lower.contains("legacy store has no") {
            return Self::IncompatibleStore { details };
        }
        if lower.contains("encryption requires")
            || lower.contains("decrypt")
            || lower.contains("encrypted")
            || lower.contains("invalid salt")
        {
            return Self::Encryption { details };
        }
        if lower.contains("cannot replay")
            || lower.contains("checksum")
            || lower.contains("corrupt")
            || lower.contains("truncated")
        {
            return Self::Corrupt { details };
        }
        if let Some(io) = error
            .chain()
            .find_map(|cause| cause.downcast_ref::<std::io::Error>())
        {
            return match io.kind() {
                ErrorKind::NotFound => Self::NotFound { details },
                ErrorKind::AlreadyExists => Self::AlreadyExists { details },
                _ => Self::Io { details },
            };
        }
        if lower.contains("must ")
            || lower.contains("invalid")
            || lower.contains("exceeds")
            || lower.contains("reserved")
            || lower.contains("too large")
            || lower.contains("out of range")
        {
            return Self::InvalidInput { details };
        }
        Self::Internal { details }
    }
}

impl From<std::io::Error> for HmsError {
    fn from(error: std::io::Error) -> Self {
        let details = error.to_string();
        match error.kind() {
            ErrorKind::NotFound => Self::NotFound { details },
            ErrorKind::AlreadyExists => Self::AlreadyExists { details },
            _ => Self::Io { details },
        }
    }
}

impl From<anyhow::Error> for HmsError {
    fn from(error: anyhow::Error) -> Self {
        Self::from_engine(&error)
    }
}
