// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#[cfg(not(target_arch = "wasm32"))]
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use super::config::HmsConfig;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingSpace {
    pub model: String,
    pub revision: String,
    pub dimensions: usize,
    pub normalization: String,
    pub metric: String,
}

#[cfg(any(feature = "local-embed", feature = "local-models"))]
impl EmbeddingSpace {
    pub(crate) fn effective(
        model: String,
        dimensions: usize,
        normalization: &str,
        descriptor: &impl Serialize,
    ) -> Result<Self> {
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(serde_json::to_vec(descriptor)?);
        Ok(Self {
            model,
            revision: format!(
                "hms-embedding-v1:{}",
                digest
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            ),
            dimensions,
            normalization: normalization.into(),
            metric: "cosine".into(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[cfg(not(target_arch = "wasm32"))]
struct StoreSchema {
    version: u32,
    dimensions: usize,
    text_encoder: String,
    vector_encoder: String,
    metric: String,
    encrypted: bool,
    meaning_enabled: bool,
    embedding_space: Option<EmbeddingSpace>,
}

/// Checks the store's `store.json` against `config` (writing it for a new store) and returns
/// its schema version, which fixes the journal format the store is written in.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn validate_store(path: &Path, dimensions: usize, config: &HmsConfig) -> Result<u32> {
    let mut expected = StoreSchema {
        version: super::format::STORE_SCHEMA_VERSION,
        dimensions,
        text_encoder: config.text_encoder.name().into(),
        vector_encoder: "signed-projection-v2".into(),
        metric: "jaccard".into(),
        encrypted: config.security.encryption_enabled,
        meaning_enabled: config.meaning.enabled,
        embedding_space: config.embedding_space.clone(),
    };
    let schema_path = path.join("store.json");
    if schema_path.exists() {
        let bytes = std::fs::read(schema_path)?;
        if let Some(version) = serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| v["version"].as_u64())
            .filter(|&v| v > u64::from(super::format::STORE_SCHEMA_VERSION))
        {
            bail!(
                "store schema version {version} is newer than the supported version {}; open it with a release that supports it",
                super::format::STORE_SCHEMA_VERSION
            );
        }
        let actual: StoreSchema = serde_json::from_slice(&bytes)?;
        // Every schema from the oldest supported one opens; only the version differs between
        // them (it selects the journal format).
        if (super::format::STORE_SCHEMA_OLDEST..=super::format::STORE_SCHEMA_VERSION)
            .contains(&actual.version)
        {
            expected.version = actual.version;
        }
        if actual != expected {
            bail!("incompatible store schema: dimensions, encoder, embedding model, encryption, or meaning configuration changed; reopen with the original configuration or re-encode source documents into a new store");
        }
    } else {
        if path.join("vectors_data.bin").exists() {
            bail!("legacy store has no embedding schema; preserve it and use hms-admin reencode with original source documents to create a new store");
        }
        super::durable_file::atomic_write(&schema_path, &serde_json::to_vec_pretty(&expected)?)?;
    }
    Ok(expected.version)
}
