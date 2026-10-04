use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use super::config::HmsConfig;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbeddingSpace {
    pub model: String,
    pub revision: String,
    pub dimensions: usize,
    pub normalization: String,
    pub metric: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
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

pub(crate) fn validate_store(path: &Path, dimensions: usize, config: &HmsConfig) -> Result<()> {
    let expected = StoreSchema {
        version: 2,
        dimensions,
        text_encoder: "multiscale-words-v1".into(),
        vector_encoder: "signed-projection-v2".into(),
        metric: "jaccard".into(),
        encrypted: config.security.encryption_enabled,
        meaning_enabled: config.meaning.enabled,
        embedding_space: config.embedding_space.clone(),
    };
    let schema_path = path.join("store.json");
    if schema_path.exists() {
        let actual: StoreSchema = serde_json::from_slice(&std::fs::read(schema_path)?)?;
        if actual != expected {
            bail!("incompatible store schema: dimensions, encoder, embedding model, encryption, or meaning configuration changed; reopen with the original configuration or re-encode source documents into a new store");
        }
    } else {
        if path.join("vectors_data.bin").exists() {
            bail!("legacy store has no embedding schema; preserve it and use hms-admin reencode with original source documents to create a new store");
        }
        super::durable_file::atomic_write(&schema_path, &serde_json::to_vec_pretty(&expected)?)?;
    }
    Ok(())
}
