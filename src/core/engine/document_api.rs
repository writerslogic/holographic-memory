use anyhow::{ensure, Result};
use rayon::prelude::*;

use super::{
    mutation::{validate_id, Mutation},
    HmsCore, ShardSet,
};
use crate::core::documents::{self, DocumentInput, DocumentResult, SearchOptions};
use crate::core::types::MemorizeBatchItem;

impl HmsCore {
    pub fn security_status(&self) -> serde_json::Value {
        serde_json::json!({
            "securityCompiled": cfg!(feature = "security"),
            "encryptionActive": self.config.security.encryption_enabled,
            "signingActive": self.config.security.signing_enabled,
            "auditActive": self.config.security.audit_enabled,
            "differentialPrivacyActive": self.config.privacy.dp_enabled,
            "epsilon": self.config.privacy.epsilon,
            "privacyMechanism": "clipped-full-domain-laplace-top-k",
            "embeddingSpace": self.config.embedding_space,
        })
    }

    pub fn memorize_document(&self, input: DocumentInput) -> Result<u32> {
        let document = documents::prepare(
            input,
            self.dimensions,
            self.config.embedding_space.as_ref().map(|s| s.dimensions),
        )?;
        let count = document.chunks.len() as u32;
        self.commit(&[Mutation::Document(document)])?;
        Ok(count)
    }

    pub fn search_documents(
        &self,
        text: &str,
        options: &SearchOptions,
    ) -> Result<Vec<DocumentResult>> {
        let _transaction = self.mutation_gate.read();
        documents::search(
            &self.documents.read(),
            text,
            options,
            self.config.embedding_space.as_ref().map(|s| s.dimensions),
        )
    }

    pub fn delete_document(&self, id: &str) -> Result<bool> {
        validate_id(id)?;
        let exists = self.documents.read().contains_key(id);
        self.commit(&[Mutation::DeleteDocument { id: id.into() }])?;
        Ok(exists)
    }

    pub(super) fn remove_document_from_memory(&self, id: &str, shards: &ShardSet) -> Result<()> {
        if let Some(old) = self.documents.write().remove(id) {
            for chunk in old.chunks {
                shards.remove(&chunk.id, self.dimensions)?;
            }
        }
        Ok(())
    }

    /// One bounded transaction; either every item is logged or none is.
    pub fn memorize_batch(&self, items: &[MemorizeBatchItem]) -> Result<()> {
        ensure!(
            !items.is_empty() && items.len() <= 4096,
            "batch requires 1..=4096 items"
        );
        ensure!(
            items
                .iter()
                .map(|item| item.id.len() + item.text.len())
                .sum::<usize>()
                <= documents::MAX_DOCUMENT_BYTES,
            "batch exceeds 8 MiB; split the input into bounded batches"
        );
        for item in items {
            super::mutation::validate_public_id(&item.id)?;
            ensure!(
                !item.id.starts_with(documents::CHUNK_PREFIX),
                "chunk IDs are reserved"
            );
        }
        let mutations: Vec<_> = items
            .par_iter()
            .map(|item| Mutation::Vector {
                id: item.id.clone(),
                vector: self.encode_text(&item.text),
            })
            .collect();
        self.commit(&mutations)
    }

    pub fn query_vector(&self, dense: &[f32], k: u32) -> Result<Vec<crate::RetrievalResult>> {
        self.validate_dense(dense)?;
        let vector = crate::EntangledHVec::from_dense(dense, self.dimensions);
        Ok(self.query(&vector, k))
    }

    pub(super) fn validate_dense(&self, dense: &[f32]) -> Result<()> {
        ensure!(
            !dense.is_empty()
                && dense.len() <= 65536
                && dense.iter().all(|v| v.is_finite())
                && dense.iter().any(|&v| v != 0.0),
            "dense vectors require 1..=65536 finite values and nonzero magnitude"
        );
        if let Some(space) = &self.config.embedding_space {
            ensure!(
                dense.len() == space.dimensions,
                "dense embedding dimensions do not match the store schema"
            );
        }
        Ok(())
    }
}
