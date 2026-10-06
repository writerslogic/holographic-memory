// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

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
        #[cfg(feature = "local-models")]
        if let Some(stages) = self.model_stages() {
            return self.memorize_document_with_models(input, &stages);
        }
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
        // A text query goes through the installed model stages; a vector-only query does not.
        #[cfg(feature = "local-models")]
        if let Some(stages) = self.model_stages().filter(|_| !text.trim().is_empty()) {
            return self.search_documents_with_models(text, "", options, &stages);
        }
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

#[cfg(feature = "local-models")]
impl HmsCore {
    /// Installs (or with `None` removes) the on-device model stages `memorize_document` and
    /// `search_documents` use. An embedder must match the store's embedding space.
    pub fn set_model_stages(
        &self,
        stages: Option<std::sync::Arc<crate::core::models::ModelStages>>,
    ) -> Result<()> {
        if let Some(e) = stages.as_ref().and_then(|s| s.embedder.as_ref()) {
            let space = self.config.embedding_space.as_ref().map(|s| s.dimensions);
            ensure!(
                space == Some(e.dimensions()),
                "the embedder's {} dimensions do not match the store embedding space {space:?}",
                e.dimensions()
            );
        }
        *self.model_stages.write() = stages;
        Ok(())
    }

    pub fn model_stages(&self) -> Option<std::sync::Arc<crate::core::models::ModelStages>> {
        self.model_stages.read().clone()
    }

    fn memorize_document_with_models(
        &self,
        mut input: DocumentInput,
        stages: &crate::core::models::ModelStages,
    ) -> Result<u32> {
        let chunks: Vec<String> = documents::chunk_document(&input)?
            .into_iter()
            .map(|c| c.text)
            .collect();
        let keys = stages.fact_keys(&chunks)?;
        if let (Some(embedder), None) = (&stages.embedder, &input.embeddings) {
            let texts = keys.as_deref().unwrap_or(&chunks);
            input.embeddings = Some(
                embedder
                    .embed(texts, false)?
                    .into_iter()
                    .map(|e| e.into_iter().map(f64::from).collect())
                    .collect(),
            );
        }
        let document = documents::prepare_with_keys(
            input,
            self.dimensions,
            self.config.embedding_space.as_ref().map(|s| s.dimensions),
            keys.as_deref(),
        )?;
        let count = document.chunks.len() as u32;
        self.commit(&[Mutation::Document(document)])?;
        Ok(count)
    }

    /// Search through the model stages: the optional LLM rewrite (`today` is the date relative
    /// times resolve against), a hybrid search per query variant with on-device query
    /// embeddings, weighted RRF across variants, then the re-rank of the fused head.
    pub fn search_documents_with_models(
        &self,
        text: &str,
        today: &str,
        options: &SearchOptions,
        stages: &crate::core::models::ModelStages,
    ) -> Result<Vec<DocumentResult>> {
        let k = options.k.unwrap_or(10) as usize;
        let pool = k.max(stages.params.rerank_n) as u32;
        let variants = stages.query_variants(text, today)?;
        let embeddings = match (&stages.embedder, &options.embedding) {
            (Some(e), None) => {
                let queries: Vec<&str> = variants.iter().map(|v| v.0.as_str()).collect();
                Some(e.embed(&queries, true)?)
            }
            _ => None,
        };
        let mut lists = Vec::with_capacity(variants.len());
        for (i, (query, weight)) in variants.iter().enumerate() {
            let mut opts = options.clone();
            opts.k = Some(pool);
            opts.lexical_weight = opts.lexical_weight.or(Some(stages.params.lexical_weight));
            opts.semantic_weight = opts.semantic_weight.or(Some(stages.params.semantic_weight));
            if let Some(e) = &embeddings {
                opts.embedding = Some(e[i].iter().copied().map(f64::from).collect());
            }
            let _transaction = self.mutation_gate.read();
            let list = documents::search(
                &self.documents.read(),
                query,
                &opts,
                self.config.embedding_space.as_ref().map(|s| s.dimensions),
            )?;
            lists.push((*weight, list));
        }
        stages.fuse(text, lists, k)
    }
}
