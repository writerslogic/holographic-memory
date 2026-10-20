// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use rayon::prelude::*;
use std::sync::Arc;

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
            self.config.text_encoder,
        )?;
        let count = document.chunks.len() as u32;
        self.commit(&[Mutation::Document(Arc::new(document))])?;
        Ok(count)
    }

    pub fn memorize_documents(&self, inputs: Vec<DocumentInput>) -> Result<Vec<u32>> {
        #[cfg(feature = "local-models")]
        let stages = self.model_stages();
        #[cfg(feature = "local-models")]
        let inferred_dimensions = stages
            .as_ref()
            .and_then(|stages| stages.embedder.as_ref())
            .map(|embedder| embedder.dimensions());
        #[cfg(not(feature = "local-models"))]
        let inferred_dimensions = None;
        super::document_batch::validate_inputs(&inputs, inferred_dimensions)?;
        let mutations = inputs
            .into_par_iter()
            .map(|input| {
                #[cfg(feature = "local-models")]
                if let Some(stages) = stages.as_deref() {
                    return self
                        .prepare_document_with_models(input, stages)
                        .map(|document| Mutation::Document(Arc::new(document)));
                }
                documents::prepare(
                    input,
                    self.dimensions,
                    self.config.embedding_space.as_ref().map(|s| s.dimensions),
                    self.config.text_encoder,
                )
                .map(|document| Mutation::Document(Arc::new(document)))
            })
            .collect::<Result<Vec<_>>>()?;
        super::document_batch::validate_prepared(&mutations)?;
        let counts = mutations
            .iter()
            .map(|mutation| match mutation {
                Mutation::Document(document) => document.chunks.len() as u32,
                _ => unreachable!("document preparation produced a non-document mutation"),
            })
            .collect();
        self.commit(&mutations)?;
        Ok(counts)
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
        let prepared = documents::prepare_search(
            text,
            options,
            self.config.embedding_space.as_ref().map(|s| s.dimensions),
        )?;
        let captured = {
            let _transaction = self.mutation_gate.read();
            documents::capture_search(&self.documents.read(), vec![prepared])
        };
        Ok(documents::execute_search(&captured, 0))
    }

    pub fn document_index_trained(&self) -> bool {
        self.documents.read().dense_index_trained()
    }

    pub fn document_embedding_count(&self) -> usize {
        self.documents.read().embedded_count()
    }

    pub fn train_document_index(&self) -> Result<()> {
        let Some(space) = &self.config.embedding_space else {
            return Ok(());
        };
        let snapshot = {
            let _transaction = self.mutation_gate.read();
            documents::capture_dense_index(&self.documents.read(), space.dimensions)?
        };
        let Some(snapshot) = snapshot else {
            return Ok(());
        };
        let Some(trained) = documents::build_dense_index(snapshot)? else {
            return Ok(());
        };
        let published = {
            let _transaction = self.mutation_gate.read();
            documents::publish_dense_index(&mut self.documents.write(), &trained)
        };
        ensure!(
            published,
            "store changed during document index training; retry maintenance"
        );
        Ok(())
    }

    /// Whether a document with this id currently resolves in the opened store.
    pub fn has_document(&self, id: &str) -> bool {
        self.documents.read().contains_key(id)
    }

    pub fn delete_document(&self, id: &str) -> Result<bool> {
        validate_id(id)?;
        let exists = self.documents.read().contains_key(id);
        self.commit(&[Mutation::DeleteDocument { id: id.into() }])?;
        Ok(exists)
    }

    pub(super) fn remove_document_from_memory(&self, id: &str, shards: &ShardSet) -> Result<()> {
        if let Some(old) = self.documents.write().remove(id) {
            for chunk in &old.chunks {
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
        if let Some(stages) = &stages {
            stages.params.validate()?;
        }
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
        input: DocumentInput,
        stages: &crate::core::models::ModelStages,
    ) -> Result<u32> {
        let document = self.prepare_document_with_models(input, stages)?;
        let count = document.chunks.len() as u32;
        self.commit(&[Mutation::Document(Arc::new(document))])?;
        Ok(count)
    }

    fn prepare_document_with_models(
        &self,
        mut input: DocumentInput,
        stages: &crate::core::models::ModelStages,
    ) -> Result<documents::StoredDocument> {
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
        documents::prepare_with_keys(
            input,
            self.dimensions,
            self.config.embedding_space.as_ref().map(|s| s.dimensions),
            keys.as_deref(),
            self.config.text_encoder,
        )
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
        stages.params.validate()?;
        let k = options.k.unwrap_or(10) as usize;
        ensure!((1..=1000).contains(&k), "require 1 <= k <= 1000");
        let pool = k.max(stages.params.rerank_n) as u32;
        let variants = stages.query_variants(text, today)?;
        let embeddings = match (&stages.embedder, &options.embedding) {
            (Some(e), None) => {
                let queries: Vec<&str> = variants.iter().map(|v| v.0.as_str()).collect();
                Some(e.embed(&queries, true)?)
            }
            _ => None,
        };
        let mut prepared = Vec::with_capacity(variants.len());
        for (i, (query, _)) in variants.iter().enumerate() {
            let mut opts = options.clone();
            opts.k = Some(pool);
            opts.lexical_weight = opts.lexical_weight.or(Some(stages.params.lexical_weight));
            opts.semantic_weight = opts.semantic_weight.or(Some(stages.params.semantic_weight));
            if let Some(e) = &embeddings {
                opts.embedding = Some(e[i].iter().copied().map(f64::from).collect());
            }
            prepared.push(documents::prepare_search(
                query,
                &opts,
                self.config.embedding_space.as_ref().map(|s| s.dimensions),
            )?);
        }
        let captured = {
            let _transaction = self.mutation_gate.read();
            documents::capture_search(&self.documents.read(), prepared)
        };
        let lists = variants
            .iter()
            .enumerate()
            .map(|(i, (_, weight))| (*weight, documents::execute_search(&captured, i)))
            .collect();
        stages.fuse(text, lists, k)
    }
}
