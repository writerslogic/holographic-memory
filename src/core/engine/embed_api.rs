// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The local embedder in the engine (feature `local-embed`): with an embedder installed,
//! `memorize_document(s)` embeds chunks that arrive without embeddings and `search_documents`
//! embeds text queries that arrive without one. With `TextEncoder::LocalEmbedV1` the sparse
//! vectors of `encode_text` and of stored chunks are sketches of the same embeddings.

use anyhow::{ensure, Result};
use std::sync::Arc;

use super::WarmCore;
use crate::core::config::WarmConfig;
use crate::core::documents::{self, DocumentInput, SearchOptions};
use crate::core::embed::Embedder;
use crate::core::encoding::TextEncoder;
use crate::core::entangled::EntangledHVec;

/// Builds the embedder `TextEncoder::LocalEmbedV1` needs and fills in the store embedding space.
pub(super) fn prepare(mut config: WarmConfig) -> Result<(WarmConfig, Option<Arc<Embedder>>)> {
    if config.text_encoder != TextEncoder::LocalEmbedV1 {
        return Ok((config, None));
    }
    let embedder = Arc::new(Embedder::new(
        &config.local_embed.clone().unwrap_or_default(),
    )?);
    match &config.embedding_space {
        None => config.embedding_space = Some(embedder.space()),
        Some(space) => check_space(space, &embedder)?,
    }
    Ok((config, Some(embedder)))
}

fn check_space(space: &crate::core::EmbeddingSpace, embedder: &Embedder) -> Result<()> {
    ensure!(
        *space == embedder.space(),
        "the store embedding space ({} {} {}d) does not match the embedder ({} {} {}d)",
        space.model,
        space.revision,
        space.dimensions,
        embedder.space().model,
        embedder.space().revision,
        embedder.dimensions()
    );
    Ok(())
}

impl WarmCore {
    /// Installs (or with `None` removes) the embedder used by the document API. The store's
    /// embedding space must be the embedder's own (`Embedder::space`).
    pub fn set_text_embedder(&self, embedder: Option<Arc<Embedder>>) -> Result<()> {
        match &embedder {
            Some(e) => match &self.config.embedding_space {
                Some(space) => check_space(space, e)?,
                None => anyhow::bail!(
                    "configure the store embedding space with Embedder::space() before installing an embedder"
                ),
            },
            None => ensure!(
                self.config.text_encoder != TextEncoder::LocalEmbedV1,
                "the local-embed-v1 text encoder cannot run without its embedder"
            ),
        }
        *self.text_embedder.write() = embedder;
        Ok(())
    }

    pub fn text_embedder(&self) -> Option<Arc<Embedder>> {
        self.text_embedder.read().clone()
    }

    pub(super) fn encode_texts_with_embedder(&self, texts: &[&str]) -> Result<Vec<EntangledHVec>> {
        let embedder = self
            .text_embedder()
            .ok_or_else(|| anyhow::anyhow!("local text encoder has no installed embedder"))?;
        embedder.validate_inputs(texts, crate::core::embed::EmbedKind::Passage)?;
        let nonempty: Vec<_> = texts
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, text)| !text.trim().is_empty())
            .collect();
        let inputs: Vec<_> = nonempty.iter().map(|(_, text)| *text).collect();
        let embedded = if inputs.is_empty() {
            Vec::new()
        } else {
            embedder.embed_passages(&inputs)?
        };
        let mut vectors =
            vec![EntangledHVec::from_indices(Vec::new(), self.dimensions); texts.len()];
        for ((index, _), dense) in nonempty.into_iter().zip(embedded) {
            vectors[index] = EntangledHVec::from_dense(&dense, self.dimensions);
        }
        Ok(vectors)
    }

    /// Fills `embeddings` of every input that has none, embedding all their chunks in one pass.
    pub(super) fn embed_document_inputs(&self, inputs: &mut [DocumentInput]) -> Result<()> {
        let Some(embedder) = self.text_embedder() else {
            return Ok(());
        };
        #[cfg(feature = "local-models")]
        if self.model_stages().is_some_and(|s| s.embedder.is_some()) {
            return Ok(());
        }
        let mut texts: Vec<String> = Vec::new();
        let mut spans = Vec::with_capacity(inputs.len());
        for input in inputs.iter() {
            if input.embeddings.is_some() {
                spans.push(None);
                continue;
            }
            let start = texts.len();
            texts.extend(
                documents::chunk_document(input)?
                    .into_iter()
                    .map(|c| c.text),
            );
            spans.push(Some(start..texts.len()));
        }
        if texts.is_empty() {
            return Ok(());
        }
        let vectors = embedder.embed_passages(&texts)?;
        for (input, span) in inputs.iter_mut().zip(spans) {
            if let Some(span) = span {
                input.embeddings = Some(
                    vectors[span]
                        .iter()
                        .map(|v| v.iter().copied().map(f64::from).collect())
                        .collect(),
                );
            }
        }
        Ok(())
    }

    /// The query embedding for a text search that carries none, when an embedder is installed.
    pub(super) fn embed_search_text(
        &self,
        text: &str,
        options: &SearchOptions,
    ) -> Result<Option<Vec<f64>>> {
        if options.embedding.is_some() || text.trim().is_empty() {
            return Ok(None);
        }
        let Some(embedder) = self.text_embedder() else {
            return Ok(None);
        };
        #[cfg(feature = "local-models")]
        if self.model_stages().is_some_and(|s| s.embedder.is_some()) {
            return Ok(None);
        }
        let v = embedder.embed_query(&[text])?;
        Ok(Some(v[0].iter().copied().map(f64::from).collect()))
    }
}
