// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use candle_core::{DType, Device, D};

use super::qwen3::Qwen3;
use super::{encode, encoder_batches, last_hidden, load, prompts, ModelSource};

/// Qwen3-Embedding: queries carry the task instruction, documents none; last-token pooling and
/// L2 normalization, inputs truncated to 2048 tokens (the reference `max_seq_length`).
pub struct Embedder {
    model: Qwen3,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    dim: usize,
    batch: usize,
    query_task: EmbeddingQueryTask,
    spaces: [crate::core::EmbeddingSpace; 2],
}

/// Query instruction; document embeddings are identical for both tasks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbeddingQueryTask {
    #[default]
    UserMessages,
    Conversation,
}

const MAX_TOKENS: usize = 2048;
const BATCH_TOKENS: usize = 16384;
/// Default padded batch on Metal/CUDA: on an M4 (Metal), 96 user turns embedded in 3.5 s at
/// batch 4 against 4.3 s at 1, 3.7 s at 2, 4.0 s at 8 and 5.6 s at 16
/// (`benchmarks/results/local_models_batching.json`). CUDA is unmeasured.
const GPU_BATCH: usize = 4;

impl Embedder {
    pub fn load(source: &ModelSource, device: &Device) -> Result<Self> {
        let l = load(source, device, super::dtype_for(device, false), true)?;
        let mut tokenizer = l.tokenizer;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("tokenizer truncation: {e}"))?;
        tokenizer.with_padding(None);
        let dim = l.config.hidden_size;
        let space = |instruction: String| {
            crate::core::EmbeddingSpace::effective(
                l.model_name.clone(),
                dim,
                "l2",
                &(
                    "qwen3-last-token-v1",
                    &source.revision,
                    &l.identity,
                    MAX_TOKENS,
                    instruction,
                    format!("{:?}", super::dtype_for(device, false)),
                ),
            )
        };
        let spaces = [
            space(prompts::embed_query(""))?,
            space(prompts::conversation_embed_query(""))?,
        ];
        Ok(Self {
            spaces,
            model: Qwen3::new(&l.config, l.vb, false)?,
            tokenizer,
            device: device.clone(),
            dim,
            batch: super::default_encode_batch(device, GPU_BATCH),
            query_task: EmbeddingQueryTask::default(),
        })
    }

    pub fn set_query_task(&mut self, task: EmbeddingQueryTask) {
        self.query_task = task;
    }

    pub fn query_task(&self) -> EmbeddingQueryTask {
        self.query_task
    }

    pub fn dimensions(&self) -> usize {
        self.dim
    }

    pub fn space(&self) -> crate::core::EmbeddingSpace {
        self.spaces[match self.query_task {
            EmbeddingQueryTask::UserMessages => 0,
            EmbeddingQueryTask::Conversation => 1,
        }]
        .clone()
    }

    /// Sets the most sequences per forward pass (at least 1).
    pub fn set_batch(&mut self, batch: usize) {
        self.batch = batch.max(1);
    }

    /// L2-normalized embeddings, one per text, in input order.
    pub fn embed<S: AsRef<str>>(&self, texts: &[S], is_query: bool) -> Result<Vec<Vec<f32>>> {
        let ids: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| {
                let t = t.as_ref();
                if is_query {
                    let query = match self.query_task {
                        EmbeddingQueryTask::UserMessages => prompts::embed_query(t),
                        EmbeddingQueryTask::Conversation => prompts::conversation_embed_query(t),
                    };
                    encode(&self.tokenizer, &query, true)
                } else {
                    encode(&self.tokenizer, t, true)
                }
            })
            .collect::<Result<_>>()?;
        ensure!(ids.iter().all(|x| !x.is_empty()), "empty token sequence");
        let lens: Vec<usize> = ids.iter().map(Vec::len).collect();
        let mut out = vec![Vec::new(); texts.len()];
        for batch in encoder_batches(&self.device, &lens, BATCH_TOKENS, self.batch) {
            let last = last_hidden(&self.model, &ids, &batch)?.to_dtype(DType::F32)?;
            let norm = last
                .sqr()?
                .sum_keepdim(D::Minus1)?
                .sqrt()?
                .clamp(1e-12, f64::MAX)?;
            let rows = last.broadcast_div(&norm)?.to_vec2::<f32>()?;
            for (&i, row) in batch.iter().zip(rows) {
                out[i] = row;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn tiny(seed: u64) -> Embedder {
        let space =
            crate::core::EmbeddingSpace::effective("tiny-qwen".into(), 32, "l2", &seed).unwrap();
        let conversation = crate::core::EmbeddingSpace::effective(
            "tiny-qwen".into(),
            32,
            "l2",
            &(seed, "conversation"),
        )
        .unwrap();
        Embedder {
            model: super::super::qwen3::testing::tiny(seed, true),
            tokenizer: tokenizers::Tokenizer::new(
                tokenizers::models::wordlevel::WordLevel::default(),
            ),
            device: Device::Cpu,
            dim: 32,
            batch: 1,
            query_task: EmbeddingQueryTask::UserMessages,
            spaces: [space, conversation],
        }
    }

    #[test]
    fn effective_model_installation_rejects_same_dimension_different_identity() {
        let dir = tempfile::tempdir().unwrap();
        let original = tiny(1);
        let config = crate::core::HmsConfig {
            embedding_space: Some(original.space()),
            ..Default::default()
        };
        let core = crate::HmsCore::new(1024, Some(dir.path().display().to_string()), Some(config))
            .unwrap();
        let original = Arc::new(crate::core::models::ModelStages {
            embedder: Some(original),
            ..Default::default()
        });
        core.set_model_stages(Some(original.clone())).unwrap();
        let incompatible = Arc::new(crate::core::models::ModelStages {
            embedder: Some(tiny(2)),
            ..Default::default()
        });
        assert!(core.set_model_stages(Some(incompatible)).is_err());
        assert!(Arc::ptr_eq(&core.model_stages().unwrap(), &original));
        let mut task_changed = tiny(1);
        task_changed.set_query_task(EmbeddingQueryTask::Conversation);
        assert!(core
            .set_model_stages(Some(Arc::new(crate::core::models::ModelStages {
                embedder: Some(task_changed),
                ..Default::default()
            })))
            .is_err());
    }
}
