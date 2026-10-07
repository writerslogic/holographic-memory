// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

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
}

const MAX_TOKENS: usize = 2048;
const BATCH_TOKENS: usize = 16384;

impl Embedder {
    pub fn load(source: &ModelSource, device: &Device) -> Result<Self> {
        let l = load(source, device, super::dtype_for(device, false))?;
        let mut tokenizer = l.tokenizer;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("tokenizer truncation: {e}"))?;
        tokenizer.with_padding(None);
        let dim = l.config.hidden_size;
        Ok(Self {
            model: Qwen3::new(&l.config, l.vb, false)?,
            tokenizer,
            device: device.clone(),
            dim,
            batch: super::default_encode_batch(device),
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dim
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
                    encode(&self.tokenizer, &prompts::embed_query(t), true)
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
