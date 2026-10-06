// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_transformers::models::qwen3::Model;

use super::{encode, equal_length_batches, load, prompts, ModelSource};

/// Qwen3-Embedding: queries carry the task instruction, documents none; last-token pooling and
/// L2 normalization, inputs truncated to 2048 tokens (the reference `max_seq_length`).
pub struct Embedder {
    model: Model,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    dim: usize,
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
            model: Model::new(&l.config, l.vb)?,
            tokenizer,
            device: device.clone(),
            dim,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dim
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
        for batch in equal_length_batches(&lens, BATCH_TOKENS, super::max_batch(&self.device)) {
            let len = lens[batch[0]];
            let flat: Vec<u32> = batch.iter().flat_map(|&i| ids[i].iter().copied()).collect();
            let input = Tensor::from_vec(flat, (batch.len(), len), &self.device)?;
            // A clone shares the weights and starts with an empty KV cache.
            let hidden = self.model.clone().forward(&input, 0)?;
            let last = hidden
                .narrow(1, len - 1, 1)?
                .squeeze(1)?
                .to_dtype(DType::F32)?;
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
