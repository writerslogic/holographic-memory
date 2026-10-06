// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_transformers::models::qwen3::ModelForCausalLM;

use super::{encode, equal_length_batches, load, prompts, token_id, ModelSource};

/// Qwen3-Reranker: p(yes) from a softmax over the `no`/`yes` logits after the model-card
/// prompt; the query/document body is cut to `4096 - prefix - suffix` tokens.
pub struct Reranker {
    model: ModelForCausalLM,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    yes: u32,
    no: u32,
}

const MAX_TOKENS: usize = 4096;
const BATCH_TOKENS: usize = 12288;

impl Reranker {
    pub fn load(source: &ModelSource, device: &Device) -> Result<Self> {
        let l = load(source, device, super::dtype_for(device, false))?;
        let mut tokenizer = l.tokenizer;
        tokenizer.with_padding(None);
        let _ = tokenizer.with_truncation(None);
        let prefix = encode(&tokenizer, prompts::RERANK_PREFIX, false)?;
        let suffix = encode(&tokenizer, prompts::RERANK_SUFFIX, false)?;
        Ok(Self {
            yes: token_id(&tokenizer, "yes")?,
            no: token_id(&tokenizer, "no")?,
            model: ModelForCausalLM::new(&l.config, l.vb)?,
            tokenizer,
            device: device.clone(),
            prefix,
            suffix,
        })
    }

    /// Relevance in [0, 1] for each (query, document) pair, in input order.
    pub fn score<Q: AsRef<str>, T: AsRef<str>>(&self, pairs: &[(Q, T)]) -> Result<Vec<f32>> {
        let cap = MAX_TOKENS - self.prefix.len() - self.suffix.len();
        let ids: Vec<Vec<u32>> = pairs
            .iter()
            .map(|(q, d)| {
                let mut body = encode(
                    &self.tokenizer,
                    &prompts::rerank_body(q.as_ref(), d.as_ref()),
                    true,
                )?;
                body.truncate(cap);
                let mut x = self.prefix.clone();
                x.extend(body);
                x.extend(&self.suffix);
                Ok(x)
            })
            .collect::<Result<_>>()?;
        let lens: Vec<usize> = ids.iter().map(Vec::len).collect();
        let mut out = vec![0.0; pairs.len()];
        for batch in equal_length_batches(&lens, BATCH_TOKENS, super::max_batch(&self.device)) {
            let len = lens[batch[0]];
            let flat: Vec<u32> = batch.iter().flat_map(|&i| ids[i].iter().copied()).collect();
            let input = Tensor::from_vec(flat, (batch.len(), len), &self.device)?;
            let logits = self.model.clone().forward(&input, 0)?.squeeze(1)?;
            let no = logits.narrow(D::Minus1, self.no as usize, 1)?;
            let yes = logits.narrow(D::Minus1, self.yes as usize, 1)?;
            let pair = Tensor::cat(&[no, yes], 1)?.to_dtype(DType::F32)?;
            let p = candle_nn::ops::log_softmax(&pair, 1)?
                .narrow(1, 1, 1)?
                .exp()?;
            for (&i, v) in batch.iter().zip(p.flatten_all()?.to_vec1::<f32>()?) {
                out[i] = v;
            }
        }
        Ok(out)
    }
}
