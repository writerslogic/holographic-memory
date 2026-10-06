// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_transformers::models::qwen3::ModelForCausalLM;

use super::{encode, load, prompts, token_id, ModelSource, QueryPlan};

/// Greedy (temperature 0) generation with a Qwen3 instruct model, the reference's sampling:
/// at most 1536 new tokens, a 12288-token context, stop at `<|im_end|>` or the model's EOS.
pub struct Generator {
    model: ModelForCausalLM,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    stop: Vec<u32>,
}

pub const MAX_NEW_TOKENS: usize = 1536;
pub const MAX_CONTEXT: usize = 12288;

impl Generator {
    pub fn load(source: &ModelSource, device: &Device) -> Result<Self> {
        let mut l = load(source, device, super::dtype_for(device, true))?;
        // The rotary table only needs the positions a generation can reach (the 2507 models
        // declare 262144). Uncapped and in f32, Metal produced NaN logits; capped and in bf16 it
        // decodes correctly. Which of the two changes was required was not isolated.
        l.config.max_position_embeddings = l.config.max_position_embeddings.min(MAX_CONTEXT);
        let mut tokenizer = l.tokenizer;
        tokenizer.with_padding(None);
        let _ = tokenizer.with_truncation(None);
        let mut stop = vec![token_id(&tokenizer, "<|im_end|>")?];
        if let Ok(gen) = std::fs::read(source.dir.join("generation_config.json")) {
            let v: serde_json::Value = serde_json::from_slice(&gen)?;
            match v.get("eos_token_id") {
                Some(serde_json::Value::Array(a)) => {
                    stop.extend(a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32))
                }
                Some(x) => stop.extend(x.as_u64().map(|x| x as u32)),
                None => {}
            }
        }
        Ok(Self {
            model: ModelForCausalLM::new(&l.config, l.vb)?,
            tokenizer,
            device: device.clone(),
            stop,
        })
    }

    /// The model's reply to one user message, special tokens removed.
    pub fn complete(&self, user: &str) -> Result<String> {
        let prompt = encode(&self.tokenizer, &prompts::chat(user), false)?;
        ensure!(
            !prompt.is_empty() && prompt.len() < MAX_CONTEXT,
            "prompt of {} tokens exceeds the {MAX_CONTEXT}-token context",
            prompt.len()
        );
        let budget = MAX_NEW_TOKENS.min(MAX_CONTEXT - prompt.len());
        let mut model = self.model.clone();
        let mut input = Tensor::new(prompt.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut offset = 0;
        let mut out = Vec::new();
        while out.len() < budget {
            let logits = model.forward(&input, offset)?;
            offset += input.dim(1)?;
            let next = logits
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .argmax(D::Minus1)?
                .to_scalar::<u32>()?;
            if self.stop.contains(&next) {
                break;
            }
            out.push(next);
            input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
        }
        self.tokenizer
            .decode(&out, true)
            .map_err(|e| anyhow::anyhow!("decoding: {e}"))
    }
}

/// Personal facts from one conversation's user turns (the tuned fact prompt).
pub struct FactExtractor<'a>(pub &'a Generator);

impl FactExtractor<'_> {
    /// The raw model output and the parsed (turn index, fact) pairs.
    pub fn extract<S: AsRef<str>>(&self, user_turns: &[S]) -> Result<(String, prompts::Facts)> {
        let raw = self.0.complete(&prompts::fact_prompt(user_turns))?;
        let facts = prompts::parse_facts(prompts::parse_json(&raw).as_ref(), user_turns.len());
        Ok((raw, facts))
    }
}

/// Search rewrite, sub-queries and time range for one question (the tuned query prompt).
pub struct QueryRewriter<'a>(pub &'a Generator);

impl QueryRewriter<'_> {
    /// `today` is the question date as the user would state it.
    pub fn rewrite(&self, today: &str, question: &str) -> Result<(String, QueryPlan)> {
        let raw = self.0.complete(&prompts::query_prompt(today, question))?;
        let plan = prompts::parse_query_plan(prompts::parse_json(&raw).as_ref());
        Ok((raw, plan))
    }
}
