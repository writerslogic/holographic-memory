// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{ensure, Result};
use candle_core::{Device, Tensor};

use super::qwen3::{argmax_rows, Cache, Positions, Qwen3};
use super::{encode, load, prompts, token_id, ModelSource, QueryPlan};

/// Greedy (temperature 0) generation with a Qwen3 instruct model, the reference's sampling:
/// at most 1536 new tokens, a 12288-token context, stop at `<|im_end|>` or the model's EOS.
///
/// [`Generator::generate`] decodes up to [`Generator::batch`] sequences together (continuous
/// batching): each prompt is prefilled alone, its cache joins the running batch left-padded to
/// the batch length, finished rows leave and queued prompts take their place.
pub struct Generator {
    model: Qwen3,
    tokenizer: tokenizers::Tokenizer,
    stop: Vec<u32>,
    batch: usize,
}

pub const MAX_NEW_TOKENS: usize = 1536;
pub const MAX_CONTEXT: usize = 12288;
/// Soft cap on the padded decode batch (rows x cached length) a new row may join: 32768
/// tokens is 4.8 GB of bf16 KV for Qwen3-4B.
const DECODE_TOKENS: usize = 32768;

/// One reply and the number of tokens generated for it (the stop token excluded).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Completion {
    pub text: String,
    pub tokens: usize,
}

/// A sequence in the running batch.
struct Row {
    /// Input index.
    index: usize,
    /// Leading cache positions that are padding.
    pad: usize,
    /// RoPE position of the token fed next.
    pos: usize,
    out: Vec<u32>,
    budget: usize,
}

/// Default decode batch: one on the CPU (its fused attention has no padding mask), 8 on a GPU.
pub fn default_decode_batch(device: &Device) -> usize {
    if device.is_cpu() {
        1
    } else {
        8
    }
}

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
            model: Qwen3::new(&l.config, l.vb, true)?,
            tokenizer,
            stop,
            batch: default_decode_batch(device),
        })
    }

    /// Sequences decoded together by [`Generator::generate`].
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Sets the decode batch (at least 1). Batches above one use the padded attention path,
    /// which on the CPU replaces the fused kernel with the standard one.
    pub fn set_batch(&mut self, batch: usize) {
        self.batch = batch.max(1);
    }

    /// The model's reply to one user message, special tokens removed.
    pub fn complete(&self, user: &str) -> Result<String> {
        Ok(self.generate(&[user])?.remove(0).text)
    }

    /// Replies to several user messages, in input order.
    pub fn complete_batch<S: AsRef<str>>(&self, users: &[S]) -> Result<Vec<String>> {
        Ok(self.generate(users)?.into_iter().map(|c| c.text).collect())
    }

    /// Replies and generated-token counts, in input order.
    pub fn generate<S: AsRef<str>>(&self, users: &[S]) -> Result<Vec<Completion>> {
        let prompts: Vec<Vec<u32>> = users
            .iter()
            .map(|u| encode(&self.tokenizer, &prompts::chat(u.as_ref()), false))
            .collect::<Result<_>>()?;
        for p in &prompts {
            ensure!(
                !p.is_empty() && p.len() < MAX_CONTEXT,
                "prompt of {} tokens exceeds the {MAX_CONTEXT}-token context",
                p.len()
            );
        }
        decode(
            &self.model,
            &self.stop,
            self.batch,
            MAX_NEW_TOKENS,
            &prompts,
        )?
        .into_iter()
        .map(|out| {
            Ok(Completion {
                tokens: out.len(),
                text: self
                    .tokenizer
                    .decode(&out, true)
                    .map_err(|e| anyhow::anyhow!("decoding: {e}"))?,
            })
        })
        .collect()
    }
}

/// Greedy token ids per prompt (at most `max_new`), at most `batch` rows decoded together. A
/// batch of one row without padding runs candle's single-sequence path, so `batch == 1`
/// reproduces the unbatched decoder exactly.
fn decode(
    model: &Qwen3,
    stop: &[u32],
    batch: usize,
    max_new: usize,
    prompts: &[Vec<u32>],
) -> Result<Vec<Vec<u32>>> {
    let device = model.device().clone();
    let mut done: Vec<Option<Vec<u32>>> = vec![None; prompts.len()];
    let mut rows: Vec<Row> = Vec::new();
    let mut cache = Cache::default();
    let mut next = 0;
    loop {
        // Admit queued prompts: prefill alone, then join the batch left-padded.
        while rows.len() < batch && next < prompts.len() {
            let t = cache.seq_len();
            let len = prompts[next].len();
            if !rows.is_empty() && (rows.len() + 1) * t.max(len) > DECODE_TOKENS {
                break;
            }
            let index = next;
            next += 1;
            let budget = max_new.min(MAX_CONTEXT - len);
            if budget == 0 {
                done[index] = Some(Vec::new());
                continue;
            }
            let mut row_cache = Cache::default();
            let input = Tensor::new(prompts[index].as_slice(), &device)?.unsqueeze(0)?;
            let hidden = model.forward(&input, Positions::Uniform(0), &mut row_cache)?;
            let first = argmax_rows(&model.logits(&hidden.narrow(1, len - 1, 1)?)?)?[0];
            if stop.contains(&first) {
                done[index] = Some(Vec::new());
                continue;
            }
            if budget == 1 {
                done[index] = Some(vec![first]);
                continue;
            }
            let mut pad = 0;
            if rows.is_empty() {
                cache = row_cache;
            } else {
                if len <= t {
                    pad = t - len;
                    row_cache = row_cache.left_pad(pad)?;
                } else {
                    cache = cache.left_pad(len - t)?;
                    rows.iter_mut().for_each(|r| r.pad += len - t);
                }
                cache = Cache::cat(&[cache, row_cache])?;
            }
            rows.push(Row {
                index,
                pad,
                pos: len,
                out: vec![first],
                budget,
            });
        }
        if rows.is_empty() {
            break;
        }

        // One decode step for every row.
        let b = rows.len();
        let t = cache.seq_len();
        let last: Vec<u32> = rows.iter().map(|r| r.out[r.out.len() - 1]).collect();
        let input = Tensor::from_vec(last, (b, 1), &device)?;
        let hidden = if b == 1 && rows[0].pad == 0 {
            model.forward(&input, Positions::Uniform(t), &mut cache)?
        } else {
            let pos: Vec<u32> = rows.iter().map(|r| r.pos as u32).collect();
            let positions = Tensor::from_vec(pos, (b, 1), &device)?;
            let keep: Vec<bool> = rows
                .iter()
                .flat_map(|r| (0..=t).map(move |j| j >= r.pad))
                .collect();
            let mask = model.additive_mask(&keep, &[b, 1, 1, t + 1])?;
            let pos = Positions::PerRow {
                positions: &positions,
                mask: &mask,
            };
            model.forward(&input, pos, &mut cache)?
        };
        let toks = argmax_rows(&model.logits(&hidden)?)?;

        // Retire finished rows; drop their cache rows and padding no remaining row needs.
        let mut kept = Vec::with_capacity(b);
        let mut live = Vec::with_capacity(b);
        for (r, (mut row, tok)) in rows.drain(..).zip(toks).enumerate() {
            row.pos += 1;
            let finished = stop.contains(&tok) || {
                row.out.push(tok);
                row.out.len() == row.budget
            };
            if finished {
                done[row.index] = Some(row.out);
            } else {
                kept.push(r as u32);
                live.push(row);
            }
        }
        rows = live;
        if kept.len() == b {
            continue;
        }
        if rows.is_empty() {
            cache = Cache::default();
            continue;
        }
        let idx = Tensor::new(kept.as_slice(), &device)?;
        cache = cache.map(|x| x.index_select(&idx, 0))?;
        let trim = rows.iter().map(|r| r.pad).min().unwrap_or(0);
        if trim > 0 {
            let t = cache.seq_len();
            cache = cache.map(|x| x.narrow(2, trim, t - trim)?.contiguous())?;
            rows.iter_mut().for_each(|r| r.pad -= trim);
        }
    }
    done.into_iter()
        .map(|d| d.ok_or_else(|| anyhow::anyhow!("a sequence was not decoded")))
        .collect()
}

/// Personal facts from one conversation's user turns (the tuned fact prompt).
pub struct FactExtractor<'a>(pub &'a Generator);

impl FactExtractor<'_> {
    /// The raw model output and the parsed (turn index, fact) pairs.
    pub fn extract<S: AsRef<str>>(&self, user_turns: &[S]) -> Result<(String, prompts::Facts)> {
        Ok(self.extract_batch(&[user_turns])?.remove(0))
    }

    /// [`FactExtractor::extract`] for several conversations, decoded as one batch.
    pub fn extract_batch<S: AsRef<str>, T: AsRef<[S]>>(
        &self,
        sessions: &[T],
    ) -> Result<Vec<(String, prompts::Facts)>> {
        let prompts: Vec<String> = sessions
            .iter()
            .map(|s| prompts::fact_prompt(s.as_ref()))
            .collect();
        let raws = self.0.complete_batch(&prompts)?;
        Ok(raws
            .into_iter()
            .zip(sessions)
            .map(|(raw, s)| {
                let n = s.as_ref().len();
                let facts = prompts::parse_facts(prompts::parse_json(&raw).as_ref(), n);
                (raw, facts)
            })
            .collect())
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

#[cfg(test)]
mod tests {
    use super::super::qwen3::testing::tiny;
    use super::*;

    #[test]
    fn batched_decode_matches_one_at_a_time() {
        let prompts: Vec<Vec<u32>> = vec![
            vec![1, 2, 3, 4, 5, 6, 7],
            vec![9, 8],
            vec![10, 11, 12, 13],
            vec![20],
            vec![30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        ];
        for flash in [true, false] {
            let model = tiny(7, flash);
            // Stop on a token the model emits, so rows finish at different steps.
            let probe = decode(&model, &[], 1, 24, &prompts[..1]).unwrap();
            let stop = [probe[0][5]];
            let one = decode(&model, &stop, 1, 24, &prompts).unwrap();
            let lens: Vec<usize> = one.iter().map(Vec::len).collect();
            assert!(
                lens.iter().any(|&n| n < 24) && lens.iter().any(|&n| n > 3),
                "{lens:?}"
            );
            for batch in [2, 3, 8] {
                let got = decode(&model, &stop, batch, 24, &prompts).unwrap();
                assert_eq!(got, one, "batch {batch}, cpu flash {flash}");
            }
        }
    }
}
