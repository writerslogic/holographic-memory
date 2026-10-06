// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! On-device model stages (feature `local-models`): Qwen3 embedding, Qwen3 re-ranking and a
//! Qwen3 instruct LLM for fact extraction and query rewriting, run with candle on the CPU, Metal
//! (macOS) or CUDA (`--cfg hms_cuda` builds only). Weights load from local directories; this
//! module never downloads anything and makes no network calls.

mod embed;
mod llm;
pub mod prompts;
mod rerank;
mod stages;

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub use embed::Embedder;
pub use llm::{FactExtractor, Generator, QueryRewriter};
pub use prompts::QueryPlan;
pub use rerank::Reranker;
pub use stages::{ModelStages, StageParams};

/// Revisions the LongMemEval pipeline was tuned with (`benchmarks/public/longmemeval_modal.py`).
pub const PINNED: &[(&str, &str)] = &[
    (
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    ),
    (
        "Qwen/Qwen3-Reranker-0.6B",
        "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    ),
    (
        "Qwen/Qwen3-4B-Instruct-2507",
        "cdbee75f17c01a7cc42f958dc650907174af0554",
    ),
    (
        "Qwen/Qwen3-Embedding-8B",
        "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    ),
    (
        "Qwen/Qwen3-Reranker-8B",
        "77d193c791ed757ca307ee72715aa132723da912",
    ),
    (
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
    ),
];

/// A model directory and the revision its files must come from.
#[derive(Clone, Debug)]
pub struct ModelSource {
    pub dir: PathBuf,
    /// Expected commit sha; checked against the directory's recorded revision.
    pub revision: String,
}

impl ModelSource {
    pub fn new(dir: impl Into<PathBuf>, revision: impl Into<String>) -> Self {
        Self {
            dir: dir.into(),
            revision: revision.into(),
        }
    }
}

/// The revision a local model directory records: `REVISION` (one line), or the commit line
/// `hf download --revision <sha> --local-dir <dir>` writes to
/// `.cache/huggingface/download/config.json.metadata`.
pub fn recorded_revision(dir: &Path) -> Result<String> {
    let explicit = dir.join("REVISION");
    let hub = dir.join(".cache/huggingface/download/config.json.metadata");
    for path in [explicit, hub] {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Some(line) = text.lines().next().map(str::trim).filter(|l| !l.is_empty()) {
                return Ok(line.to_string());
            }
        }
    }
    bail!(
        "{} records no revision (expected REVISION or the hf download metadata)",
        dir.display()
    )
}

pub fn check_revision(source: &ModelSource) -> Result<()> {
    let found = recorded_revision(&source.dir)?;
    ensure!(
        found == source.revision,
        "{} is revision {found}, expected {}",
        source.dir.display(),
        source.revision
    );
    Ok(())
}

/// CUDA when compiled in and present, else Metal when present, else CPU.
/// `HMS_MODEL_DEVICE=cpu` forces the CPU.
pub fn default_device() -> Result<Device> {
    if std::env::var("HMS_MODEL_DEVICE").is_ok_and(|v| v == "cpu") {
        return Ok(Device::Cpu);
    }
    if candle_core::utils::cuda_is_available() {
        return Ok(Device::new_cuda(0)?);
    }
    if candle_core::utils::metal_is_available() {
        return Ok(Device::new_metal(0)?);
    }
    Ok(Device::Cpu)
}

/// f32 on the CPU; on a GPU the dtype the reference ran with (`half` for the
/// encoder stages on T4, bf16 for the LLM; the encoders stay f32 on Metal).
fn dtype_for(device: &Device, llm: bool) -> DType {
    match device {
        Device::Cuda(_) | Device::Metal(_) if llm => DType::BF16,
        Device::Cuda(_) => DType::F16,
        _ => DType::F32,
    }
}

struct Loaded {
    config: candle_transformers::models::qwen3::Config,
    tokenizer: tokenizers::Tokenizer,
    vb: VarBuilder<'static>,
}

fn load(source: &ModelSource, device: &Device, dtype: DType) -> Result<Loaded> {
    check_revision(source)?;
    let dir = &source.dir;
    let config = serde_json::from_slice(
        &std::fs::read(dir.join("config.json")).context("reading config.json")?,
    )
    .context("parsing config.json")?;
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    ensure!(!files.is_empty(), "{} has no safetensors", dir.display());
    // SAFETY: the weights are memory-mapped read-only; the caller owns the model directory and
    // must not modify the files while the model is loaded.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device)? };
    // Qwen3-Embedding checkpoints are saved without the `model.` prefix of the causal LM.
    let vb = if vb.contains_tensor("model.embed_tokens.weight") {
        vb
    } else {
        vb.rename_f(|name: &str| name.strip_prefix("model.").unwrap_or(name).to_string())
    };
    Ok(Loaded {
        config,
        tokenizer,
        vb,
    })
}

fn token_id(tokenizer: &tokenizers::Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .with_context(|| format!("tokenizer has no {token:?} token"))
}

fn encode(tokenizer: &tokenizers::Tokenizer, text: &str, special: bool) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, special)
        .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
        .get_ids()
        .to_vec())
}

/// Sequences per forward pass. candle 0.11's Qwen3 builds its explicit causal mask (used on
/// Metal and CUDA) for a batch of one, so on those devices a batch of two returns a wrong
/// second row (measured on Metal: NaN or cosine 0.17 to the reference). The CPU path masks
/// inside its fused kernel and batches correctly.
fn max_batch(device: &Device) -> usize {
    if device.is_cpu() {
        64
    } else {
        1
    }
}

/// Indices grouped by token length (no padding mask exists in candle's Qwen3, so a batch only
/// holds sequences of one length), each group split so a batch stays within `budget` tokens.
fn equal_length_batches(lens: &[usize], budget: usize, max_batch: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lens.len()).collect();
    order.sort_by_key(|&i| (lens[i], i));
    let mut out: Vec<Vec<usize>> = Vec::new();
    for i in order {
        match out.last_mut() {
            Some(b)
                if lens[b[0]] == lens[i]
                    && b.len() < max_batch
                    && (b.len() + 1) * lens[i] <= budget =>
            {
                b.push(i)
            }
            _ => out.push(vec![i]),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revision_is_read_and_checked() {
        let dir = tempfile::tempdir().unwrap();
        assert!(recorded_revision(dir.path()).is_err());
        let meta = dir.path().join(".cache/huggingface/download");
        std::fs::create_dir_all(&meta).unwrap();
        std::fs::write(meta.join("config.json.metadata"), "abc\netag\n1.0\n").unwrap();
        assert!(check_revision(&ModelSource::new(dir.path(), "abc")).is_ok());
        let err = check_revision(&ModelSource::new(dir.path(), "def")).unwrap_err();
        assert!(err.to_string().contains("expected def"));
        std::fs::write(dir.path().join("REVISION"), "def\n").unwrap();
        assert!(check_revision(&ModelSource::new(dir.path(), "def")).is_ok());
    }

    #[test]
    fn batches_hold_one_length_within_budget() {
        let lens = [3, 5, 3, 3, 5, 9];
        let b = equal_length_batches(&lens, 6, 64);
        assert_eq!(b, vec![vec![0, 2], vec![3], vec![1], vec![4], vec![5]]);
        let all: usize = b.iter().map(Vec::len).sum();
        assert_eq!(all, lens.len());
        assert_eq!(equal_length_batches(&lens, 100, 2)[0], vec![0, 2]);
    }
}
