// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The model pack: one self-describing file holding a BERT-family encoder (quantized weights,
//! tokenizer, architecture, pooling and prompt prefixes). A pack is verified by the sha256 of its
//! whole content, which the registry pins; the format itself is canonical (fixed field and
//! tensor order, contiguous data), so converting the same upstream files always produces the
//! same bytes.
//!
//! Layout: `HMSEMB01` | u64 LE header length | header JSON | tokenizer.json bytes | tensor data
//! in header order, each tensor immediately after the previous one.

use std::io::Read;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::kernels::{f16_to_f32, f32_to_f16, Linear, Weights};

type NamedTensors = std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>;
pub(crate) const MAGIC: &[u8; 8] = b"HMSEMB01";
const MAX_HEADER: u64 = 1 << 20;
const MAX_TOKENIZER: u64 = 64 << 20;
const MAX_TENSOR_BYTES: u64 = 2 << 30;

/// How a pack stores its weight matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    F32,
    F16,
    /// Symmetric int8 with one scale per output row.
    Int8,
}

impl Quantization {
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Int8 => "int8",
        }
    }
}

impl std::str::FromStr for Quantization {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "int8" => Ok(Self::Int8),
            _ => bail!("unknown quantization {s:?} (expected f32, f16 or int8)"),
        }
    }
}

/// How token states become one vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pooling {
    /// Mean over all tokens including the special tokens (sentence-transformers `mean_tokens`).
    Mean,
    /// The first token's state.
    Cls,
}

impl std::str::FromStr for Pooling {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "mean" => Ok(Self::Mean),
            "cls" => Ok(Self::Cls),
            _ => bail!("unknown pooling {s:?} (expected mean or cls)"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Arch {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub max_positions: usize,
    pub type_vocab: usize,
    pub layer_norm_eps: f32,
    /// First position id (0 for BERT, `pad_token_id + 1` for RoBERTa and XLM-R).
    pub position_offset: usize,
}

impl Arch {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            (1..=8192).contains(&self.hidden)
                && (1..=128).contains(&self.layers)
                && (1..=128).contains(&self.heads)
                && self.hidden.is_multiple_of(self.heads)
                && (1..=65536).contains(&self.intermediate)
                && (1..=2_000_000).contains(&self.vocab)
                && (1..=65536).contains(&self.max_positions)
                && (1..=16).contains(&self.type_vocab)
                && self.position_offset < self.max_positions
                && self.layer_norm_eps.is_finite()
                && self.layer_norm_eps > 0.0,
            "implausible model architecture {self:?}"
        );
        Ok(())
    }
}

/// Provenance and prompt conventions stored with the weights.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    /// Upstream repository, e.g. `BAAI/bge-small-en-v1.5`.
    pub model: String,
    /// Upstream commit the weights were converted from.
    pub revision: String,
    pub license: String,
    pub pooling: Pooling,
    pub normalize: bool,
    pub query_prefix: String,
    pub passage_prefix: String,
    /// Longest sequence (tokens, specials included) the model should be run at.
    pub max_length: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TensorEntry {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    len: u64,
    /// int8 tensors: values per scale (divides the row length); 0 otherwise.
    #[serde(default)]
    group: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Header {
    format: u32,
    meta: Meta,
    arch: Arch,
    quantization: Quantization,
    tokenizer_len: u64,
    tensors: Vec<TensorEntry>,
}

/// The decoded contents of a verified pack.
pub(crate) struct Loaded {
    pub(crate) meta: Meta,
    pub(crate) arch: Arch,
    pub(crate) quantization: Quantization,
    pub(crate) tokenizer: Vec<u8>,
    pub(crate) tensors: std::collections::HashMap<String, Tensor>,
    pub(crate) sha256: String,
    pub(crate) file_bytes: u64,
}

pub(crate) enum Tensor {
    F32(Vec<f32>),
    F16(Vec<u16>),
    I8 {
        q: Vec<u8>,
        scale: Vec<f32>,
        group: usize,
    },
}

impl Loaded {
    fn take(&mut self, name: &str, expect: &[usize]) -> Result<Tensor> {
        let t = self
            .tensors
            .remove(name)
            .with_context(|| format!("pack is missing tensor {name:?}"))?;
        let len = match &t {
            Tensor::F32(v) => v.len(),
            Tensor::F16(v) => v.len(),
            Tensor::I8 { q, .. } => q.len(),
        };
        ensure!(
            len == expect.iter().product::<usize>(),
            "tensor {name:?} has {len} values, expected shape {expect:?}"
        );
        Ok(t)
    }

    pub(crate) fn vector(&mut self, name: &str, n: usize) -> Result<Vec<f32>> {
        match self.take(name, &[n])? {
            Tensor::F32(v) => Ok(v),
            _ => bail!("tensor {name:?} must be stored as f32"),
        }
    }

    pub(crate) fn matrix_f32(&mut self, name: &str, rows: usize, cols: usize) -> Result<Vec<f32>> {
        match self.take(name, &[rows, cols])? {
            Tensor::F32(v) => Ok(v),
            _ => bail!("tensor {name:?} must be stored as f32"),
        }
    }

    pub(crate) fn weights(&mut self, name: &str, rows: usize, cols: usize) -> Result<Weights> {
        Ok(match self.take(name, &[rows, cols])? {
            Tensor::F32(v) => Weights::F32(v),
            Tensor::F16(v) => Weights::F16(v),
            Tensor::I8 { q, scale, group } => {
                ensure!(
                    group > 0 && cols.is_multiple_of(group) && scale.len() == rows * (cols / group),
                    "tensor {name:?} has wrong scale count"
                );
                Weights::I8 { q, scale, group }
            }
        })
    }

    pub(crate) fn linear(&mut self, base: &str, n: usize, k: usize) -> Result<Linear> {
        Ok(Linear {
            w: self.weights(&format!("{base}.w"), n, k)?,
            bias: self.vector(&format!("{base}.b"), n)?,
            n,
            k,
        })
    }
}

/// The pack's content hash is not the pinned one: the file is corrupt, truncated or a different
/// model. Nothing from it was used.
#[derive(Debug, thiserror::Error)]
#[error("sha256 mismatch for {}: expected {expected}, file is {actual}; the file is corrupt or not the pinned model, delete it and fetch it again", path.display())]
pub struct Sha256Mismatch {
    pub path: std::path::PathBuf,
    pub expected: String,
    pub actual: String,
}

/// Little-endian `f32` words; a trailing partial word is ignored.
pub(super) fn f32s_le(raw: &[u8]) -> Vec<f32> {
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}

/// Little-endian `u16` words; a trailing partial word is ignored.
fn u16s_le(raw: &[u8]) -> impl Iterator<Item = u16> + '_ {
    raw.as_chunks::<2>()
        .0
        .iter()
        .map(|&b| u16::from_le_bytes(b))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// Streams the sha256 of a file without holding it in memory.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Hashes everything it reads so the digest always covers exactly the bytes that were parsed.
struct Tee {
    inner: std::io::BufReader<std::fs::File>,
    hash: Sha256,
    read: u64,
}

impl Tee {
    fn exact(&mut self, len: u64, what: &str) -> Result<Vec<u8>> {
        let len = usize::try_from(len).context("section too large")?;
        let mut v = vec![0u8; len];
        self.inner
            .read_exact(&mut v)
            .with_context(|| format!("pack is truncated in {what}"))?;
        self.hash.update(&v);
        self.read += len as u64;
        Ok(v)
    }
}

fn dtype_size(dtype: &str) -> Result<usize> {
    Ok(match dtype {
        "f32" => 4,
        "f16" => 2,
        "i8" => 1,
        _ => bail!("unsupported tensor dtype {dtype:?}"),
    })
}

/// Reads and verifies a pack. With `expected_sha256`, a mismatch fails before any tensor is used.
pub(crate) fn read(path: &Path, expected_sha256: Option<&str>) -> Result<Loaded> {
    let file =
        std::fs::File::open(path).with_context(|| format!("open pack {}", path.display()))?;
    let total = file.metadata()?.len();
    let mut t = Tee {
        inner: std::io::BufReader::with_capacity(1 << 20, file),
        hash: Sha256::new(),
        read: 0,
    };
    let magic = t.exact(8, "magic")?;
    ensure!(
        magic == MAGIC,
        "{} is not a holographic-memory embedding pack",
        path.display()
    );
    let hl = u64::from_le_bytes(t.exact(8, "header length")?.try_into().expect("8 bytes"));
    ensure!(
        hl <= MAX_HEADER && hl <= total,
        "pack header length {hl} is invalid"
    );
    let header: Header = serde_json::from_slice(&t.exact(hl, "header")?).context("pack header")?;
    ensure!(
        header.format == 1,
        "unsupported pack format {}",
        header.format
    );
    header.arch.validate()?;
    ensure!(
        header.tokenizer_len <= MAX_TOKENIZER && header.tokenizer_len <= total,
        "pack tokenizer length is invalid"
    );
    let tokenizer = t.exact(header.tokenizer_len, "tokenizer")?;
    let mut tensors = std::collections::HashMap::new();
    for e in &header.tensors {
        let elems = e
            .shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .context("tensor shape overflows")?;
        let size = dtype_size(&e.dtype)?;
        let data = (elems as u64)
            .checked_mul(size as u64)
            .filter(|&b| b <= MAX_TENSOR_BYTES)
            .context("tensor is too large")?;
        let scale_bytes = if e.dtype == "i8" {
            ensure!(
                e.shape.len() == 2 && e.group > 0 && e.shape[1].is_multiple_of(e.group),
                "int8 tensor {:?} needs a 2-D shape and a group dividing its row length",
                e.name
            );
            (e.shape[0] * (e.shape[1] / e.group)) as u64 * 4
        } else {
            0
        };
        ensure!(
            e.len == data + scale_bytes,
            "tensor {:?} length does not match its shape",
            e.name
        );
        ensure!(
            t.read + e.len <= total,
            "pack is truncated: tensor {:?} extends past the end",
            e.name
        );
        let raw = t.exact(e.len, &e.name)?;
        let tensor = match e.dtype.as_str() {
            "f32" => Tensor::F32(f32s_le(&raw)),
            "f16" => Tensor::F16(u16s_le(&raw).collect()),
            _ => {
                let (q, s) = raw.split_at(data as usize);
                Tensor::I8 {
                    group: e.group,
                    q: q.to_vec(),
                    scale: f32s_le(s),
                }
            }
        };
        ensure!(
            tensors.insert(e.name.clone(), tensor).is_none(),
            "duplicate tensor {:?}",
            e.name
        );
    }
    ensure!(
        t.read == total,
        "pack has {} trailing bytes",
        total - t.read
    );
    let sha256 = hex(&t.hash.finalize());
    if let Some(want) = expected_sha256 {
        if !sha256.eq_ignore_ascii_case(want) {
            return Err(Sha256Mismatch {
                path: path.to_path_buf(),
                expected: want.to_string(),
                actual: sha256,
            }
            .into());
        }
    }
    Ok(Loaded {
        meta: header.meta,
        arch: header.arch,
        quantization: header.quantization,
        tokenizer,
        tensors,
        sha256,
        file_bytes: total,
    })
}

/// A tensor to write: f32 values with a shape; matrices are quantized when `matrix` is set.
pub(crate) struct Source {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) values: Vec<f32>,
    pub(crate) matrix: bool,
}

fn quantize_i8(values: &[f32], rows: usize, group: usize) -> (Vec<u8>, Vec<f32>) {
    let mut q = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len() / group);
    let _ = rows;
    for g in values.chunks_exact(group) {
        let max = g.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
        scales.push(scale);
        q.extend(
            g.iter()
                .map(|&v| ((v / scale).round().clamp(-127.0, 127.0) as i8) as u8),
        );
    }
    (q, scales)
}

/// Values per int8 scale unless a caller chooses otherwise (see benchmarks/results/embedding_selection_v1.json).
pub const DEFAULT_INT8_GROUP: usize = 64;

/// Serializes a pack. The output is canonical: the same inputs give the same bytes.
pub(crate) fn build(
    meta: Meta,
    arch: Arch,
    quantization: Quantization,
    int8_group: usize,
    tokenizer: &[u8],
    sources: &[Source],
) -> Result<Vec<u8>> {
    arch.validate()?;
    let mut entries = Vec::new();
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    for s in sources {
        ensure!(
            s.values.len() == s.shape.iter().product::<usize>(),
            "tensor {:?} shape mismatch",
            s.name
        );
        ensure!(
            s.values.iter().all(|v| v.is_finite()),
            "tensor {:?} has non-finite values",
            s.name
        );
        let mut group = 0;
        let (dtype, blob) = match (s.matrix, quantization) {
            (true, Quantization::Int8) => {
                let cols = s.shape[1];
                group = if int8_group > 0 && cols.is_multiple_of(int8_group) {
                    int8_group
                } else {
                    cols
                };
                let (q, scales) = quantize_i8(&s.values, s.shape[0], group);
                let mut b = q;
                for sc in scales {
                    b.extend_from_slice(&sc.to_le_bytes());
                }
                ("i8", b)
            }
            (true, Quantization::F16) => (
                "f16",
                s.values
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect(),
            ),
            _ => (
                "f32",
                s.values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        };
        entries.push(TensorEntry {
            name: s.name.clone(),
            dtype: dtype.into(),
            shape: s.shape.clone(),
            len: blob.len() as u64,
            group,
        });
        blobs.push(blob);
    }
    let header = serde_json::to_vec(&Header {
        format: 1,
        meta,
        arch,
        quantization,
        tokenizer_len: tokenizer.len() as u64,
        tensors: entries,
    })?;
    let mut out = Vec::with_capacity(
        16 + header.len() + tokenizer.len() + blobs.iter().map(Vec::len).sum::<usize>(),
    );
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(tokenizer);
    for b in blobs {
        out.extend_from_slice(&b);
    }
    Ok(out)
}

/// Reads a safetensors file as f32 tensors by name (F32, F16 and BF16 are accepted).
pub(crate) fn read_safetensors(path: &Path) -> Result<NamedTensors> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(bytes.len() >= 8, "safetensors file is too short");
    let hl = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    ensure!(
        hl <= bytes.len() - 8 && hl <= 100 << 20,
        "safetensors header length is invalid"
    );
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + hl]).context("safetensors header")?;
    let base = 8 + hl;
    let mut out = std::collections::HashMap::new();
    for (name, v) in header
        .as_object()
        .context("safetensors header is not an object")?
    {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().context("tensor dtype")?;
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .context("tensor shape")?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize).context("shape entry"))
            .collect::<Result<_>>()?;
        let off = v["data_offsets"].as_array().context("data_offsets")?;
        let (a, b) = (
            off.first().and_then(|x| x.as_u64()).context("offset")? as usize,
            off.get(1).and_then(|x| x.as_u64()).context("offset")? as usize,
        );
        ensure!(
            a <= b && base + b <= bytes.len(),
            "tensor {name:?} lies outside the file"
        );
        let raw = &bytes[base + a..base + b];
        let values: Vec<f32> = match dtype {
            "F32" => f32s_le(raw),
            "F16" => u16s_le(raw).map(f16_to_f32).collect(),
            "BF16" => u16s_le(raw)
                .map(|h| f32::from_bits(u32::from(h) << 16))
                .collect(),
            // Index buffers such as position_ids carry no weights.
            "I64" | "I32" | "U8" | "BOOL" => continue,
            other => bail!("tensor {name:?} has unsupported dtype {other}"),
        };
        ensure!(
            values.len() == shape.iter().product::<usize>(),
            "tensor {name:?} size does not match its shape"
        );
        out.insert(name.clone(), (shape, values));
    }
    Ok(out)
}

/// Fields of the upstream `config.json` the converter needs.
#[derive(Deserialize)]
struct HfConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    type_vocab_size: Option<usize>,
    #[serde(default)]
    layer_norm_eps: Option<f32>,
    #[serde(default)]
    pad_token_id: Option<usize>,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    position_embedding_type: Option<String>,
}

/// Converts a Hugging Face BERT, RoBERTa or XLM-R directory (`config.json`, `tokenizer.json`,
/// `model.safetensors`) to pack bytes.
pub fn convert_hf_dir(
    dir: &Path,
    meta: Meta,
    quantization: Quantization,
    int8_group: usize,
) -> Result<Vec<u8>> {
    let cfg: HfConfig = serde_json::from_slice(
        &std::fs::read(dir.join("config.json")).context("read config.json")?,
    )
    .context("parse config.json")?;
    let model_type = cfg.model_type.as_deref().unwrap_or("bert");
    ensure!(
        matches!(model_type, "bert" | "roberta" | "xlm-roberta"),
        "model_type {model_type:?} is not supported (BERT, RoBERTa and XLM-RoBERTa encoders only)"
    );
    ensure!(
        matches!(cfg.hidden_act.as_deref().unwrap_or("gelu"), "gelu"),
        "hidden_act {:?} is not supported (exact GELU only)",
        cfg.hidden_act
    );
    ensure!(
        matches!(
            cfg.position_embedding_type.as_deref().unwrap_or("absolute"),
            "absolute"
        ),
        "only absolute position embeddings are supported"
    );
    let roberta = model_type != "bert";
    let arch = Arch {
        hidden: cfg.hidden_size,
        layers: cfg.num_hidden_layers,
        heads: cfg.num_attention_heads,
        intermediate: cfg.intermediate_size,
        vocab: cfg.vocab_size,
        max_positions: cfg.max_position_embeddings,
        type_vocab: cfg.type_vocab_size.unwrap_or(1),
        layer_norm_eps: cfg.layer_norm_eps.unwrap_or(1e-12),
        position_offset: if roberta {
            cfg.pad_token_id.unwrap_or(1) + 1
        } else {
            0
        },
    };
    let mut raw = read_safetensors(&dir.join("model.safetensors"))?;
    // Sentence-transformers repos nest the encoder under a prefix; find it from the embedding table.
    let word = raw
        .keys()
        .find(|k| k.ends_with("embeddings.word_embeddings.weight"))
        .context("no embeddings.word_embeddings.weight in model.safetensors")?
        .clone();
    let prefix = word
        .trim_end_matches("embeddings.word_embeddings.weight")
        .to_string();
    let mut get = |suffix: &str| -> Result<(Vec<usize>, Vec<f32>)> {
        let alt = suffix
            .replace(".weight", ".gamma")
            .replace(".bias", ".beta");
        raw.remove(&format!("{prefix}{suffix}"))
            .or_else(|| raw.remove(&format!("{prefix}{alt}")))
            .with_context(|| format!("model.safetensors has no tensor {prefix}{suffix}"))
    };
    let h = arch.hidden;
    let mut sources = Vec::new();
    let mut push = |name: &str,
                    (shape, values): (Vec<usize>, Vec<f32>),
                    matrix: bool,
                    expect: &[usize]|
     -> Result<()> {
        ensure!(
            shape == expect,
            "tensor {name} has shape {shape:?}, expected {expect:?}"
        );
        sources.push(Source {
            name: name.into(),
            shape,
            values,
            matrix,
        });
        Ok(())
    };
    push(
        "word_emb",
        get("embeddings.word_embeddings.weight")?,
        true,
        &[arch.vocab, h],
    )?;
    push(
        "pos_emb",
        get("embeddings.position_embeddings.weight")?,
        false,
        &[arch.max_positions, h],
    )?;
    let type_emb = get("embeddings.token_type_embeddings.weight")?;
    push("type_emb", type_emb, false, &[arch.type_vocab, h])?;
    push("emb_ln.w", get("embeddings.LayerNorm.weight")?, false, &[h])?;
    push("emb_ln.b", get("embeddings.LayerNorm.bias")?, false, &[h])?;
    for l in 0..arch.layers {
        let p = format!("encoder.layer.{l}.");
        let mut qkv_w = Vec::with_capacity(3 * h * h);
        let mut qkv_b = Vec::with_capacity(3 * h);
        for part in ["query", "key", "value"] {
            let (s, w) = get(&format!("{p}attention.self.{part}.weight"))?;
            ensure!(s == [h, h], "{p}{part} has shape {s:?}");
            qkv_w.extend(w);
            let (s, b) = get(&format!("{p}attention.self.{part}.bias"))?;
            ensure!(s == [h], "{p}{part} bias has shape {s:?}");
            qkv_b.extend(b);
        }
        let name = |s: &str| format!("l{l}.{s}");
        push(&name("qkv.w"), (vec![3 * h, h], qkv_w), true, &[3 * h, h])?;
        push(&name("qkv.b"), (vec![3 * h], qkv_b), false, &[3 * h])?;
        push(
            &name("ao.w"),
            get(&format!("{p}attention.output.dense.weight"))?,
            true,
            &[h, h],
        )?;
        push(
            &name("ao.b"),
            get(&format!("{p}attention.output.dense.bias"))?,
            false,
            &[h],
        )?;
        push(
            &name("aln.w"),
            get(&format!("{p}attention.output.LayerNorm.weight"))?,
            false,
            &[h],
        )?;
        push(
            &name("aln.b"),
            get(&format!("{p}attention.output.LayerNorm.bias"))?,
            false,
            &[h],
        )?;
        push(
            &name("f1.w"),
            get(&format!("{p}intermediate.dense.weight"))?,
            true,
            &[arch.intermediate, h],
        )?;
        push(
            &name("f1.b"),
            get(&format!("{p}intermediate.dense.bias"))?,
            false,
            &[arch.intermediate],
        )?;
        push(
            &name("f2.w"),
            get(&format!("{p}output.dense.weight"))?,
            true,
            &[h, arch.intermediate],
        )?;
        push(
            &name("f2.b"),
            get(&format!("{p}output.dense.bias"))?,
            false,
            &[h],
        )?;
        push(
            &name("fln.w"),
            get(&format!("{p}output.LayerNorm.weight"))?,
            false,
            &[h],
        )?;
        push(
            &name("fln.b"),
            get(&format!("{p}output.LayerNorm.bias"))?,
            false,
            &[h],
        )?;
    }
    let tokenizer = std::fs::read(dir.join("tokenizer.json")).context("read tokenizer.json")?;
    build(meta, arch, quantization, int8_group, &tokenizer, &sources)
}
