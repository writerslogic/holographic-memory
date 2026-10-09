// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The forward pass is adapted from candle-transformers 0.11 `models/qwen3.rs`
// (https://github.com/huggingface/candle), Copyright The HuggingFace candle contributors,
// licensed Apache-2.0 OR MIT. Changes: the KV cache lives outside the weights so one loaded
// model serves many sequences, and attention takes per-row positions and a key mask so a
// batch can hold sequences of different lengths.

//! Qwen3 decoder with batch-correct masking. candle's version builds its causal mask for a
//! batch of one and has no padding mask; with [`Positions::Uniform`] this module computes
//! the CPU computes what candle does for one sequence (same kernels in the same order), and
//! [`Positions::PerRow`] adds left-padded rows at their own RoPE positions.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::attention::{flash_attn, AttnMask};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::qwen3::Config;
use candle_transformers::models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm};
use candle_transformers::utils::repeat_kv;

/// Where the rows of a forward pass sit.
pub enum Positions<'a> {
    /// Every row starts at `offset` (the cache length) and has no padding: candle's path,
    /// including the fused CPU attention kernel.
    Uniform(usize),
    /// `positions` (B, L) u32 RoPE positions per token; `mask` an additive mask broadcastable
    /// to (B, 1, L, T) over the cached plus new keys (0 to attend, -inf to skip). Every query
    /// row must keep at least one key, or softmax returns NaN.
    PerRow {
        positions: &'a Tensor,
        mask: &'a Tensor,
    },
}

/// Keys and values of every layer, each (B, kv_heads, T, head_dim).
#[derive(Clone, Default)]
pub struct Cache {
    kv: Vec<Option<(Tensor, Tensor)>>,
}

impl Cache {
    /// Cached sequence length (0 when empty).
    pub fn seq_len(&self) -> usize {
        self.kv
            .first()
            .and_then(Option::as_ref)
            .map_or(0, |(k, _)| k.dim(2).unwrap_or(0))
    }

    /// Applies `f` to every cached key and value tensor.
    pub fn map(&self, f: impl Fn(&Tensor) -> Result<Tensor>) -> Result<Self> {
        let kv = self
            .kv
            .iter()
            .map(|e| match e {
                Some((k, v)) => Ok(Some((f(k)?, f(v)?))),
                None => Ok(None),
            })
            .collect::<Result<_>>()?;
        Ok(Self { kv })
    }

    /// Rows of several caches stacked along the batch dimension; all must have one length.
    pub fn cat(parts: &[Cache]) -> Result<Self> {
        let layers = parts.first().map_or(0, |c| c.kv.len());
        let mut kv = Vec::with_capacity(layers);
        for l in 0..layers {
            let mut ks = Vec::with_capacity(parts.len());
            let mut vs = Vec::with_capacity(parts.len());
            for p in parts {
                match &p.kv[l] {
                    Some((k, v)) => {
                        ks.push(k);
                        vs.push(v);
                    }
                    None => candle_core::bail!("cat of an empty cache"),
                }
            }
            kv.push(Some((Tensor::cat(&ks, 0)?, Tensor::cat(&vs, 0)?)));
        }
        Ok(Self { kv })
    }

    /// Prepends `n` zero positions to every row (finite zeros, never read through the mask).
    pub fn left_pad(&self, n: usize) -> Result<Self> {
        if n == 0 {
            return Ok(self.clone());
        }
        self.map(|t| {
            let (b, h, _, d) = t.dims4()?;
            let z = Tensor::zeros((b, h, n, d), t.dtype(), t.device())?;
            Tensor::cat(&[&z, t], 2)
        })
    }
}

#[derive(Clone)]
struct Rotary {
    sin: Tensor,
    cos: Tensor,
}

impl Rotary {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// cos/sin for the pass: (L, D/2) for uniform rows, (B, L, D/2) gathered per row.
    fn tables(&self, pos: &Positions, b: usize, l: usize) -> Result<(Tensor, Tensor)> {
        match pos {
            Positions::Uniform(offset) => Ok((
                self.cos.narrow(0, *offset, l)?,
                self.sin.narrow(0, *offset, l)?,
            )),
            Positions::PerRow { positions, .. } => {
                let idx = positions.flatten_all()?;
                let half = self.cos.dim(1)?;
                Ok((
                    self.cos.index_select(&idx, 0)?.reshape((b, l, half))?,
                    self.sin.index_select(&idx, 0)?.reshape((b, l, half))?,
                ))
            }
        }
    }
}

#[derive(Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if cfg.use_sliding_window {
            candle_core::bail!("sliding window is not supported")
        }
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let bias = cfg.attention_bias;
        Ok(Self {
            q_proj: linear_b(cfg.hidden_size, num_heads * head_dim, bias, vb.pp("q_proj"))?,
            k_proj: linear_b(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                bias,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_b(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                bias,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_b(num_heads * head_dim, cfg.hidden_size, bias, vb.pp("o_proj"))?,
            q_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            // The config's hidden_size is not always heads * head_dim.
            hidden_size: head_dim * num_heads,
        })
    }

    /// `flash`: candle's fused CPU kernel (causal, no padding); otherwise standard attention
    /// with the additive `mask`.
    fn forward(
        &self,
        x: &Tensor,
        (cos, sin): (&Tensor, &Tensor),
        flash: bool,
        mask: Option<&Tensor>,
        kv: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((
            b,
            self.num_heads,
            l,
            self.head_dim,
        ))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((
            b,
            self.num_kv_heads,
            l,
            self.head_dim,
        ))?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
        let offset = kv.as_ref().map_or(Ok(0), |(k, _)| k.dim(2))?;
        // Same concatenation as candle's `ConcatKvCache` (dim 2). The inputs must be
        // contiguous: `Tensor::cat` of a strided tensor returns a strided result, so the
        // whole cache would be re-copied element-wise every decode step.
        let (k, v) = (k.contiguous()?, v.contiguous()?);
        let (k, v) = match kv.take() {
            Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
            None => (k, v),
        };
        *kv = Some((k.clone(), v.clone()));

        if flash {
            return self.cpu_flash(&q, &k, &v, offset, b, l);
        }
        let metal_sdpa = x.device().is_metal()
            && matches!(q.dtype(), DType::BF16 | DType::F16 | DType::F32)
            && matches!(self.head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512)
            && !(l > 1 && self.head_dim == 512 && q.dtype() == DType::F32)
            && (l > 1 || mask.is_none());
        if metal_sdpa {
            // Candle's vector SDPA ignores masks; masked single-token rows use the path below.
            let mask = mask
                .map(|m| m.broadcast_as((b, self.num_heads, l, k.dim(2)?)))
                .transpose()?;
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            return candle_nn::ops::sdpa(&q, &k, &v, mask.as_ref(), false, scale, 1.0)?
                .transpose(1, 2)?
                .reshape((b, l, self.hidden_size))?
                .apply(&self.o_proj);
        }
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    /// candle's CPU path: the fused kernel, causal with the cache offset, computed in f32.
    fn cpu_flash(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        offset: usize,
        b: usize,
        l: usize,
    ) -> Result<Tensor> {
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mask = AttnMask::causal_with_offset(offset);
        let ctx = match q.dtype() {
            DType::F32 => flash_attn::<f32>(&q, &k, &v, scale, mask, None, None)?,
            other => flash_attn::<f32>(
                &q.to_dtype(DType::F32)?,
                &k.to_dtype(DType::F32)?,
                &v.to_dtype(DType::F32)?,
                scale,
                mask,
                None,
                None,
            )?
            .to_dtype(other)?,
        };
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

#[derive(Clone)]
struct Layer {
    attn: Attention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

/// Qwen3 weights (shared by clones); the optional LM head makes it the causal LM.
#[derive(Clone)]
pub struct Qwen3 {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Option<Linear>,
    rotary: Rotary,
    device: Device,
    dtype: DType,
    /// Uniform passes on the CPU use candle's fused kernel (off only in tests, to run the
    /// GPU's masked path on the CPU).
    cpu_flash: bool,
}

impl Qwen3 {
    /// `lm_head`: build the causal-LM head (tied to the embeddings when the config says so).
    pub fn new(cfg: &Config, vb: VarBuilder, lm_head: bool) -> Result<Self> {
        if vb.dtype() == DType::F64 {
            candle_core::bail!("Qwen3 does not support f64");
        }
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let vb_l = vb.pp("model.layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let vb = vb_l.pp(i);
                Ok(Layer {
                    attn: Attention::new(cfg, vb.pp("self_attn"))?,
                    mlp: Mlp::new(cfg, vb.pp("mlp"))?,
                    ln1: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
                    ln2: RmsNorm::new(
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                        vb.pp("post_attention_layernorm"),
                    )?,
                })
            })
            .collect::<Result<_>>()?;
        let lm_head = match (lm_head, cfg.tie_word_embeddings) {
            (false, _) => None,
            (true, true) => Some(Linear::from_weights(
                embed_tokens.embeddings().clone(),
                None,
            )),
            (true, false) => Some(linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp("lm_head"),
            )?),
        };
        Ok(Self {
            rotary: Rotary::new(vb.dtype(), cfg, vb.device())?,
            embed_tokens,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            cpu_flash: true,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Additive mask in the model dtype: 0 where `keep`, -inf elsewhere, shaped `shape`.
    pub fn additive_mask(&self, keep: &[bool], shape: &[usize]) -> Result<Tensor> {
        let v: Vec<f32> = keep
            .iter()
            .map(|&k| if k { 0.0 } else { f32::NEG_INFINITY })
            .collect();
        Tensor::from_vec(v, shape, &self.device)?.to_dtype(self.dtype)
    }

    /// candle's causal mask for one row: (1, 1, L, L + offset).
    fn causal_mask(&self, l: usize, offset: usize) -> Result<Tensor> {
        let t = l + offset;
        let keep: Vec<bool> = (0..l)
            .flat_map(|i| (0..t).map(move |j| j <= i + offset))
            .collect();
        self.additive_mask(&keep, &[1, 1, l, t])
    }

    /// Final hidden states (B, L, hidden); appends the new keys and values to `cache`.
    pub fn forward(&self, input: &Tensor, pos: Positions, cache: &mut Cache) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        cache.kv.resize(self.layers.len(), None);
        let flash = self.cpu_flash && self.device.is_cpu() && matches!(pos, Positions::Uniform(_));
        let causal = match pos {
            Positions::Uniform(offset) if !flash && l > 1 => Some(self.causal_mask(l, offset)?),
            _ => None,
        };
        let mask = match &pos {
            Positions::PerRow { mask, .. } => Some(*mask),
            Positions::Uniform(_) => causal.as_ref(),
        };
        let (cos, sin) = self.rotary.tables(&pos, b, l)?;
        let mut h = self.embed_tokens.forward(input)?;
        for (layer, kv) in self.layers.iter().zip(cache.kv.iter_mut()) {
            let a = layer
                .attn
                .forward(&layer.ln1.forward(&h)?, (&cos, &sin), flash, mask, kv)?;
            let x = (h + a)?;
            let m = layer.mlp.forward(&layer.ln2.forward(&x)?)?;
            h = (x + m)?;
        }
        self.norm.forward(&h)
    }

    /// LM-head logits of the given hidden rows (..., hidden).
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        match &self.lm_head {
            Some(head) => hidden.apply(head),
            None => candle_core::bail!("model was loaded without an LM head"),
        }
    }

    /// Hidden state at the last token of each row: `last[i]` is row i's index in (B, L, H).
    pub fn gather_last(hidden: &Tensor, last: &[usize]) -> Result<Tensor> {
        let (b, l, hd) = hidden.dims3()?;
        if last.len() != b || last.iter().any(|&i| i >= l) {
            candle_core::bail!("gather_last: bad indices for ({b}, {l})");
        }
        let idx: Vec<u32> = last
            .iter()
            .enumerate()
            .map(|(r, &i)| (r * l + i) as u32)
            .collect();
        let idx = Tensor::new(idx.as_slice(), hidden.device())?;
        hidden.reshape((b * l, hd))?.index_select(&idx, 0)
    }
}

/// Greedy next token per row from (B, 1, V) or (B, V) logits.
pub fn argmax_rows(logits: &Tensor) -> Result<Vec<u32>> {
    let l = if logits.rank() == 3 {
        logits.squeeze(1)?
    } else {
        logits.clone()
    };
    l.to_dtype(DType::F32)?.argmax(D::Minus1)?.to_vec1::<u32>()
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// A tiny Qwen3 on the CPU with weights from a fixed LCG. `cpu_flash = false` runs the
    /// masked attention path that Metal and CUDA use.
    pub(crate) fn tiny(seed: u64, cpu_flash: bool) -> Qwen3 {
        let cfg: Config = serde_json::from_value(serde_json::json!({
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
            "num_hidden_layers": 2, "num_attention_heads": 4, "head_dim": 8,
            "attention_bias": false, "num_key_value_heads": 2,
            "max_position_embeddings": 256, "sliding_window": null, "max_window_layers": 2,
            "tie_word_embeddings": true, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
            "use_sliding_window": false, "hidden_act": "silu"
        }))
        .unwrap();
        let dev = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        // Create the variables, overwrite them in name order, then build on the set weights.
        Qwen3::new(&cfg, vb.clone(), false).unwrap();
        let data = varmap.data().lock().unwrap();
        let mut names: Vec<&String> = data.keys().collect();
        names.sort();
        let mut state = seed;
        for name in names {
            let var = &data[name];
            let vals: Vec<f32> = (0..var.elem_count())
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 1.5
                })
                .collect();
            var.set(&Tensor::from_vec(vals, var.shape(), &dev).unwrap())
                .unwrap();
        }
        drop(data);
        let mut model = Qwen3::new(&cfg, vb, true).unwrap();
        model.cpu_flash = cpu_flash;
        model
    }
}
