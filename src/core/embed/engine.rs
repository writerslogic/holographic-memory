// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! BERT-family encoder forward pass (BERT, RoBERTa, XLM-R: absolute positions, post-LN layers,
//! exact GELU). Sequences are never padded: tokens of a batch are stacked into one matrix for the
//! linear layers and attention runs per sequence and head, so a sequence's output does not depend
//! on what it is batched with.

use anyhow::Result;
use rayon::prelude::*;

use super::kernels::{dot_head, gelu, layer_norm, softmax, Linear, Weights};
use super::pack::{Arch, Loaded, Pooling};

struct Layer {
    qkv: Linear,
    attn_out: Linear,
    attn_ln: (Vec<f32>, Vec<f32>),
    ffn_in: Linear,
    ffn_out: Linear,
    ffn_ln: (Vec<f32>, Vec<f32>),
}

pub(crate) struct Encoder {
    arch: Arch,
    word: Weights,
    pos: Vec<f32>,
    type0: Vec<f32>,
    emb_ln: (Vec<f32>, Vec<f32>),
    layers: Vec<Layer>,
}

/// Reusable activation buffers; the soft batch budget always admits one full sequence.
#[derive(Default)]
pub(crate) struct Scratch {
    h: Vec<f32>,
    qkv: Vec<f32>,
    ctx: Vec<f32>,
    proj: Vec<f32>,
    inter: Vec<f32>,
    lin: Vec<f32>,
}

impl Encoder {
    pub(crate) fn from_pack(p: &mut Loaded) -> Result<Self> {
        let a = p.arch.clone();
        let h = a.hidden;
        let mut layers = Vec::with_capacity(a.layers);
        let word = p.weights("word_emb", a.vocab, h)?;
        let pos = p.matrix_f32("pos_emb", a.max_positions, h)?;
        let type_emb = p.matrix_f32("type_emb", a.type_vocab, h)?;
        let emb_ln = (p.vector("emb_ln.w", h)?, p.vector("emb_ln.b", h)?);
        for l in 0..a.layers {
            let n = |s: &str| format!("l{l}.{s}");
            layers.push(Layer {
                qkv: p.linear(&n("qkv"), 3 * h, h)?,
                attn_out: p.linear(&n("ao"), h, h)?,
                attn_ln: (p.vector(&n("aln.w"), h)?, p.vector(&n("aln.b"), h)?),
                ffn_in: p.linear(&n("f1"), a.intermediate, h)?,
                ffn_out: p.linear(&n("f2"), h, a.intermediate)?,
                ffn_ln: (p.vector(&n("fln.w"), h)?, p.vector(&n("fln.b"), h)?),
            });
        }
        Ok(Self {
            arch: a,
            word,
            pos,
            type0: type_emb[..h].to_vec(),
            emb_ln,
            layers,
        })
    }

    pub(crate) fn weight_bytes(&self) -> usize {
        self.word.bytes()
            + (self.pos.len() + self.type0.len()) * 4
            + self
                .layers
                .iter()
                .map(|l| {
                    l.qkv.w.bytes()
                        + l.attn_out.w.bytes()
                        + l.ffn_in.w.bytes()
                        + l.ffn_out.w.bytes()
                })
                .sum::<usize>()
    }

    /// Pooled (not normalized) vectors for sequences of token ids, in input order.
    /// Every sequence must be non-empty and at most `max_positions - position_offset` long.
    pub(crate) fn encode(
        &self,
        seqs: &[&[u32]],
        pooling: Pooling,
        s: &mut Scratch,
    ) -> Vec<Vec<f32>> {
        let a = &self.arch;
        let h = a.hidden;
        let total: usize = seqs.iter().map(|q| q.len()).sum();
        let mut starts = Vec::with_capacity(seqs.len() + 1);
        starts.push(0usize);
        for q in seqs {
            starts.push(starts.last().copied().unwrap_or(0) + q.len());
        }
        s.h.clear();
        s.h.resize(total * h, 0.0);
        let mut row = vec![0f32; h];
        for (q, &start) in seqs.iter().zip(&starts) {
            for (i, &id) in q.iter().enumerate() {
                self.word
                    .row_into((id as usize).min(a.vocab - 1), h, &mut row);
                let p = &self.pos[(a.position_offset + i) * h..(a.position_offset + i + 1) * h];
                let out = &mut s.h[(start + i) * h..(start + i + 1) * h];
                for j in 0..h {
                    out[j] = row[j] + p[j] + self.type0[j];
                }
            }
        }
        layer_norm(
            &mut s.h,
            h,
            &self.emb_ln.0,
            &self.emb_ln.1,
            a.layer_norm_eps,
        );
        let heads = a.heads;
        let dh = h / heads;
        let scale = 1.0 / (dh as f32).sqrt();
        for layer in &self.layers {
            s.qkv.resize(total * 3 * h, 0.0);
            layer.qkv.forward(&s.h, total, &mut s.qkv, &mut s.lin);
            s.ctx.clear();
            s.ctx.resize(total * h, 0.0);
            let qkv = &s.qkv;
            // One task per (sequence, head); each returns its own context block.
            let tasks: Vec<(usize, usize)> = (0..seqs.len())
                .flat_map(|i| (0..heads).map(move |hd| (i, hd)))
                .collect();
            let blocks: Vec<Vec<f32>> = tasks
                .par_iter()
                .map(|&(i, hd)| {
                    let (start, len) = (starts[i], seqs[i].len());
                    let at = |t: usize, part: usize| {
                        &qkv[(start + t) * 3 * h + part * h + hd * dh..][..dh]
                    };
                    let mut scores = vec![0f32; len];
                    let mut out = vec![0f32; len * dh];
                    for qi in 0..len {
                        let qv = at(qi, 0);
                        for (ki, sc) in scores.iter_mut().enumerate() {
                            *sc = dot_head(qv, at(ki, 1)) * scale;
                        }
                        softmax(&mut scores);
                        let o = &mut out[qi * dh..(qi + 1) * dh];
                        for (vi, &w) in scores.iter().enumerate() {
                            for (x, &v) in o.iter_mut().zip(at(vi, 2)) {
                                *x += w * v;
                            }
                        }
                    }
                    out
                })
                .collect();
            for (&(i, hd), block) in tasks.iter().zip(&blocks) {
                for t in 0..seqs[i].len() {
                    let dst = (starts[i] + t) * h + hd * dh;
                    s.ctx[dst..dst + dh].copy_from_slice(&block[t * dh..(t + 1) * dh]);
                }
            }
            s.proj.resize(total * h, 0.0);
            layer
                .attn_out
                .forward(&s.ctx, total, &mut s.proj, &mut s.lin);
            for (x, &r) in s.proj.iter_mut().zip(&s.h) {
                *x += r;
            }
            layer_norm(
                &mut s.proj,
                h,
                &layer.attn_ln.0,
                &layer.attn_ln.1,
                a.layer_norm_eps,
            );
            s.inter.resize(total * a.intermediate, 0.0);
            layer
                .ffn_in
                .forward(&s.proj, total, &mut s.inter, &mut s.lin);
            s.inter.par_iter_mut().for_each(|x| *x = gelu(*x));
            layer.ffn_out.forward(&s.inter, total, &mut s.h, &mut s.lin);
            for (x, &r) in s.h.iter_mut().zip(&s.proj) {
                *x += r;
            }
            layer_norm(
                &mut s.h,
                h,
                &layer.ffn_ln.0,
                &layer.ffn_ln.1,
                a.layer_norm_eps,
            );
        }
        seqs.iter()
            .zip(&starts)
            .map(|(q, &start)| match pooling {
                Pooling::Cls => s.h[start * h..(start + 1) * h].to_vec(),
                Pooling::Mean => {
                    let mut acc = vec![0f32; h];
                    for t in 0..q.len() {
                        for (x, &v) in acc
                            .iter_mut()
                            .zip(&s.h[(start + t) * h..(start + t + 1) * h])
                        {
                            *x += v;
                        }
                    }
                    let inv = 1.0 / q.len() as f32;
                    acc.iter_mut().for_each(|x| *x *= inv);
                    acc
                }
            })
            .collect()
    }
}
