// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Random-weight model packs for tests: a 2-layer, 32-wide BERT with a 25-word vocabulary. They
//! exercise the whole pipeline (pack, tokenizer, inference, store integration) without real
//! weights or network access. Not for production use.

use std::path::{Path, PathBuf};

use super::pack::{self, Arch, Meta, Pooling, Quantization, Source};
use super::sha256_hex;

pub const WORDS: &[&str] = &[
    "the", "cat", "sat", "on", "mat", "dog", "ran", "in", "park", "bird", "flew", "over", "tree",
    "fish", "swam", "river", "sun", "moon", "star", "night", "day",
];

pub fn tiny_tokenizer() -> Vec<u8> {
    let vocab: serde_json::Map<String, serde_json::Value> = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        .iter()
        .chain(WORDS)
        .enumerate()
        .map(|(i, t)| ((*t).to_string(), serde_json::json!(i)))
        .collect();
    serde_json::to_vec(&serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": true,
            "handle_chinese_chars": true,
            "strip_accents": null,
            "lowercase": true
        },
        "pre_tokenizer": { "type": "BertPreTokenizer" },
        "post_processor": { "type": "BertProcessing", "sep": ["[SEP]", 3], "cls": ["[CLS]", 2] },
        "decoder": null,
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocab
        }
    }))
    .expect("tokenizer json")
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

pub fn tiny_arch() -> Arch {
    Arch {
        hidden: 32,
        layers: 2,
        heads: 4,
        intermediate: 64,
        vocab: 4 + WORDS.len(),
        max_positions: 64,
        type_vocab: 2,
        layer_norm_eps: 1e-12,
        position_offset: 0,
    }
}

pub fn tiny_pack_bytes(quant: Quantization, pooling: Pooling) -> Vec<u8> {
    tiny_pack_bytes_with_group(quant, pooling, 16)
}

/// Like [`tiny_pack_bytes`] with a chosen int8 group (0 or a non-divisor means one scale per row).
pub fn tiny_pack_bytes_with_group(quant: Quantization, pooling: Pooling, group: usize) -> Vec<u8> {
    let a = tiny_arch();
    let mut rng = Lcg(7);
    let mut src = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, matrix: bool, scale: f32, ln: Option<f32>| {
        let n: usize = shape.iter().product();
        let values = (0..n)
            .map(|_| ln.map_or(rng.next() * scale, |c| c + rng.next() * 0.05))
            .collect();
        src.push(Source {
            name,
            shape,
            values,
            matrix,
        });
    };
    let h = a.hidden;
    add("word_emb".into(), vec![a.vocab, h], true, 0.5, None);
    add("pos_emb".into(), vec![a.max_positions, h], false, 0.2, None);
    add("type_emb".into(), vec![a.type_vocab, h], false, 0.2, None);
    add("emb_ln.w".into(), vec![h], false, 0.0, Some(1.0));
    add("emb_ln.b".into(), vec![h], false, 0.0, Some(0.0));
    for l in 0..a.layers {
        let n = |s: &str| format!("l{l}.{s}");
        add(n("qkv.w"), vec![3 * h, h], true, 0.3, None);
        add(n("qkv.b"), vec![3 * h], false, 0.05, None);
        add(n("ao.w"), vec![h, h], true, 0.3, None);
        add(n("ao.b"), vec![h], false, 0.05, None);
        add(n("aln.w"), vec![h], false, 0.0, Some(1.0));
        add(n("aln.b"), vec![h], false, 0.0, Some(0.0));
        add(n("f1.w"), vec![a.intermediate, h], true, 0.3, None);
        add(n("f1.b"), vec![a.intermediate], false, 0.05, None);
        add(n("f2.w"), vec![h, a.intermediate], true, 0.3, None);
        add(n("f2.b"), vec![h], false, 0.05, None);
        add(n("fln.w"), vec![h], false, 0.0, Some(1.0));
        add(n("fln.b"), vec![h], false, 0.0, Some(0.0));
    }
    let meta = Meta {
        model: "test/tiny".into(),
        revision: "0".into(),
        license: "Apache-2.0".into(),
        pooling,
        normalize: true,
        query_prefix: String::new(),
        passage_prefix: String::new(),
        max_length: 32,
    };
    pack::build(meta, a, quant, group, &tiny_tokenizer(), &src).unwrap()
}

pub fn write_tiny(dir: &Path, quant: Quantization, pooling: Pooling) -> (PathBuf, String) {
    let bytes = tiny_pack_bytes(quant, pooling);
    let path = dir.join(format!("tiny-{}.hmsemb", quant.name()));
    std::fs::write(&path, &bytes).unwrap();
    (path, sha256_hex(&bytes))
}
