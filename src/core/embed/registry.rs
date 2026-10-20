// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The models this build knows by name. Each entry pins the upstream commit and the sha256 of
//! every upstream file it converts from, and the sha256 of the resulting pack, so a model id
//! always means exactly one set of weights.

use std::path::Path;

use anyhow::{bail, Result};

use super::pack::{convert_hf_dir, Meta, Pooling, Quantization};

#[derive(Debug)]
pub struct PinnedFile {
    pub name: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

#[derive(Debug)]
pub struct ModelSpec {
    /// Short stable id used in options and file names.
    pub id: &'static str,
    /// Upstream Hugging Face repository.
    pub repo: &'static str,
    /// Upstream commit the pins refer to.
    pub revision: &'static str,
    pub license: &'static str,
    pub pooling: Pooling,
    pub query_prefix: &'static str,
    pub passage_prefix: &'static str,
    pub max_length: usize,
    pub quantization: Quantization,
    /// Values per int8 scale (ignored for f16 and f32 packs).
    pub int8_group: usize,
    /// Upstream files the pack is converted from.
    pub files: &'static [PinnedFile],
    /// sha256 of the converted pack.
    pub pack_sha256: &'static str,
    pub pack_bytes: u64,
}

impl ModelSpec {
    pub fn pack_name(&self) -> String {
        format!("{}-{}.hmsemb", self.id, self.quantization.name())
    }

    pub fn find(id: &str) -> Result<&'static ModelSpec> {
        find(id)
    }

    /// Converts the upstream files in `dir` to this model's pack bytes.
    pub fn convert(&self, dir: &Path) -> Result<Vec<u8>> {
        convert_hf_dir(
            dir,
            Meta {
                model: self.repo.into(),
                revision: self.revision.into(),
                license: self.license.into(),
                pooling: self.pooling,
                normalize: true,
                query_prefix: self.query_prefix.into(),
                passage_prefix: self.passage_prefix.into(),
                max_length: self.max_length,
            },
            self.quantization,
            self.int8_group,
        )
    }
}

/// The id `EmbedOptions::default()` selects (chosen by benchmarks/results/embedding_selection_v1.json).
pub const DEFAULT_MODEL: &str = "granite-embedding-30m-english";

/// Every model this build knows. Pins come from `benchmarks/embedding/pin_model.py`.
pub static MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "gte-small",
        repo: "thenlper/gte-small",
        revision: "17e1f347d17fe144873b1201da91788898c639cd",
        license: "MIT",
        pooling: Pooling::Mean,
        query_prefix: "",
        passage_prefix: "",
        max_length: 512,
        quantization: Quantization::Int8,
        int8_group: 64,
        files: &[
            PinnedFile {
                name: "config.json",
                sha256: "00273cdfb3e05c7304d180270bdef8adae45023e7bdfc396338bdd815943dc59",
                size: 583,
            },
            PinnedFile {
                name: "tokenizer.json",
                sha256: "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
                size: 711661,
            },
            PinnedFile {
                name: "model.safetensors",
                sha256: "9a1eb90bbac323ea08aa5629b624fe6ae75db121b904799c2266a1e2c2de22d2",
                size: 66746168,
            },
        ],
        pack_sha256: "402cf8ba3c43a65ff2ccf414665644213ed37f7a53396736ef343bc2847c5401",
        pack_bytes: 36768634,
    },
    ModelSpec {
        id: "bge-small-en-v1.5",
        repo: "BAAI/bge-small-en-v1.5",
        revision: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        license: "MIT",
        pooling: Pooling::Cls,
        query_prefix: "Represent this sentence for searching relevant passages: ",
        passage_prefix: "",
        max_length: 512,
        quantization: Quantization::Int8,
        int8_group: 64,
        files: &[
            PinnedFile {
                name: "config.json",
                sha256: "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
                size: 743,
            },
            PinnedFile {
                name: "tokenizer.json",
                sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
                size: 711396,
            },
            PinnedFile {
                name: "model.safetensors",
                sha256: "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
                size: 133466304,
            },
        ],
        pack_sha256: "e9597f7d16a22f1b17cbd4ead05abcb254a1240b3ec7e33bed953415e1924f86",
        pack_bytes: 36768429,
    },
    ModelSpec {
        id: "granite-embedding-30m-english",
        repo: "ibm-granite/granite-embedding-30m-english",
        revision: "9b5b096411652ec1189c68fcfb90d0a82c5b45af",
        license: "Apache-2.0",
        pooling: Pooling::Cls,
        query_prefix: "",
        passage_prefix: "",
        max_length: 512,
        quantization: Quantization::Int8,
        int8_group: 64,
        files: &[
            PinnedFile {
                name: "config.json",
                sha256: "875a866ab8bd1bdf0aa304977ce49c03c97323546ea67cb24110c00214cff984",
                size: 683,
            },
            PinnedFile {
                name: "tokenizer.json",
                sha256: "33465117406b9007673e8ba283f7f1383d9b5094df947481af60eec94ed7d7bd",
                size: 1355881,
            },
            PinnedFile {
                name: "model.safetensors",
                sha256: "3b1fcdc9c5eb954f603bc386474e321505ff29c6c67f21e3aa8db3d2d1a533cf",
                size: 60602024,
            },
        ],
        pack_sha256: "6a1cc24734a57e4524ecdab1466b2c14535a059b393c76c418ca458ec29602dd",
        pack_bytes: 34065777,
    },
    ModelSpec {
        id: "all-minilm-l6-v2",
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        revision: "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        license: "Apache-2.0",
        pooling: Pooling::Mean,
        query_prefix: "",
        passage_prefix: "",
        max_length: 256,
        quantization: Quantization::Int8,
        int8_group: 64,
        files: &[
            PinnedFile {
                name: "config.json",
                sha256: "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
                size: 612,
            },
            PinnedFile {
                name: "tokenizer.json",
                sha256: "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
                size: 466247,
            },
            PinnedFile {
                name: "model.safetensors",
                sha256: "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
                size: 90868376,
            },
        ],
        pack_sha256: "4af8c11c2b14be256a86620d43b82d044033f2d769bffac8105c20b6e6377f48",
        pack_bytes: 25117924,
    },
    ModelSpec {
        id: "e5-small-v2",
        repo: "intfloat/e5-small-v2",
        revision: "ffb93f3bd4047442299a41ebb6fa998a38507c52",
        license: "MIT",
        pooling: Pooling::Mean,
        query_prefix: "query: ",
        passage_prefix: "passage: ",
        max_length: 512,
        quantization: Quantization::Int8,
        int8_group: 64,
        files: &[
            PinnedFile {
                name: "config.json",
                sha256: "5dfb0363cd0243be179c03bcaafd1542d0fbb95e8cbcf575fff3e229342adc2f",
                size: 615,
            },
            PinnedFile {
                name: "tokenizer.json",
                sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
                size: 711396,
            },
            PinnedFile {
                name: "model.safetensors",
                sha256: "45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1",
                size: 133466304,
            },
        ],
        pack_sha256: "9ea3519b7dd12c614ec6f5370865d1b2d907415ad3dfdec513c40104afc4d078",
        pack_bytes: 36768387,
    },
];

pub(crate) fn find(id: &str) -> Result<&'static ModelSpec> {
    let id = if id == "default" { DEFAULT_MODEL } else { id };
    match MODELS.iter().find(|m| m.id == id || m.repo == id) {
        Some(m) => Ok(m),
        None => bail!(
            "unknown embedding model {id:?}; known models: {}; or pass the path of a .hmsemb pack",
            MODELS.iter().map(|m| m.id).collect::<Vec<_>>().join(", ")
        ),
    }
}
