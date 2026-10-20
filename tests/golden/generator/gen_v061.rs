// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Writes the canonical golden store with the code of the tag this file is copied into
//! (built there as an example): `gen_v061 <out-dir> <corpus.json> <dim> [passphrase]`.

#[path = "ops.rs"]
mod ops;

use holographic_memory::core::HmsConfig;
use holographic_memory::HmsCore;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 4,
        "usage: gen_v061 <out-dir> <corpus.json> <dim> [passphrase]"
    );
    let dim: u32 = args[3].parse()?;
    let mut config = HmsConfig::default();
    if let Some(passphrase) = args.get(4) {
        config.security.encryption_enabled = true;
        config.security.encryption_passphrase = Some(passphrase.clone());
    }
    let corpus = ops::load_corpus(std::path::Path::new(&args[2]))?;
    {
        let core = HmsCore::new(dim, Some(args[1].clone()), Some(config))?;
        ops::populate(&core, corpus, dim as usize, true)?;
    }
    Ok(())
}
