// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `warm-embed`: build, fetch, verify and run local embedding model packs.
//!
//! Air-gapped installs: run `warm-embed fetch <id> --out DIR` on a connected machine, copy DIR,
//! and set `WARM_EMBED_MODEL_DIR=DIR` (or `WARM_EMBED_PACK=DIR/<pack>`) on the offline machine.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use writerslogic_warm::core::embed::{
    convert_hf_dir, sha256_file, EmbedKind, EmbedOptions, Embedder, Meta, Pooling, Quantization,
    DEFAULT_INT8_GROUP, MODELS,
};

#[derive(Parser)]
#[command(
    name = "warm-embed",
    about = "Local embedding model packs for writerslogic-warm"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List registry models.
    List,
    /// Convert a Hugging Face directory (config.json, tokenizer.json, model.safetensors) to a pack.
    Pack {
        #[arg(long)]
        from_dir: PathBuf,
        #[arg(long)]
        model: String,
        #[arg(long)]
        revision: String,
        #[arg(long)]
        license: String,
        #[arg(long, default_value = "mean")]
        pooling: Pooling,
        #[arg(long, default_value = "int8")]
        quantization: Quantization,
        /// Values per int8 scale (a divisor of every matrix row length).
        #[arg(long, default_value_t = DEFAULT_INT8_GROUP)]
        int8_group: usize,
        #[arg(long, default_value = "")]
        query_prefix: String,
        #[arg(long, default_value = "")]
        passage_prefix: String,
        #[arg(long, default_value_t = 512)]
        max_length: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Fetch a registry model's pinned upstream files, convert them and write the pack into DIR.
    Fetch {
        id: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Check a pack's structure and, with --sha256, its hash.
    Verify {
        pack: PathBuf,
        #[arg(long)]
        sha256: Option<String>,
    },
    /// Embed lines of text from stdin (or --input) and write one JSON array per line.
    Embed {
        #[arg(long, default_value = "default")]
        model: String,
        #[arg(long)]
        query: bool,
        /// Input file; `-` or absent reads stdin.
        #[arg(long)]
        input: Option<PathBuf>,
        /// Each input line is a JSON string (texts containing newlines).
        #[arg(long)]
        json_lines: bool,
        #[arg(long, default_value_t = 0)]
        threads: usize,
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        #[arg(long)]
        max_length: Option<usize>,
        /// Write raw little-endian f32 rows here instead of JSON lines.
        #[arg(long)]
        raw_out: Option<PathBuf>,
        /// Print load and embed timings to stderr.
        #[arg(long)]
        timings: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::List => {
            for m in MODELS {
                println!("{}\t{}\t{}\t{}", m.id, m.repo, m.license, m.pack_name());
            }
        }
        Command::Pack {
            from_dir,
            model,
            revision,
            license,
            pooling,
            quantization,
            int8_group,
            query_prefix,
            passage_prefix,
            max_length,
            out,
        } => {
            let meta = Meta {
                model,
                revision,
                license,
                pooling,
                normalize: true,
                query_prefix,
                passage_prefix,
                max_length,
            };
            let bytes = convert_hf_dir(&from_dir, meta, quantization, int8_group)?;
            std::fs::write(&out, &bytes).with_context(|| format!("write {}", out.display()))?;
            println!(
                "{}  {}  {} bytes",
                writerslogic_warm::core::embed::sha256_hex(&bytes),
                out.display(),
                bytes.len()
            );
        }
        Command::Fetch { id, out } => fetch(&id, &out)?,
        Command::Verify { pack, sha256 } => {
            let got = sha256_file(&pack)?;
            if let Some(want) = sha256 {
                if !got.eq_ignore_ascii_case(&want) {
                    bail!("sha256 mismatch: expected {want}, file is {got}");
                }
            }
            let e = Embedder::from_pack(&pack, Some(&got), &EmbedOptions::default())?;
            println!(
                "ok {got} {} rev {} {} dim {}",
                e.model(),
                e.revision(),
                e.quantization().name(),
                e.dimensions()
            );
        }
        Command::Embed {
            model,
            query,
            input,
            json_lines,
            threads,
            batch_size,
            max_length,
            raw_out,
            timings,
        } => {
            let t0 = Instant::now();
            let e = Embedder::new(&EmbedOptions {
                model,
                threads,
                batch_size,
                max_length,
                ..EmbedOptions::default()
            })?;
            let load = t0.elapsed();
            let lines: Vec<String> = match input.filter(|p| p.as_os_str() != "-") {
                Some(p) => std::io::BufReader::new(std::fs::File::open(p)?)
                    .lines()
                    .collect::<Result<_, _>>()?,
                None => std::io::stdin().lock().lines().collect::<Result<_, _>>()?,
            };
            let lines: Vec<String> = if json_lines {
                lines
                    .iter()
                    .map(|l| serde_json::from_str(l).context("input line is not a JSON string"))
                    .collect::<Result<_>>()?
            } else {
                lines
            };
            let kind = if query {
                EmbedKind::Query
            } else {
                EmbedKind::Passage
            };
            let t1 = Instant::now();
            let vectors = e.embed(&lines, kind)?;
            let run = t1.elapsed();
            if timings {
                eprintln!(
                    "{{\"load_ms\":{:.1},\"embed_ms\":{:.1},\"texts\":{},\"ms_per_text\":{:.3},\"threads\":{}}}",
                    load.as_secs_f64() * 1e3,
                    run.as_secs_f64() * 1e3,
                    lines.len(),
                    run.as_secs_f64() * 1e3 / lines.len().max(1) as f64,
                    e.threads()
                );
            }
            match raw_out {
                Some(p) => {
                    let mut w = std::io::BufWriter::new(std::fs::File::create(p)?);
                    for v in &vectors {
                        for x in v {
                            w.write_all(&x.to_le_bytes())?;
                        }
                    }
                    w.flush()?;
                }
                None => {
                    let mut w = std::io::BufWriter::new(std::io::stdout().lock());
                    for v in &vectors {
                        writeln!(w, "{}", serde_json::to_string(v)?)?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "local-embed-download")]
fn fetch(id: &str, out: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let spec = writerslogic_warm::core::embed::ModelSpec::find(id)?;
    let path = writerslogic_warm::core::embed::fetch_pack(spec, out)?;
    println!("{}  {}", sha256_file(&path)?, path.display());
    Ok(())
}

#[cfg(not(feature = "local-embed-download"))]
fn fetch(_id: &str, _out: &std::path::Path) -> Result<()> {
    bail!("this build has no downloader: rebuild with --features local-embed-download")
}
