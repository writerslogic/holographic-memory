// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use writerslogic_warm::core::admin::{
    inspect_store, migrate_store, reencode_documents, verify_store,
};

#[derive(Parser)]
#[command(
    name = "warm-admin",
    about = "Inspect, verify, and migrate WARM stores"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Read format and utilization metadata without verifying every checksum.
    Inspect { store: PathBuf },
    /// Scan every arena frame (compression and CRC32), verify the manifest checksums, and
    /// open the store to confirm every logged document resolves.
    Verify {
        store: PathBuf,
        /// Environment variable holding the passphrase of an encrypted store.
        #[arg(long)]
        encryption_passphrase_env: Option<String>,
    },
    /// Create a locked, verified, atomically published current-format copy.
    Migrate {
        source: PathBuf,
        destination: PathBuf,
    },
    /// Re-encode original DocumentInput JSONL into a new store; never modifies the source.
    Reencode {
        input: PathBuf,
        destination: PathBuf,
        #[arg(long, default_value_t = 16384)]
        dimensions: u32,
        #[arg(long)]
        encryption_passphrase_env: Option<String>,
        /// Text encoder for the new store; capped union requires --features experimental.
        #[arg(long, default_value = "multiscale-words-v1")]
        text_encoder: writerslogic_warm::core::encoding::TextEncoder,
    },
}

fn main() -> Result<()> {
    let output = match Cli::parse().command {
        Command::Inspect { store } => serde_json::to_value(inspect_store(store, false)?)?,
        Command::Verify {
            store,
            encryption_passphrase_env,
        } => serde_json::to_value(verify_store(store, encryption_passphrase_env)?)?,
        Command::Migrate {
            source,
            destination,
        } => serde_json::to_value(migrate_store(source, destination)?)?,
        Command::Reencode {
            input,
            destination,
            dimensions,
            encryption_passphrase_env,
            text_encoder,
        } => {
            let mut config = writerslogic_warm::core::WarmConfig {
                text_encoder,
                ..Default::default()
            };
            config.security.encryption_enabled = encryption_passphrase_env.is_some();
            config.security.encryption_passphrase_env = encryption_passphrase_env;
            serde_json::to_value(reencode_documents(input, destination, dimensions, config)?)?
        }
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_subcommand() {
        for args in [
            vec!["warm-admin", "inspect", "store"],
            vec!["warm-admin", "verify", "store"],
            vec![
                "warm-admin",
                "verify",
                "store",
                "--encryption-passphrase-env",
                "WARM_PASS",
            ],
            vec!["warm-admin", "migrate", "source", "destination"],
        ] {
            Cli::try_parse_from(args).expect("valid administration command");
        }
    }
}
