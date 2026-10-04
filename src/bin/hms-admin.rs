// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::Result;
use clap::{Parser, Subcommand};
use holographic_memory::core::admin::{inspect_store, migrate_store, reencode_documents};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hms-admin", about = "Inspect, verify, and migrate HMS stores")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Read format and utilization metadata without verifying every checksum.
    Inspect { store: PathBuf },
    /// Scan every arena frame and validate compression and CRC32 checksums.
    Verify { store: PathBuf },
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
    },
}

fn main() -> Result<()> {
    let output = match Cli::parse().command {
        Command::Inspect { store } => serde_json::to_value(inspect_store(store, false)?)?,
        Command::Verify { store } => serde_json::to_value(inspect_store(store, true)?)?,
        Command::Migrate {
            source,
            destination,
        } => serde_json::to_value(migrate_store(source, destination)?)?,
        Command::Reencode {
            input,
            destination,
            dimensions,
            encryption_passphrase_env,
        } => {
            let mut config = holographic_memory::core::HmsConfig::default();
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
            vec!["hms-admin", "inspect", "store"],
            vec!["hms-admin", "verify", "store"],
            vec!["hms-admin", "migrate", "source", "destination"],
        ] {
            Cli::try_parse_from(args).expect("valid administration command");
        }
    }
}
