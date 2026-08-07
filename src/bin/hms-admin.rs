// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use clap::{Parser, Subcommand};
use holographic_memory::core::admin::{inspect_store, migrate_store};
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
}

fn main() -> Result<()> {
    let output = match Cli::parse().command {
        Command::Inspect { store } => serde_json::to_value(inspect_store(store, false)?)?,
        Command::Verify { store } => serde_json::to_value(inspect_store(store, true)?)?,
        Command::Migrate {
            source,
            destination,
        } => serde_json::to_value(migrate_store(source, destination)?)?,
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
