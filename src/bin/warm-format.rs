// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! `warm-format`: validates and upgrades the persisted format of an WARM store, offline.
//!
//! Exit codes: 0 verified, 10 the store opens after the engine repairs it (torn tail, bad
//! cache), 20 committed data or structure is damaged, 30 the store was written by a newer
//! format or is not openable by this build, 64 usage error, 74 the path could not be read.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};
use writerslogic_warm::core::format::{
    self, plan_upgrade, scan, upgrade_store, verify_path, Severity, VerifyOptions,
};

#[derive(Parser)]
#[command(
    name = "warm-format",
    about = "Verify and upgrade the persisted format of an WARM store, offline",
    after_help = "Exit codes: 0 ok, 10 recoverable, 20 corrupt, 30 unsupported version, 64 usage, 74 I/O"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate a store (or a server data directory) without modifying it.
    Verify {
        path: PathBuf,
        /// Environment variable holding the passphrase of an encrypted store. Without it the
        /// frames of an encrypted store are checked for framing and CRC32 only.
        #[arg(long)]
        passphrase_env: Option<String>,
        /// Also open a private copy with the engine and compare it with the journal.
        #[arg(long)]
        deep: bool,
        /// Print the report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// List what an upgrade would do, without writing anything.
    Plan {
        path: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Write an upgraded copy of a store to a new directory; the source is never modified.
    Upgrade {
        source: PathBuf,
        destination: PathBuf,
        /// Plan and verify the source only.
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        json: bool,
    },
    /// Print the format versions this build reads and writes.
    Versions,
    /// Print the frames of a store: segment, offset, lengths, CRC32, sequence number and,
    /// for plaintext stores, the mutation kinds in each.
    Dump {
        path: PathBuf,
        /// Print the decompressed payload of frame N (0-based, across segments) and exit.
        #[arg(long)]
        frame: Option<usize>,
    },
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let code = if error.use_stderr() { 64 } else { 0 };
            let _ = error.print();
            return ExitCode::from(code);
        }
    };
    match run(cli.command) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("warm-format: {error:#}");
            ExitCode::from(74)
        }
    }
}

fn run(command: Command) -> Result<u8> {
    match command {
        Command::Verify {
            path,
            passphrase_env,
            deep,
            json,
        } => {
            let report = verify_path(
                &path,
                &VerifyOptions {
                    passphrase_env,
                    deep,
                },
            )?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_report(&report, 0);
            }
            Ok(report.exit_code as u8)
        }
        Command::Plan { path, json } => {
            let plan = plan_upgrade(&path)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan)?);
            } else {
                print_plan(&plan);
            }
            Ok(0)
        }
        Command::Upgrade {
            source,
            destination,
            dry_run,
            json,
        } => {
            let report = upgrade_store(&source, &destination, dry_run)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print_plan(&report.plan);
                match (&report.destination, dry_run) {
                    (Some(destination), _) => {
                        println!("upgraded {} frames into {destination}", report.frames)
                    }
                    (None, true) => println!("dry run: nothing written"),
                    (None, false) => println!("nothing to do"),
                }
            }
            Ok(0)
        }
        Command::Versions => {
            println!("{}", serde_json::to_string_pretty(&format::CURRENT)?);
            Ok(0)
        }
        Command::Dump { path, frame } => {
            dump(&path, frame)?;
            Ok(0)
        }
    }
}

fn print_plan(plan: &format::UpgradePlan) {
    println!(
        "{}: storage format {} (target {}), store schema {} (target {})",
        plan.source,
        plan.storage_format,
        plan.target_storage_format,
        plan.store_schema,
        plan.target_store_schema
    );
    if plan.up_to_date {
        println!("up to date");
    }
    for step in &plan.steps {
        println!(
            "  {} ({} {} -> {}): {}",
            step.id, step.artefact, step.from, step.to, step.description
        );
    }
}

fn print_report(report: &format::Report, depth: usize) {
    let pad = "  ".repeat(depth);
    println!(
        "{pad}{}: {:?} (exit {})",
        report.path, report.verdict, report.exit_code
    );
    let v = &report.versions;
    println!(
        "{pad}  storage format {:?}, schema {:?}, encoder {:?}, encrypted {:?}, dimensions {:?}",
        v.storage_format, v.store_schema, v.text_encoder, v.encrypted, v.dimensions
    );
    println!(
        "{pad}  {} segments, {} frames, {} mutations, documents {:?}, vectors {:?}, stamped {:?}",
        report.segments,
        report.frames,
        report.mutations,
        report.live_documents,
        report.live_vectors,
        report.memory_stamped_documents
    );
    if !report.checks.is_empty() {
        println!("{pad}  passed: {}", report.checks.join(", "));
    }
    for finding in &report.findings {
        let tag = match finding.severity {
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Error => "ERROR",
            Severity::Unsupported => "UNSUPPORTED",
        };
        let at = finding.offset.map(|o| format!("@{o}")).unwrap_or_default();
        println!(
            "{pad}  [{tag}] {} {}{at}: {}",
            finding.code, finding.file, finding.message
        );
    }
    for tenant in &report.tenants {
        println!("{pad}  tenant {}", tenant.name);
        print_report(&tenant.report, depth + 2);
    }
}

fn dump(path: &std::path::Path, only: Option<usize>) -> Result<()> {
    let arena = path.join("vectors_data.bin");
    let generation = writerslogic_warm::core::format::verify::resolve_generation(&arena)?;
    let mut index = 0usize;
    for segment in 0.. {
        let file = generation.join(format!("seg_{segment}.bin"));
        if !file.exists() {
            break;
        }
        let map = std::fs::read(&file)?;
        let scan = scan::scan_segment(&map, true, |frame, payload| {
            if only.is_none_or(|n| n == index) {
                if only.is_some() {
                    match scan::transaction_json(payload) {
                        Some(Ok(json)) => println!("{}", String::from_utf8_lossy(&json)),
                        _ => println!("<{} bytes, not a plaintext transaction>", payload.len()),
                    }
                } else {
                    let kinds = match scan::transaction_json(payload) {
                        Some(Ok(json)) => serde_json::from_slice::<Vec<serde_json::Value>>(&json)
                            .map(|ms| {
                                ms.iter()
                                    .filter_map(|m| {
                                        m.as_object().and_then(|o| o.keys().next().cloned())
                                    })
                                    .collect::<Vec<_>>()
                                    .join(",")
                            })
                            .unwrap_or_default(),
                        _ => "(opaque)".to_owned(),
                    };
                    println!(
                        "frame {index} seg {segment} @{} raw {} stored {} crc {:08x} seq {} {kinds}",
                        frame.offset, frame.raw_len, frame.comp_len, frame.crc, frame.seq
                    );
                }
            }
            index += 1;
        });
        if only.is_none() {
            println!(
                "segment {segment} ends: {:?} at {}",
                scan.end, scan.end_offset
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_subcommand() {
        for args in [
            vec!["warm-format", "verify", "store"],
            vec![
                "warm-format",
                "verify",
                "store",
                "--deep",
                "--json",
                "--passphrase-env",
                "P",
            ],
            vec!["warm-format", "plan", "store"],
            vec!["warm-format", "upgrade", "a", "b", "--dry-run"],
            vec!["warm-format", "versions"],
            vec!["warm-format", "dump", "store", "--frame", "3"],
        ] {
            Cli::try_parse_from(args).expect("valid command");
        }
    }
}
