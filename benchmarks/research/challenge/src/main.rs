// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Command line of the challenge package: `generate`, `run`, `audit`. See `RUN.md`.
//!
//! Exit codes: 0 PASS (or a non-judging command finished), 1 FAIL, 2 error.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;

use warm_research_challenge::audit;
use warm_research_challenge::eval::{self, Results, RunOptions};
use warm_research_challenge::store::DEFAULT_DIMENSIONS;
use warm_research_challenge::synth::{self, GenConfig, Split, FROZEN_SEED};

const USAGE: &str = "\
warm-research-challenge <command> [options]

commands
  generate  --out FILE [--split tune|test] [--families N] [--conflict-fraction F] [--seed S]
            write the dataset (families, records, queries with gold) as JSON
  run       --out FILE [--split tune|test] [--families N] [--conflict-fraction F] [--seed S]
            [--repeats R] [--warmup W] [--dims D] [--store-dir DIR] [--wipe-sample N]
            ingest, evaluate every method, check erasure, write results JSON, print PASS/FAIL
  audit     (--trace FILE.jsonl | --synthetic --examples FILE.jsonl) --out FILE
            report the fraction of queries carrying a usable (entity, attribute) key

defaults: split test, families 1000, conflict-fraction 0.2, seed 0x5EEDC0DE20260001,
          repeats 5, warmup 1, dims 1024, wipe-sample 10, temporary store directory";

const SWITCHES: [&str; 2] = ["synthetic", "help"];

struct Args {
    command: String,
    values: BTreeMap<String, String>,
    switches: BTreeSet<String>,
}

impl Args {
    fn parse(raw: Vec<String>) -> Result<Self> {
        let mut raw = raw.into_iter();
        let command = raw.next().unwrap_or_else(|| "help".to_owned());
        let mut values = BTreeMap::new();
        let mut switches = BTreeSet::new();
        while let Some(arg) = raw.next() {
            let Some(name) = arg.strip_prefix("--") else {
                bail!("unexpected argument {arg:?}\n{USAGE}");
            };
            if SWITCHES.contains(&name) {
                switches.insert(name.to_owned());
                continue;
            }
            let value = raw
                .next()
                .ok_or_else(|| anyhow!("--{name} needs a value\n{USAGE}"))?;
            values.insert(name.to_owned(), value);
        }
        Ok(Self {
            command,
            values,
            switches,
        })
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    fn required(&self, name: &str) -> Result<&str> {
        self.get(name)
            .ok_or_else(|| anyhow!("--{name} is required\n{USAGE}"))
    }

    fn parsed<T>(&self, name: &str, default: T) -> Result<T>
    where
        T: FromStr,
        T::Err: Display,
    {
        match self.get(name) {
            None => Ok(default),
            Some(value) => value
                .parse::<T>()
                .map_err(|error| anyhow!("--{name} {value:?}: {error}")),
        }
    }

    fn seed(&self) -> Result<u64> {
        match self.get("seed") {
            None => Ok(FROZEN_SEED),
            Some(value) => {
                let parsed = match value.strip_prefix("0x") {
                    Some(hex) => u64::from_str_radix(hex, 16),
                    None => value.parse::<u64>(),
                };
                parsed.map_err(|error| anyhow!("--seed {value:?}: {error}"))
            }
        }
    }

    fn switch(&self, name: &str) -> bool {
        self.switches.contains(name)
    }
}

fn gen_config(args: &Args) -> Result<GenConfig> {
    let split_name = args.get("split").unwrap_or("test");
    let split = Split::parse(split_name)
        .ok_or_else(|| anyhow!("--split must be tune or test, got {split_name:?}"))?;
    let conflict_fraction: f64 = args.parsed("conflict-fraction", 0.2)?;
    if !(0.0..=1.0).contains(&conflict_fraction) {
        bail!("--conflict-fraction must be within [0, 1]");
    }
    Ok(GenConfig {
        families: args.parsed("families", 1000)?,
        conflict_fraction,
        split,
        seed: args.seed()?,
    })
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("cannot create {}", parent.display()))?;
        }
    }
    let json = serde_json::to_vec_pretty(value).context("cannot serialize the output")?;
    std::fs::write(path, json).with_context(|| format!("cannot write {}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

fn print_results(results: &Results) {
    println!(
        "dataset: {} families ({} conflict), {} records, {} queries, split {}",
        results.dataset.families,
        results.dataset.conflict_families,
        results.dataset.records,
        results.dataset.queries,
        results.split
    );
    println!(
        "ingest: {} facts in {:.0} ms; arena: {} families in {:.0} ms",
        results.ingest.facts_created,
        results.ingest.elapsed_ms,
        results.arena.families,
        results.arena.elapsed_ms
    );
    println!(
        "{:<36} {:>8} {:>8} {:>7} {:>8} {:>10} {:>10}",
        "method", "correct", "ans-only", "stale", "miss-exc", "p50 us", "p95 us"
    );
    for name in warm_research_challenge::baselines::METHODS {
        if let Some(report) = results.methods.get(name) {
            println!(
                "{:<36} {:>7.1}% {:>7.1}% {:>6.1}% {:>7.1}% {:>10.1} {:>10.1}",
                name,
                report.correctness * 100.0,
                report.answer_only_accuracy * 100.0,
                report.stale_answer_rate * 100.0,
                report.missed_exception_rate * 100.0,
                report.latency.p50_us,
                report.latency.p95_us
            );
        }
    }
    println!(
        "challenge: support overridden on {:.1}% of queries; statuses {:?}; defeats {:?}",
        results.challenge.support_overridden_rate * 100.0,
        results.challenge.status_counts,
        results.challenge.defeat_counts
    );
    println!(
        "erase: {} families wiped, verified {}, p95 {:.1} us{}",
        results.erase.families_wiped,
        results.erase.verified,
        results.erase.wipe_p95_us,
        if results.erase.failures.is_empty() {
            String::new()
        } else {
            format!("; failures: {}", results.erase.failures.join(" | "))
        }
    );
    println!("sqlite baseline: {}", results.sqlite_baseline);
    println!("{}", results.verdict.summary);
}

fn run() -> Result<i32> {
    let args = Args::parse(std::env::args().skip(1).collect())?;
    if args.switch("help") {
        println!("{USAGE}");
        return Ok(0);
    }
    match args.command.as_str() {
        "generate" => {
            let out = PathBuf::from(args.required("out")?);
            let config = gen_config(&args)?;
            let dataset = synth::generate(&config);
            println!(
                "generated {} families ({} conflict), {} records, {} queries",
                dataset.families.len(),
                dataset.conflict_families(),
                dataset.records(),
                dataset.queries.len()
            );
            write_json(&out, &dataset)?;
            Ok(0)
        }
        "run" => {
            let out = PathBuf::from(args.required("out")?);
            let options = RunOptions {
                gen: gen_config(&args)?,
                repeats: args.parsed("repeats", 5)?,
                warmup: args.parsed("warmup", 1)?,
                store_dir: args.get("store-dir").map(PathBuf::from),
                dimensions: args.parsed("dims", DEFAULT_DIMENSIONS)?,
                wipe_sample: args.parsed("wipe-sample", 10)?,
            };
            let results = eval::run_experiment(&options)?;
            write_json(&out, &results)?;
            print_results(&results);
            Ok(if results.verdict.pass { 0 } else { 1 })
        }
        "audit" => {
            let out = PathBuf::from(args.required("out")?);
            let report = if args.switch("synthetic") {
                let examples = PathBuf::from(args.required("examples")?);
                let queries = audit::synthetic_examples(args.seed()?);
                audit::write_jsonl(&examples, &queries)?;
                println!("wrote {} synthetic queries to {}", queries.len(), examples.display());
                audit::audit_file(&examples, true)?
            } else {
                let trace = PathBuf::from(args.required("trace")?);
                audit::audit_file(&trace, false)?
            };
            write_json(&out, &report)?;
            println!(
                "audit: {} queries, {} keyed by metadata, {} keyed by rule, {} unkeyed; usable {:.1}% (threshold {:.0}%) -> {}{}",
                report.queries,
                report.keyed_by_metadata,
                report.keyed_by_rule,
                report.unkeyed,
                report.usable_fraction * 100.0,
                report.threshold * 100.0,
                if report.pass { "PASS" } else { "FAIL" },
                if report.synthetic {
                    " (SYNTHETIC TRACE: not evidence; replace with a real agent trace)"
                } else {
                    ""
                }
            );
            Ok(if report.pass { 0 } else { 1 })
        }
        "help" | "-h" | "--help" => {
            println!("{USAGE}");
            Ok(0)
        }
        other => bail!("unknown command {other:?}\n{USAGE}"),
    }
}

fn main() {
    let code = match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            2
        }
    };
    std::process::exit(code);
}
