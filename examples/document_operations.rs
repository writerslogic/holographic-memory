// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Barrier,
    },
    time::Instant,
};
use writerslogic_warm::{DocumentInput, SearchOptions, WarmCore};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 2049)]
    documents: usize,
    #[arg(long, default_value_t = 4)]
    readers: usize,
    #[arg(long)]
    out: PathBuf,
}

fn input(id: usize, version: usize) -> Result<DocumentInput> {
    Ok(serde_json::from_value(json!({
        "id":format!("d{id:08}"), "text":format!("common token{id} revision{version}"),
        "sourceUri":format!("fixture:source:{}", id % 8),
        "metadata":{"group":id % 8}, "version":version.to_string()
    }))?)
}

fn query(warm: &WarmCore, origin: Instant, reader: usize, sequence: usize) -> Result<Value> {
    let group = (reader + sequence) % 8;
    let options = SearchOptions {
        filter: Some(json!({"group":group})),
        source_uri: Some(format!("fixture:source:{group}")),
        ..Default::default()
    };
    let start = origin.elapsed().as_nanos();
    let results = warm.search_documents("common", &options)?;
    let end = origin.elapsed().as_nanos();
    Ok(json!({"reader":reader,"sequence":sequence,"group":group,
        "start_ns":start,"end_ns":end,"results":results}))
}

fn phase(warm: &WarmCore, origin: Instant, args: &Args, name: &str) -> Result<Value> {
    let barrier = Barrier::new(args.readers + 1);
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| -> Result<Value> {
        let mut workers = Vec::new();
        for reader in 0..args.readers {
            let barrier = &barrier;
            let done = &done;
            workers.push(scope.spawn(move || -> Result<Vec<Value>> {
                barrier.wait();
                let mut rows = Vec::new();
                let budget = if name == "compact" { 512 } else { 10000 };
                for sequence in 0..budget {
                    rows.push(query(warm, origin, reader, sequence)?);
                    if (name == "compact" && sequence + 1 == budget)
                        || (name != "compact" && sequence >= 127 && done.load(Ordering::Acquire))
                    {
                        return Ok(rows);
                    }
                }
                anyhow::bail!("reader exhausted bounded query budget before writer completed")
            }));
        }
        barrier.wait();
        let mutations = (|| -> Result<Vec<Value>> {
            let mut rows = Vec::new();
            let count = match name {
                "replace" => 64,
                "delete" => args.documents / 2 + 1,
                "compact" => 1,
                _ => anyhow::bail!("unknown phase"),
            };
            for id in 0..count {
                let start = origin.elapsed().as_nanos();
                match name {
                    "replace" => {
                        warm.memorize_document(input(id, 2)?)?;
                    }
                    "delete" => {
                        ensure!(
                            warm.delete_document(&format!("d{id:08}"))?,
                            "missing delete target"
                        );
                    }
                    "compact" => warm.compact()?,
                    _ => unreachable!(),
                }
                rows.push(json!({"id":id,"start_ns":start,"end_ns":origin.elapsed().as_nanos()}));
            }
            Ok(rows)
        })();
        done.store(true, Ordering::Release);
        let mut queries = Vec::new();
        for worker in workers {
            queries.extend(worker.join().expect("query worker panicked")?);
        }
        Ok(json!({"phase":name,"queries":queries,"operations":mutations?}))
    })
}

fn snapshot(warm: &WarmCore, n: usize) -> Result<Vec<Value>> {
    (0..n)
        .map(|id| {
            let options = SearchOptions {
                document_ids: Some(vec![format!("d{id:08}")]),
                ..Default::default()
            };
            Ok(json!({"id":id,"results":warm.search_documents("common", &options)?}))
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1023..=10000).contains(&args.documents),
        "documents outside 1023..=10000"
    );
    ensure!((1..=8).contains(&args.readers), "readers outside 1..=8");
    ensure!(!args.out.exists(), "refusing to overwrite evidence");
    let directory = tempfile::tempdir()?;
    let path = directory.path().display().to_string();
    let warm = WarmCore::new(1024, Some(path.clone()), None)?;
    let origin = Instant::now();
    for id in 0..args.documents {
        warm.memorize_document(input(id, 1)?)?;
    }
    let ingest_ns = origin.elapsed().as_nanos();
    let mut static_queries = Vec::new();
    for sequence in 0..512 {
        static_queries.push(query(&warm, origin, 0, sequence)?);
    }
    let mut phases = Vec::new();
    for name in ["replace", "delete", "compact"] {
        phases.push(phase(&warm, origin, &args, name)?);
    }
    let before_restart = snapshot(&warm, args.documents)?;
    let mut repeat_compactions_ns = Vec::new();
    for _ in 0..2 {
        let start = Instant::now();
        warm.compact()?;
        repeat_compactions_ns.push(start.elapsed().as_nanos());
    }
    drop(warm);
    let start = Instant::now();
    let restored = WarmCore::new(1024, Some(path), None)?;
    let restart_ns = start.elapsed().as_nanos();
    let after_restart = snapshot(&restored, args.documents)?;
    ensure!(
        before_restart == after_restart,
        "restart changed query outputs"
    );
    std::fs::write(
        &args.out,
        serde_json::to_vec(&json!({
            "schema_version":1,"documents":args.documents,"readers":args.readers,
            "synthetic":true,"timing_claim_eligible":false,"ingest_ns":ingest_ns,
            "static_queries":static_queries,"phases":phases,"restart_ns":restart_ns,
            "before_restart":before_restart,"after_restart":after_restart,
            "repeat_compactions_ns":repeat_compactions_ns
        }))?,
    )?;
    Ok(())
}
