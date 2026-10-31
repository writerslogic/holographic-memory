// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use clap::Parser;
use writerslogic_warm::{
    core::{engine::diagnostic, EmbeddingSpace, WarmConfig},
    DocumentInput, WarmCore, SearchOptions,
};
use serde_json::{json, Value};
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{mpsc, Barrier},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const READERS: usize = 4;
const HOT: usize = 128;
const ARRIVALS: usize = 4800;
const INTERVAL_NS: u64 = 1_666_667;
const QUEUE: usize = 64;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    seed: bool,
    #[arg(long)]
    observe: bool,
    #[arg(long)]
    documents: usize,
    #[arg(long)]
    batch: usize,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    store: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

fn create(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn write(writer: &mut impl Write, row: &Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, row)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn event(phase: &str) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    write(
        &mut stdout,
        &json!({"phase":phase,"epoch_ns":SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()}),
    )?;
    stdout.flush()?;
    Ok(())
}

fn embedding(pair: usize, revision: usize) -> Vec<f64> {
    let mut vector = vec![0.0; 128];
    vector[(pair + revision) % 128] = 1.0;
    vector
}

fn changed(original: &DocumentInput, revision: usize, number: usize) -> DocumentInput {
    let mut input = original.clone();
    input.text = format!("{} revision{revision}", original.text);
    input.version = Some(revision.to_string());
    input.source_uri = Some(format!(
        "{}:revision{revision}",
        original.source_uri.as_ref().unwrap()
    ));
    input.metadata.as_mut().unwrap()["revision"] = json!(revision);
    input.embeddings = Some(vec![embedding(number / 2, revision)]);
    input
}

fn query(core: &WarmCore, ids: Vec<String>, pair: usize, origin: Instant) -> Result<Value> {
    let count = ids.len() as u32;
    let options = SearchOptions {
        k: Some(count),
        candidate_limit: Some(count),
        document_ids: Some(ids),
        embedding: Some(embedding(pair, 0)),
        ..Default::default()
    };
    let start = origin.elapsed().as_nanos();
    let results = core.search_documents("hmsprobe", &options)?;
    let end = origin.elapsed().as_nanos();
    Ok(json!({"start_ns":start,"end_ns":end,"results":results}))
}

fn snapshot(core: &WarmCore, n: usize, origin: Instant, path: &Path) -> Result<()> {
    let mut output = create(path)?;
    for number in 0..n {
        let mut row = query(core, vec![format!("d{number:08}")], number / 2, origin)?;
        row["id"] = json!(number);
        write(&mut output, &row)?;
    }
    output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!([5182, 20728].contains(&args.documents), "unsupported size");
    ensure!([2, 64].contains(&args.batch), "unsupported batch");
    ensure!(!args.out.exists(), "refusing to overwrite evidence");
    let documents: Vec<DocumentInput> = serde_json::from_reader(File::open(&args.input)?)?;
    ensure!(
        documents.len() == if args.seed { args.documents } else { HOT },
        "fixture size mismatch"
    );
    std::fs::create_dir(&args.out)?;
    let config = WarmConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "synthetic-one-hot-control".into(),
            revision: "1".into(),
            dimensions: 128,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    };
    if args.seed {
        ensure!(!args.store.exists(), "refusing to overwrite store");
        let core = WarmCore::new(1024, Some(args.store.display().to_string()), Some(config))?;
        for batch in documents.chunks(64) {
            core.memorize_documents(batch.to_vec())?;
        }
        core.flush()?;
        return Ok(());
    }
    ensure!(args.store.exists(), "missing seeded store");
    let origin = Instant::now();
    let core = WarmCore::new(
        1024,
        Some(args.store.display().to_string()),
        Some(config.clone()),
    )?;
    for pair in 0..HOT / 2 {
        query(&core, vec![format!("d{:08}", pair * 2)], pair, origin)?;
    }
    event("static")?;
    if args.observe {
        let mut signal = String::new();
        std::io::stdin().read_line(&mut signal)?;
        ensure!(signal.trim() == "start", "missing observer signal");
    }
    diagnostic::enable(origin);
    let operations = if args.batch == 2 { 256 } else { 64 };
    let barrier = Barrier::new(READERS + 3);
    let base = origin.elapsed().as_nanos() as u64 + 20_000_000;
    event("serving")?;
    let (first_send, first_recv) = mpsc::channel();
    let compact = std::thread::scope(|scope| -> Result<Value> {
        let mut workers = Vec::new();
        let mut senders = Vec::new();
        for reader in 0..READERS {
            let (sender, receiver) = mpsc::sync_channel::<usize>(QUEUE);
            senders.push(sender);
            let core = &core;
            let barrier = &barrier;
            let batch = args.batch;
            let path = args.out.join(format!("reader-{reader}.jsonl"));
            workers.push(scope.spawn(move || -> Result<()> {
                let mut output = create(&path)?;
                barrier.wait();
                for sequence in receiver {
                    let pair = (sequence / READERS + reader) % (HOT / 2);
                    let dequeued = origin.elapsed().as_nanos();
                    let first = if batch == 64 && sequence % 16 == 0 {
                        (pair / 32) * 64
                    } else {
                        pair * 2
                    };
                    let count = if batch == 64 && sequence % 16 == 0 {
                        64
                    } else {
                        2
                    };
                    let mut row = query(
                        core,
                        (first..first + count)
                            .map(|id| format!("d{id:08}"))
                            .collect(),
                        pair,
                        origin,
                    )?;
                    row["sequence"] = json!(sequence);
                    row["reader"] = json!(reader);
                    row["pair"] = json!(pair);
                    row["dequeued_ns"] = json!(dequeued);
                    row["scheduled_ns"] = json!(base + sequence as u64 * INTERVAL_NS);
                    write(&mut output, &row)?;
                }
                output.flush()?;
                Ok(())
            }));
        }
        let producer_barrier = &barrier;
        let arrival_path = args.out.join("arrivals.jsonl");
        let producer = scope.spawn(move || -> Result<()> {
            let mut output = create(&arrival_path)?;
            producer_barrier.wait();
            for sequence in 0..ARRIVALS {
                let scheduled = base + sequence as u64 * INTERVAL_NS;
                if let Some(delay) = Duration::from_nanos(scheduled).checked_sub(origin.elapsed()) {
                    std::thread::sleep(delay);
                }
                let offered = origin.elapsed().as_nanos();
                let accepted = match senders[sequence % READERS].try_send(sequence) {
                    Ok(()) => true,
                    Err(mpsc::TrySendError::Full(_)) => false,
                    Err(mpsc::TrySendError::Disconnected(_)) => {
                        anyhow::bail!("reader disconnected")
                    }
                };
                write(
                    &mut output,
                    &json!({"sequence":sequence,"scheduled_ns":scheduled,
                        "offered_ns":offered,"attempt_end_ns":origin.elapsed().as_nanos(),
                        "accepted":accepted}),
                )?;
            }
            drop(senders);
            output.flush()?;
            Ok(())
        });
        let writer_core = &core;
        let writer_barrier = &barrier;
        let batch = args.batch;
        let path = args.out.join("operations.jsonl");
        let writer = scope.spawn(move || -> Result<()> {
            let mut output = create(&path)?;
            writer_barrier.wait();
            for revision in 1..=operations {
                let first = ((revision - 1) * batch) % HOT;
                let inputs = (first..first + batch)
                    .map(|number| changed(&documents[number], revision, number))
                    .collect();
                let start = origin.elapsed().as_nanos();
                writer_core.memorize_documents(inputs)?;
                write(
                    &mut output,
                    &json!({"revision":revision,"first":first,"count":batch,
                        "start_ns":start,"end_ns":origin.elapsed().as_nanos()}),
                )?;
                if revision == 1 {
                    first_send.send(())?;
                }
            }
            output.flush()?;
            Ok(())
        });
        barrier.wait();
        first_recv.recv()?;
        let start = origin.elapsed().as_nanos();
        core.compact()?;
        let end = origin.elapsed().as_nanos();
        producer.join().expect("producer panicked")?;
        writer.join().expect("writer panicked")?;
        for worker in workers {
            worker.join().expect("reader panicked")?;
        }
        Ok(json!({"start_ns":start,"end_ns":end}))
    })?;
    event("verification")?;
    let traces = diagnostic::take();
    let mut trace_output = create(&args.out.join("traces.json"))?;
    serde_json::to_writer(&mut trace_output, &traces)?;
    trace_output.flush()?;
    snapshot(
        &core,
        args.documents,
        origin,
        &args.out.join("before-restart.jsonl"),
    )?;
    core.flush()?;
    drop(core);
    let core = WarmCore::new(1024, Some(args.store.display().to_string()), Some(config))?;
    snapshot(
        &core,
        args.documents,
        origin,
        &args.out.join("after-restart.jsonl"),
    )?;
    let report = json!({"schema_version":1,"documents":args.documents,"batch":args.batch,
        "operations":operations,"readers":READERS,"arrivals":ARRIVALS,"interval_ns":INTERVAL_NS,
        "queue_capacity":QUEUE,"scheduled_base_ns":base,"compact":compact,
        "synthetic_embeddings":true,"retrieval_quality_measured":false});
    let mut output = create(&args.out.join("report.json"))?;
    serde_json::to_writer(&mut output, &report)?;
    output.flush()?;
    event("done")?;
    Ok(())
}
