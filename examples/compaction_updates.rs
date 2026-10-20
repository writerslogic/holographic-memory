// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{ensure, Result};
use clap::Parser;
use holographic_memory::{DocumentInput, HmsCore, SearchOptions};
use serde_json::{json, Value};
use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{mpsc, Barrier},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const PAIRS: usize = 16;
const READERS: usize = 4;
const QUERIES: usize = 1024;
const UPDATES: usize = 256;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    seed: bool,
    #[arg(long)]
    observe: bool,
    #[arg(long)]
    documents: Option<usize>,
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
    let epoch_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let mut stdout = std::io::stdout().lock();
    write(&mut stdout, &json!({"phase":phase,"epoch_ns":epoch_ns}))?;
    stdout.flush()?;
    Ok(())
}

fn changed(original: &DocumentInput, revision: usize) -> DocumentInput {
    let mut input = original.clone();
    input.text = format!("{} revision{revision}", original.text);
    input.version = Some(revision.to_string());
    input.source_uri = Some(format!(
        "{}:revision{revision}",
        original.source_uri.as_ref().unwrap()
    ));
    input.metadata.as_mut().unwrap()["revision"] = json!(revision);
    input
}

fn query(core: &HmsCore, ids: Vec<String>, origin: Instant) -> Result<Value> {
    let start = origin.elapsed().as_nanos();
    let results = core.search_documents(
        "hmsprobe",
        &SearchOptions {
            k: Some(2),
            candidate_limit: Some(2),
            document_ids: Some(ids),
            ..Default::default()
        },
    )?;
    let end = origin.elapsed().as_nanos();
    Ok(json!({"start_ns":start,"end_ns":end,"results":results}))
}

fn queries(
    core: &HmsCore,
    origin: Instant,
    reader: usize,
    count: usize,
    out: &Path,
    paced: bool,
) -> Result<()> {
    let mut output = create(out)?;
    for sequence in 0..count {
        let pair = (reader + sequence) % PAIRS;
        let mut row = query(
            core,
            vec![format!("d{:08}", pair * 2), format!("d{:08}", pair * 2 + 1)],
            origin,
        )?;
        row["reader"] = json!(reader);
        row["sequence"] = json!(sequence);
        row["pair"] = json!(pair);
        write(&mut output, &row)?;
        if paced && sequence + 1 < count {
            std::thread::sleep(Duration::from_millis(8));
        }
    }
    output.flush()?;
    Ok(())
}

fn snapshot(core: &HmsCore, n: usize, origin: Instant, out: &Path) -> Result<()> {
    let mut output = create(out)?;
    for id in 0..n {
        let mut row = query(core, vec![format!("d{id:08}")], origin)?;
        row["id"] = json!(id);
        write(&mut output, &row)?;
    }
    output.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(!args.out.exists(), "refusing to overwrite evidence");
    let mut documents: Vec<DocumentInput> = serde_json::from_reader(File::open(&args.input)?)?;
    let n = args.documents.unwrap_or(documents.len());
    ensure!((32..=5182).contains(&n), "corpus size outside 32..=5182");
    std::fs::create_dir(&args.out)?;
    if args.seed {
        ensure!(n == documents.len(), "seed corpus size mismatch");
        ensure!(!args.store.exists(), "refusing to overwrite store");
        let core = HmsCore::new(1024, Some(args.store.display().to_string()), None)?;
        for batch in documents.chunks(64) {
            core.memorize_documents(batch.to_vec())?;
        }
        core.flush()?;
        return Ok(());
    }
    ensure!(args.store.exists(), "missing seeded store");
    ensure!(
        documents.len() == PAIRS * 2,
        "serving fixture must contain 32 documents"
    );
    documents.truncate(PAIRS * 2);
    documents.shrink_to_fit();
    let origin = Instant::now();
    let core = HmsCore::new(1024, Some(args.store.display().to_string()), None)?;
    for pair in 0..PAIRS {
        query(&core, vec![format!("d{:08}", pair * 2)], origin)?;
    }
    event("static")?;
    if args.observe {
        let mut start = String::new();
        std::io::stdin().read_line(&mut start)?;
        ensure!(start.trim() == "start", "missing observer start signal");
    }
    std::thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::new();
        for reader in 0..READERS {
            let core = &core;
            let path = args.out.join(format!("static-{reader}.jsonl"));
            workers.push(scope.spawn(move || queries(core, origin, reader, 256, &path, false)));
        }
        for worker in workers {
            worker.join().expect("static reader panicked")?;
        }
        Ok(())
    })?;
    event("updates")?;
    let barrier = Barrier::new(READERS + 2);
    let serving = Barrier::new(READERS + 1);
    let (sender, receiver) = mpsc::channel();
    let compact = std::thread::scope(|scope| -> Result<Value> {
        let mut readers = Vec::new();
        for reader in 0..READERS {
            let core = &core;
            let barrier = &barrier;
            let serving = &serving;
            let path = args.out.join(format!("updates-{reader}.jsonl"));
            readers.push(scope.spawn(move || {
                barrier.wait();
                serving.wait();
                queries(core, origin, reader, QUERIES, &path, true)
            }));
        }
        let writer_core = &core;
        let writer_barrier = &barrier;
        let writer_documents = &documents;
        let writer_path = args.out.join("operations.jsonl");
        let writer = scope.spawn(move || -> Result<()> {
            let core = writer_core;
            let documents = writer_documents;
            writer_barrier.wait();
            let mut output = create(&writer_path)?;
            for revision in 1..=UPDATES {
                let pair = (revision - 1) % PAIRS;
                let start = origin.elapsed().as_nanos();
                core.memorize_documents(vec![
                    changed(&documents[pair * 2], revision),
                    changed(&documents[pair * 2 + 1], revision),
                ])?;
                write(
                    &mut output,
                    &json!({"kind":"replace","pair":pair,"revision":revision,
                    "start_ns":start,"end_ns":origin.elapsed().as_nanos()}),
                )?;
                if revision == 1 {
                    sender.send(())?;
                }
                if revision % 4 == 0 {
                    let start = origin.elapsed().as_nanos();
                    ensure!(
                        core.delete_document(&documents[pair * 2].id)?,
                        "missing deletion"
                    );
                    write(
                        &mut output,
                        &json!({"kind":"delete","pair":pair,"revision":revision,
                        "start_ns":start,"end_ns":origin.elapsed().as_nanos()}),
                    )?;
                }
            }
            output.flush()?;
            Ok(())
        });
        barrier.wait();
        let acknowledged = receiver.recv();
        let start = origin.elapsed().as_nanos();
        let notified = event("compacting");
        serving.wait();
        acknowledged?;
        notified?;
        let compacted = core.compact();
        let end = origin.elapsed().as_nanos();
        event("updates-tail")?;
        writer.join().expect("writer panicked")?;
        for reader in readers {
            reader.join().expect("reader panicked")?;
        }
        compacted?;
        Ok(json!({"start_ns":start,"end_ns":end}))
    })?;
    event("verification")?;
    snapshot(&core, n, origin, &args.out.join("before-restart.jsonl"))?;
    core.flush()?;
    drop(core);
    let start = origin.elapsed().as_nanos();
    let core = HmsCore::new(1024, Some(args.store.display().to_string()), None)?;
    let restart_ns = origin.elapsed().as_nanos() - start;
    snapshot(&core, n, origin, &args.out.join("after-restart.jsonl"))?;
    let mut output = create(&args.out.join("report.json"))?;
    write(
        &mut output,
        &json!({"schema_version":1,"documents":n,"readers":READERS,
        "queries_per_reader":QUERIES,"updates":UPDATES,"compact":compact,"restart_ns":restart_ns,
        "synthetic_updates":true,"retrieval_quality_measured":false}),
    )?;
    output.flush()?;
    event("done")?;
    Ok(())
}
