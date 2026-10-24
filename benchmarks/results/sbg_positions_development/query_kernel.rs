use anyhow::Result;
use holographic_memory::core::documents::{
    EncodeMode, IngestOptions, LexicalExecution, LexicalOptions, LexicalQuery,
};
use holographic_memory::{DocumentInput, HmsCore};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};

#[derive(Deserialize)]
struct Request {
    query: String,
    k: usize,
    count: bool,
    repeats: usize,
    verify: bool,
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("index path");
    let core = HmsCore::new(1024, Some(path), None)?;
    if let Some(corpus) = std::env::args().nth(2) {
        return ingest(&core, &corpus);
    }
    println!("ready");
    std::io::stdout().flush()?;
    for line in std::io::stdin().lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;
        let query = LexicalQuery::parse(&request.query)?;
        let options = LexicalOptions {
            k: request.k,
            count: request.count,
            ..Default::default()
        };
        let expected = core.search_lexical(&query, &options)?;
        if request.verify {
            assert_eq!(
                expected,
                core.search_lexical(
                    &query,
                    &LexicalOptions {
                        execution: LexicalExecution::Exhaustive,
                        ..options.clone()
                    }
                )?,
                "{}",
                request.query
            );
        }
        let start = std::time::Instant::now();
        for _ in 0..request.repeats {
            std::hint::black_box(core.search_lexical(std::hint::black_box(&query), &options)?);
        }
        let ns = start.elapsed().as_nanos();
        let bytes = serde_json::to_vec(&expected)?;
        let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        println!(
            "{}",
            serde_json::json!({"ns":ns,"hash":hash,"count":expected.count})
        );
        std::io::stdout().flush()?;
    }
    Ok(())
}

fn rss_kib() -> Result<u64> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let line = status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .expect("RSS line");
    Ok(line.split_whitespace().nth(1).expect("RSS value").parse()?)
}

fn ingest(core: &HmsCore, corpus: &str) -> Result<()> {
    let mut batches = Vec::new();
    let mut batch = Vec::new();
    let mut replacements = Vec::new();
    let mut count = 0;
    let mut input_hash = 0xcbf29ce484222325u64;
    for line in BufReader::new(std::fs::File::open(corpus)?).lines() {
        let row: serde_json::Value = serde_json::from_str(&line?)?;
        let text = row["text"].as_str().expect("corpus text");
        if text.trim().is_empty() {
            continue;
        }
        let document: DocumentInput = serde_json::from_value(serde_json::json!({
            "id": count.to_string(), "text": text, "chunkWords": 4096,
            "overlapWords": 0, "storeText": false
        }))?;
        for byte in text.bytes() {
            input_hash = (input_hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        if count < 128 {
            let mut replacement = document.clone();
            replacement.text = format!("aardvark {}", replacement.text);
            replacements.push(replacement);
        }
        batch.push(document);
        count += 1;
        if batch.len() == 256 {
            batches.push(std::mem::take(&mut batch));
        }
        if count == 10_000 {
            break;
        }
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    anyhow::ensure!(count == 10_000, "corpus too short");
    println!("ready");
    std::io::stdout().flush()?;
    let mut go = String::new();
    std::io::stdin().read_line(&mut go)?;
    anyhow::ensure!(go.trim() == "go", "expected go");
    let options = IngestOptions {
        encode: EncodeMode::Lexical,
        positions: true,
        whole_document: true,
        unstemmed: true,
        ..Default::default()
    };
    let start = std::time::Instant::now();
    for batch in batches {
        core.memorize_documents_with(batch, &options)?;
    }
    core.flush()?;
    let ingest_ns = start.elapsed().as_nanos();
    let rss_ingested_kib = rss_kib()?;
    let start = std::time::Instant::now();
    core.memorize_documents_with(replacements, &options)?;
    core.flush()?;
    let replace_ns = start.elapsed().as_nanos();
    let start = std::time::Instant::now();
    for id in 0..128 {
        anyhow::ensure!(core.delete_document(&id.to_string())?, "missing document");
    }
    core.flush()?;
    let delete_ns = start.elapsed().as_nanos();
    println!(
        "{}",
        serde_json::json!({"documents": count, "input_hash": input_hash, "ingest_ns": ingest_ns, "replace_ns": replace_ns, "delete_ns": delete_ns, "rss_ingested_kib": rss_ingested_kib, "rss_after_kib": rss_kib()?})
    );
    Ok(())
}
