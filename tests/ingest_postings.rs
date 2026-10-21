// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Bulk postings: a corpus memorized in one batch, in batches of 7 and one document at a time
//! yields identical search results (ids, scores, offsets, text) for 50 queries, as does every
//! store reopened from disk; deletes, re-inserts into reused slots (the merge path of a
//! posting list), replacements and a deferred encoding catch-up reach the results of a fresh
//! store holding the same documents.

use holographic_memory::core::documents::{EncodeMode, IngestOptions};
use holographic_memory::{DocumentInput, HmsCore, SearchOptions};
use serde_json::{json, Value};

const VOCABULARY: [&str; 64] = [
    "memory",
    "vector",
    "sparse",
    "index",
    "query",
    "document",
    "chunk",
    "restore",
    "backup",
    "snapshot",
    "verified",
    "running",
    "common",
    "topic",
    "rare",
    "shared",
    "birds",
    "fly",
    "distant",
    "place",
    "revision",
    "patients",
    "studies",
    "dietary",
    "fiber",
    "cancer",
    "risk",
    "trial",
    "randomized",
    "controlled",
    "effects",
    "treatment",
    "journal",
    "postings",
    "stem",
    "stemmed",
    "tantivy",
    "sqlite",
    "bm25",
    "jaccard",
    "bundle",
    "bind",
    "permute",
    "dimension",
    "threshold",
    "holographic",
    "encoder",
    "deferred",
    "lexical",
    "eager",
    "segment",
    "overlay",
    "dictionary",
    "varint",
    "delta",
    "merge",
    "bucket",
    "range",
    "thread",
    "k=10",
    "2026-10-20",
    "naïve",
    "coöperate",
    "Café",
];

/// Deterministic generator: the high bits of a 64-bit LCG.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as usize
    }
}

fn input(id: &str, text: &str, group: usize, source: usize) -> DocumentInput {
    serde_json::from_value(json!({
        "id": id,
        "text": text,
        "chunkWords": 8,
        "overlapWords": 2,
        "metadata": {"group": group, "even": group.is_multiple_of(2)},
        "sourceUri": format!("s3://corpus/{source}"),
    }))
    .expect("document input")
}

/// 150 documents of 10..=59 vocabulary words with a head-heavy word distribution (so common
/// terms have long posting lists), punctuation and non-ASCII words; every document spans
/// several chunks at 8 words per chunk.
fn corpus() -> Vec<DocumentInput> {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    (0..150)
        .map(|i| {
            let words = 10 + rng.next() % 50;
            let mut text = String::new();
            for w in 0..words {
                if w > 0 {
                    text.push(' ');
                }
                let r = rng.next() % VOCABULARY.len();
                text.push_str(VOCABULARY[(r * r) / VOCABULARY.len()]);
                if w % 9 == 8 {
                    text.push(',');
                }
            }
            if i % 5 == 0 {
                text.push_str(". Café naïve coöperate 2026-10-20 again!");
            }
            input(&format!("doc{i:03}"), &text, i % 4, i % 6)
        })
        .collect()
}

/// 50 queries of 1..=4 vocabulary words, some with a token absent from the corpus.
fn queries() -> Vec<String> {
    let mut rng = Lcg(0x2545_F491_4F6C_DD1D);
    (0..50)
        .map(|q| {
            let mut text = String::new();
            for w in 0..=(q % 4) {
                if w > 0 {
                    text.push(' ');
                }
                text.push_str(VOCABULARY[rng.next() % VOCABULARY.len()]);
            }
            if q % 10 == 9 {
                text.push_str(" zzzabsent");
            }
            text
        })
        .collect()
}

fn open(dir: &tempfile::TempDir) -> anyhow::Result<HmsCore> {
    HmsCore::new(1024, Some(dir.path().display().to_string()), None)
}

/// Unfiltered rankings (the posting-list path) with a deep candidate pool, plus a shallower
/// cut, for every query.
fn unfiltered_options() -> Vec<SearchOptions> {
    vec![
        SearchOptions {
            k: Some(50),
            candidate_limit: Some(200),
            ..Default::default()
        },
        SearchOptions {
            k: Some(5),
            ..Default::default()
        },
    ]
}

/// The unfiltered options plus metadata, source and id filters.
fn all_options() -> Vec<SearchOptions> {
    let mut options = unfiltered_options();
    options.push(SearchOptions {
        k: Some(20),
        filter: Some(json!({"group": 1})),
        ..Default::default()
    });
    options.push(SearchOptions {
        k: Some(20),
        source_uri: Some("s3://corpus/3".into()),
        ..Default::default()
    });
    options.push(SearchOptions {
        k: Some(20),
        document_ids: Some((0..150).step_by(4).map(|i| format!("doc{i:03}")).collect()),
        ..Default::default()
    });
    options
}

/// Every result list, serialized: id, document id, offsets, text, scores.
fn results(core: &HmsCore, options: &[SearchOptions]) -> anyhow::Result<Vec<Value>> {
    let mut all = Vec::new();
    for query in queries() {
        for options in options {
            all.push(serde_json::to_value(
                core.search_documents(&query, options)?,
            )?);
        }
    }
    Ok(all)
}

fn assert_same(expected: &[Value], actual: &[Value], what: &str) {
    assert_eq!(expected.len(), actual.len(), "{what}: result list count");
    let options = all_options().len();
    for (i, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected,
            actual,
            "{what}: query {} options {}",
            i / options,
            i % options
        );
    }
}

#[test]
fn batching_does_not_change_search_results_live_or_reopened() -> anyhow::Result<()> {
    let inputs = corpus();
    let dirs: Vec<tempfile::TempDir> = (0..3)
        .map(|_| tempfile::tempdir())
        .collect::<Result<_, _>>()?;
    let one = open(&dirs[0])?;
    let sevens = open(&dirs[1])?;
    let singles = open(&dirs[2])?;

    let counts = one.memorize_documents(inputs.clone())?;
    assert!(
        counts.iter().all(|&count| count >= 2),
        "every document spans several chunks"
    );
    let mut seven_counts = Vec::new();
    for batch in inputs.chunks(7) {
        seven_counts.extend(sevens.memorize_documents(batch.to_vec())?);
    }
    assert_eq!(seven_counts, counts);
    for (input, &count) in inputs.iter().zip(&counts) {
        assert_eq!(singles.memorize_document(input.clone())?, count);
    }
    for core in [&one, &sevens, &singles] {
        assert_eq!(core.document_count(), inputs.len());
        assert_eq!(
            core.vector_count(),
            counts.iter().map(|&c| u64::from(c)).sum::<u64>()
        );
    }

    let options = all_options();
    let expected = results(&one, &options)?;
    assert!(
        expected
            .iter()
            .filter(|r| !r.as_array().is_some_and(Vec::is_empty))
            .count()
            > expected.len() / 2,
        "most queries must hit"
    );
    assert_same(&expected, &results(&sevens, &options)?, "batches of 7");
    assert_same(&expected, &results(&singles, &options)?, "one at a time");

    // Reopened from the journal (one, sevens) and from a compacted store (singles).
    singles.compact()?;
    drop((one, sevens, singles));
    for (dir, what) in dirs
        .iter()
        .zip(["one batch", "batches of 7", "one at a time"])
    {
        let core = open(dir)?;
        assert_eq!(core.document_count(), inputs.len());
        assert_same(
            &expected,
            &results(&core, &options)?,
            &format!("{what}, reopened"),
        );
    }
    Ok(())
}

#[test]
fn reused_slots_replacements_and_deferred_catch_up_match_a_fresh_store() -> anyhow::Result<()> {
    let inputs = corpus();
    let dirs: Vec<tempfile::TempDir> = (0..3)
        .map(|_| tempfile::tempdir())
        .collect::<Result<_, _>>()?;
    let fresh = open(&dirs[0])?;
    let churned = open(&dirs[1])?;
    let deferred = open(&dirs[2])?;

    // `churned`: everything, then every third document deleted and re-added, half one at a
    // time and half in one batch, into the freed slots (runs below a list's final slot).
    churned.memorize_documents(inputs.clone())?;
    for input in inputs
        .iter()
        .filter(|input| doc_index(input).is_multiple_of(3))
    {
        assert!(churned.delete_document(&input.id)?);
    }
    assert_eq!(churned.document_count(), 100);
    for input in inputs
        .iter()
        .filter(|input| doc_index(input).is_multiple_of(6))
    {
        churned.memorize_document(input.clone())?;
    }
    churned.memorize_documents(
        inputs
            .iter()
            .filter(|input| doc_index(input) % 6 == 3)
            .cloned()
            .collect(),
    )?;
    assert_eq!(churned.document_count(), inputs.len());

    // `churned` then replaces one document in place (same word and chunk count, a new first
    // term) and one with more chunks; `fresh` holds that final set from one batch.
    let mut replaced = inputs.clone();
    for input in &mut replaced {
        match doc_index(input) {
            7 => {
                let mut words: Vec<&str> = input.text.split(' ').collect();
                words[0] = "mnemonic";
                input.text = words.join(" ");
            }
            11 => input
                .text
                .push_str(" appended tail of eleven more words to add chunks"),
            _ => {}
        }
    }
    churned.memorize_document(replaced[7].clone())?;
    churned.memorize_document(replaced[11].clone())?;
    fresh.memorize_documents(replaced.clone())?;
    assert_eq!(fresh.document_count(), churned.document_count());

    // `deferred`: the final document set in one batch without vectors, then the catch-up,
    // which re-installs every document with unchanged terms.
    let deferred_options = IngestOptions {
        encode: EncodeMode::Deferred,
    };
    deferred.memorize_documents_with(replaced.clone(), &deferred_options)?;
    assert_eq!(deferred.vector_count(), 0);
    assert_eq!(deferred.encode_pending_documents()?, replaced.len());
    assert_eq!(deferred.vector_count(), fresh.vector_count());

    let options = unfiltered_options();
    let expected = results(&fresh, &options)?;
    assert_same_unfiltered(&expected, &results(&churned, &options)?, "churned");
    assert_same_unfiltered(&expected, &results(&deferred, &options)?, "deferred");

    drop((fresh, churned, deferred));
    for (dir, what) in dirs.iter().zip(["fresh", "churned", "deferred"]) {
        let core = open(dir)?;
        assert_same_unfiltered(
            &expected,
            &results(&core, &options)?,
            &format!("{what}, reopened"),
        );
    }
    Ok(())
}

fn doc_index(input: &DocumentInput) -> usize {
    input.id["doc".len()..].parse().expect("document index")
}

fn assert_same_unfiltered(expected: &[Value], actual: &[Value], what: &str) {
    assert_eq!(expected.len(), actual.len(), "{what}: result list count");
    let options = unfiltered_options().len();
    for (i, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected,
            actual,
            "{what}: query {} options {}",
            i / options,
            i % options
        );
    }
}
