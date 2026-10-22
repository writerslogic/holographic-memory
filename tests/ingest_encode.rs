// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Ingest encoding: the memoized text encoder is bit-identical to its reference formulation,
//! and `IngestOptions::encode` keeps every lexical result while deferring or skipping the
//! chunk hypervectors.

use std::collections::BTreeSet;

use holographic_memory::core::documents::{EncodeMode, IngestOptions};
use holographic_memory::core::encoding::TextEncoder;
use holographic_memory::{DocumentInput, HmsCore, SearchOptions};
use serde_json::json;

const PARAGRAPHS: [&str; 6] = [
    "Café records explain backups. Restore deleted documents from verified snapshots safely.",
    "The quick brown fox jumps over the lazy dog; the dog, being lazy, does not object.",
    "running common topic rare shared birds fly distant place revision one two three four",
    "Holographic memory stores sparse hypervectors: character n-grams, words and bigrams are \
     bundled by majority vote, then combined across levels by cross-level support.",
    "ünïcödé text with punctuation!!! and NUMBERS 12345, repeated repeated repeated words",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g h i j k l m n o p q",
];

const VOCABULARY: [&str; 40] = [
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
    "Café",
    "naïve",
    "coöperate",
    "2026-10-20",
    "k=10",
    "fsync",
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
];

/// A deterministic corpus: the fixed paragraphs, generated sentences over a vocabulary with
/// punctuation, numbers and non-ASCII words (3..=40 words, enough to cross the bundle
/// thresholds either way), and the degenerate inputs.
fn corpus() -> Vec<String> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize
    };
    let mut texts: Vec<String> = PARAGRAPHS.iter().map(|p| (*p).to_string()).collect();
    for i in 0..40 {
        let words = 3 + next() % 38;
        let mut text = String::new();
        for _ in 0..words {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(VOCABULARY[next() % VOCABULARY.len()]);
        }
        if i % 7 == 0 {
            text.push_str(", again: Café!");
        }
        texts.push(text);
    }
    for degenerate in ["x", "hi", "aa", "aa aa aa", "   ", "\u{1F600} emoji"] {
        texts.push(degenerate.to_string());
    }
    texts
}

#[test]
fn memoized_encoder_matches_reference_bit_for_bit() {
    let texts = corpus();
    for encoder in [
        TextEncoder::MultiscaleWordsV1,
        TextEncoder::MultiscaleCappedUnionV1,
    ] {
        for dim in [1usize, 255, 256, 257, 1024, 4096, 16384] {
            for text in &texts {
                let reference = encoder.encode_reference(text, dim);
                let memoized = encoder.encode_text(text, dim);
                assert_eq!(
                    memoized.indices(),
                    reference.indices(),
                    "{} dim {dim} text {text:?}",
                    encoder.name()
                );
                // The second call answers from the memo tables.
                assert_eq!(
                    encoder.encode_text(text, dim).indices(),
                    reference.indices(),
                    "{} dim {dim} memoized text {text:?}",
                    encoder.name()
                );
            }
        }
    }
}

fn document(i: usize, text: &str) -> DocumentInput {
    serde_json::from_value(json!({
        "id": format!("doc{i:03}"), "text": text, "chunkWords": 6, "overlapWords": 1,
        "metadata": {"group": i % 3}
    }))
    .expect("document input")
}

fn open(dir: &tempfile::TempDir) -> anyhow::Result<HmsCore> {
    HmsCore::new(1024, Some(dir.path().display().to_string()), None)
}

/// Every chunk the vector index returns for `text`, as a set: slot order differs between a
/// store that encoded at ingest and one that encoded later, and ties are ordered by slot.
fn vector_hits(core: &HmsCore, text: &str) -> BTreeSet<(String, u64)> {
    core.query(&core.encode_text(text), 4096)
        .into_iter()
        .map(|hit| (hit.id, hit.similarity.to_bits()))
        .collect()
}

fn lexical_hits(core: &HmsCore, text: &str) -> anyhow::Result<serde_json::Value> {
    let options = SearchOptions {
        k: Some(20),
        ..Default::default()
    };
    Ok(serde_json::to_value(
        core.search_documents(text, &options)?,
    )?)
}

#[test]
fn encode_modes_keep_lexical_results_and_defer_or_skip_vectors() -> anyhow::Result<()> {
    let texts: Vec<String> = corpus()
        .into_iter()
        .filter(|text| !text.trim().is_empty())
        .take(40)
        .collect();
    let inputs: Vec<DocumentInput> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| document(i, text))
        .collect();
    let dirs: Vec<tempfile::TempDir> = (0..4)
        .map(|_| tempfile::tempdir())
        .collect::<Result<_, _>>()?;
    let eager = open(&dirs[0])?;
    let deferred = open(&dirs[1])?;
    let lexical = open(&dirs[2])?;
    let deferred_options = IngestOptions {
        encode: EncodeMode::Deferred,
        ..Default::default()
    };
    let lexical_options = IngestOptions {
        encode: EncodeMode::Lexical,
        ..Default::default()
    };

    let counts = eager.memorize_documents(inputs.clone())?;
    assert_eq!(
        deferred.memorize_documents_with(inputs.clone(), &deferred_options)?,
        counts
    );
    assert_eq!(
        lexical.memorize_documents_with(inputs.clone(), &lexical_options)?,
        counts
    );
    let chunks: u64 = counts.iter().map(|&count| u64::from(count)).sum();
    assert!(
        chunks > inputs.len() as u64,
        "corpus must span several chunks"
    );
    assert_eq!(eager.vector_count(), chunks);
    assert_eq!(deferred.vector_count(), 0);
    assert_eq!(lexical.vector_count(), 0);

    let queries = [
        "restore snapshots",
        "running common topic",
        texts[3].as_str(),
        texts[11].as_str(),
        "Café records",
        "absent token",
    ];
    for query in queries {
        let expected = lexical_hits(&eager, query)?;
        assert_eq!(
            lexical_hits(&deferred, query)?,
            expected,
            "deferred {query:?}"
        );
        assert_eq!(
            lexical_hits(&lexical, query)?,
            expected,
            "lexical {query:?}"
        );
    }

    let probe = deferred.encode_text(&texts[5]);
    assert!(deferred
        .explain_query(&probe, 10)
        .rationale
        .contains("await deferred encoding"));
    assert!(lexical
        .explain_query(&probe, 10)
        .rationale
        .contains("lexical-only"));
    assert!(lexical.query(&probe, 10).is_empty());

    // The first vector query installs every deferred encoding; the results match a store
    // that encoded at ingest, chunk for chunk and bit for bit.
    for query in queries {
        assert_eq!(
            vector_hits(&deferred, query),
            vector_hits(&eager, query),
            "{query:?}"
        );
        assert!(vector_hits(&lexical, query).is_empty());
    }
    assert_eq!(deferred.vector_count(), chunks);
    assert_eq!(deferred.encode_pending_documents()?, 0);
    assert!(!deferred
        .explain_query(&probe, 10)
        .rationale
        .contains("await deferred encoding"));

    // The installed encodings survive a reopen (journal replay) and a compaction.
    drop(deferred);
    let deferred = open(&dirs[1])?;
    assert_eq!(deferred.vector_count(), chunks);
    for query in queries {
        assert_eq!(vector_hits(&deferred, query), vector_hits(&eager, query));
    }
    deferred.compact()?;
    drop(deferred);
    let deferred = open(&dirs[1])?;
    assert_eq!(deferred.vector_count(), chunks);
    assert_eq!(
        vector_hits(&deferred, &texts[3]),
        vector_hits(&eager, &texts[3])
    );

    // Deferred documents that are never queried stay pending across a reopen, and a document
    // replaced while pending takes the replacement's encoding, not the stale one.
    let staged = open(&dirs[3])?;
    staged.memorize_documents_with(inputs.clone(), &deferred_options)?;
    drop(staged);
    let staged = open(&dirs[3])?;
    assert_eq!(staged.vector_count(), 0);
    let mut replacement = inputs[9].clone();
    replacement.text = texts[21].clone();
    assert_eq!(
        staged.memorize_document_with(replacement, &deferred_options)?,
        counts[21]
    );
    let expected_vectors = chunks - u64::from(counts[9]) + u64::from(counts[21]);
    assert_eq!(
        u64::from(staged.maintain_indices()?.vector_count),
        expected_vectors
    );
    assert_eq!(staged.vector_count(), expected_vectors);
    assert_eq!(staged.encode_pending_documents()?, 0);
    let replaced: BTreeSet<_> = vector_hits(&staged, &texts[21])
        .into_iter()
        .filter(|(id, _)| id.contains(":doc009:"))
        .map(|(_, similarity)| similarity)
        .collect();
    let original: BTreeSet<_> = vector_hits(&eager, &texts[21])
        .into_iter()
        .filter(|(id, _)| id.contains(":doc021:"))
        .map(|(_, similarity)| similarity)
        .collect();
    assert_eq!(replaced, original);

    // A lexical store stays lexical across a reopen; memorizing a document again eagerly
    // indexes it, and deleting one leaves the counters consistent.
    drop(lexical);
    let lexical = open(&dirs[2])?;
    assert_eq!(lexical.vector_count(), 0);
    assert_eq!(lexical.memorize_document(inputs[7].clone())?, counts[7]);
    assert_eq!(lexical.vector_count(), u64::from(counts[7]));
    let expected: BTreeSet<_> = vector_hits(&eager, &texts[7])
        .into_iter()
        .filter(|(id, _)| id.contains(":doc007:"))
        .collect();
    assert_eq!(vector_hits(&lexical, &texts[7]), expected);
    assert!(lexical.delete_document("doc008")?);
    assert!(lexical
        .explain_query(&probe, 10)
        .rationale
        .contains("lexical-only"));

    // Deferred encoding needs the text to come back to.
    let mut hidden = inputs[0].clone();
    hidden.store_text = Some(false);
    assert!(lexical
        .memorize_document_with(hidden.clone(), &deferred_options)
        .is_err());
    assert!(lexical
        .memorize_document_with(hidden, &lexical_options)
        .is_ok());
    assert!("deferred".parse::<EncodeMode>().is_ok());
    assert!("later".parse::<EncodeMode>().is_err());
    Ok(())
}
