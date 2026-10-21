// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::{compact_chunks, Embedding, EntangledHVec, StoredChunk, StoredDocument, VectorState};
use std::collections::BTreeMap;

#[test]
fn rebuilt_terms_preserve_complete_document_bytes_and_legacy_frequencies() {
    for count in [0, 1, 10, 11, 12, 127, 128, 129, 4095, 4096, 4097] {
        let mut terms = BTreeMap::new();
        for index in 0..count {
            terms.insert(
                format!("term:{index:05}:é\0"),
                match index % 3 {
                    0 => 0,
                    1 => 1,
                    _ => u32::MAX,
                },
            );
        }
        let mut document = StoredDocument {
            id: "document:é".into(),
            source_uri: Some("source:é".into()),
            version: "1".into(),
            metadata: serde_json::json!({"nested":[null, false, "é"]}),
            chunks: vec![StoredChunk {
                id: "chunk:é".into(),
                text: Some("é\0 running".into()),
                text_at: None,
                start_byte: 0,
                end_byte: 11,
                vector: EntangledHVec::new_deterministic(1024, count as u64),
                embedding: Some(Embedding::from_f32(vec![1.0, -0.0, 0.0])),
                terms: terms.into(),
                word_count: 0,
                stemmed: count % 2 == 0,
                vector_state: VectorState::Encoded,
            }],
        };
        let expected = serde_json::to_vec(&document).unwrap();
        compact_chunks(&mut document.chunks);
        assert_eq!(serde_json::to_vec(&document).unwrap(), expected);
        compact_chunks(&mut document.chunks);
        assert_eq!(serde_json::to_vec(&document).unwrap(), expected);
        let restored: StoredDocument = serde_json::from_slice(&expected).unwrap();
        assert_eq!(serde_json::to_vec(&restored).unwrap(), expected);
    }
}

/// The memoized, hash-counted tokenizer returns exactly what the per-occurrence reference
/// returns: same keys, same counts, for ASCII, mixed case, non-ASCII (final sigma), and words
/// seen before the memo was populated and after.
#[test]
fn memoized_terms_match_the_reference_tokenizer() {
    fn reference(text: &str, stem: bool) -> std::collections::BTreeMap<String, u32> {
        let stemmer = rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English);
        let mut out = std::collections::BTreeMap::new();
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
        {
            let lower = word.to_lowercase();
            let key = if stem {
                stemmer.stem(&lower).into_owned()
            } else {
                lower
            };
            *out.entry(key).or_insert(0) += 1;
        }
        out
    }
    let texts = [
        "Running runners RUN quickly; the runner's running-shoes ran.",
        "ΟΔΟΣ Οδός ΣΟΦΟΣ naïve Café CAFÉ straße STRASSE 42 x2 İstanbul",
        "",
        "!!! ,,, ---",
        "Running runners RUN quickly; the runner's running-shoes ran.",
    ];
    for stem in [false, true] {
        for text in texts {
            assert_eq!(
                super::terms(text, stem),
                reference(text, stem),
                "stem={stem} text={text:?}"
            );
        }
    }
}
