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

/// Ingest's interned counting gives every chunk exactly the term table and word count of
/// `terms(text, true)` with the first line counted `title_weight` times in total: with and
/// without a batch interner, for repeated, mixed-case and non-ASCII words, and after the
/// interner starts over at its capacity.
#[test]
fn interned_chunk_terms_match_terms_with_title_weighting() {
    fn expected(text: &str, title_weight: u32) -> (Vec<(String, u32)>, usize) {
        let mut terms = super::terms(text, true);
        if title_weight > 1 {
            for (term, count) in super::terms(text.lines().next().unwrap_or_default(), true) {
                *terms.entry(term).or_insert(0) += count * (title_weight - 1);
            }
        }
        let words = terms.values().map(|&n| n as usize).sum();
        (terms.into_iter().collect(), words)
    }
    fn actual(text: &str, title_weight: u32) -> (Vec<(String, u32)>, usize) {
        let (terms, words) = super::chunk_terms(text, title_weight);
        terms.assert_valid_storage();
        let pairs = (&terms)
            .into_iter()
            .map(|(k, &v)| (k.to_owned(), v))
            .collect();
        (pairs, words)
    }
    let many: String = (0..70_000).map(|i| format!("w{i} ")).collect();
    let texts = [
        "Running runners RUN quickly;\nthe runner's running-shoes ran. Running",
        "ΟΔΟΣ Οδός ΣΟΦΟΣ naïve\r\nCafé CAFÉ straße STRASSE 42 x2 İstanbul",
        "",
        "!!! ,,, ---\nwords after an empty title",
        "abcdefgh abcdefghi abcdefgg abcdefg abcdefghij zzzzzzzzz a aa\u{0}aa",
        &many,
        "Running runners RUN quickly;\nthe runner's running-shoes ran. Running",
    ];
    for title_weight in [1, 2, 16] {
        for text in texts {
            assert_eq!(actual(text, title_weight), expected(text, title_weight));
        }
        let mut memo = super::StemMemo::default();
        let _installed = memo.install();
        for text in texts.iter().chain(&texts) {
            for words in text.split_inclusive(' ').collect::<Vec<_>>().chunks(1000) {
                let chunk = words.concat();
                assert_eq!(actual(&chunk, title_weight), expected(&chunk, title_weight));
            }
        }
    }
}

/// The byte scanner finds the same whitespace-separated words as `char::is_whitespace` on
/// text mixing ASCII and Unicode separators (NBSP, NEL, em space, ideographic space).
#[test]
fn chunk_boundaries_match_char_whitespace_on_mixed_text() {
    let text = "α\u{a0}beta \u{85}gamma\u{2003}δ\tε\u{3000}zeta\u{b}eta  θ";
    let reference: Vec<&str> = text
        .split(char::is_whitespace)
        .filter(|w| !w.is_empty())
        .collect();
    let input: super::DocumentInput = serde_json::from_value(serde_json::json!({
        "id": "d", "text": text, "chunkWords": 1, "overlapWords": 0
    }))
    .unwrap();
    let chunks: Vec<String> = super::chunk_document(&input)
        .unwrap()
        .into_iter()
        .map(|c| c.text)
        .collect();
    assert_eq!(chunks, reference);
}
