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
        let (terms, words) = super::chunk_terms(text, title_weight, true);
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

/// Words up to and past the 16-byte inline key, multibyte ones included, map back to their
/// own ids and never to another word's.
#[test]
fn word_map_keys_words_across_the_inline_limit() {
    let mut words: Vec<String> = (1..=34)
        .flat_map(|len| {
            [
                "a".repeat(len),
                format!("{}b", "a".repeat(len - 1)),
                "é".repeat(len / 2 + 1),
            ]
        })
        .collect();
    words.sort();
    words.dedup();
    let mut map = super::WordMap::default();
    for (id, word) in words.iter().enumerate() {
        assert_eq!(map.get(word), None, "{word:?}");
        map.insert(word, id as u32);
    }
    assert_eq!(map.len(), words.len());
    for (id, word) in words.iter().enumerate() {
        assert_eq!(map.get(word), Some(id as u32), "{word:?}");
    }
    assert_eq!(map.get(&"a".repeat(35)), None);
}

/// The interner a store keeps between batches, taken back warm and reset once it holds
/// `STEM_MEMO_CAPACITY` words, counts every chunk as a fresh one does; of two interners
/// returned together the pool keeps the one that has seen more words.
#[test]
fn pooled_interner_counts_later_batches_like_a_fresh_one() {
    let texts = [
        "Running runners RUN quickly;\nthe runner's running-shoes ran. Running",
        "ΟΔΟΣ Οδός ΣΟΦΟΣ naïve\r\nCafé CAFÉ straße STRASSE 42 x2 İstanbul",
        "abcdefghijklmnop abcdefghijklmnopq éééééééé ééééééééx Abcdefghijklmnopq",
    ];
    let table = |text: &str| {
        let (terms, words) = super::chunk_terms(text, 2, true);
        let pairs: Vec<(String, u32)> = (&terms)
            .into_iter()
            .map(|(k, &v)| (k.to_owned(), v))
            .collect();
        (pairs, words)
    };
    let fresh: Vec<_> = texts.iter().map(|text| table(text)).collect();
    let pool = super::StemMemoPool::default();
    for batch in 0..3 {
        let mut memo = pool.take();
        let _installed = memo.install();
        if batch == 1 {
            for start in (0..=super::STEM_MEMO_CAPACITY).step_by(1000) {
                let chunk: String = (start..start + 1000).map(|i| format!("w{i} ")).collect();
                super::chunk_terms(&chunk, 1, true);
            }
        }
        for (text, expected) in texts.iter().zip(&fresh) {
            assert_eq!(&table(text), expected, "batch {batch}");
        }
    }
    let warm = pool.take();
    drop(pool.take());
    drop(warm);
    assert!(pool
        .kept
        .lock()
        .as_ref()
        .is_some_and(|kept| kept.0.words.len() > 0));

    // With keeping off, the kept interner is freed and returned ones are not kept.
    pool.set_keep(false);
    assert!(pool.kept.lock().is_none());
    {
        let mut memo = pool.take();
        let _installed = memo.install();
        assert_eq!(&table(texts[0]), &fresh[0]);
    }
    assert!(pool.kept.lock().is_none());
    pool.set_keep(true);
    drop(pool.take());
    assert!(pool.kept.lock().is_some());
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

/// Positions built from the interned term of each word are byte for byte the positions
/// the reference analyzer builds, through the interner's word cache, title weighting,
/// Unicode lowercasing and both stemming modes.
#[test]
fn interned_positions_match_the_reference_analyzer() {
    let long = format!("{} tail", "Repeat repeats repeated ".repeat(400));
    let texts = [
        ("Running runs ran; the RUNNER ran\nand ran again", 3),
        ("ÜBER über Σοφία ΣΟΦΊΑΣ naïve café CAFÉ 東京 東京", 1),
        ("title line\nbody title body", 2),
        ("", 1),
        ("--- ...", 1),
        (long.as_str(), 1),
    ];
    for stem in [true, false] {
        let mut interner = super::Interner::default();
        for whole_document in [false, true] {
            for (text, title_weight) in texts {
                let (terms, _) =
                    interner.chunk_terms(text, title_weight, stem, Some(whole_document));
                let reference =
                    super::positions::Positions::from_text(text, stem, &terms, whole_document);
                assert_eq!(
                    serde_json::to_value(terms.positions.as_deref().unwrap()).unwrap(),
                    serde_json::to_value(&reference).unwrap(),
                    "{text:?} stem {stem}"
                );
                let (plain, _) = interner.chunk_terms(text, title_weight, stem, None);
                assert!(plain.positions.is_none());
            }
        }
    }
}
