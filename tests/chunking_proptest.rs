// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Properties of `chunk_document` that hold for every input, not only the hand-written case in
//! `tests/documents.rs`.

use proptest::prelude::*;
use writerslogic_warm::DocumentInput;

fn input(text: &str, chunk_words: u32, overlap_words: u32) -> DocumentInput {
    serde_json::from_value(serde_json::json!({
        "id": "doc", "text": text, "chunkWords": chunk_words, "overlapWords": overlap_words
    }))
    .unwrap()
}

/// Words and whitespace of several kinds, with multi-byte text such as `Café`, CJK and emoji.
fn mixed_text() -> impl Strategy<Value = String> {
    let piece = prop_oneof![
        "[a-z]{1,8}",
        Just("Café".to_string()),
        Just("日本語".to_string()),
        Just("naïve—ok".to_string()),
        Just("🦀".to_string()),
        Just(" ".to_string()),
        Just("\t".to_string()),
        Just("\n\n".to_string()),
        Just("\u{3000}".to_string()),
        Just("\u{a0}".to_string()),
    ];
    prop::collection::vec(piece, 0..80).prop_map(|pieces| pieces.concat())
}

fn check(text: &str, size: u32, overlap: u32) -> Result<(), TestCaseError> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let result = writerslogic_warm::chunk_document(&input(text, size, overlap));
    if words.is_empty() {
        prop_assert!(result.is_err(), "a text without words must be rejected");
        return Ok(());
    }
    let chunks = result.map_err(|e| TestCaseError::fail(e.to_string()))?;
    let (size, step) = (size as usize, (size - overlap) as usize);
    for (j, chunk) in chunks.iter().enumerate() {
        let (start, end) = (chunk.start_byte as usize, chunk.end_byte as usize);
        // The byte range falls on character boundaries and slices the text to the chunk's text.
        prop_assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
        prop_assert_eq!(&text[start..end], chunk.text.as_str());
        // Chunk j holds the words from j * step on, in their order in the input.
        let first = j * step;
        let last = (first + size).min(words.len());
        prop_assert_eq!(
            chunk.text.split_whitespace().collect::<Vec<_>>(),
            words[first..last].to_vec()
        );
    }
    // The last chunk reaches the last word, and the one before it did not.
    let last = chunks.len() - 1;
    prop_assert!(last * step + size >= words.len());
    prop_assert!(last == 0 || (last - 1) * step + size < words.len());
    Ok(())
}

proptest! {
    #[test]
    fn chunks_slice_the_text_and_cover_every_word(text in mixed_text(), size in 1u32..=12, overlap in 0u32..12) {
        check(&text, size, overlap % size)?;
    }

    #[test]
    fn chunks_of_arbitrary_unicode_text(text in any::<String>(), size in 1u32..=12, overlap in 0u32..12) {
        check(&text, size, overlap % size)?;
    }

    #[test]
    fn an_overlap_not_below_the_chunk_size_is_an_error(text in mixed_text(), size in 1u32..=12, extra in 0u32..4) {
        prop_assert!(writerslogic_warm::chunk_document(&input(&text, size, size + extra)).is_err());
    }
}
