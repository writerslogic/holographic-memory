// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::{compact_chunks, EntangledHVec, StoredChunk, StoredDocument};
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
                embedding: Some(vec![1.0, -0.0, 0.0]),
                terms: terms.into(),
                word_count: 0,
                stemmed: count % 2 == 0,
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
