// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `count_documents` counts exactly the documents an unfiltered lexical search returns, over
//! multi-chunk, replaced and deleted documents, and `IngestOptions::unstemmed` documents match
//! only the words they contain as written.

use std::collections::BTreeSet;

use writerslogic_warm::core::documents::{EncodeMode, IngestOptions};
use writerslogic_warm::{DocumentInput, SearchOptions, WarmCore};

fn input(id: &str, text: &str) -> DocumentInput {
    DocumentInput {
        id: id.into(),
        text: text.into(),
        source_uri: None,
        version: None,
        metadata: None,
        chunk_words: Some(3),
        overlap_words: Some(0),
        store_text: Some(false),
        embeddings: None,
        title_weight: None,
    }
}

fn matching_ids(core: &WarmCore, query: &str) -> BTreeSet<String> {
    let options = SearchOptions {
        k: Some(1000),
        candidate_limit: Some(4096),
        ..Default::default()
    };
    let hits = core.search_documents(query, &options).unwrap();
    assert!(hits.len() < 1000, "the search must not be truncated");
    hits.into_iter().map(|hit| hit.document_id).collect()
}

#[test]
fn count_matches_the_documents_a_lexical_search_returns() {
    let dir = tempfile::tempdir().unwrap();
    let core = WarmCore::new(64, Some(dir.path().display().to_string()), None).unwrap();
    let words = ["red", "green", "blue", "cyan", "gold", "grey", "pink"];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = |bound: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % bound as u64) as usize
    };
    let mut documents = Vec::new();
    for id in 0..300 {
        let length = 1 + next(12);
        let text: Vec<&str> = (0..length).map(|_| words[next(words.len())]).collect();
        documents.push(input(&format!("d{id}"), &text.join(" ")));
    }
    let options = IngestOptions {
        encode: EncodeMode::Lexical,
        ..Default::default()
    };
    core.memorize_documents_with(documents, &options).unwrap();
    // Replacements change chunk counts and free slots; deletions leave holes.
    for id in (0..300).step_by(7) {
        let text = if id % 2 == 0 {
            "gold"
        } else {
            "cyan pink red cyan pink red"
        };
        core.memorize_documents_with(vec![input(&format!("d{id}"), text)], &options)
            .unwrap();
    }
    for id in (3..300).step_by(11) {
        core.delete_document(&format!("d{id}")).unwrap();
    }
    for query in [
        "red",
        "gold",
        "pink cyan",
        "blue grey green",
        "absent",
        "red absent",
    ] {
        let expected = matching_ids(&core, query).len();
        assert_eq!(core.count_documents(query).unwrap(), expected, "{query}");
    }
    assert_eq!(core.count_documents("absent").unwrap(), 0);
}

#[test]
fn unstemmed_documents_match_only_their_own_words() {
    let dir = tempfile::tempdir().unwrap();
    let core = WarmCore::new(64, Some(dir.path().display().to_string()), None).unwrap();
    let stemmed = IngestOptions {
        encode: EncodeMode::Lexical,
        ..Default::default()
    };
    let unstemmed = IngestOptions {
        unstemmed: true,
        ..stemmed.clone()
    };
    core.memorize_documents_with(vec![input("plural", "observatories")], &unstemmed)
        .unwrap();
    core.memorize_documents_with(vec![input("singular", "observatory")], &unstemmed)
        .unwrap();
    core.memorize_documents_with(vec![input("stemmed", "observatories")], &stemmed)
        .unwrap();
    let both: BTreeSet<String> = ["singular", "stemmed"].map(String::from).into();
    assert_eq!(matching_ids(&core, "observatory"), both);
    assert_eq!(core.count_documents("observatory").unwrap(), 2);
    let plural: BTreeSet<String> = ["plural", "stemmed"].map(String::from).into();
    assert_eq!(matching_ids(&core, "observatories"), plural);
    // The unstemmed form survives a reopen from the journal.
    drop(core);
    let core = WarmCore::new(64, Some(dir.path().display().to_string()), None).unwrap();
    assert_eq!(matching_ids(&core, "observatory"), both);
}
