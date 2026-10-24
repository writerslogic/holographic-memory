// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::core::documents::{self, DocumentInput, EncodeMode, IngestOptions};
use crate::core::encoding::TextEncoder;
use crate::HmsCore;
use proptest::prelude::*;

fn input(id: &str, text: &str) -> DocumentInput {
    serde_json::from_value(serde_json::json!({"id":id,"text":text,"chunkWords":4096,"overlapWords":0,"storeText":false})).unwrap()
}

fn options(positions: bool, unstemmed: bool) -> IngestOptions {
    IngestOptions {
        encode: EncodeMode::Lexical,
        positions,
        unstemmed,
        ..Default::default()
    }
}

pub(super) fn insert(
    store: &mut DocumentStore,
    id: &str,
    text: &str,
    positions: bool,
    unstemmed: bool,
) {
    let document = documents::prepare_with_options(
        input(id, text),
        64,
        None,
        None,
        TextEncoder::default(),
        &options(positions, unstemmed),
    )
    .unwrap();
    store.insert(id.to_owned(), document);
}

fn run(store: &DocumentStore, query: &str, k: usize) -> LexicalResults {
    execute(
        store,
        &LexicalQuery::parse(query).unwrap(),
        &LexicalOptions {
            k,
            count: true,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn boolean_truth_table_and_phrase_offsets() {
    let mut store = DocumentStore::default();
    for (id, text) in [
        ("a", "red red blue fox"),
        ("b", "red fox blue"),
        ("c", "blue fox"),
        ("d", "green"),
    ] {
        insert(&mut store, id, text, true, true);
    }
    for (query, expected) in [
        ("", 0),
        ("-red", 0),
        ("red blue", 3),
        ("+red +blue", 2),
        ("+red absent", 2),
        ("+absent red", 0),
        ("+red -blue", 0),
        ("+red -absent", 2),
        ("+red -red", 0),
        ("+red red red", 2),
        ("\"red blue\"", 1),
        ("\"blue red\"", 0),
        ("\"red red\"", 1),
        ("\"red fox blue\"", 1),
        ("+\"red blue\" +fox", 1),
        ("+fox -\"red blue\"", 2),
        ("+fox \"absent blue\"", 3),
        ("\"red absent\"", 0),
        ("-\"red blue\"", 0),
    ] {
        let actual = run(&store, query, 100);
        assert_eq!(actual.count, Some(expected), "{query}");
        assert_eq!(actual.hits.len(), expected, "{query}");
    }
    assert_eq!(run(&store, "red red", 100), run(&store, "red", 100));
    assert!(run(&store, "red", 0).hits.is_empty());
    let phrase = LexicalQuery::parse("\"red red\"").unwrap();
    let clause = &phrase.clauses[0];
    insert(&mut store, "e", "red red red red", true, true);
    assert_eq!(
        frequency(
            clause,
            &store.documents["e"].chunks[0],
            true,
            &mut word_ids(&phrase)[0]
        ),
        3
    );
    assert_eq!(
        frequency(
            clause,
            &store.documents["e"].chunks[0],
            false,
            &mut word_ids(&phrase)[0]
        ),
        1
    );
}

#[test]
fn single_optional_phrases_filter_candidates_without_changing_counts() {
    for boundary in [127, 128, 129] {
        let mut store = DocumentStore::default();
        for id in 0..boundary {
            insert(
                &mut store,
                &format!("common{id:03}"),
                "common common filler common",
                true,
                true,
            );
        }
        for (id, text) in [
            ("adjacent", "rare common rare common"),
            ("reversed", "common rare"),
            ("separated", "rare filler common rare"),
        ] {
            insert(&mut store, id, text, true, true);
        }
        for (text, expected) in [
            ("\"rare common\"", 1),
            ("\"common common\"", boundary),
            ("\"rare rare\"", 0),
            ("\"rare absent\"", 0),
            ("+common \"rare rare\"", boundary + 3),
            ("\"rare rare\" common", boundary + 3),
        ] {
            let query = LexicalQuery::parse(text).unwrap();
            for k in [0, 1, 10, 1000] {
                for count in [false, true] {
                    let options = LexicalOptions {
                        k,
                        count,
                        execution: LexicalExecution::DocumentAtATime,
                        ..Default::default()
                    };
                    let actual = execute(&store, &query, &options).unwrap();
                    assert_eq!(actual.count, count.then_some(expected), "{text}");
                    assert_eq!(actual.hits.len(), expected.min(k), "{text}");
                    let reference = execute(
                        &store,
                        &query,
                        &LexicalOptions {
                            execution: LexicalExecution::Exhaustive,
                            ..options
                        },
                    )
                    .unwrap();
                    assert_eq!(actual, reference, "{text}, k={k}, count={count}");
                }
            }
        }
    }
}

#[test]
fn parser_rejects_unsupported_syntax_and_limits() {
    for text in [
        "+",
        "-",
        "+ red",
        "\"red",
        "\"\"",
        "red*",
        "a:b",
        "(red)",
        "red^2",
        "\"red blue\"~1",
        "red/blue",
    ] {
        assert!(LexicalQuery::parse(text).is_err(), "{text}");
    }
    assert!(LexicalQuery::parse(&vec!["red"; 1024].join(" ")).is_ok());
    assert!(LexicalQuery::parse(&vec!["red"; 1025].join(" ")).is_err());
    assert!(LexicalQuery::parse(&"x".repeat(MAX_DOCUMENT_BYTES + 1)).is_err());
}

#[test]
fn phrases_require_actual_positions_and_follow_ingest_analyzer() {
    let mut store = DocumentStore::default();
    insert(&mut store, "legacy", "red blue", false, true);
    assert!(execute(
        &store,
        &LexicalQuery::parse("\"red blue\"").unwrap(),
        &LexicalOptions::default()
    )
    .unwrap_err()
    .to_string()
    .contains("unsupported index"));
    assert_eq!(run(&store, "+red +blue", 10).count, Some(1));
    store.remove("legacy");
    insert(&mut store, "stemmed", "Running FOXES running", true, false);
    insert(&mut store, "raw", "running foxes running", true, true);
    assert_eq!(run(&store, "\"run fox\"", 10).count, Some(1));
    assert_eq!(run(&store, "\"running foxes\"", 10).count, Some(2));
}

#[test]
fn live_visibility_chunk_deduplication_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().display().to_string();
    let core = HmsCore::new(64, Some(path.clone()), None).unwrap();
    let ingest = options(true, true);
    let mut document = input("multi", "red blue red blue");
    document.chunk_words = Some(2);
    core.memorize_documents_with(vec![document, input("other", "blue red")], &ingest)
        .unwrap();
    let query = LexicalQuery::parse("\"red blue\"").unwrap();
    assert_eq!(core.count_lexical(&query).unwrap(), 1);
    let held = core
        .search_lexical(
            &query,
            &LexicalOptions {
                k: 10,
                count: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(held.hits.len(), 2);
    core.delete_document("multi").unwrap();
    core.memorize_documents_with(vec![input("other", "red blue")], &ingest)
        .unwrap();
    core.flush().unwrap();
    drop(core);
    let core = HmsCore::new(64, Some(path), None).unwrap();
    assert_eq!(core.count_lexical(&query).unwrap(), 1);
    assert_eq!(
        core.search_lexical(&query, &LexicalOptions::default())
            .unwrap()
            .hits[0]
            .document_id,
        "other"
    );
    assert_eq!(held.hits[0].document_id, "multi");
    core.compact().unwrap();
    assert_eq!(core.count_lexical(&query).unwrap(), 1);
}

#[test]
fn phrase_ordinals_follow_replacement_slot_reuse_and_replay() {
    fn check(core: &HmsCore) {
        for text in [
            "\"red red\"",
            "\"running foxes\"",
            "\"run fox\"",
            "+\"red red\" +foxes",
            "+red -\"running foxes\"",
        ] {
            let query = LexicalQuery::parse(text).unwrap();
            for k in [0, 10] {
                for count in [false, true] {
                    let options = LexicalOptions {
                        k,
                        count,
                        ..Default::default()
                    };
                    let expected = core
                        .search_lexical(
                            &query,
                            &LexicalOptions {
                                execution: LexicalExecution::Exhaustive,
                                ..options.clone()
                            },
                        )
                        .unwrap();
                    assert_eq!(
                        core.search_lexical(&query, &options).unwrap(),
                        expected,
                        "{text}"
                    );
                }
            }
        }
    }
    for count in [127, 128, 129] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().display().to_string();
        let core = HmsCore::new(64, Some(path.clone()), None).unwrap();
        for unstemmed in [false, true] {
            let documents = (0..count)
                .map(|i| input(&format!("{unstemmed}-{i}"), "red red running foxes zebra"))
                .collect();
            core.memorize_documents_with(documents, &options(true, unstemmed))
                .unwrap();
        }
        check(&core);
        for text in [
            "aardvark red red running foxes",
            "aardvark running red foxes red",
        ] {
            for unstemmed in [false, true] {
                let documents = (0..count)
                    .map(|i| input(&format!("{unstemmed}-{i}"), text))
                    .collect();
                core.memorize_documents_with(documents, &options(true, unstemmed))
                    .unwrap();
            }
            check(&core);
        }
        for i in 0..count {
            core.delete_document(&format!("false-{i}")).unwrap();
        }
        let mut moved = input("true-0", "running foxes red red running foxes");
        moved.chunk_words = Some(2);
        core.memorize_documents_with(vec![moved], &options(true, false))
            .unwrap();
        let reused = (0..count)
            .map(|i| input(&format!("reused-{i}"), "zebra red red running foxes"))
            .collect();
        core.memorize_documents_with(reused, &options(true, true))
            .unwrap();
        check(&core);
        core.flush().unwrap();
        drop(core);
        let core = HmsCore::new(64, Some(path), None).unwrap();
        check(&core);
    }
}

#[test]
fn ties_and_block_boundary_sizes() {
    for count in [0, 1, 127, 128, 129, 255, 256, 257] {
        let mut store = DocumentStore::default();
        for id in (0..count).rev() {
            insert(&mut store, &format!("d{id:04}"), "red red blue", true, true);
        }
        for k in [0, 1, 10, 127, 128, 129, 1000] {
            let result = run(&store, "red", k);
            assert_eq!(result.count, Some(count));
            assert_eq!(result.hits.len(), count.min(k));
            assert!(result
                .hits
                .windows(2)
                .all(|w| w[0].document_id < w[1].document_id));
        }
    }
}

#[test]
fn bm25_parameters_and_nonfinite_scores_are_explicit() {
    let mut store = DocumentStore::default();
    insert(&mut store, "a", "red red red red", true, true);
    let query = LexicalQuery::parse("red").unwrap();
    for (k1, b) in [(0.0, 0.0), (0.0, 1.0), (1.2, 0.75), (2.0, 1.0)] {
        let result = execute(
            &store,
            &query,
            &LexicalOptions {
                bm25_k1: k1,
                bm25_b: b,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.hits[0].score.is_finite());
    }
    for (k1, b) in [
        (f64::NAN, 0.0),
        (1.2, f64::NAN),
        (1.2, -0.01),
        (-1.0, 0.75),
        (1.2, 1.01),
        (f64::MAX, 0.0),
    ] {
        assert!(execute(
            &store,
            &query,
            &LexicalOptions {
                bm25_k1: k1,
                bm25_b: b,
                ..Default::default()
            }
        )
        .is_err());
    }
}

fn check_model(store: &DocumentStore, texts: &[(String, Vec<String>)], left: usize, right: usize) {
    let words = ["red", "blue", "green", "gold", "absent"];
    let (a, b) = (words[left % words.len()], words[right % words.len()]);
    for query in [
        a.to_owned(),
        format!("+{a} -\"{a} {b}\""),
        format!("{a} {b}"),
        format!("+{a} +{b}"),
        format!("+{a} {b}"),
        format!("+{a} -{b}"),
        format!("\"{a} {b}\""),
        format!("+\"{a} {b}\" +green"),
    ] {
        let expected = texts
            .iter()
            .filter(|(_, tokens)| {
                let has_a = tokens.iter().any(|t| t == a);
                let has_b = tokens.iter().any(|t| t == b);
                if !query.contains(' ') {
                    has_a
                } else if query.contains("-\"") {
                    has_a && !tokens.windows(2).any(|p| p == [a, b])
                } else if query.starts_with("+\"") {
                    tokens.windows(2).any(|p| p == [a, b]) && tokens.iter().any(|t| t == "green")
                } else if query.starts_with('"') {
                    tokens.windows(2).any(|p| p == [a, b])
                } else if query.contains(" -") {
                    has_a && !has_b
                } else if query.contains(" +") {
                    has_a && has_b
                } else if query.starts_with('+') {
                    has_a
                } else {
                    has_a || has_b
                }
            })
            .count();
        assert_eq!(run(store, &query, 10).count, Some(expected), "{query}");
        let parsed = LexicalQuery::parse(&query).unwrap();
        for k in [0, 1, 3, 10, 129, 1000] {
            let reference = execute(
                store,
                &parsed,
                &LexicalOptions {
                    k,
                    count: true,
                    execution: LexicalExecution::Exhaustive,
                    ..Default::default()
                },
            )
            .unwrap();
            for execution in [
                LexicalExecution::Auto,
                LexicalExecution::DocumentAtATime,
                LexicalExecution::BlockMaxWand,
            ] {
                for count in [false, true] {
                    let actual = execute(
                        store,
                        &parsed,
                        &LexicalOptions {
                            k,
                            count,
                            execution,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    assert_eq!(actual.hits, reference.hits, "{query}, k={k}, {execution:?}");
                    assert_eq!(
                        actual.count,
                        count.then_some(expected),
                        "{query}, k={k}, {execution:?}"
                    );
                }
            }
        }
    }
}

/// Whether every block bound of the test vocabulary's lists holds for the store's lengths.
fn bounds_cover(store: &DocumentStore) -> bool {
    let words: Vec<u32> = (0..store.slot_count)
        .map(|slot| store.slot_word_count(slot))
        .collect();
    (0..2).all(|form| {
        ["red", "blue", "green", "gold"].iter().all(|word| {
            store.postings[form]
                .get(word)
                .is_none_or(|list| list.bounds_cover(&words))
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn differential_insert_replace_delete_sequences(
        corpus in prop::collection::vec(prop::collection::vec(0usize..4, 1..20), 1..140),
        left in 0usize..5, right in 0usize..5,
    ) {
        let words = ["red", "blue", "green", "gold"];
        let mut store = DocumentStore::default();
        let mut texts = Vec::new();
        for (i, row) in corpus.iter().enumerate() {
            let tokens: Vec<String> = row.iter().map(|&w| words[w].to_owned()).collect();
            let id = format!("d{i:04}");
            insert(&mut store, &id, &tokens.join(" "), true, i % 2 == 0);
            texts.push((id, tokens));
        }
        check_model(&store, &texts, left, right);
        for i in (0..corpus.len()).step_by(7) {
            let id = format!("d{i:04}");
            store.remove(&id);
            texts.retain(|(key, _)| key != &id);
        }
        prop_assert!(bounds_cover(&store));
        for i in (0..corpus.len()).step_by(11) {
            let id = format!("d{i:04}");
            insert(&mut store, &id, "gold red blue gold", true, true);
            texts.retain(|(key, _)| key != &id);
            texts.push((id, ["gold", "red", "blue", "gold"].map(str::to_owned).to_vec()));
        }
        check_model(&store, &texts, left, right);
        prop_assert!(bounds_cover(&store));
    }
}

#[test]
fn whole_articles_preserve_conjunction_exclusion_and_cross_boundary_phrases() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().display().to_string();
    let core = HmsCore::new(64, Some(path.clone()), None).unwrap();
    let mut article = vec!["filler"; 4095];
    article.extend(["red", "blue", "fox"]);
    let ingest = IngestOptions {
        whole_document: true,
        ..options(true, true)
    };
    core.memorize_documents_with(vec![input("article", &article.join(" "))], &ingest)
        .unwrap();
    for (query, expected) in [("+red +fox", 1), ("+red -fox", 0), ("\"red blue\"", 1)] {
        assert_eq!(
            core.count_lexical(&LexicalQuery::parse(query).unwrap())
                .unwrap(),
            expected
        );
    }
    for length in [65_535, 65_536, 65_537] {
        core.memorize_documents_with(
            vec![input("wide", &format!("{} red blue", "x".repeat(length)))],
            &ingest,
        )
        .unwrap();
    }
    core.flush().unwrap();
    drop(core);
    let core = HmsCore::new(64, Some(path), None).unwrap();
    assert_eq!(
        core.count_lexical(&LexicalQuery::parse("\"red blue\"").unwrap())
            .unwrap(),
        2
    );
    core.compact().unwrap();
    core.delete_document("wide").unwrap();
    assert_eq!(
        core.count_lexical(&LexicalQuery::parse("+red -fox").unwrap())
            .unwrap(),
        0
    );
    assert!(core
        .memorize_documents_with(
            vec![input("bad", "red blue")],
            &IngestOptions {
                whole_document: true,
                ..Default::default()
            }
        )
        .is_err());
}

#[test]
fn shared_whole_article_batches_preserve_pinned_owners_at_group_boundaries() {
    let ingest = IngestOptions {
        whole_document: true,
        ..options(true, true)
    };
    let template = documents::prepare_with_options(
        input("template", "red blue red who"),
        64,
        None,
        None,
        TextEncoder::default(),
        &ingest,
    )
    .unwrap();
    for count in [
        documents::MAX_BATCH_ITEMS - 1,
        documents::MAX_BATCH_ITEMS,
        documents::MAX_BATCH_ITEMS + 1,
    ] {
        let batch: Vec<_> = (0..count)
            .map(|i| {
                let mut document = template.clone();
                document.id = format!("d{i:05}");
                std::sync::Arc::new(document)
            })
            .collect();
        let pinned = std::sync::Arc::clone(&batch[0]);
        let before = serde_json::to_vec(pinned.as_ref()).unwrap();
        let mut store = DocumentStore::default();
        store.insert_batch(batch);
        assert_eq!(run(&store, "+\"red blue\" +who", 10).count, Some(count));
        assert_eq!(run(&store, "red blue", 10).count, Some(count));
        store.remove("d00000");
        assert_eq!(run(&store, "red blue", 10).count, Some(count - 1));
        drop(store);
        assert_eq!(serde_json::to_vec(pinned.as_ref()).unwrap(), before);
        assert_eq!(pinned.chunks[0].terms.get("red"), Some(&2));
    }
}
