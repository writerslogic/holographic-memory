// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The local embedder through the engine: ingest, search and persistence with
//! `TextEncoder::LocalEmbedV1`, using a random-weight pack so no model download is needed.
//! Retrieval quality on real weights is measured by benchmarks/embedding, not here.
#![cfg(feature = "local-embed")]

use std::sync::Arc;

use holographic_memory::core::embed::testing::{write_tiny, WORDS};
use holographic_memory::core::embed::{EmbedLimits, EmbedOptions, Embedder, Pooling, Quantization};
use holographic_memory::core::encoding::TextEncoder;
use holographic_memory::core::HmsConfig;
use holographic_memory::{DocumentInput, HmsCore, SearchOptions};

fn input(id: &str, text: &str) -> DocumentInput {
    DocumentInput {
        id: id.into(),
        text: text.into(),
        source_uri: None,
        version: None,
        metadata: None,
        chunk_words: None,
        overlap_words: None,
        store_text: None,
        embeddings: None,
        title_weight: None,
    }
}

fn corpus() -> Vec<(String, String)> {
    // Distinct texts so each document's text identifies it.
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let mut i = 0usize;
    while out.len() < 40 {
        let words: Vec<&str> = (0..6 + i % 5)
            .map(|j| WORDS[(i * 7 + j * 3 + j * j * (i % 4 + 1)) % WORDS.len()])
            .collect();
        let text = words.join(" ");
        if seen.insert(text.clone()) {
            out.push((format!("doc-{}", out.len()), text));
        }
        i += 1;
    }
    out
}

fn config(path: &std::path::Path, sha: &str) -> HmsConfig {
    HmsConfig {
        text_encoder: TextEncoder::LocalEmbedV1,
        local_embed: Some(EmbedOptions {
            model: path.to_string_lossy().into_owned(),
            sha256: Some(sha.into()),
            threads: 2,
            ..EmbedOptions::default()
        }),
        ..HmsConfig::default()
    }
}

#[test]
fn embedding_limits_fail_before_mutation_and_batches_share_one_admission() {
    use holographic_memory::MemorizeBatchItem;
    let dir = tempfile::tempdir().unwrap();
    let (pack, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let mut cfg = config(&pack, &sha);
    cfg.local_embed.as_mut().unwrap().limits = EmbedLimits {
        max_input_bytes: 64,
        max_concurrent_calls: 1,
        ..Default::default()
    };
    let path = dir.path().join("bounded").display().to_string();
    let core = HmsCore::new(1024, Some(path.clone()), Some(cfg.clone())).unwrap();
    let items: Vec<_> = (0..8)
        .map(|i| MemorizeBatchItem {
            id: format!("original-{i}"),
            text: "cat".into(),
        })
        .collect();
    core.memorize_batch(&items).unwrap();
    assert_eq!(core.vector_count(), 8);
    let oversized = "cat ".repeat(17);
    assert!(core.try_encode_text(&oversized).is_err());
    assert!(std::panic::catch_unwind(
        std::panic::AssertUnwindSafe(|| core.encode_text(&oversized))
    )
    .is_err());
    assert!(core.memorize_meaning("meaning", &oversized).is_err());
    assert!(core
        .memorize_triplet(
            "triple".into(),
            "cat".into(),
            oversized.clone(),
            "dog".into()
        )
        .is_err());
    assert!(core
        .memorize_sequence("sequence".into(), &["cat".into(), oversized.clone()])
        .is_err());
    assert!(core
        .memorize_batch(&[
            MemorizeBatchItem {
                id: "tentative".into(),
                text: "dog".into()
            },
            MemorizeBatchItem {
                id: "rejected".into(),
                text: oversized.clone()
            },
        ])
        .is_err());
    assert_eq!(core.vector_count(), 8);
    assert!(core
        .query_triplet("cat".into(), oversized.clone(), 10)
        .is_err());
    assert!(core.try_find_analogy("cat", "dog", &oversized, 10).is_err());
    assert!(core.query_sequence(&[oversized], 10).is_err());
    let query = core.try_encode_text("cat").unwrap();
    assert_eq!(core.query(&query, 16).len(), 8);
    assert!(core
        .query(&query, 16)
        .iter()
        .all(|hit| hit.similarity == 1.0));
    assert!(core.try_encode_text(" \t").unwrap().indices().is_empty());
    core.flush().unwrap();
    drop(core);
    let core = HmsCore::new(1024, Some(path), Some(cfg)).unwrap();
    assert_eq!(core.vector_count(), 8);
    assert_eq!(core.query(&query, 16).len(), 8);
    assert!(core
        .query(&query, 16)
        .iter()
        .all(|hit| hit.similarity == 1.0));
}

#[test]
fn corpus_round_trip_through_ingest_and_search() {
    let dir = tempfile::tempdir().unwrap();
    let (pack, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let store = dir.path().join("store");
    let core = HmsCore::new(
        2048,
        Some(store.to_string_lossy().into()),
        Some(config(&pack, &sha)),
    )
    .unwrap();
    let docs = corpus();
    let counts = core
        .memorize_documents(docs.iter().map(|(id, t)| input(id, t)).collect())
        .unwrap();
    assert!(counts.iter().all(|&c| c == 1));

    // Pure semantic search (lexical weight 0) must rank each document first for its own text:
    // the query embedding and the chunk embedding come from the same model on the same string.
    let options = SearchOptions {
        k: Some(3),
        lexical_weight: Some(0.0),
        ..SearchOptions::default()
    };
    for (id, text) in &docs {
        let hits = core.search_documents(text, &options).unwrap();
        assert_eq!(
            &hits[0].document_id, id,
            "own text must retrieve its document"
        );
    }
    // The sparse text vectors are sketches of the embedding, so identical text is identical and
    // different text differs.
    let a = core.encode_text(&docs[0].1);
    assert_eq!(a.indices(), core.encode_text(&docs[0].1).indices());
    assert_ne!(a.indices(), core.encode_text(&docs[1].1).indices());
    assert!(core.encode_text("   ").indices().is_empty());

    // The store persists with its embedding space; reopening with the same pack works and
    // search results are identical.
    let before = core.search_documents(&docs[5].1, &options).unwrap();
    drop(core);
    let core = HmsCore::new(
        2048,
        Some(store.to_string_lossy().into()),
        Some(config(&pack, &sha)),
    )
    .unwrap();
    let after = core.search_documents(&docs[5].1, &options).unwrap();
    assert_eq!(
        before
            .iter()
            .map(|h| (&h.id, h.score.to_bits()))
            .collect::<Vec<_>>(),
        after
            .iter()
            .map(|h| (&h.id, h.score.to_bits()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_store_rejects_a_different_model_pack() {
    let dir = tempfile::tempdir().unwrap();
    let (int8, sha8) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let (f32p, sha32) = write_tiny(dir.path(), Quantization::F32, Pooling::Mean);
    let store = dir.path().join("store");
    let path = Some(store.to_string_lossy().into_owned());
    let core = HmsCore::new(1024, path.clone(), Some(config(&int8, &sha8))).unwrap();
    core.memorize_document(input("a", "the cat sat on the mat"))
        .unwrap();
    drop(core);
    // Same architecture, different quantization: the embedding space revision differs.
    let err = HmsCore::new(1024, path, Some(config(&f32p, &sha32)))
        .err()
        .expect("must reject")
        .to_string();
    assert!(err.contains("incompatible store schema"), "{err}");
}

#[test]
fn effective_options_and_same_metadata_different_weights_cannot_relabel_a_store() {
    let dir = tempfile::tempdir().unwrap();
    let (pack, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let store = dir.path().join("store");
    let path = Some(store.display().to_string());
    let original = config(&pack, &sha);
    let core = HmsCore::new(1024, path.clone(), Some(original.clone())).unwrap();
    core.memorize_document(input("fact", "the cat sat on the mat"))
        .unwrap();
    drop(core);
    let base = original.local_embed.clone().unwrap();
    for options in [
        EmbedOptions {
            pooling: Some(Pooling::Cls),
            ..base.clone()
        },
        EmbedOptions {
            normalize: Some(false),
            ..base.clone()
        },
        EmbedOptions {
            max_length: Some(8),
            ..base.clone()
        },
        EmbedOptions {
            query_prefix: Some("cat ".into()),
            ..base.clone()
        },
        EmbedOptions {
            passage_prefix: Some("cat ".into()),
            ..base
        },
    ] {
        let mut changed = original.clone();
        changed.local_embed = Some(options);
        assert!(HmsCore::new(1024, path.clone(), Some(changed))
            .err()
            .expect("must reject changed embedding function")
            .to_string()
            .contains("incompatible store schema"));
    }
    let bytes = holographic_memory::core::embed::testing::tiny_pack_bytes_with_group(
        Quantization::Int8,
        Pooling::Mean,
        32,
    );
    let changed_pack = dir.path().join("other-int8.hmsemb");
    std::fs::write(&changed_pack, &bytes).unwrap();
    let changed_sha = holographic_memory::core::embed::sha256_hex(&bytes);
    assert!(HmsCore::new(
        1024,
        path.clone(),
        Some(config(&changed_pack, &changed_sha))
    )
    .is_err());
    let reopened = HmsCore::new(1024, path, Some(original)).unwrap();
    assert_eq!(reopened.document_count(), 1);
}

#[test]
fn unnormalized_model_outputs_keep_truthful_identity_and_cosine_search() {
    let dir = tempfile::tempdir().unwrap();
    let (pack, sha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Mean);
    let store = dir.path().join("store");
    let mut cfg = config(&pack, &sha);
    cfg.local_embed.as_mut().unwrap().normalize = Some(false);
    let path = Some(store.display().to_string());
    let core = HmsCore::new(1024, path.clone(), Some(cfg.clone())).unwrap();
    core.memorize_document(input("cat", "the cat sat on the mat"))
        .unwrap();
    core.memorize_document(input("dog", "the dog ran in the park"))
        .unwrap();
    let query = SearchOptions {
        k: Some(1),
        lexical_weight: Some(0.0),
        ..Default::default()
    };
    assert_eq!(
        core.search_documents("the cat sat on the mat", &query)
            .unwrap()[0]
            .document_id,
        "cat"
    );
    drop(core);
    let schema: serde_json::Value =
        serde_json::from_slice(&std::fs::read(store.join("store.json")).unwrap()).unwrap();
    assert_eq!(schema["embedding_space"]["normalization"], "none");
    let reopened = HmsCore::new(1024, path.clone(), Some(cfg)).unwrap();
    assert_eq!(
        reopened
            .search_documents("the dog ran in the park", &query)
            .unwrap()[0]
            .document_id,
        "dog"
    );
    drop(reopened);
    assert!(HmsCore::new(1024, path, Some(config(&pack, &sha))).is_err());
}

#[test]
fn encoder_without_its_embedder_fails_at_open_not_at_query() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = HmsConfig {
        text_encoder: TextEncoder::LocalEmbedV1,
        local_embed: Some(EmbedOptions {
            model: dir
                .path()
                .join("missing.hmsemb")
                .to_string_lossy()
                .into_owned(),
            ..EmbedOptions::default()
        }),
        ..HmsConfig::default()
    };
    let err = HmsCore::new(
        1024,
        Some(dir.path().join("s").to_string_lossy().into()),
        Some(cfg),
    )
    .err()
    .expect("must fail")
    .to_string();
    assert!(err.contains("missing.hmsemb"), "{err}");
}

#[test]
fn a_lexical_store_can_adopt_an_embedder_and_default_encoding_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let (pack, sha) = write_tiny(dir.path(), Quantization::F32, Pooling::Cls);
    let embedder = Arc::new(
        Embedder::new(&EmbedOptions {
            model: pack.to_string_lossy().into(),
            sha256: Some(sha),
            ..EmbedOptions::default()
        })
        .unwrap(),
    );
    let cfg = HmsConfig {
        embedding_space: Some(embedder.space()),
        ..HmsConfig::default()
    };
    assert_eq!(cfg.text_encoder.name(), "multiscale-words-v1");
    let core = HmsCore::new(
        2048,
        Some(dir.path().join("s").to_string_lossy().into()),
        Some(cfg),
    )
    .unwrap();
    let lexical_before = core.encode_text("the cat sat on the mat");
    core.set_text_embedder(Some(embedder)).unwrap();
    // The text encoder stays lexical: installing an embedder adds dense vectors, nothing else.
    assert_eq!(
        lexical_before.indices(),
        core.encode_text("the cat sat on the mat").indices()
    );
    core.memorize_documents(corpus().iter().map(|(id, t)| input(id, t)).collect())
        .unwrap();
    assert_eq!(core.document_embedding_count(), 40);
    let (id, text) = &corpus()[9];
    let hits = core
        .search_documents(
            text,
            &SearchOptions {
                k: Some(1),
                lexical_weight: Some(0.0),
                ..SearchOptions::default()
            },
        )
        .unwrap();
    assert_eq!(&hits[0].document_id, id);
    // A mismatched embedder is refused.
    let (other, osha) = write_tiny(dir.path(), Quantization::Int8, Pooling::Cls);
    let other = Arc::new(
        Embedder::new(&EmbedOptions {
            model: other.to_string_lossy().into(),
            sha256: Some(osha),
            ..EmbedOptions::default()
        })
        .unwrap(),
    );
    assert!(core.set_text_embedder(Some(other)).is_err());
    // Removing it is allowed for a lexical store.
    core.set_text_embedder(None).unwrap();
}
