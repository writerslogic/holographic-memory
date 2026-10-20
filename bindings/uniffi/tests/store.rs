// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use hms_mobile::{chunk_document, DocumentInput, HmsError, HmsStore, SearchOptions, StoreOptions};

fn doc(id: &str, text: &str) -> DocumentInput {
    DocumentInput {
        id: id.into(),
        text: text.into(),
        source_uri: None,
        version: None,
        metadata_json: None,
        chunk_words: None,
        overlap_words: None,
        store_text: None,
        embeddings: None,
        title_weight: None,
    }
}

fn small() -> StoreOptions {
    StoreOptions {
        dimensions: 1024,
        ..Default::default()
    }
}

fn open(dir: &std::path::Path) -> Arc<HmsStore> {
    HmsStore::open_blocking(dir.join("store").display().to_string(), Some(small())).unwrap()
}

fn top_ids(store: &HmsStore, query: &str, options: Option<SearchOptions>) -> Vec<String> {
    store
        .search_blocking(query.into(), options)
        .unwrap()
        .into_iter()
        .map(|hit| hit.document_id)
        .collect()
}

#[test]
fn ingest_search_delete_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    assert_eq!(
        store
            .ingest_blocking(doc("a", "the quick brown fox jumps"))
            .unwrap(),
        1
    );
    store
        .ingest_blocking(doc("b", "a slow green turtle crawls"))
        .unwrap();
    assert_eq!(top_ids(&store, "turtle", None), ["b"]);
    let hit = &store.search_blocking("fox".into(), None).unwrap()[0];
    assert_eq!(hit.text.as_deref(), Some("the quick brown fox jumps"));
    assert!(hit.lexical_score > 0.0);

    let stats = store.stats().unwrap();
    assert_eq!((stats.document_count, stats.chunk_count), (2, 2));
    assert!(stats.disk_bytes > 0 && !stats.encrypted);

    assert!(store.delete_document_blocking("a".into()).unwrap());
    assert!(!store.delete_document_blocking("a".into()).unwrap());
    assert!(!store.has_document("a".into()).unwrap());
    store.shutdown();
    drop(store);

    let reopened = open(dir.path());
    assert_eq!(reopened.stats().unwrap().document_count, 1);
    assert_eq!(top_ids(&reopened, "turtle", None), ["b"]);
    assert!(top_ids(&reopened, "fox", None).is_empty());
}

#[test]
fn replacing_a_document_by_id_replaces_its_content() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    store.ingest_blocking(doc("a", "alpha original")).unwrap();
    store
        .ingest_blocking(doc("a", "bravo replacement"))
        .unwrap();
    assert_eq!(store.stats().unwrap().document_count, 1);
    assert!(top_ids(&store, "alpha", None).is_empty());
    assert_eq!(top_ids(&store, "bravo", None), ["a"]);
}

#[test]
fn metadata_filter_source_and_k() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    for (id, group) in [("a", "x"), ("b", "y"), ("c", "x")] {
        let mut input = doc(id, "shared term");
        input.metadata_json = Some(format!(r#"{{"group":"{group}"}}"#));
        input.source_uri = Some(format!("file:{id}"));
        store.ingest_blocking(input).unwrap();
    }
    let filtered = SearchOptions {
        filter_json: Some(r#"{"group":"x"}"#.into()),
        ..Default::default()
    };
    let mut ids = top_ids(&store, "shared", Some(filtered));
    ids.sort();
    assert_eq!(ids, ["a", "c"]);
    let sourced = SearchOptions {
        source_uri: Some("file:b".into()),
        ..Default::default()
    };
    assert_eq!(top_ids(&store, "shared", Some(sourced)), ["b"]);
    let limited = SearchOptions {
        k: Some(1),
        ..Default::default()
    };
    assert_eq!(top_ids(&store, "shared", Some(limited)).len(), 1);
    let hit = &store.search_blocking("shared".into(), None).unwrap()[0];
    assert!(hit.metadata_json.contains("group"));
}

#[test]
fn bm25_options_change_scores_and_are_validated() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    store.ingest_blocking(doc("short", "needle")).unwrap();
    store
        .ingest_blocking(doc(
            "long",
            &format!("needle needle needle {}", "filler ".repeat(200)),
        ))
        .unwrap();
    let score = |options: Option<SearchOptions>, id: &str| {
        store
            .search_blocking("needle".into(), options)
            .unwrap()
            .into_iter()
            .find(|hit| hit.document_id == id)
            .unwrap()
            .score
    };
    let no_length_norm = SearchOptions {
        bm25_b: Some(0.0),
        ..Default::default()
    };
    assert_ne!(score(None, "long"), score(Some(no_length_norm), "long"));
    let bad = SearchOptions {
        bm25_b: Some(1.5),
        ..Default::default()
    };
    assert!(matches!(
        store.search_blocking("needle".into(), Some(bad)),
        Err(HmsError::InvalidInput { .. })
    ));
}

#[test]
fn title_weight_lifts_a_title_match() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    let body = "filler words about nothing in particular ".repeat(30);
    let mut titled = doc("titled", &format!("zebra\n{body}"));
    titled.title_weight = Some(8);
    store.ingest_blocking(titled).unwrap();
    store
        .ingest_blocking(doc("plain", &format!("zebra\n{body}")))
        .unwrap();
    let hits = store.search_blocking("zebra".into(), None).unwrap();
    assert_eq!(hits[0].document_id, "titled");
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn bad_input_is_a_typed_error() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    let mut input = doc("a", "text");
    input.metadata_json = Some("[1,2]".into());
    assert!(matches!(
        store.ingest_blocking(input),
        Err(HmsError::InvalidInput { .. })
    ));
    assert!(matches!(
        store.ingest_blocking(doc("", "text")),
        Err(HmsError::InvalidInput { .. })
    ));
    assert!(matches!(
        HmsStore::open_blocking(
            dir.path().join("tiny").display().to_string(),
            Some(StoreOptions {
                dimensions: 0,
                ..Default::default()
            })
        ),
        Err(HmsError::InvalidInput { .. })
    ));
}

#[test]
fn chunking_reports_source_offsets() {
    let mut input = doc("a", &"word ".repeat(500));
    input.chunk_words = Some(100);
    let chunks = chunk_document(input).unwrap();
    assert!(chunks.len() >= 5);
    assert!(chunks.windows(2).all(|w| w[0].end_byte <= w[1].end_byte));
}

#[test]
fn a_directory_opens_once() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    let second = HmsStore::open_blocking(dir.path().join("store").display().to_string(), None);
    assert!(matches!(second, Err(HmsError::StoreLocked { .. })));
    store.shutdown();
    drop(store);
    open(dir.path());
}

#[test]
fn closed_handles_fail_with_closed() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    store.shutdown();
    assert!(matches!(store.stats(), Err(HmsError::Closed)));
    assert!(matches!(
        store.ingest_blocking(doc("a", "x")),
        Err(HmsError::Closed)
    ));
}

#[test]
fn encrypted_store_requires_its_key_and_hides_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("secure").display().to_string();
    let keyed = |key: Option<&str>| StoreOptions {
        dimensions: 1024,
        encryption_key: key.map(str::to_owned),
        ..Default::default()
    };
    let secret = "uniquemarkerphrase correct horse battery staple";
    {
        let store =
            HmsStore::open_blocking(path.clone(), Some(keyed(Some("passphrase-1")))).unwrap();
        store.ingest_blocking(doc("a", secret)).unwrap();
        assert!(store.stats().unwrap().encrypted);
        store.compact_blocking().unwrap();
    }
    let mut plaintext_files = Vec::new();
    for entry in walk(std::path::Path::new(&path)) {
        let bytes = std::fs::read(&entry).unwrap();
        if bytes
            .windows(b"uniquemarkerphrase".len())
            .any(|w| w == b"uniquemarkerphrase")
        {
            plaintext_files.push(entry);
        }
    }
    assert!(
        plaintext_files.is_empty(),
        "plaintext found in {plaintext_files:?}"
    );

    assert!(matches!(
        HmsStore::open_blocking(path.clone(), Some(keyed(None))),
        Err(HmsError::IncompatibleStore { .. })
    ));
    let wrong = HmsStore::open_blocking(path.clone(), Some(keyed(Some("not-the-key"))));
    assert!(
        matches!(
            wrong,
            Err(HmsError::Encryption { .. }) | Err(HmsError::Corrupt { .. })
        ),
        "wrong key gave {:?}",
        wrong.err()
    );
    let store = HmsStore::open_blocking(path, Some(keyed(Some("passphrase-1")))).unwrap();
    assert_eq!(top_ids(&store, "uniquemarkerphrase", None), ["a"]);
}

fn walk(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(walk(&path));
        } else {
            out.push(path);
        }
    }
    out
}

fn populate(store: &HmsStore) {
    for i in 0..40 {
        let mut input = doc(
            &format!("d{i}"),
            &format!("document number{i} about topic{} and shared", i % 5),
        );
        input.metadata_json = Some(format!(r#"{{"n":{i}}}"#));
        store.ingest_blocking(input).unwrap();
    }
}

#[test]
fn snapshot_restores_identical_results_and_survives_later_writes() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    populate(&store);
    let before: Vec<_> = (0..5)
        .map(|t| top_ids(&store, &format!("topic{t}"), None))
        .collect();
    let snap = dir.path().join("snap");
    let info = store.snapshot_blocking(snap.display().to_string()).unwrap();
    assert_eq!(info.document_count, 40);
    assert!(info.file_count > 0 && info.total_bytes > 0);
    store
        .ingest_blocking(doc("late", "added after the snapshot"))
        .unwrap();

    let restored = HmsStore::restore_blocking(
        snap.display().to_string(),
        dir.path().join("restored").display().to_string(),
        Some(small()),
    )
    .unwrap();
    assert_eq!(restored.stats().unwrap().document_count, 40);
    assert!(!restored.has_document("late".into()).unwrap());
    let after: Vec<_> = (0..5)
        .map(|t| top_ids(&restored, &format!("topic{t}"), None))
        .collect();
    assert_eq!(before, after);

    // The snapshot is not tied to the live store.
    store.shutdown();
    drop(store);
    std::fs::remove_dir_all(dir.path().join("store")).unwrap();
    drop(restored);
    let again = HmsStore::restore_blocking(
        snap.display().to_string(),
        dir.path().join("again").display().to_string(),
        Some(small()),
    )
    .unwrap();
    assert_eq!(again.stats().unwrap().document_count, 40);
}

#[test]
fn snapshot_refuses_bad_destinations_and_restore_detects_tampering() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    populate(&store);
    let snap = dir.path().join("snap");
    store.snapshot_blocking(snap.display().to_string()).unwrap();
    assert!(matches!(
        store.snapshot_blocking(snap.display().to_string()),
        Err(HmsError::AlreadyExists { .. })
    ));
    assert!(matches!(
        store.snapshot_blocking(dir.path().join("store/inside").display().to_string()),
        Err(HmsError::InvalidInput { .. })
    ));

    let victim = walk(&snap)
        .into_iter()
        .filter(|p| p.file_name().unwrap() != "hms-snapshot.json")
        .max_by_key(|p| std::fs::metadata(p).unwrap().len())
        .unwrap();
    let mut bytes = std::fs::read(&victim).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    std::fs::write(&victim, bytes).unwrap();
    let target = dir.path().join("tampered");
    assert!(matches!(
        HmsStore::restore_blocking(
            snap.display().to_string(),
            target.display().to_string(),
            Some(small())
        ),
        Err(HmsError::Corrupt { .. })
    ));
    assert!(!target.exists(), "a failed restore leaves nothing behind");
    assert!(matches!(
        HmsStore::restore_blocking(
            dir.path().join("missing").display().to_string(),
            dir.path().join("x").display().to_string(),
            None
        ),
        Err(HmsError::NotFound { .. })
    ));
}

#[test]
fn snapshot_while_searching_and_ingesting_stays_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let store = open(dir.path());
    populate(&store);
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut workers = Vec::new();
    for t in 0..3 {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        workers.push(std::thread::spawn(move || {
            let mut i = 0;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                store
                    .ingest_blocking(doc(
                        &format!("w{t}-{i}"),
                        &format!("worker{t} item{i} shared"),
                    ))
                    .unwrap();
                store.search_blocking("shared".into(), None).unwrap();
                i += 1;
            }
        }));
    }
    let mut counts = Vec::new();
    for n in 0..3 {
        let snap = dir.path().join(format!("snap{n}"));
        let info = store.snapshot_blocking(snap.display().to_string()).unwrap();
        counts.push((snap, info.document_count));
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for worker in workers {
        worker.join().unwrap();
    }
    store.shutdown();
    drop(store);
    for (n, (snap, expected)) in counts.into_iter().enumerate() {
        let restored = HmsStore::restore_blocking(
            snap.display().to_string(),
            dir.path().join(format!("r{n}")).display().to_string(),
            Some(small()),
        )
        .unwrap();
        assert_eq!(restored.stats().unwrap().document_count, expected);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_forms_run_off_the_caller_thread() {
    let dir = tempfile::tempdir().unwrap();
    let store = HmsStore::open(
        dir.path().join("store").display().to_string(),
        Some(small()),
    )
    .await
    .unwrap();
    let counts = Arc::clone(&store)
        .ingest_batch(vec![doc("a", "async alpha"), doc("b", "async bravo")])
        .await
        .unwrap();
    assert_eq!(counts, [1, 1]);
    let hits = Arc::clone(&store)
        .search("bravo".into(), None)
        .await
        .unwrap();
    assert_eq!(hits[0].document_id, "b");
    let info = Arc::clone(&store)
        .snapshot(dir.path().join("snap").display().to_string())
        .await
        .unwrap();
    assert_eq!(info.document_count, 2);
    assert!(Arc::clone(&store)
        .delete_document("a".into())
        .await
        .unwrap());
}

#[test]
fn semantic_search_with_caller_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let store = HmsStore::open_blocking(
        dir.path().join("store").display().to_string(),
        Some(StoreOptions {
            dimensions: 1024,
            embedding_model: Some("fixture".into()),
            embedding_revision: Some("1".into()),
            embedding_dimensions: Some(3),
            ..Default::default()
        }),
    )
    .unwrap();
    for (id, text, v) in [
        ("x", "unrelated words alpha", vec![1.0, 0.0, 0.0]),
        ("y", "other words beta", vec![0.0, 1.0, 0.0]),
    ] {
        let mut input = doc(id, text);
        input.embeddings = Some(vec![v]);
        store.ingest_blocking(input).unwrap();
    }
    assert_eq!(store.stats().unwrap().embedded_chunk_count, 2);
    let hits = store
        .search_blocking(
            String::new(),
            Some(SearchOptions {
                embedding: Some(vec![0.0, 1.0, 0.0]),
                semantic_weight: Some(1.0),
                lexical_weight: Some(0.0),
                ..Default::default()
            }),
        )
        .unwrap();
    assert_eq!(hits[0].document_id, "y");
    assert!(hits[0].semantic_score.unwrap() > 0.99);
}
