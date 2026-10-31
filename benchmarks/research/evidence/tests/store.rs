// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use warm_research_evidence::contract::{
    Claim, Interval, Modality, Selection, Source, Span, MAX_SOURCES, MAX_SOURCE_BYTES,
};
use warm_research_evidence::store::{
    validate_bundle, EvidenceStore, MAX_CONTROL_BYTES, MAX_METADATA_BYTES, MAX_PENDING_DELETIONS,
    MAX_REVOKED_VERSIONS, STORE_DIMENSIONS,
};
use writerslogic_warm::core::documents::{EncodeMode, IngestOptions};
use writerslogic_warm::core::memory_api::{ForgetOptions, ForgetScope};
use writerslogic_warm::{DocumentInput, WarmCore, Memory};
use serde_json::{json, Value};

fn source(id: &str, text: &str) -> Source {
    Source {
        id: id.into(),
        version: "v1".into(),
        text: text.into(),
        observed_at: 10,
    }
}

fn span(source: &Source) -> Span {
    Span {
        source_id: source.id.clone(),
        source_version: source.version.clone(),
        start: 0,
        end: source.text.len(),
        quote: source.text.clone(),
    }
}

fn claim(id: &str, source: &Source) -> Claim {
    Claim {
        id: id.into(),
        subject: "person:alice".into(),
        predicate: "attended".into(),
        event_key: "event:one".into(),
        event_candidates: vec!["event:one".into()],
        value: None,
        event_time: None,
        valid_from: 10,
        valid_to: None,
        positive: true,
        modality: Modality::Actual,
        evidence: vec![span(source)],
        supersedes: Vec::new(),
    }
}

fn selection() -> Selection {
    Selection {
        subject: "person:alice".into(),
        predicate: "attended".into(),
        modality: Modality::Actual,
        event_window: None,
    }
}

fn hex(value: &str) -> String {
    value.bytes().map(|byte| format!("{byte:02x}")).collect()
}

fn document_id(namespace: &str, source_id: &str) -> String {
    format!("evidence:v1:{}:{}", hex(namespace), hex(source_id))
}

fn metadata(namespace: &str, source: &Source, claims: &[Claim]) -> Value {
    json!({
        "schema_version": 1,
        "namespace": namespace,
        "source_id": source.id,
        "source_version": source.version,
        "observed_at": source.observed_at,
        "claims": claims,
    })
}

fn raw_document(namespace: &str, source: &Source, claims: &[Claim]) -> DocumentInput {
    DocumentInput {
        id: document_id(namespace, &source.id),
        text: source.text.clone(),
        source_uri: None,
        version: Some(format!("evidence:v1:{}", hex(&source.version))),
        metadata: Some(metadata(namespace, source, claims)),
        chunk_words: None,
        overlap_words: None,
        store_text: Some(true),
        embeddings: None,
        title_weight: None,
    }
}

fn inject(path: &std::path::Path, documents: Vec<DocumentInput>) {
    let core = WarmCore::new(STORE_DIMENSIONS, Some(path.to_str().unwrap().into()), None).unwrap();
    core.memorize_documents_with(
        documents,
        &IngestOptions {
            encode: EncodeMode::Lexical,
            positions: true,
            whole_document: true,
            ..Default::default()
        },
    )
    .unwrap();
}

fn pin(source: &Source) -> Value {
    json!({"source_id": source.id, "source_version": source.version})
}

fn control_document(namespace: &str, revoked: &[Value], pending: &[Value]) -> DocumentInput {
    DocumentInput {
        id: format!("evidence-control:v1:{}", hex(namespace)),
        text: "Evidence deletion replay fence.".into(),
        source_uri: None,
        version: Some("evidence-control:v1".into()),
        metadata: Some(json!({
            "schema_version": 1,
            "namespace": namespace,
            "revoked": revoked,
            "pending": pending,
        })),
        chunk_words: None,
        overlap_words: None,
        store_text: Some(true),
        embeddings: None,
        title_weight: None,
    }
}

fn stored_control(path: &std::path::Path, namespace: &str) -> Option<Value> {
    let core = WarmCore::new(STORE_DIMENSIONS, Some(path.to_str().unwrap().into()), None).unwrap();
    core.get_document(&format!("evidence-control:v1:{}", hex(namespace)))
        .map(|record| record.metadata)
}

#[test]
fn reopens_exact_source_bytes_versions_and_claims() {
    let directory = tempfile::tempdir().unwrap();
    let mut source = source("source:é", "  Alice attended the café.\n\t");
    source.version = "mem1:{\"v\":1,\"ingest\":0}".into();
    let claim = claim("claim:one", &source);
    {
        let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
        store
            .put(std::slice::from_ref(&source), std::slice::from_ref(&claim))
            .unwrap();
        assert_eq!(
            store.lexical("café attended", 10).unwrap(),
            [source.id.clone()]
        );
        assert_eq!(
            store.matching(&selection(), 10).unwrap(),
            [source.id.clone()]
        );
    }
    let store = EvidenceStore::open(directory.path(), "research").unwrap();
    assert_eq!(
        store.snapshot().unwrap(),
        (vec![source.clone()], vec![claim.clone()])
    );
    assert_eq!(
        store.load(std::slice::from_ref(&source.id)).unwrap(),
        (vec![source], vec![claim])
    );
    assert!(store.load(&["missing".into()]).is_err());
}

#[test]
fn replacement_removes_old_claims_and_rejects_stale_cross_source_evidence_atomically() {
    let directory = tempfile::tempdir().unwrap();
    let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
    let a = source("a", "Alice attended the concert.");
    let b = source("b", "The concert was Tuesday.");
    let old = claim("old", &a);
    let mut dependent = claim("dependent", &b);
    dependent.evidence.push(span(&a));
    store
        .put(&[a.clone(), b.clone()], &[old, dependent.clone()])
        .unwrap();
    let before = store.snapshot().unwrap();
    let updated = Source {
        version: "v2".into(),
        text: "Alice attended the festival.".into(),
        ..a.clone()
    };
    let new = claim("new", &updated);
    assert!(store
        .put(std::slice::from_ref(&updated), std::slice::from_ref(&new))
        .is_err());
    assert_eq!(store.snapshot().unwrap(), before);
    dependent.evidence[1] = span(&updated);
    store
        .put(&[updated.clone(), b], &[new.clone(), dependent])
        .unwrap();
    let (sources, claims) = store.snapshot().unwrap();
    assert_eq!(sources[0], updated);
    assert!(!claims.iter().any(|claim| claim.id == "old"));
    assert!(claims.contains(&new));
    assert!(store.lexical("festival", 10).unwrap().contains(&"a".into()));
    assert!(store
        .lexical("concert", 10)
        .unwrap()
        .iter()
        .all(|id| id != "a"));
    let mut invalid_version = sources[0].clone();
    invalid_version.text = "Different bytes under the same version.".into();
    assert!(store.put(&[invalid_version], &[]).is_err());
}

#[test]
fn dependency_closure_and_deletion_preserve_surviving_references_on_reopen() {
    let directory = tempfile::tempdir().unwrap();
    {
        let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
        let a = source("a", "Alice attended the concert.");
        let b = source("b", "The date was corrected to Wednesday.");
        let c = source("c", "The venue confirmed Wednesday.");
        let old = claim("old", &a);
        let mut corrected = claim("corrected", &b);
        corrected.supersedes.push(old.id.clone());
        corrected.evidence.push(span(&c));
        store.put(&[a, b, c], &[old, corrected]).unwrap();
        let loaded = store.load(&["b".into()]).unwrap();
        assert_eq!(
            loaded
                .0
                .iter()
                .map(|source| source.id.as_str())
                .collect::<Vec<_>>(),
            ["a", "b", "c"]
        );
        assert_eq!(loaded.1.len(), 2);
        assert_eq!(store.load(&["a".into()]).unwrap(), loaded);
        let before = store.snapshot().unwrap();
        assert!(store.forget_sources(&["a".into()]).is_err());
        assert!(store.forget_sources(&["c".into()]).is_err());
        assert_eq!(store.snapshot().unwrap(), before);
        assert_eq!(
            store
                .forget_sources(&["a".into(), "b".into(), "c".into(), "a".into()])
                .unwrap(),
            3
        );
        assert_eq!(
            store
                .forget_sources(&["a".into(), "b".into(), "c".into()])
                .unwrap(),
            0
        );
        assert!(store.lexical("concert Wednesday", 10).unwrap().is_empty());
        assert!(store.matching(&selection(), 10).unwrap().is_empty());
    }
    let store = EvidenceStore::open(directory.path(), "research").unwrap();
    assert_eq!(store.snapshot().unwrap(), (Vec::new(), Vec::new()));
    assert!(store.load(&["a".into()]).is_err());
}

#[test]
fn namespaces_and_tuple_identity_do_not_collide_or_displace_results() {
    let directory = tempfile::tempdir().unwrap();
    let first = source("c:d", "A quiet needle.");
    let second = source("d", "needle needle needle needle");
    {
        let mut store = EvidenceStore::open(directory.path(), "ab").unwrap();
        store
            .put(
                std::slice::from_ref(&first),
                &[claim("same-claim-id", &first)],
            )
            .unwrap();
    }
    {
        let mut store = EvidenceStore::open(directory.path(), "ab:c").unwrap();
        store
            .put(
                std::slice::from_ref(&second),
                &[claim("same-claim-id", &second)],
            )
            .unwrap();
        assert_eq!(store.snapshot().unwrap().0, std::slice::from_ref(&second));
        assert_eq!(
            store.lexical("needle", 1).unwrap(),
            std::slice::from_ref(&second.id)
        );
    }
    {
        let mut store = EvidenceStore::open(directory.path(), "ab").unwrap();
        assert_eq!(store.snapshot().unwrap().0, std::slice::from_ref(&first));
        assert_eq!(
            store.lexical("needle", 1).unwrap(),
            std::slice::from_ref(&first.id)
        );
        store
            .forget_sources(std::slice::from_ref(&first.id))
            .unwrap();
    }
    let store = EvidenceStore::open(directory.path(), "ab:c").unwrap();
    assert_eq!(store.snapshot().unwrap().0, [second]);
}

#[test]
fn rejects_duplicate_global_claim_ids_dangling_links_cycles_and_incompatible_scopes() {
    let directory = tempfile::tempdir().unwrap();
    let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
    let a = source("a", "Alice attended.");
    let b = source("b", "Alice confirmed.");
    let first = claim("duplicate", &a);
    store
        .put(std::slice::from_ref(&a), std::slice::from_ref(&first))
        .unwrap();
    let before = store.snapshot().unwrap();
    assert!(store
        .put(std::slice::from_ref(&b), &[claim("duplicate", &b)])
        .is_err());
    let mut second = claim("second", &b);
    second.supersedes = vec!["missing".into()];
    assert!(store
        .put(std::slice::from_ref(&b), std::slice::from_ref(&second))
        .is_err());
    second.supersedes = vec![first.id.clone()];
    second.subject = "person:bob".into();
    assert!(store
        .put(std::slice::from_ref(&b), std::slice::from_ref(&second))
        .is_err());
    second.subject = first.subject.clone();
    let mut cyclic = first.clone();
    cyclic.supersedes = vec![second.id.clone()];
    assert!(store
        .put(&[a.clone(), b.clone()], &[cyclic, second.clone()])
        .is_err());
    assert_eq!(store.snapshot().unwrap(), before);
    assert!(store.put(&[a.clone(), a], &[]).is_err());
    assert!(store.put(&[], &[second]).is_err());
    assert_eq!(store.snapshot().unwrap(), before);
}

#[test]
fn rejects_malformed_scoped_metadata_and_duplicate_metadata_source_ids() {
    let source = source("a", "Alice attended the café.");
    let claim = claim("claim", &source);
    for malformed in 0..7 {
        let directory = tempfile::tempdir().unwrap();
        let mut document = raw_document("research", &source, std::slice::from_ref(&claim));
        let metadata = document.metadata.as_mut().unwrap();
        match malformed {
            0 => {
                metadata.as_object_mut().unwrap().remove("namespace");
            }
            1 => metadata["namespace"] = json!("different"),
            2 => metadata["schema_version"] = json!(2),
            3 => document.version = Some("different".into()),
            4 => metadata["claims"][0]["evidence"][0]["source_version"] = json!("stale"),
            5 => metadata["claims"][0]["evidence"][0]["quote"] = json!("unrelated"),
            _ => document.id.push_str("00"),
        }
        inject(directory.path(), vec![document]);
        assert!(
            EvidenceStore::open(directory.path(), "research").is_err(),
            "malformation {malformed}"
        );
    }
    let directory = tempfile::tempdir().unwrap();
    let original = raw_document("research", &source, std::slice::from_ref(&claim));
    let mut duplicate = original.clone();
    duplicate.id.push_str("00");
    inject(directory.path(), vec![original, duplicate]);
    assert!(EvidenceStore::open(directory.path(), "research").is_err());
}

#[test]
fn invalid_spans_and_missing_cross_source_evidence_leave_the_batch_unwritten() {
    let directory = tempfile::tempdir().unwrap();
    let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
    let source = source("a", "é Alice attended.");
    let mut invalid = claim("invalid", &source);
    invalid.evidence[0].start = 1;
    assert!(store
        .put(std::slice::from_ref(&source), &[invalid.clone()])
        .is_err());
    invalid.evidence[0] = Span {
        source_id: "missing".into(),
        ..span(&source)
    };
    assert!(store
        .put(std::slice::from_ref(&source), &[invalid])
        .is_err());
    assert_eq!(store.snapshot().unwrap(), (Vec::new(), Vec::new()));
}

fn claim_with_metadata_bytes(source: &Source, target: usize) -> Claim {
    let mut claim = claim("large", source);
    claim.evidence = (0..8)
        .map(|start| Span {
            start,
            end: source.text.len(),
            quote: source.text[start..].into(),
            ..span(source)
        })
        .collect();
    for _ in 0..8 {
        let size = serde_json::to_vec(&metadata("bounds", source, std::slice::from_ref(&claim)))
            .unwrap()
            .len();
        if size == target {
            return claim;
        }
        let last = claim.evidence.last_mut().unwrap();
        last.end = if size > target {
            last.end - (size - target)
        } else {
            last.end + target - size
        };
        last.quote = source.text[last.start..last.end].into();
    }
    panic!("could not construct exact metadata boundary");
}

#[test]
fn enforces_serialized_metadata_limit_minus_one_limit_and_plus_one() {
    let directory = tempfile::tempdir().unwrap();
    let mut store = EvidenceStore::open(directory.path(), "bounds").unwrap();
    let source = source("s", &"x".repeat(MAX_METADATA_BYTES / 8));
    for size in [MAX_METADATA_BYTES - 1, MAX_METADATA_BYTES] {
        let claim = claim_with_metadata_bytes(&source, size);
        validate_bundle(
            "bounds",
            std::slice::from_ref(&source),
            std::slice::from_ref(&claim),
        )
        .unwrap();
        store
            .put(std::slice::from_ref(&source), std::slice::from_ref(&claim))
            .unwrap();
        assert_eq!(store.snapshot().unwrap().1, [claim]);
    }
    let before = store.snapshot().unwrap();
    let oversized = claim_with_metadata_bytes(&source, MAX_METADATA_BYTES + 1);
    assert!(validate_bundle(
        "bounds",
        std::slice::from_ref(&source),
        std::slice::from_ref(&oversized)
    )
    .is_err());
    assert!(store
        .put(
            std::slice::from_ref(&source),
            std::slice::from_ref(&oversized)
        )
        .is_err());
    assert_eq!(store.snapshot().unwrap(), before);
    drop(store);
    assert_eq!(
        EvidenceStore::open(directory.path(), "bounds")
            .unwrap()
            .snapshot()
            .unwrap(),
        before
    );
}

#[test]
fn enforces_source_text_limit_minus_one_limit_and_plus_one() {
    let directory = tempfile::tempdir().unwrap();
    let mut store = EvidenceStore::open(directory.path(), "bounds").unwrap();
    for bytes in [MAX_SOURCE_BYTES - 1, MAX_SOURCE_BYTES] {
        let source = Source {
            version: format!("v{bytes}"),
            ..source("s", &"x".repeat(bytes))
        };
        store.put(std::slice::from_ref(&source), &[]).unwrap();
        assert_eq!(store.snapshot().unwrap().0, [source]);
    }
    let before = store.snapshot().unwrap();
    let source = source("s", &"x".repeat(MAX_SOURCE_BYTES + 1));
    assert!(store.put(&[source], &[]).is_err());
    assert_eq!(store.snapshot().unwrap(), before);
}

#[test]
fn paginates_to_source_limit_and_preserves_all_scoped_lexical_candidates() {
    let directory = tempfile::tempdir().unwrap();
    let mut store = EvidenceStore::open(directory.path(), "bounds").unwrap();
    let sources: Vec<_> = (0..MAX_SOURCES - 1)
        .map(|i| source(&format!("s{i:04}"), "needle"))
        .collect();
    store.put(&sources, &[]).unwrap();
    assert_eq!(store.snapshot().unwrap().0.len(), MAX_SOURCES - 1);
    let last = source("last", "needle");
    store.put(&[last], &[]).unwrap();
    assert_eq!(store.snapshot().unwrap().0.len(), MAX_SOURCES);
    let lexical = store.lexical("needle", MAX_SOURCES).unwrap();
    assert_eq!(lexical.len(), MAX_SOURCES);
    assert_eq!(
        lexical
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        MAX_SOURCES
    );
    assert!(store.put(&[source("overflow", "needle")], &[]).is_err());
    assert_eq!(store.snapshot().unwrap().0.len(), MAX_SOURCES);
    assert!(store.lexical("needle", MAX_SOURCES + 1).is_err());
    assert!(store.matching(&selection(), MAX_SOURCES + 1).is_err());
    assert!(store.load(&vec!["last".into(); MAX_SOURCES + 1]).is_err());
}

#[test]
fn matching_retains_unknown_and_outside_times_for_exact_execution() {
    let directory = tempfile::tempdir().unwrap();
    let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
    let a = source("a", "Alice attended a concert.");
    let b = source("b", "Alice planned a concert.");
    let mut actual = claim("actual", &a);
    actual.event_time = Some(Interval {
        start: 100,
        end: 101,
    });
    let mut planned = claim("planned", &b);
    planned.modality = Modality::Planned;
    store.put(&[a, b], &[actual, planned]).unwrap();
    let mut selection = selection();
    selection.event_window = Some(Interval { start: 1, end: 2 });
    assert_eq!(store.matching(&selection, 10).unwrap(), ["a"]);
    selection.modality = Modality::Planned;
    assert_eq!(store.matching(&selection, 10).unwrap(), ["b"]);
    assert!(store.matching(&selection, 0).unwrap().is_empty());
}

#[test]
fn same_canonical_event_mentions_share_executor_supersession_validation() {
    let directory = tempfile::tempdir().unwrap();
    let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
    let a = source("a", "Alice attended the concert.");
    let b = source("b", "The same concert was on Wednesday.");
    let mut predecessor = claim("predecessor", &a);
    predecessor.event_key = "mention:a".into();
    let mut successor = claim("successor", &b);
    successor.event_key = "mention:b".into();
    successor.supersedes.push(predecessor.id.clone());
    successor.evidence.push(span(&a));
    store
        .put(std::slice::from_ref(&a), std::slice::from_ref(&predecessor))
        .unwrap();
    assert!(validate_bundle(
        "research",
        std::slice::from_ref(&b),
        std::slice::from_ref(&successor)
    )
    .is_err());
    store
        .put(std::slice::from_ref(&b), std::slice::from_ref(&successor))
        .unwrap();
    let loaded = store.load(std::slice::from_ref(&a.id)).unwrap();
    assert_eq!(loaded.0, [a, b]);
    assert_eq!(loaded.1, [predecessor, successor]);
    validate_bundle("research", &loaded.0, &loaded.1).unwrap();
    assert!(validate_bundle("", &loaded.0, &loaded.1).is_err());
}

#[test]
fn accepted_deletion_fences_old_versions_and_allows_a_fresh_version_after_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let original = source("a", "Alice attended the concert.");
    {
        let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
        store
            .put(std::slice::from_ref(&original), &[claim("old", &original)])
            .unwrap();
        assert_eq!(
            store
                .forget_sources(std::slice::from_ref(&original.id))
                .unwrap(),
            1
        );
        assert_eq!(
            store
                .forget_sources(std::slice::from_ref(&original.id))
                .unwrap(),
            0
        );
        assert!(store
            .put(std::slice::from_ref(&original), &[claim("old", &original)])
            .is_err());
        assert!(store.snapshot().unwrap().0.is_empty());
        assert!(store
            .lexical("deletion replay fence", 10)
            .unwrap()
            .is_empty());
    }
    assert_eq!(
        stored_control(directory.path(), "research").unwrap(),
        json!({
            "schema_version": 1,
            "namespace": "research",
            "revoked": [pin(&original)],
            "pending": [],
        })
    );
    let fresh = Source {
        version: "v2".into(),
        text: "Alice attended the festival.".into(),
        ..original.clone()
    };
    {
        let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
        assert!(store.put(std::slice::from_ref(&original), &[]).is_err());
        store
            .put(std::slice::from_ref(&fresh), &[claim("fresh", &fresh)])
            .unwrap();
    }
    let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
    assert_eq!(store.snapshot().unwrap().0, [fresh]);
    assert!(store.put(&[original], &[]).is_err());
}

#[test]
fn pending_deletion_recovers_after_control_delete_and_purge_crash_boundaries() {
    let forgotten = source("a", "Alice attended the concert.");
    let survivor = source("b", "A separate festival remained.");
    for phase in 0..3 {
        let directory = tempfile::tempdir().unwrap();
        {
            let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
            store
                .put(
                    &[forgotten.clone(), survivor.clone()],
                    &[claim("a-claim", &forgotten), claim("b-claim", &survivor)],
                )
                .unwrap();
        }
        inject(
            directory.path(),
            vec![control_document(
                "research",
                &[pin(&forgotten)],
                &[pin(&forgotten)],
            )],
        );
        if phase > 0 {
            let core = std::sync::Arc::new(
                WarmCore::new(
                    STORE_DIMENSIONS,
                    Some(directory.path().to_str().unwrap().into()),
                    None,
                )
                .unwrap(),
            );
            let memory = Memory::new(core);
            memory
                .forget(
                    &ForgetScope::Ids {
                        ids: vec![document_id("research", &forgotten.id)],
                    },
                    &ForgetOptions {
                        purge: phase == 2,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        {
            let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
            assert_eq!(
                store.snapshot().unwrap().0,
                std::slice::from_ref(&survivor),
                "phase {phase}"
            );
            assert!(store.load(std::slice::from_ref(&forgotten.id)).is_err());
            assert!(store.put(std::slice::from_ref(&forgotten), &[]).is_err());
            assert_eq!(
                store
                    .forget_sources(std::slice::from_ref(&forgotten.id))
                    .unwrap(),
                0
            );
        }
        let control = stored_control(directory.path(), "research").unwrap();
        assert_eq!(control["pending"], json!([]));
        assert_eq!(control["revoked"], json!([pin(&forgotten)]));
        let store = EvidenceStore::open(directory.path(), "research").unwrap();
        assert_eq!(store.snapshot().unwrap().0, std::slice::from_ref(&survivor));
    }
}

#[test]
fn rejected_dependency_cut_creates_no_replay_fence() {
    let directory = tempfile::tempdir().unwrap();
    let a = source("a", "Alice attended.");
    let b = source("b", "The attendance was confirmed.");
    let first = claim("first", &a);
    let mut dependent = claim("dependent", &b);
    dependent.supersedes.push(first.id.clone());
    {
        let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
        store
            .put(&[a.clone(), b.clone()], &[first.clone(), dependent.clone()])
            .unwrap();
        assert!(store.forget_sources(std::slice::from_ref(&a.id)).is_err());
    }
    assert!(stored_control(directory.path(), "research").is_none());
    let mut store = EvidenceStore::open(directory.path(), "research").unwrap();
    store.put(&[a, b], &[first, dependent]).unwrap();
    assert_eq!(store.snapshot().unwrap().0.len(), 2);
}

#[test]
fn corrupt_control_and_mismatched_pending_versions_fail_before_deletion() {
    let original = source("a", "Alice attended.");
    for corruption in 0..10 {
        let directory = tempfile::tempdir().unwrap();
        let mut control = control_document("research", &[], &[]);
        let metadata = control.metadata.as_mut().unwrap();
        match corruption {
            0 => metadata["schema_version"] = json!(2),
            1 => metadata["namespace"] = json!("other"),
            2 => metadata["pending"] = json!([pin(&original)]),
            3 => metadata["revoked"] = json!([pin(&original), pin(&original)]),
            4 => {
                metadata["revoked"] = json!([pin(&original)]);
                metadata["pending"] = json!([pin(&original), pin(&original)]);
            }
            5 => metadata["revoked"] = json!([{"source_id": "a", "source_version": ""}]),
            6 => metadata["forgotten_text"] = json!("forbidden payload field"),
            7 => control.text = "Unexpected control text".into(),
            8 => control.version = Some("wrong".into()),
            _ => {
                let wrong = Source {
                    version: "v2".into(),
                    ..original.clone()
                };
                metadata["revoked"] = json!([pin(&wrong)]);
                metadata["pending"] = json!([pin(&wrong)]);
            }
        }
        inject(
            directory.path(),
            vec![raw_document("research", &original, &[]), control],
        );
        assert!(
            EvidenceStore::open(directory.path(), "research").is_err(),
            "corruption {corruption}"
        );
        let core = WarmCore::new(
            STORE_DIMENSIONS,
            Some(directory.path().to_str().unwrap().into()),
            None,
        )
        .unwrap();
        assert!(
            core.has_document(&document_id("research", &original.id)),
            "corruption {corruption}"
        );
    }
}

fn control_with_bytes(target: usize) -> DocumentInput {
    let mut revoked: Vec<_> = (0..128)
        .map(|i| {
            json!({
                "source_id": format!("r{i:04}{}", "x".repeat(251)),
                "source_version": "v".repeat(256),
            })
        })
        .collect();
    let size = serde_json::to_vec(&control_document("bounds", &revoked, &[]).metadata.unwrap())
        .unwrap()
        .len();
    let mut remaining = size.checked_sub(target).unwrap();
    for pin in &mut revoked {
        let remove = remaining.min(255);
        pin["source_version"] = json!("v".repeat(256 - remove));
        remaining -= remove;
    }
    assert_eq!(remaining, 0);
    let document = control_document("bounds", &revoked, &[]);
    assert_eq!(
        serde_json::to_vec(document.metadata.as_ref().unwrap())
            .unwrap()
            .len(),
        target
    );
    document
}

#[test]
fn deletion_control_preflights_byte_growth_before_fencing_or_deleting_sources() {
    for size in [MAX_CONTROL_BYTES - 1, MAX_CONTROL_BYTES] {
        let directory = tempfile::tempdir().unwrap();
        inject(directory.path(), vec![control_with_bytes(size)]);
        let live = source("available", "A current source.");
        {
            let mut store = EvidenceStore::open(directory.path(), "bounds").unwrap();
            assert!(store.snapshot().unwrap().0.is_empty());
            store.put(std::slice::from_ref(&live), &[]).unwrap();
            assert!(store
                .forget_sources(std::slice::from_ref(&live.id))
                .is_err());
            assert_eq!(store.snapshot().unwrap().0, [live]);
        }
        let control = stored_control(directory.path(), "bounds").unwrap();
        assert_eq!(serde_json::to_vec(&control).unwrap().len(), size);
        assert_eq!(control["pending"], json!([]));
    }
}

#[test]
fn deletion_control_enforces_revocation_count_before_mutating_sources() {
    for count in [
        MAX_REVOKED_VERSIONS - 1,
        MAX_REVOKED_VERSIONS,
        MAX_REVOKED_VERSIONS + 1,
    ] {
        let directory = tempfile::tempdir().unwrap();
        let revoked: Vec<_> = (0..count)
            .map(|i| json!({"source_id": format!("r{i:04}"), "source_version": "v1"}))
            .collect();
        inject(
            directory.path(),
            vec![control_document("bounds", &revoked, &[])],
        );
        if count > MAX_REVOKED_VERSIONS {
            assert!(EvidenceStore::open(directory.path(), "bounds").is_err());
            continue;
        }
        let live = source("available", "A current source.");
        {
            let mut store = EvidenceStore::open(directory.path(), "bounds").unwrap();
            store.put(std::slice::from_ref(&live), &[]).unwrap();
            let result = store.forget_sources(std::slice::from_ref(&live.id));
            if count == MAX_REVOKED_VERSIONS {
                assert!(result.is_err());
                assert_eq!(store.snapshot().unwrap().0, [live]);
            } else {
                assert_eq!(result.unwrap(), 1);
                assert!(store.snapshot().unwrap().0.is_empty());
            }
        }
        let control = stored_control(directory.path(), "bounds").unwrap();
        assert_eq!(
            control["revoked"].as_array().unwrap().len(),
            MAX_REVOKED_VERSIONS
        );
        assert_eq!(control["pending"], json!([]));
    }
}

#[test]
fn deletion_control_enforces_pending_count_and_recovers_absent_source_retries() {
    for count in [
        MAX_PENDING_DELETIONS - 1,
        MAX_PENDING_DELETIONS,
        MAX_PENDING_DELETIONS + 1,
    ] {
        let directory = tempfile::tempdir().unwrap();
        let pins: Vec<_> = (0..count)
            .map(|i| json!({"source_id": format!("r{i:04}"), "source_version": "v1"}))
            .collect();
        inject(
            directory.path(),
            vec![control_document("bounds", &pins, &pins)],
        );
        if count > MAX_PENDING_DELETIONS {
            assert!(EvidenceStore::open(directory.path(), "bounds").is_err());
            assert_eq!(
                stored_control(directory.path(), "bounds").unwrap()["pending"]
                    .as_array()
                    .unwrap()
                    .len(),
                count
            );
        } else {
            {
                let store = EvidenceStore::open(directory.path(), "bounds").unwrap();
                assert!(store.snapshot().unwrap().0.is_empty());
            }
            assert_eq!(
                stored_control(directory.path(), "bounds").unwrap()["pending"],
                json!([])
            );
        }
    }
}

#[test]
fn excessive_pending_cut_leaves_every_source_and_the_control_record_unchanged() {
    let directory = tempfile::tempdir().unwrap();
    let sources: Vec<_> = (0..MAX_PENDING_DELETIONS + 1)
        .map(|i| source(&format!("source{i:04}"), "A current source."))
        .collect();
    {
        let mut store = EvidenceStore::open(directory.path(), "bounds").unwrap();
        store.put(&sources, &[]).unwrap();
        let ids: Vec<_> = sources.iter().map(|source| source.id.clone()).collect();
        assert!(store.forget_sources(&ids).is_err());
        assert_eq!(store.snapshot().unwrap().0, sources);
    }
    assert!(stored_control(directory.path(), "bounds").is_none());
}
