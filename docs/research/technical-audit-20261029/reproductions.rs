// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Barrier};

use holographic_memory::core::documents::{VectorIndexMode, VectorQueryOptions};
use holographic_memory::core::memory_api::{
    FactView, RememberInput, Resolution, SupersessionPolicy,
};
use holographic_memory::core::qgraph::filter::{FilterBuildParams, FilteredIndex};
use holographic_memory::{DocumentInput, EmbeddingSpace, HmsCore, Memory};

struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("hms-audit-{name}-{}", std::process::id()));
        assert!(!path.exists());
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> String {
        self.0.display().to_string()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn exact_vector_top1_omits_the_higher_cosine() {
    let scratch = Scratch::new("exact");
    let config = holographic_memory::core::HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "audit-fixture".into(),
            revision: "1".into(),
            dimensions: 2,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    };
    let core = HmsCore::new(256, Some(scratch.path()), Some(config)).unwrap();
    for i in 0..10 {
        let (id, vector) = if i == 9 {
            ("z".to_string(), vec![1.0, 0.0001])
        } else {
            (format!("a{i}"), vec![1.0, 0.0])
        };
        core.memorize_document(DocumentInput {
            id,
            text: "vector fixture".into(),
            source_uri: None,
            version: None,
            metadata: None,
            chunk_words: None,
            overlap_words: None,
            store_text: None,
            embeddings: Some(vec![vector]),
            title_weight: None,
        })
        .unwrap();
    }
    let options = VectorQueryOptions {
        mode: VectorIndexMode::Exact,
        ..Default::default()
    };
    let first = core.search_vectors(&[1.0, 0.0001], 1, &options).unwrap();
    let all = core.search_vectors(&[1.0, 0.0001], 10, &options).unwrap();
    println!(
        "exact k=1: {} {:.12}; k=10 best: {} {:.12}",
        first[0].document_id, first[0].score, all[0].document_id, all[0].score
    );
    assert_ne!(first[0].document_id, all[0].document_id);
    assert!(first[0].score < all[0].score);
}

struct PlanningBarrier(Barrier);

impl SupersessionPolicy for PlanningBarrier {
    fn resolve(&self, _: &FactView, _: &[FactView]) -> Resolution {
        self.0.wait();
        Resolution::default()
    }
}

#[test]
fn two_memory_wrappers_both_accept_conflicting_id() {
    let scratch = Scratch::new("conflict");
    let core = Arc::new(HmsCore::new(256, Some(scratch.path()), None).unwrap());
    let policy = Arc::new(PlanningBarrier(Barrier::new(2)));
    let a = Memory::new(Arc::clone(&core)).with_supersession(policy.clone());
    let b = Memory::new(Arc::clone(&core)).with_supersession(policy);
    let write = |memory: Memory, text: &'static str| {
        std::thread::spawn(move || {
            memory.remember(RememberInput {
                id: Some("shared-id".into()),
                text: text.into(),
                subject: Some("same-slot".into()),
                ..Default::default()
            })
        })
    };
    let t1 = write(a, "first conflicting fact");
    let t2 = write(b, "second conflicting fact");
    let r1 = t1.join().unwrap();
    let r2 = t2.join().unwrap();
    println!("same-id concurrent outcomes: {:?}; {:?}", r1, r2);
    assert!(r1.as_ref().is_ok_and(|r| r.created));
    assert!(r2.as_ref().is_ok_and(|r| r.created));
    assert_eq!(core.document_count(), 1);
}

#[test]
fn filtered_index_norm_overflows_in_debug() {
    let outcome = std::panic::catch_unwind(|| {
        FilteredIndex::build(
            &vec![0u8; 131_072],
            131_072,
            &[0, 1],
            &[0],
            1,
            &FilterBuildParams::default(),
        )
    });
    println!(
        "filtered index caught arithmetic panic: {}",
        outcome.is_err()
    );
    assert!(outcome.is_err());
}

#[test]
fn cose_accepts_unprocessed_critical_header() {
    use coset::{
        iana::Algorithm, CborSerializable, CoseSign1Builder, HeaderBuilder,
        RegisteredLabelWithPrivate,
    };
    use ed25519_dalek::{Signer, SigningKey};
    let scratch = Scratch::new("cose-critical");
    let manager =
        holographic_memory::core::provenance::ProvenanceManager::new(&scratch.0.join("key"), None)
            .unwrap();
    let key = SigningKey::from_bytes(manager.signing_key().as_bytes());
    let protected = HeaderBuilder::new()
        .algorithm(Algorithm::EdDSA)
        .add_critical_label(RegisteredLabelWithPrivate::Text("audit-unknown".into()))
        .build();
    let envelope = CoseSign1Builder::new()
        .protected(protected)
        .payload(b"fixture".to_vec())
        .create_signature(b"", |data| key.sign(data).to_bytes().to_vec())
        .build()
        .to_vec()
        .unwrap();
    let accepted = holographic_memory::core::provenance::cose::verify_and_extract(
        &manager.verifying_key(),
        &envelope,
    );
    println!(
        "COSE with absent unknown critical parameter accepted: {}",
        accepted.is_ok()
    );
    assert_eq!(accepted.unwrap(), b"fixture");
}

#[test]
fn provenance_reuses_a_deleted_credentials_revocation_slot() {
    use holographic_memory::core::provenance::ProvenanceManager;
    let scratch = Scratch::new("status-slot");
    let key = scratch.0.join("key");
    let old_sequence = {
        let manager = ProvenanceManager::new(&key, Some(&scratch.0)).unwrap();
        let record = manager
            .create_fact_provenance("old", b"old fact", None)
            .unwrap();
        manager.revoke_credential(record.sequence).unwrap();
        manager.record_deletion("old", None).unwrap();
        record.sequence
    };
    let manager = ProvenanceManager::new(&key, Some(&scratch.0)).unwrap();
    let new_record = manager
        .create_fact_provenance("new", b"new fact", None)
        .unwrap();
    println!(
        "old status slot: {old_sequence}; new status slot: {}; new credential already revoked: {}",
        new_record.sequence,
        manager.is_revoked(new_record.sequence)
    );
    assert_eq!(new_record.sequence, old_sequence);
    assert!(manager.is_revoked(new_record.sequence));
}

#[test]
fn cose_accepts_an_unauthenticated_algorithm_header() {
    use coset::{iana::Algorithm, CborSerializable, CoseSign1Builder, HeaderBuilder};
    use ed25519_dalek::{Signer, SigningKey};
    let scratch = Scratch::new("cose-algorithm");
    let manager =
        holographic_memory::core::provenance::ProvenanceManager::new(&scratch.0.join("key"), None)
            .unwrap();
    let key = SigningKey::from_bytes(manager.signing_key().as_bytes());
    let envelope = CoseSign1Builder::new()
        .unprotected(HeaderBuilder::new().algorithm(Algorithm::EdDSA).build())
        .payload(b"fixture".to_vec())
        .create_signature(b"", |data| key.sign(data).to_bytes().to_vec())
        .build()
        .to_vec()
        .unwrap();
    let accepted = holographic_memory::core::provenance::cose::verify_and_extract(
        &manager.verifying_key(),
        &envelope,
    );
    println!(
        "COSE with algorithm only in unprotected header accepted: {}",
        accepted.is_ok()
    );
    assert_eq!(accepted.unwrap(), b"fixture");
}
