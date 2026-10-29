use std::sync::Arc;

use super::index::Index;
use super::*;
use crate::core::HmsCore;

fn fact(id: &str, text: &str) -> RememberInput {
    RememberInput {
        id: Some(id.into()),
        text: text.into(),
        ..RememberInput::default()
    }
}

#[test]
fn concurrent_wrappers_cannot_both_create_different_facts_with_one_id() {
    struct Synchronized(std::sync::Barrier);
    impl SupersessionPolicy for Synchronized {
        fn resolve(&self, _: &FactView, _: &[FactView]) -> Resolution {
            self.0.wait();
            Resolution::default()
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let core = Arc::new(HmsCore::new(256, Some(dir.path().display().to_string()), None).unwrap());
    let policy = Arc::new(Synchronized(std::sync::Barrier::new(2)));
    let workers: Vec<_> = ["first fact", "second fact"]
        .into_iter()
        .map(|text| {
            let memory = Memory::new(core.clone()).with_supersession(policy.clone());
            std::thread::spawn(move || {
                memory.remember(RememberInput {
                    subject: Some("slot".into()),
                    ..fact("shared", text)
                })
            })
        })
        .collect();
    let results: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|r| r.as_ref().is_err_and(|e| matches!(
                e.downcast_ref::<MemoryError>(),
                Some(MemoryError::Conflict(_))
            )))
            .count(),
        1
    );
    assert_eq!(core.document_count(), 1);
}

#[test]
fn concurrent_subject_updates_replan_against_the_committed_predecessor() {
    struct Synchronized {
        barrier: std::sync::Barrier,
        calls: std::sync::atomic::AtomicUsize,
    }
    impl SupersessionPolicy for Synchronized {
        fn resolve(&self, incoming: &FactView, group: &[FactView]) -> Resolution {
            if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) < 2 {
                self.barrier.wait();
            }
            LatestBySubject.resolve(incoming, group)
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let core = Arc::new(HmsCore::new(256, Some(dir.path().display().to_string()), None).unwrap());
    let policy = Arc::new(Synchronized {
        barrier: std::sync::Barrier::new(2),
        calls: Default::default(),
    });
    let workers: Vec<_> = [10, 20]
        .into_iter()
        .map(|time| {
            let memory = Memory::new(core.clone()).with_supersession(policy.clone());
            std::thread::spawn(move || {
                memory.remember(RememberInput {
                    subject: Some("slot".into()),
                    event_time: Some(time),
                    ..fact(&time.to_string(), "shared subject")
                })
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap().unwrap();
    }
    let memory = Memory::new(core);
    let old = memory.get("10").unwrap().unwrap();
    let new = memory.get("20").unwrap().unwrap();
    assert_eq!(old.status, FactStatus::Superseded);
    assert_eq!(new.status, FactStatus::Current);
    assert_eq!(new.supersedes, vec!["10"]);
}

#[test]
fn conditional_changes_never_overwrite_a_raw_core_write() {
    let dir = tempfile::tempdir().unwrap();
    let core = Arc::new(HmsCore::new(256, Some(dir.path().display().to_string()), None).unwrap());
    let memory = Memory::new(core.clone());
    memory.remember(fact("shared", "original")).unwrap();
    let revision = core.revision();
    let replacement: crate::DocumentInput = serde_json::from_value(serde_json::json!({
        "id":"shared", "text":"raw replacement"
    }))
    .unwrap();
    core.memorize_document(replacement.clone()).unwrap();
    assert!(!core
        .change_memory_documents(vec![replacement], &["shared".into()], revision)
        .unwrap());
    assert_eq!(
        core.get_document("shared").unwrap().text.as_deref(),
        Some("raw replacement")
    );
}

#[test]
fn forgetting_policy_replans_instead_of_deleting_a_replaced_source() {
    struct ReplaceDuringSelection(Arc<HmsCore>, std::sync::atomic::AtomicBool);
    impl ForgettingPolicy for ReplaceDuringSelection {
        fn select(&self, _: &[FactSummary], _: Millis) -> Vec<String> {
            if !self.1.swap(true, std::sync::atomic::Ordering::SeqCst) {
                self.0
                    .memorize_document(
                        serde_json::from_value(serde_json::json!({
                            "id":"source", "text":"foreign replacement"
                        }))
                        .unwrap(),
                    )
                    .unwrap();
            }
            vec!["source".into()]
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let core = Arc::new(HmsCore::new(256, Some(dir.path().display().to_string()), None).unwrap());
    let memory = Memory::new(core.clone());
    memory.remember(fact("source", "outdated fact")).unwrap();
    let report = memory
        .apply_forgetting_with(
            &ReplaceDuringSelection(core.clone(), false.into()),
            &ForgetOptions::default(),
        )
        .unwrap();
    assert!(report.forgotten.is_empty());
    assert_eq!(
        core.get_document("source").unwrap().text.as_deref(),
        Some("foreign replacement")
    );
}

#[test]
fn continuous_foreign_writes_return_contention_without_a_stale_commit() {
    struct AlwaysWrite(Arc<HmsCore>);
    impl SupersessionPolicy for AlwaysWrite {
        fn resolve(&self, _: &FactView, _: &[FactView]) -> Resolution {
            self.0
                .memorize_document(
                    serde_json::from_value(serde_json::json!({
                        "id":"foreign", "text":"writer advances revision"
                    }))
                    .unwrap(),
                )
                .unwrap();
            Resolution::default()
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let core = Arc::new(HmsCore::new(256, Some(dir.path().display().to_string()), None).unwrap());
    let memory = Memory::new(core.clone()).with_supersession(Arc::new(AlwaysWrite(core.clone())));
    let error = memory
        .remember(RememberInput {
            subject: Some("subject".into()),
            ..fact("planned", "must not be committed")
        })
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<MemoryError>(),
        Some(MemoryError::Concurrent(_))
    ));
    assert!(!core.has_document("planned"));
    assert!(core.has_document("foreign"));
}

#[test]
fn consolidation_replans_when_its_source_changes_during_the_policy_callback() {
    struct ReplaceDuringPlan(Arc<HmsCore>, std::sync::atomic::AtomicBool);
    impl ConsolidationPolicy for ReplaceDuringPlan {
        fn plan(&self, facts: &[FactSummary], _: Millis) -> Vec<ConsolidationAction> {
            if !facts.iter().any(|fact| fact.id == "source") {
                return vec![];
            }
            if !self.1.swap(true, std::sync::atomic::Ordering::SeqCst) {
                self.0
                    .memorize_document(
                        serde_json::from_value(serde_json::json!({
                            "id":"source", "text":"changed foreign source"
                        }))
                        .unwrap(),
                    )
                    .unwrap();
            }
            vec![ConsolidationAction {
                sources: vec!["source".into()],
                summary: fact("summary", "stale summary"),
                forget_sources: true,
            }]
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let core = Arc::new(HmsCore::new(256, Some(dir.path().display().to_string()), None).unwrap());
    let memory = Memory::new(core.clone());
    memory.remember(fact("source", "original fact")).unwrap();
    let result = memory
        .consolidate_with(
            &ReplaceDuringPlan(core.clone(), false.into()),
            &ConsolidateOptions {
                allow_destructive: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(result.created.is_empty());
    assert!(!core.has_document("summary"));
    assert_eq!(
        core.get_document("source").unwrap().text.as_deref(),
        Some("changed foreign source")
    );
}

#[test]
fn replacing_a_fact_leaves_the_index_equal_to_a_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let core = Arc::new(HmsCore::new(64, Some(dir.path().display().to_string()), None).unwrap());
    let config = MemoryConfig {
        on_id_conflict: IdConflict::Replace,
        ..MemoryConfig::default()
    };
    let memory = Memory::with_config(core.clone(), config).unwrap();
    memory
        .remember(RememberInput {
            subject: Some("s".into()),
            entities: Some(vec!["e1".into(), "e2".into()]),
            valid_from: Some(10),
            valid_to: Some(20),
            ..fact("a", "alpha one")
        })
        .unwrap();
    memory.remember(fact("b", "beta one")).unwrap();
    memory
        .remember(RememberInput {
            supersedes: Some(vec!["b".into()]),
            ..fact("c", "gamma one")
        })
        .unwrap();
    memory.ensure_fresh();
    let tracked = |memory: &Memory| {
        let live = memory.index.read();
        let tracked = live.revision == Some(core.revision());
        if tracked {
            assert_eq!(live.canonical(), Index::rebuild(&core).canonical());
        }
        tracked
    };

    // Unlinked: updated in place, the old subject, entities and window gone.
    memory
        .remember(RememberInput {
            subject: Some("t".into()),
            entities: Some(vec!["e2".into(), "e3".into()]),
            ..fact("a", "alpha two")
        })
        .unwrap();
    assert!(tracked(&memory));
    // Unlinked before, linked by the new stamp: still in place.
    memory
        .remember(RememberInput {
            supersedes: Some(vec!["c".into()]),
            ..fact("a", "alpha three")
        })
        .unwrap();
    assert!(tracked(&memory));
    // Linked (c supersedes b): only a rebuild reproduces other facts' links to it.
    memory.remember(fact("b", "beta two")).unwrap();
    assert!(!tracked(&memory));
    memory.ensure_fresh();
    assert!(tracked(&memory));
}
