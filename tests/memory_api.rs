// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Invariants of the memory API: history is never lost, as-of queries are consistent and
//! independent of arrival order, forget is complete for its scope, retries are idempotent,
//! and mutations survive restart and an abrupt process death.

use std::collections::BTreeSet;
use std::sync::Arc;

use holographic_memory::core::memory_api::*;
use holographic_memory::core::HmsConfig;
use holographic_memory::{DocumentInput, EmbeddingSpace, HmsCore, SearchOptions};
use serde_json::json;

const DIM: u32 = 4096;

fn retry_contention<T>(mut call: impl FnMut() -> anyhow::Result<T>) -> T {
    for _ in 0..200 {
        match call() {
            Ok(value) => return value,
            Err(error)
                if matches!(
                    error.downcast_ref::<MemoryError>(),
                    Some(MemoryError::Concurrent(_))
                ) =>
            {
                std::thread::yield_now();
            }
            Err(error) => panic!("unexpected memory failure: {error:#}"),
        }
    }
    panic!("finite concurrent writers did not make progress");
}

fn open(path: &std::path::Path) -> Arc<HmsCore> {
    Arc::new(HmsCore::new(DIM, Some(path.display().to_string()), None).unwrap())
}

fn memory_at(path: &std::path::Path, clock: &Arc<ManualClock>) -> Memory {
    Memory::new(open(path)).with_clock(clock.clone())
}

fn latest_by_subject(path: &std::path::Path, clock: &Arc<ManualClock>) -> Memory {
    let config = MemoryConfig {
        supersession: SupersessionSpec::LatestBySubject,
        ..MemoryConfig::default()
    };
    Memory::with_config(open(path), config)
        .unwrap()
        .with_clock(clock.clone())
}

fn fact(id: &str, text: &str) -> RememberInput {
    RememberInput {
        id: Some(id.into()),
        text: text.into(),
        ..RememberInput::default()
    }
}

fn ids(results: &[Recalled]) -> Vec<&str> {
    results.iter().map(|r| r.id.as_str()).collect()
}

fn recall(memory: &Memory, query: &str, options: RecallOptions) -> Vec<Recalled> {
    memory.recall(query, &options).unwrap()
}

#[test]
fn knowledge_eligibility_preserves_unstamped_documents_before_selection() {
    let dir = tempfile::tempdir().unwrap();
    let core = open(dir.path());
    core.memorize_document(
        serde_json::from_value(json!({
            "id": "plain", "text": "station timetable available", "sourceUri": "fixture:plain"
        }))
        .unwrap(),
    )
    .unwrap();
    let clock = Arc::new(ManualClock::new(2000));
    let memory = Memory::new(core).with_clock(clock);
    memory
        .remember(fact("future", "station timetable priority priority"))
        .unwrap();
    let hits = memory
        .recall(
            "station timetable priority",
            &RecallOptions {
                known_at: Some(1000),
                history: Some(HistoryMode::All),
                k: Some(1),
                candidate_limit: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(ids(&hits), ["plain"]);
    assert_eq!(hits[0].text.as_deref(), Some("station timetable available"));
}

#[test]
fn supersession_keeps_history_and_defaults_to_current_only() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(1_000));
    let memory = memory_at(dir.path(), &clock);
    memory
        .remember(fact("city-1", "Alice lives in Paris"))
        .unwrap();
    clock.advance(10);
    let mut newer = fact("city-2", "Alice lives in Berlin");
    newer.supersedes = Some(vec!["city-1".into()]);
    let outcome = memory.remember(newer).unwrap();
    assert_eq!(outcome.supersedes, vec!["city-1".to_string()]);

    let current = recall(&memory, "where does Alice live", RecallOptions::default());
    assert_eq!(ids(&current), ["city-2"]);
    assert_eq!(current[0].status, FactStatus::Current);

    let all = recall(
        &memory,
        "where does Alice live",
        RecallOptions {
            history: Some(HistoryMode::All),
            ..RecallOptions::default()
        },
    );
    assert_eq!(all.len(), 2);
    let old = all.iter().find(|r| r.id == "city-1").unwrap();
    assert_eq!(old.status, FactStatus::Superseded);
    assert_eq!(old.superseded_by, vec!["city-2".to_string()]);
    assert_eq!(old.text.as_deref(), Some("Alice lives in Paris"));

    let historical = recall(
        &memory,
        "Alice",
        RecallOptions {
            history: Some(HistoryMode::Historical),
            ..RecallOptions::default()
        },
    );
    assert_eq!(ids(&historical), ["city-1"]);

    let chain = memory.history("city-1").unwrap();
    assert_eq!(
        chain.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        ["city-1", "city-2"]
    );
    // The superseded fact is untouched: still a stored document with its own text.
    assert!(memory.core().has_document("city-1"));
    assert_eq!(
        memory.get("city-1").unwrap().unwrap().text.as_deref(),
        Some("Alice lives in Paris")
    );
}

#[test]
fn default_history_is_tunable_per_config() {
    let dir = tempfile::tempdir().unwrap();
    let config = MemoryConfig {
        default_history: HistoryMode::All,
        ..MemoryConfig::default()
    };
    let memory = Memory::with_config(open(dir.path()), config).unwrap();
    memory.remember(fact("a", "alpha topic")).unwrap();
    let mut b = fact("b", "alpha topic updated");
    b.supersedes = Some(vec!["a".into()]);
    memory.remember(b).unwrap();
    assert_eq!(recall(&memory, "alpha", RecallOptions::default()).len(), 2);
    let current_only = RecallOptions {
        history: Some(HistoryMode::Current),
        ..RecallOptions::default()
    };
    assert_eq!(ids(&recall(&memory, "alpha", current_only)), ["b"]);
}

/// All permutations of `items`.
fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(i);
        for mut tail in permutations(&rest) {
            tail.insert(0, head.clone());
            out.push(tail);
        }
    }
    out
}

#[test]
fn as_of_results_are_independent_of_arrival_order_and_match_the_timeline() {
    // Four facts of one slot, valid from 100, 200, 300, 400. Whatever order they are
    // remembered in, the current fact at time t is the latest one that has begun by t.
    let timeline: [(&str, i64); 4] = [("v1", 100), ("v2", 200), ("v3", 300), ("v4", 400)];
    for order in permutations(&timeline) {
        let dir = tempfile::tempdir().unwrap();
        let clock = Arc::new(ManualClock::new(10_000));
        let memory = latest_by_subject(dir.path(), &clock);
        for (id, from) in &order {
            clock.advance(1);
            memory
                .remember(RememberInput {
                    id: Some((*id).into()),
                    text: format!("slot value {id}"),
                    subject: Some("slot".into()),
                    valid_from: Some(*from),
                    ..RememberInput::default()
                })
                .unwrap();
        }
        for t in [50, 100, 150, 199, 200, 250, 399, 400, 10_000] {
            let expected: Vec<&str> = timeline
                .iter()
                .filter(|(_, from)| *from <= t)
                .map(|(id, _)| *id)
                .next_back()
                .into_iter()
                .collect();
            let got = recall(
                &memory,
                "slot value",
                RecallOptions {
                    as_of: Some(t),
                    ..RecallOptions::default()
                },
            );
            assert_eq!(ids(&got), expected, "order {order:?} as of {t}");
        }
        // Nothing was lost along the way.
        let history = memory.history("v1").unwrap();
        assert_eq!(history.len(), 4, "order {order:?}");
    }
}

#[test]
fn validity_windows_and_known_at_are_separate_axes() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(5_000));
    let memory = memory_at(dir.path(), &clock);
    memory
        .remember(RememberInput {
            id: Some("promo".into()),
            text: "promotion code SAVE10 applies".into(),
            valid_from: Some(1_000),
            valid_to: Some(2_000),
            ..RememberInput::default()
        })
        .unwrap();
    let at = |t: i64, history| RecallOptions {
        as_of: Some(t),
        history: Some(history),
        ..RecallOptions::default()
    };
    assert!(recall(&memory, "promotion", at(999, HistoryMode::Current)).is_empty());
    assert_eq!(
        recall(&memory, "promotion", at(1_000, HistoryMode::Current)).len(),
        1
    );
    assert_eq!(
        recall(&memory, "promotion", at(1_999, HistoryMode::Current)).len(),
        1
    );
    assert!(recall(&memory, "promotion", at(2_000, HistoryMode::Current)).is_empty());
    let expired = recall(&memory, "promotion", at(2_500, HistoryMode::All));
    assert_eq!(expired[0].status, FactStatus::Expired);
    let early = recall(&memory, "promotion", at(500, HistoryMode::All));
    assert_eq!(early[0].status, FactStatus::NotYetValid);

    // Knowledge time: the store ingested the fact at 5_000, so it knew nothing at 4_000.
    let known = |t: i64| RecallOptions {
        as_of: Some(1_500),
        known_at: Some(t),
        ..RecallOptions::default()
    };
    assert!(recall(&memory, "promotion", known(4_000)).is_empty());
    assert_eq!(recall(&memory, "promotion", known(5_000)).len(), 1);

    // A supersession ingested later does not exist yet in an earlier view of the store.
    clock.set(9_000);
    memory
        .remember(RememberInput {
            id: Some("promo2".into()),
            text: "promotion code SAVE20 applies".into(),
            valid_from: Some(1_500),
            supersedes: Some(vec!["promo".into()]),
            ..RememberInput::default()
        })
        .unwrap();
    let view = |known_at| RecallOptions {
        as_of: Some(1_700),
        known_at,
        ..RecallOptions::default()
    };
    assert_eq!(
        ids(&recall(&memory, "promotion", view(Some(6_000)))),
        ["promo"]
    );
    assert_eq!(
        ids(&recall(&memory, "promotion", view(Some(9_000)))),
        ["promo2"]
    );
    assert_eq!(ids(&recall(&memory, "promotion", view(None))), ["promo2"]);
}

#[test]
fn late_arriving_older_fact_is_born_history() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(10_000));
    let memory = latest_by_subject(dir.path(), &clock);
    let slot = |id: &str, from| RememberInput {
        id: Some(id.into()),
        text: format!("residence {id}"),
        subject: Some("alice/city".into()),
        valid_from: Some(from),
        ..RememberInput::default()
    };
    memory.remember(slot("new", 500)).unwrap();
    let late = memory.remember(slot("old", 100)).unwrap();
    assert_eq!(late.superseded_by, vec!["new".to_string()]);
    assert_eq!(
        ids(&recall(&memory, "residence", RecallOptions::default())),
        ["new"]
    );
    let at_150 = recall(
        &memory,
        "residence",
        RecallOptions {
            as_of: Some(150),
            ..RecallOptions::default()
        },
    );
    assert_eq!(ids(&at_150), ["old"]);
}

#[test]
fn retries_are_idempotent_and_conflicts_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(100));
    let memory = memory_at(dir.path(), &clock);
    let mut input = fact("f", "the capital of France is Paris");
    input.entities = Some(vec!["France".into()]);
    let first = memory.remember(input.clone()).unwrap();
    assert!(first.created);
    let revision = memory.core().revision();
    clock.advance(500);
    let again = memory.remember(input.clone()).unwrap();
    assert!(!again.created);
    assert_eq!(
        again.ingest_time, first.ingest_time,
        "a retry keeps the original ingest time"
    );
    assert_eq!(memory.core().revision(), revision, "a retry writes nothing");

    let mut changed = input.clone();
    changed.text = "the capital of France is Lyon".into();
    assert!(memory.remember(changed).is_err());

    // Without an id the id is derived from the content, so a retried call stores one fact.
    let anonymous = RememberInput {
        text: "tea is hot".into(),
        ..RememberInput::default()
    };
    let a = memory.remember(anonymous.clone()).unwrap();
    let b = memory.remember(anonymous).unwrap();
    assert_eq!(a.id, b.id);
    assert!(a.created && !b.created);

    // Retrying a supersession adds no duplicate link.
    let mut newer = fact("g", "the capital of France is still Paris");
    newer.supersedes = Some(vec!["f".into()]);
    memory.remember(newer.clone()).unwrap();
    memory.remember(newer).unwrap();
    assert_eq!(memory.get("f").unwrap().unwrap().superseded_by, ["g"]);
    assert_eq!(memory.history("f").unwrap().len(), 2);

    // A batch that names the same id twice is refused whole.
    assert!(memory
        .remember_many(vec![fact("dup", "x one"), fact("dup", "x two")])
        .is_err());
    assert!(!memory.core().has_document("dup"));
}

#[test]
fn replace_conflict_mode_overwrites_and_reindexes() {
    let dir = tempfile::tempdir().unwrap();
    let config = MemoryConfig {
        on_id_conflict: IdConflict::Replace,
        ..MemoryConfig::default()
    };
    let memory = Memory::with_config(open(dir.path()), config).unwrap();
    memory.remember(fact("f", "first wording")).unwrap();
    memory.remember(fact("f", "second wording")).unwrap();
    let got = recall(&memory, "wording", RecallOptions::default());
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].text.as_deref(), Some("second wording"));
}

#[test]
fn invalid_input_is_rejected_without_writing() {
    let dir = tempfile::tempdir().unwrap();
    let memory = memory_at(dir.path(), &Arc::new(ManualClock::new(1)));
    let revision = memory.core().revision();
    for bad in [
        fact("x", "   "),
        fact("hms:x", "text"),
        RememberInput {
            valid_from: Some(5),
            valid_to: Some(5),
            ..fact("x", "text")
        },
        RememberInput {
            metadata: Some(json!({"_mem": 1})),
            ..fact("x", "text")
        },
        RememberInput {
            supersedes: Some(vec!["missing".into()]),
            ..fact("x", "text")
        },
        RememberInput {
            supersedes: Some(vec!["x".into()]),
            ..fact("x", "text")
        },
        RememberInput {
            metadata: Some(json!([1])),
            ..fact("x", "text")
        },
    ] {
        assert!(memory.remember(bad).is_err());
    }
    assert_eq!(memory.core().revision(), revision);
    // A failing fact fails the whole batch.
    assert!(memory
        .remember_many(vec![fact("ok", "fine text"), fact("hms:no", "bad id")])
        .is_err());
    assert!(!memory.core().has_document("ok"));
}

#[test]
fn forget_is_complete_for_each_scope_and_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(1_000));
    let memory = latest_by_subject(dir.path(), &clock);
    let secret = "zebracrossing";
    let add = |id: &str, subject: Option<&str>, entity: Option<&str>, tenant: &str| {
        memory
            .remember(RememberInput {
                id: Some(id.into()),
                text: format!("{secret} {id} note"),
                subject: subject.map(str::to_owned),
                entities: entity.map(|e| vec![e.to_owned()]),
                metadata: Some(json!({"tenant": tenant})),
                valid_from: Some(clock.now_millis()),
                ..RememberInput::default()
            })
            .unwrap();
        clock.advance(10);
    };
    add("s1", Some("slot/a"), None, "t1");
    add("s2", Some("slot/a"), None, "t1");
    add("s3", Some("slot/a"), None, "t1");
    add("e1", None, Some("Acme"), "t2");
    add("e2", None, Some("Acme"), "t2");
    add("m1", None, None, "t3");
    add("m2", None, None, "t3");
    add("keep", None, None, "t4");

    let remaining = |memory: &Memory| -> BTreeSet<String> {
        recall(
            memory,
            secret,
            RecallOptions {
                k: Some(50),
                history: Some(HistoryMode::All),
                ..RecallOptions::default()
            },
        )
        .into_iter()
        .map(|r| r.id)
        .collect()
    };
    let raw = |memory: &Memory| -> BTreeSet<String> {
        memory
            .core()
            .search_documents(
                secret,
                &SearchOptions {
                    k: Some(50),
                    ..Default::default()
                },
            )
            .unwrap()
            .into_iter()
            .map(|r| r.document_id)
            .collect()
    };
    assert_eq!(remaining(&memory).len(), 8);

    // Dry run changes nothing.
    let dry = memory
        .forget(
            &ForgetScope::Subject {
                subject: "slot/a".into(),
            },
            &ForgetOptions {
                dry_run: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(dry.forgotten, ["s1", "s2", "s3"]);
    assert_eq!(remaining(&memory).len(), 8);

    // One fact of a chain: its predecessor is current again.
    let report = memory
        .forget(
            &ForgetScope::Ids {
                ids: vec!["s3".into()],
            },
            &ForgetOptions::default(),
        )
        .unwrap();
    assert_eq!(report.forgotten, ["s3"]);
    assert_eq!(
        memory.get("s2").unwrap().unwrap().status,
        FactStatus::Current
    );
    assert!(!memory.core().has_document("s3"));

    // The rest of the chain, by subject.
    let report = memory
        .forget(
            &ForgetScope::Subject {
                subject: "slot/a".into(),
            },
            &ForgetOptions::default(),
        )
        .unwrap();
    assert_eq!(report.forgotten, ["s1", "s2"]);

    memory
        .forget(
            &ForgetScope::Entity {
                entity: "Acme".into(),
            },
            &ForgetOptions::default(),
        )
        .unwrap();
    memory
        .forget(
            &ForgetScope::Matching {
                filter: json!({"tenant": "t3"}),
            },
            &ForgetOptions::default(),
        )
        .unwrap();
    assert_eq!(remaining(&memory), BTreeSet::from(["keep".to_string()]));
    assert_eq!(raw(&memory), BTreeSet::from(["keep".to_string()]));
    for gone in ["s1", "s2", "s3", "e1", "e2", "m1", "m2"] {
        assert!(!memory.core().has_document(gone), "{gone}");
        assert!(memory.get(gone).unwrap().is_none());
        assert!(memory.history(gone).is_err());
        assert!(memory
            .explain(secret, gone, &RecallOptions::default())
            .is_err());
    }
    // Repeating removes nothing more.
    let again = memory
        .forget(
            &ForgetScope::Entity {
                entity: "Acme".into(),
            },
            &ForgetOptions::default(),
        )
        .unwrap();
    assert!(again.forgotten.is_empty());

    // Time-based scopes.
    clock.set(50_000);
    add("old1", None, None, "t5");
    clock.set(60_000);
    add("new1", None, None, "t5");
    let report = memory
        .forget(
            &ForgetScope::IngestedBefore { time: 55_000 },
            &ForgetOptions::default(),
        )
        .unwrap();
    assert!(report.forgotten.contains(&"keep".to_string()));
    assert!(report.forgotten.contains(&"old1".to_string()));
    assert!(!report.forgotten.contains(&"new1".to_string()));
    assert_eq!(remaining(&memory), BTreeSet::from(["new1".to_string()]));

    // And the removal survives a restart.
    drop(memory);
    let memory = latest_by_subject(dir.path(), &clock);
    assert_eq!(remaining(&memory), BTreeSet::from(["new1".to_string()]));
    assert_eq!(raw(&memory), BTreeSet::from(["new1".to_string()]));
}

#[test]
fn forget_with_chain_removes_every_version_and_purge_rewrites_the_journal() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(1));
    let memory = latest_by_subject(dir.path(), &clock);
    for (i, word) in ["quartzite", "feldspar", "obsidian"].iter().enumerate() {
        clock.advance(1);
        memory
            .remember(RememberInput {
                id: Some(format!("v{i}")),
                text: format!("mineral {word}"),
                subject: Some("rock".into()),
                valid_from: Some(i as i64 * 10),
                ..RememberInput::default()
            })
            .unwrap();
    }
    memory
        .remember(fact("other", "unrelated mineral basalt"))
        .unwrap();
    // Without purge the journal still holds the text; with it, the text is gone from disk.
    let on_disk = |needle: &str| -> bool {
        std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_file())
            .any(|entry| {
                std::fs::read(entry.path())
                    .map(|bytes| bytes.windows(needle.len()).any(|w| w == needle.as_bytes()))
                    .unwrap_or(false)
            })
    };
    assert!(on_disk("feldspar"));
    let report = memory
        .forget(
            &ForgetScope::Ids {
                ids: vec!["v1".into()],
            },
            &ForgetOptions {
                include_chain: true,
                purge: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(report.forgotten, ["v0", "v1", "v2"]);
    assert!(report.purged);
    // The journal is rewritten without the text at once; the in-memory text segment file
    // (`documents.text`) is rebuilt from the journal the next time the store opens.
    drop(memory);
    let memory = latest_by_subject(dir.path(), &clock);
    for word in ["quartzite", "feldspar", "obsidian"] {
        assert!(!on_disk(word), "{word} survived the purge");
    }
    assert!(on_disk("basalt"));
    assert_eq!(
        ids(&recall(&memory, "mineral", RecallOptions::default())),
        ["other"]
    );
}

#[test]
fn forgetting_a_middle_version_does_not_resurrect_an_older_one() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(10_000));
    let memory = latest_by_subject(dir.path(), &clock);
    for i in 0..4 {
        clock.advance(1);
        memory
            .remember(RememberInput {
                id: Some(format!("v{i}")),
                text: format!("setting is value {i}"),
                subject: Some("setting".into()),
                valid_from: Some(100 * (i + 1)),
                ..RememberInput::default()
            })
            .unwrap();
    }
    memory
        .forget(
            &ForgetScope::Ids {
                ids: vec!["v1".into(), "v2".into()],
            },
            &ForgetOptions::default(),
        )
        .unwrap();
    let check = |memory: &Memory| {
        assert_eq!(
            ids(&recall(memory, "setting", RecallOptions::default())),
            ["v3"]
        );
        let chain: Vec<String> = memory
            .history("v0")
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(chain, ["v0", "v3"]);
        assert_eq!(
            memory.get("v0").unwrap().unwrap().status,
            FactStatus::Superseded
        );
        let early = recall(
            memory,
            "setting",
            RecallOptions {
                as_of: Some(150),
                ..RecallOptions::default()
            },
        );
        assert_eq!(ids(&early), ["v0"]);
    };
    check(&memory);
    drop(memory);
    check(&latest_by_subject(dir.path(), &clock));
}

#[test]
fn default_recall_is_document_search_when_nothing_is_constrained() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Memory::new(open(dir.path()));
    for i in 0..40 {
        memory
            .remember(fact(
                &format!("d{i}"),
                &format!("note {i} about topic{} and shared words", i % 5),
            ))
            .unwrap();
    }
    let options = RecallOptions {
        k: Some(7),
        ..RecallOptions::default()
    };
    let recalled = memory.recall("topic3 shared", &options).unwrap();
    let raw = memory
        .core()
        .search_documents(
            "topic3 shared",
            &SearchOptions {
                k: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(recalled.len(), raw.len());
    for (a, b) in recalled.iter().zip(&raw) {
        assert_eq!(a.id, b.document_id);
        assert_eq!(
            a.score.to_bits(),
            b.score.to_bits(),
            "scores are bit-identical"
        );
    }
}

#[test]
fn plain_documents_and_old_stores_work_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    {
        // A store written with no memory API at all.
        let core = open(dir.path());
        core.memorize_document(
            serde_json::from_value::<DocumentInput>(json!({
                "id": "legacy", "text": "legacy document about glaciers",
                "metadata": {"kind": "old"}
            }))
            .unwrap(),
        )
        .unwrap();
    }
    let memory = Memory::new(open(dir.path()));
    let got = recall(&memory, "glaciers", RecallOptions::default());
    assert_eq!(ids(&got), ["legacy"]);
    assert_eq!(got[0].status, FactStatus::Current);
    assert_eq!(got[0].metadata, json!({"kind": "old"}));
    assert!(got[0].ingest_time.is_none());
    // A plain document can be superseded and forgotten like a fact.
    let mut next = fact("fresh", "new document about glaciers");
    next.supersedes = Some(vec!["legacy".into()]);
    memory.remember(next).unwrap();
    assert_eq!(
        ids(&recall(&memory, "glaciers", RecallOptions::default())),
        ["fresh"]
    );
    assert_eq!(memory.history("legacy").unwrap().len(), 2);
}

#[test]
fn state_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(10_000));
    {
        let memory = latest_by_subject(dir.path(), &clock);
        for (i, city) in ["Paris", "Rome", "Oslo"].iter().enumerate() {
            clock.advance(1);
            memory
                .remember(RememberInput {
                    id: Some(format!("c{i}")),
                    text: format!("Alice lives in {city}"),
                    subject: Some("alice/city".into()),
                    valid_from: Some(100 * (i as i64 + 1)),
                    entities: Some(vec!["Alice".into()]),
                    metadata: Some(json!({"src": "chat"})),
                    ..RememberInput::default()
                })
                .unwrap();
        }
    }
    let memory = latest_by_subject(dir.path(), &clock);
    assert_eq!(
        ids(&recall(&memory, "Alice lives", RecallOptions::default())),
        ["c2"]
    );
    let chain = memory.history("c1").unwrap();
    assert_eq!(chain.len(), 3);
    assert_eq!(chain[0].status, FactStatus::Superseded);
    assert_eq!(chain[0].metadata, json!({"src": "chat"}));
    let at_250 = recall(
        &memory,
        "Alice lives",
        RecallOptions {
            as_of: Some(250),
            ..RecallOptions::default()
        },
    );
    assert_eq!(ids(&at_250), ["c1"]);
    let by_entity = recall(
        &memory,
        "lives",
        RecallOptions {
            entities: Some(vec!["Alice".into()]),
            ..RecallOptions::default()
        },
    );
    assert_eq!(ids(&by_entity), ["c2"]);
    // A new fact after restart still resolves against the persisted chain.
    clock.advance(1);
    memory
        .remember(RememberInput {
            id: Some("c3".into()),
            text: "Alice lives in Bergen".into(),
            subject: Some("alice/city".into()),
            valid_from: Some(1_000),
            ..RememberInput::default()
        })
        .unwrap();
    assert_eq!(memory.get("c2").unwrap().unwrap().superseded_by, ["c3"]);
}

#[test]
fn explain_reports_scores_terms_validity_and_chain() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(1_000));
    let memory = memory_at(dir.path(), &clock);
    memory
        .remember(fact("a", "the migration to postgres finished"))
        .unwrap();
    clock.advance(5);
    let mut b = fact("b", "the migration to postgres was reverted");
    b.supersedes = Some(vec!["a".into()]);
    memory.remember(b).unwrap();
    memory.remember(fact("c", "lunch menu for friday")).unwrap();

    let options = RecallOptions {
        explain: true,
        ..RecallOptions::default()
    };
    let results = memory
        .recall("postgres migration reverted", &options)
        .unwrap();
    assert_eq!(results[0].id, "b");
    let explanation = results[0].explanation.as_ref().unwrap();
    assert!(explanation.retrieved);
    assert_eq!(explanation.rank, Some(1));
    assert_eq!(
        explanation.matched_terms,
        ["postgres", "migration", "reverted"]
    );
    assert!(explanation.lexical_score.unwrap() > 0.0);
    assert_eq!(explanation.final_score, Some(results[0].score));
    assert_eq!(explanation.status, FactStatus::Current);
    assert_eq!(explanation.chain.len(), 1);
    assert_eq!(explanation.chain[0].id, "a");
    assert_eq!(explanation.chain[0].relation, ChainRelation::Predecessor);
    // No explanation unless asked.
    assert!(memory
        .recall("postgres", &RecallOptions::default())
        .unwrap()
        .iter()
        .all(|r| r.explanation.is_none()));

    // Explaining a superseded fact shows its successor; it is retrieved with history included.
    let old = memory
        .explain(
            "postgres migration finished",
            "a",
            &RecallOptions::default(),
        )
        .unwrap();
    assert!(old.retrieved);
    assert_eq!(old.status, FactStatus::Superseded);
    assert_eq!(old.chain[0].relation, ChainRelation::Successor);
    // A fact that shares no query terms is not retrieved, and says so.
    let miss = memory
        .explain("postgres migration", "c", &RecallOptions::default())
        .unwrap();
    assert!(!miss.retrieved);
    assert!(miss.matched_terms.is_empty());
}

#[test]
fn explain_reports_semantic_scores_with_an_embedding_space() {
    let dir = tempfile::tempdir().unwrap();
    let config = HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "test".into(),
            revision: "1".into(),
            dimensions: 4,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..HmsConfig::default()
    };
    let core =
        Arc::new(HmsCore::new(DIM, Some(dir.path().display().to_string()), Some(config)).unwrap());
    let memory = Memory::new(core);
    let with = |id: &str, text: &str, e: [f64; 4]| RememberInput {
        embedding: Some(e.to_vec()),
        ..fact(id, text)
    };
    memory
        .remember(with("x", "apples are red", [1.0, 0.0, 0.0, 0.0]))
        .unwrap();
    memory
        .remember(with("y", "bananas are yellow", [0.0, 1.0, 0.0, 0.0]))
        .unwrap();
    let options = RecallOptions {
        embedding: Some(vec![0.0, 1.0, 0.0, 0.0]),
        explain: true,
        ..RecallOptions::default()
    };
    let got = memory.recall("fruit", &options).unwrap();
    assert_eq!(got[0].id, "y");
    let explanation = got[0].explanation.as_ref().unwrap();
    assert!((explanation.semantic_score.unwrap() - 1.0).abs() < 1e-9);
    assert!(explanation.matched_terms.is_empty());
}

#[test]
fn recency_and_historical_weight_rescale_scores_deterministically() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(1_000_000));
    let config = MemoryConfig {
        recency: RecencySpec::Exponential {
            half_life_ms: 1_000,
            basis: RecencyBasis::Event,
            floor: 0.0,
        },
        ..MemoryConfig::default()
    };
    let memory = Memory::with_config(open(dir.path()), config)
        .unwrap()
        .with_clock(clock.clone());
    let at = |id: &str, event: i64| RememberInput {
        event_time: Some(event),
        ..fact(id, "identical words about launch")
    };
    memory.remember(at("old", 1_000_000 - 5_000)).unwrap();
    memory.remember(at("fresh", 1_000_000)).unwrap();
    let got = memory
        .recall(
            "launch",
            &RecallOptions {
                explain: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(ids(&got), ["fresh", "old"]);
    let factor = got[1].explanation.as_ref().unwrap().recency_factor;
    assert!((factor - 0.5f64.powi(5)).abs() < 1e-12);
    let again = memory.recall("launch", &RecallOptions::default()).unwrap();
    assert_eq!(ids(&again), ids(&got));
}

#[test]
fn forgetting_and_consolidation_defaults_do_nothing_and_policies_are_opt_in() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(10_000));
    let memory = memory_at(dir.path(), &clock);
    let timed = |id: &str, to: Option<i64>| RememberInput {
        valid_to: to,
        ..fact(id, "reminder about the standup meeting")
    };
    memory.remember(timed("expired", Some(2_000))).unwrap();
    memory
        .remember(RememberInput {
            text: "completely different Standup meeting".into(),
            ..timed("open", None)
        })
        .unwrap();
    // Defaults: nothing selected, nothing planned.
    assert!(memory
        .apply_forgetting(&ForgetOptions::default())
        .unwrap()
        .forgotten
        .is_empty());
    assert!(memory
        .consolidate(&ConsolidateOptions::default())
        .unwrap()
        .created
        .is_empty());

    let policy = ExpiredFor { grace_ms: 1_000 };
    let dry = memory
        .apply_forgetting_with(
            &policy,
            &ForgetOptions {
                dry_run: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(dry.forgotten, ["expired"]);
    assert!(memory.core().has_document("expired"));
    let real = memory
        .apply_forgetting_with(&policy, &ForgetOptions::default())
        .unwrap();
    assert_eq!(real.forgotten, ["expired"]);
    assert!(!memory.core().has_document("expired"));

    // Consolidation is non-destructive: duplicates are superseded, not deleted, and a
    // repeated run adds nothing.
    let dup = |id: &str| fact(id, "Water boils at 100 degrees");
    memory.remember(dup("w1")).unwrap();
    clock.advance(1);
    memory
        .remember(RememberInput {
            text: "water  boils at 100 DEGREES".into(),
            ..dup("w2")
        })
        .unwrap();
    let report = memory
        .consolidate_with(&CollapseDuplicates, &ConsolidateOptions::default())
        .unwrap();
    assert_eq!(report.created.len(), 1);
    assert!(report.forgotten.is_empty());
    assert_eq!(
        memory.get("w1").unwrap().unwrap().status,
        FactStatus::Superseded
    );
    assert!(memory.core().has_document("w2"));
    let repeat = memory
        .consolidate_with(&CollapseDuplicates, &ConsolidateOptions::default())
        .unwrap();
    assert!(repeat.created.is_empty());
    let current = recall(&memory, "water boils", RecallOptions::default());
    assert_eq!(current.len(), 1);
    assert_eq!(current[0].id, report.created[0].id);

    // Forgetting policies built in and custom.
    struct Everything;
    impl ForgettingPolicy for Everything {
        fn select(&self, facts: &[FactSummary], _: i64) -> Vec<String> {
            facts.iter().map(|f| f.id.clone()).collect()
        }
    }
    let dry = memory
        .apply_forgetting_with(
            &Everything,
            &ForgetOptions {
                dry_run: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(dry.forgotten.len() >= 4);
    let capped = memory
        .apply_forgetting_with(&MaxFacts { limit: 2 }, &ForgetOptions::default())
        .unwrap();
    assert_eq!(memory.summaries(false).len(), 2);
    // The non-current facts went first, so the current ones remain.
    assert!(capped.forgotten.iter().all(|id| id != &current[0].id));
}

#[test]
fn destructive_consolidation_needs_an_explicit_opt_in() {
    struct Merge;
    impl ConsolidationPolicy for Merge {
        fn plan(&self, facts: &[FactSummary], _: i64) -> Vec<ConsolidationAction> {
            let sources: Vec<String> = facts.iter().map(|f| f.id.clone()).collect();
            vec![ConsolidationAction {
                summary: RememberInput {
                    text: "summary of notes".into(),
                    ..Default::default()
                },
                sources,
                forget_sources: true,
            }]
        }
    }
    for allow in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let memory = Memory::new(open(dir.path()));
        memory.remember(fact("n1", "note one")).unwrap();
        memory.remember(fact("n2", "note two")).unwrap();
        let report = memory
            .consolidate_with(
                &Merge,
                &ConsolidateOptions {
                    allow_destructive: allow,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(report.forgotten.len(), if allow { 2 } else { 0 });
        assert_eq!(memory.core().has_document("n1"), !allow);
    }
}

#[test]
fn entity_extractor_and_observer_are_optional_hooks() {
    use std::sync::Mutex;
    struct Log(Mutex<Vec<String>>);
    impl MemoryObserver for Log {
        fn on_remembered(&self, fact: &Remembered, _: &RememberInput) {
            self.0
                .lock()
                .unwrap()
                .push(format!("remembered {}", fact.id));
        }
        fn on_forgotten(&self, ids: &[String]) {
            self.0
                .lock()
                .unwrap()
                .push(format!("forgotten {}", ids.join(",")));
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let log = Arc::new(Log(Mutex::new(Vec::new())));
    let memory = Memory::new(open(dir.path()))
        .with_entity_extractor(Arc::new(|text: &str| {
            text.split_whitespace()
                .filter(|word| word.chars().next().is_some_and(char::is_uppercase))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        }))
        .with_observer(log.clone());
    memory
        .remember(fact("a", "Bob met Carol in Lisbon"))
        .unwrap();
    memory
        .remember(fact("a", "Bob met Carol in Lisbon"))
        .unwrap();
    let got = recall(
        &memory,
        "met",
        RecallOptions {
            entities: Some(vec!["Carol".into()]),
            ..Default::default()
        },
    );
    assert_eq!(ids(&got), ["a"]);
    assert!(recall(
        &memory,
        "met",
        RecallOptions {
            entities: Some(vec!["Zed".into()]),
            ..Default::default()
        }
    )
    .is_empty());
    memory
        .forget(
            &ForgetScope::Entity {
                entity: "Bob".into(),
            },
            &ForgetOptions::default(),
        )
        .unwrap();
    // The retry did not notify twice.
    assert_eq!(*log.0.lock().unwrap(), ["remembered a", "forgotten a"]);
}

#[test]
fn custom_supersession_policy_is_validated() {
    struct Bad;
    impl SupersessionPolicy for Bad {
        fn resolve(&self, _: &FactView, _: &[FactView]) -> Resolution {
            Resolution {
                supersedes: vec!["not-in-subject".into()],
                superseded_by: vec![],
            }
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let memory = Memory::new(open(dir.path())).with_supersession(Arc::new(Bad));
    let result = memory.remember(RememberInput {
        subject: Some("s".into()),
        ..fact("a", "text here")
    });
    assert!(result.is_err());
    assert!(!memory.core().has_document("a"));
}

/// A clock that runs one armed action the next time it is read. Recall reads the clock
/// between its search and its index check, so the action lands a commit in that window.
struct InterleavingClock {
    armed: std::sync::Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

impl Clock for InterleavingClock {
    fn now_millis(&self) -> Millis {
        let action = self.armed.lock().unwrap().take();
        if let Some(action) = action {
            action();
        }
        1_000
    }
}

#[test]
fn recall_judges_status_from_the_revision_it_searched() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(InterleavingClock {
        armed: std::sync::Mutex::new(None),
    });
    let config = MemoryConfig {
        supersession: SupersessionSpec::LatestBySubject,
        ..MemoryConfig::default()
    };
    let memory = Arc::new(
        Memory::with_config(open(dir.path()), config)
            .unwrap()
            .with_clock(clock.clone()),
    );
    let version = |id: &str, from: i64| RememberInput {
        subject: Some("worker".into()),
        valid_from: Some(from),
        ..fact(id, &format!("worker reports {id}"))
    };
    memory.remember(version("v0", 1)).unwrap();
    let writer = Arc::downgrade(&memory);
    *clock.armed.lock().unwrap() = Some(Box::new(move || {
        writer
            .upgrade()
            .unwrap()
            .remember(version("v1", 2))
            .unwrap();
    }));
    let got = memory
        .recall("worker reports", &RecallOptions::default())
        .unwrap();
    assert!(
        clock.armed.lock().unwrap().is_none(),
        "the commit did not interleave"
    );
    let current: Vec<&str> = got.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(current, ["v1"]);
}

#[test]
fn concurrent_writers_and_readers_stay_consistent_with_foreign_writes() {
    let dir = tempfile::tempdir().unwrap();
    let core = open(dir.path());
    let config = MemoryConfig {
        supersession: SupersessionSpec::LatestBySubject,
        ..MemoryConfig::default()
    };
    let memory = Arc::new(Memory::with_config(core.clone(), config).unwrap());
    const WRITERS: usize = 4;
    const VERSIONS: usize = 12;
    std::thread::scope(|scope| {
        for writer in 0..WRITERS {
            let memory = memory.clone();
            scope.spawn(move || {
                for version in 0..VERSIONS {
                    retry_contention(|| {
                        memory.remember(RememberInput {
                            id: Some(format!("w{writer}-v{version}")),
                            text: format!("worker {writer} reports version {version}"),
                            subject: Some(format!("worker/{writer}")),
                            valid_from: Some(version as i64),
                            ..RememberInput::default()
                        })
                    });
                }
            });
        }
        // A foreign writer using the raw document API on the same core.
        let foreign = core.clone();
        scope.spawn(move || {
            for i in 0..10 {
                foreign
                    .memorize_document(
                        serde_json::from_value::<DocumentInput>(
                            json!({"id": format!("raw{i}"), "text": "raw document about workers"}),
                        )
                        .unwrap(),
                    )
                    .unwrap();
            }
        });
        for _ in 0..2 {
            let memory = memory.clone();
            scope.spawn(move || {
                for _ in 0..40 {
                    let got = retry_contention(|| {
                        memory.recall(
                            "worker reports version",
                            &RecallOptions {
                                k: Some(20),
                                ..Default::default()
                            },
                        )
                    });
                    // Current-only recall never shows two versions of one worker.
                    let mut seen = BTreeSet::new();
                    for r in &got {
                        if let Some(subject) = &r.subject {
                            assert!(
                                seen.insert(subject.clone()),
                                "two current facts for {subject}"
                            );
                        }
                    }
                }
            });
        }
    });
    let got = memory
        .recall(
            "worker reports version",
            &RecallOptions {
                k: Some(20),
                candidate_limit: Some(200),
                ..Default::default()
            },
        )
        .unwrap();
    let current: BTreeSet<&str> = got
        .iter()
        .filter(|r| r.subject.is_some())
        .map(|r| r.id.as_str())
        .collect();
    let expected: BTreeSet<String> = (0..WRITERS)
        .map(|w| format!("w{w}-v{}", VERSIONS - 1))
        .collect();
    assert_eq!(current, expected.iter().map(String::as_str).collect());
    for writer in 0..WRITERS {
        assert_eq!(
            memory.history(&format!("w{writer}-v0")).unwrap().len(),
            VERSIONS
        );
    }
    // A second handle over the same core agrees.
    let other = Memory::with_config(
        core,
        MemoryConfig {
            supersession: SupersessionSpec::LatestBySubject,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(other.history("w0-v5").unwrap().len(), VERSIONS);
}

/// Child half of the crash test: remembers, then dies without unwinding or flushing.
#[test]
fn crash_child() {
    let Ok(path) = std::env::var("HMS_MEMORY_CRASH_DIR") else {
        return;
    };
    let clock = Arc::new(ManualClock::new(1));
    let memory = latest_by_subject(std::path::Path::new(&path), &clock);
    for version in 0..5i64 {
        clock.advance(1);
        memory
            .remember(RememberInput {
                id: Some(format!("c{version}")),
                text: format!("crash fact version {version}"),
                subject: Some("crash".into()),
                valid_from: Some(version),
                ..RememberInput::default()
            })
            .unwrap();
        println!("acked c{version}");
    }
    memory
        .forget(
            &ForgetScope::Ids {
                ids: vec!["c1".into()],
            },
            &ForgetOptions::default(),
        )
        .unwrap();
    println!("acked forget");
    std::process::abort();
}

#[test]
fn acknowledged_mutations_survive_an_abrupt_process_death() {
    let dir = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crash_child", "--nocapture", "--test-threads=1"])
        .env("HMS_MEMORY_CRASH_DIR", dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!output.status.success(), "the child must die by abort");
    assert!(stdout.contains("acked forget"), "{stdout}");
    let clock = Arc::new(ManualClock::new(100));
    let memory = latest_by_subject(dir.path(), &clock);
    // c1 was forgotten; the others and their chain survive.
    assert!(!memory.core().has_document("c1"));
    let history: Vec<String> = memory
        .history("c4")
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(history, ["c0", "c2", "c3", "c4"]);
    assert_eq!(
        ids(&recall(&memory, "crash fact", RecallOptions::default())),
        ["c4"]
    );
    // The store is writable after recovery and keeps resolving the chain.
    clock.advance(1);
    memory
        .remember(RememberInput {
            id: Some("c5".into()),
            text: "crash fact version 5".into(),
            subject: Some("crash".into()),
            valid_from: Some(5),
            ..RememberInput::default()
        })
        .unwrap();
    assert_eq!(
        ids(&recall(&memory, "crash fact", RecallOptions::default())),
        ["c5"]
    );
}

#[test]
fn a_stamp_too_large_for_the_version_string_falls_back_to_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let entities: Vec<String> = (0..100).map(|i| format!("entity-number-{i:04}")).collect();
    {
        let memory = Memory::new(open(dir.path()));
        memory
            .remember(RememberInput {
                entities: Some(entities.clone()),
                metadata: Some(json!({"k": "v"})),
                ..fact("big", "fact with many entities")
            })
            .unwrap();
        let mut next = fact("big2", "newer fact with many entities");
        next.supersedes = Some(vec!["big".into()]);
        memory.remember(next).unwrap();
    }
    let memory = Memory::new(open(dir.path()));
    let got = recall(
        &memory,
        "entities",
        RecallOptions {
            entities: Some(vec!["entity-number-0099".into()]),
            history: Some(HistoryMode::All),
            ..Default::default()
        },
    );
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].id, "big");
    assert_eq!(got[0].status, FactStatus::Superseded);
    assert_eq!(got[0].metadata, json!({"k": "v"}));
    assert_eq!(got[0].entities.len(), 100);
}

#[test]
fn json_dispatch_carries_every_operation_and_classifies_errors() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Memory::new(open(dir.path()));
    let call = |op: &str, request: serde_json::Value| memory.call(op, request);
    let out = call(
        "remember",
        json!({"id": "a", "text": "alpha beta", "entities": ["X"]}),
    )
    .unwrap();
    assert_eq!(out["remembered"][0]["created"], json!(true));
    let out = call(
        "remember",
        json!({"facts": [{"id": "b", "text": "alpha gamma", "supersedes": ["a"]}]}),
    )
    .unwrap();
    assert_eq!(out["remembered"][0]["supersedes"], json!(["a"]));
    let out = call(
        "recall",
        json!({"query": "alpha", "history": "all", "explain": true}),
    )
    .unwrap();
    assert_eq!(out["results"].as_array().unwrap().len(), 2);
    assert!(out["results"][0]["explanation"].is_object());
    assert_eq!(
        call("get", json!({"id": "a"})).unwrap()["fact"]["status"],
        json!("superseded")
    );
    assert_eq!(
        call("get", json!({"id": "zzz"})).unwrap()["fact"],
        json!(null)
    );
    assert_eq!(
        call("history", json!({"id": "a"})).unwrap()["history"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert!(
        call("explain", json!({"query": "alpha", "id": "b"})).unwrap()["explanation"]["retrieved"]
            .as_bool()
            .unwrap()
    );
    // Maintenance defaults to a dry run.
    let report = call("applyForgetting", json!({})).unwrap();
    assert_eq!(report["dryRun"], json!(true));
    assert_eq!(
        call("consolidate", json!(null)).unwrap()["dryRun"],
        json!(true)
    );
    let report = call(
        "forget",
        json!({"scope": {"kind": "entity", "entity": "X"}, "includeChain": true}),
    )
    .unwrap();
    assert_eq!(report["forgotten"], json!(["a", "b"]));

    let class = |result: anyhow::Result<serde_json::Value>| match result
        .unwrap_err()
        .downcast_ref::<MemoryError>()
    {
        Some(MemoryError::Invalid(_)) => "invalid",
        Some(MemoryError::NotFound(_)) => "not-found",
        Some(MemoryError::Conflict(_)) => "conflict",
        Some(MemoryError::Concurrent(_)) => "concurrent",
        None => "internal",
    };
    assert_eq!(class(call("nope", json!({}))), "invalid");
    assert_eq!(
        class(call("recall", json!({"query": "x", "asOff": 1}))),
        "invalid"
    );
    assert_eq!(class(call("recall", json!({"k": 3}))), "invalid");
    assert_eq!(
        class(call("recall", json!({"query": "x", "k": 0}))),
        "invalid"
    );
    assert_eq!(class(call("history", json!({"id": "gone"}))), "not-found");
    assert_eq!(
        class(call("explain", json!({"query": "x", "id": "gone"}))),
        "not-found"
    );
    call("remember", json!({"id": "c", "text": "one"})).unwrap();
    assert_eq!(
        class(call("remember", json!({"id": "c", "text": "two"}))),
        "conflict"
    );
    assert_eq!(
        class(call("remember", json!({"facts": [], "text": "x"}))),
        "invalid"
    );
    assert_eq!(
        class(call("forget", json!({"scope": {"kind": "nothing"}}))),
        "invalid"
    );
}

#[test]
fn identical_inputs_give_byte_identical_results_in_separate_stores() {
    let run = || {
        let dir = tempfile::tempdir().unwrap();
        let clock = Arc::new(ManualClock::new(50_000));
        let config = MemoryConfig {
            supersession: SupersessionSpec::LatestBySubject,
            recency: RecencySpec::Exponential {
                half_life_ms: 5_000,
                basis: RecencyBasis::Event,
                floor: 0.1,
            },
            ..MemoryConfig::default()
        };
        let memory = Memory::with_config(open(dir.path()), config)
            .unwrap()
            .with_clock(clock.clone());
        let mut written = Vec::new();
        for i in 0..30 {
            clock.advance(7);
            let outcome = memory
                .remember(RememberInput {
                    text: format!("report {i} about service{} latency", i % 4),
                    subject: Some(format!("service/{}", i % 4)),
                    event_time: Some(40_000 + i * 300),
                    valid_from: Some(40_000 + i * 300),
                    entities: Some(vec![format!("svc{}", i % 4)]),
                    ..RememberInput::default()
                })
                .unwrap();
            written.push(outcome);
        }
        let mut out = vec![serde_json::to_value(&written).unwrap()];
        for query in ["service1 latency", "report 7", "latency"] {
            for history in [HistoryMode::Current, HistoryMode::All] {
                let got = memory
                    .recall(
                        query,
                        &RecallOptions {
                            k: Some(12),
                            history: Some(history),
                            explain: true,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                out.push(serde_json::to_value(&got).unwrap());
            }
        }
        serde_json::to_string(&out).unwrap()
    };
    let first = run();
    assert_eq!(first, run());
    // Auto ids are content hashes, so they are stable across stores and runs.
    assert!(first.contains("\"id\":\"mem:"));
}

#[cfg(feature = "security")]
#[test]
fn encrypted_stores_hold_facts_and_hide_text_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = HmsConfig::default();
    config.security.encryption_enabled = true;
    config.security.encryption_passphrase = Some("memory-api-test-passphrase".into());
    let open_encrypted = || {
        Arc::new(
            HmsCore::new(
                DIM,
                Some(dir.path().display().to_string()),
                Some(config.clone()),
            )
            .unwrap(),
        )
    };
    let clock = Arc::new(ManualClock::new(1));
    {
        let memory = Memory::new(open_encrypted()).with_clock(clock.clone());
        memory
            .remember(fact("e1", "confidential marmalade recipe"))
            .unwrap();
        let mut newer = fact("e2", "confidential marmalade recipe v2");
        newer.supersedes = Some(vec!["e1".into()]);
        memory.remember(newer).unwrap();
    }
    // `documents.text` is the engine's plaintext text cache, outside the encrypted journal
    // (an engine property that predates the memory API); every other file must be opaque.
    for entry in std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name() != "documents.text")
    {
        if let Ok(bytes) = std::fs::read(entry.path()) {
            assert!(
                !bytes.windows(9).any(|w| w == b"marmalade"),
                "{:?} holds plaintext",
                entry.path()
            );
        }
    }
    let memory = Memory::new(open_encrypted());
    assert_eq!(
        ids(&recall(&memory, "marmalade", RecallOptions::default())),
        ["e2"]
    );
    assert_eq!(memory.history("e1").unwrap().len(), 2);
}
