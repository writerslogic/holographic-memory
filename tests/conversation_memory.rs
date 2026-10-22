// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "conversation-memory")]

use std::path::Path;
use std::sync::{Arc, Barrier};

use holographic_memory::core::memory_api::{
    ConversationEvidence, ConversationInput, ConversationMemory, ConversationRecallOptions,
    ConversationRole, ConversationTurn,
};
use holographic_memory::HmsCore;

fn open(path: &Path) -> ConversationMemory {
    ConversationMemory::new(HmsCore::new(4096, Some(path.display().to_string()), None).unwrap())
        .unwrap()
}

fn turn(id: &str, role: ConversationRole, text: &str) -> ConversationTurn {
    ConversationTurn {
        source_id: id.into(),
        role,
        text: text.into(),
        observation_time: 1_704_067_200_000,
    }
}

fn session(id: &str, version: &str, turns: Vec<ConversationTurn>) -> ConversationInput {
    ConversationInput {
        session_id: id.into(),
        version: version.into(),
        turns,
    }
}

fn options(roles: Vec<ConversationRole>) -> ConversationRecallOptions {
    ConversationRecallOptions {
        roles,
        k: 1,
        candidate_limit: 32,
        max_sources: 16,
        max_bytes: 128 * 1024,
        max_rounds: None,
        ..ConversationRecallOptions::default()
    }
}

fn texts(packet: &ConversationEvidence) -> Vec<&str> {
    packet
        .sources
        .iter()
        .map(|source| source.text.as_str())
        .collect()
}

fn source_ids(packet: &ConversationEvidence) -> Vec<&str> {
    packet
        .sources
        .iter()
        .map(|source| source.source_id.as_str())
        .collect()
}

#[test]
fn assistant_only_named_fact_is_searchable_without_becoming_a_user_assertion() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    let original = "  Suggested option: quartzneedle calibration at 17 hertz; this is a recommendation, not an adoption.\n";
    memory
        .remember(session(
            "assistant-only-session",
            "v1",
            vec![turn(
                "assistant-report",
                ConversationRole::Assistant,
                original,
            )],
        ))
        .unwrap();
    memory
        .remember(session(
            "other-session",
            "v1",
            vec![turn(
                "user-request",
                ConversationRole::User,
                "Please suggest an option for the instrument.",
            )],
        ))
        .unwrap();
    let packet = memory
        .recall(
            "quartzneedle calibration",
            &options(vec![ConversationRole::Assistant]),
        )
        .unwrap();
    assert_eq!(packet.sources.len(), 1);
    assert_eq!(packet.anchors.len(), 1);
    assert_eq!(packet.anchors[0].source_id, "assistant-report");
    let source = &packet.sources[0];
    assert_eq!(source.role, ConversationRole::Assistant);
    assert_eq!(source.text, original);
    assert_eq!(source.session_id, "assistant-only-session");
    assert_eq!(source.version, "v1");
    assert_eq!(source.turn_index, 0);
    assert!(source.premise_record_ids.is_empty());
}

#[test]
fn followup_closes_over_opening_and_nearest_user_premises_in_its_own_session_version() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    for (session_id, version, marker) in [
        ("thread", "v0", "WRONG_VERSION"),
        ("different-thread", "v1", "WRONG_SESSION"),
    ] {
        memory
            .remember(session(
                session_id,
                version,
                vec![turn("opening", ConversationRole::User, marker)],
            ))
            .unwrap();
    }
    let input = session(
        "thread",
        "v1",
        vec![
            turn(
                "opening",
                ConversationRole::User,
                "The two destinations are the harbor museum and ridge observatory.",
            ),
            turn(
                "first-reply",
                ConversationRole::Assistant,
                "The destinations have different schedules.",
            ),
            turn(
                "second-user",
                ConversationRole::User,
                "The second destination is the one I selected.",
            ),
            turn(
                "second-reply",
                ConversationRole::Assistant,
                "The selection is understood.",
            ),
            turn(
                "nearest-user",
                ConversationRole::User,
                "Its appointment is the late evening slot.",
            ),
            turn(
                "ordinary-reply",
                ConversationRole::Assistant,
                "The appointment has been noted.",
            ),
            turn(
                "followup",
                ConversationRole::Assistant,
                "The heliotropevoucher applies to that appointment.",
            ),
        ],
    );
    memory.remember(input).unwrap();
    let packet = memory
        .recall(
            "heliotropevoucher",
            &options(vec![ConversationRole::Assistant]),
        )
        .unwrap();
    assert_eq!(packet.anchors.len(), 1);
    assert_eq!(packet.anchors[0].source_id, "followup");
    let anchor = packet
        .sources
        .iter()
        .find(|source| source.source_id == "followup")
        .unwrap();
    let mut premises: Vec<_> = anchor
        .premise_record_ids
        .iter()
        .map(|id| {
            packet
                .sources
                .iter()
                .find(|source| &source.record_id == id)
                .unwrap()
        })
        .collect();
    premises.sort_by_key(|source| source.turn_index);
    assert_eq!(
        premises
            .iter()
            .map(|source| source.source_id.as_str())
            .collect::<Vec<_>>(),
        ["opening", "second-user", "second-reply", "nearest-user"]
    );
    assert!(packet
        .sources
        .iter()
        .all(|source| source.session_id == "thread" && source.version == "v1"));
    assert!(!source_ids(&packet).contains(&"first-reply"));
    assert!(!texts(&packet).contains(&"WRONG_VERSION"));
    assert!(!texts(&packet).contains(&"WRONG_SESSION"));
}

#[test]
fn a_user_confirmation_and_its_following_answer_retain_the_original_assistant_referent() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    memory
        .remember(session(
            "confirmation-thread",
            "old-version",
            vec![
                turn(
                    "question",
                    ConversationRole::User,
                    "Please select a trail shelter.",
                ),
                turn(
                    "option",
                    ConversationRole::Assistant,
                    "The old suggestion was a different shelter.",
                ),
            ],
        ))
        .unwrap();
    let referent = "The selected option is the Fenwick Glade shelter.";
    let confirmation = "Yes, that one. Track my saffronlatch confirmation.";
    let followup = "Your nightorchid booking note uses that confirmed option.";
    memory
        .remember(session(
            "confirmation-thread",
            "current-version",
            vec![
                turn(
                    "question",
                    ConversationRole::User,
                    "Please select a trail shelter for the park.",
                ),
                turn("option", ConversationRole::Assistant, referent),
                turn("confirmation", ConversationRole::User, confirmation),
                turn("following-answer", ConversationRole::Assistant, followup),
            ],
        ))
        .unwrap();
    for (query, role, anchor_id, expected_count) in [
        ("saffronlatch", ConversationRole::User, "confirmation", 3),
        (
            "nightorchid",
            ConversationRole::Assistant,
            "following-answer",
            4,
        ),
    ] {
        let mut opts = options(vec![role]);
        let packet = memory.recall(query, &opts).unwrap();
        assert_eq!(packet.anchors.len(), 1);
        assert_eq!(packet.anchors[0].source_id, anchor_id);
        assert_eq!(packet.sources.len(), expected_count);
        assert!(packet.sources.iter().all(|source| {
            source.session_id == "confirmation-thread" && source.version == "current-version"
        }));
        let original = packet
            .sources
            .iter()
            .find(|source| source.source_id == "option")
            .unwrap();
        assert_eq!(original.role, ConversationRole::Assistant);
        assert_eq!(original.text, referent);
        let user = packet
            .sources
            .iter()
            .find(|source| source.source_id == "confirmation")
            .unwrap();
        assert_eq!(user.role, ConversationRole::User);
        assert_eq!(user.text, confirmation);
        assert_eq!(original.turn_index + 1, user.turn_index);
        assert!(user.premise_record_ids.contains(&original.record_id));
        let anchor = packet
            .sources
            .iter()
            .find(|source| source.source_id == anchor_id)
            .unwrap();
        assert!(anchor.premise_record_ids.contains(&original.record_id));
        if role == ConversationRole::Assistant {
            assert!(anchor.premise_record_ids.contains(&user.record_id));
        }
        opts.max_bytes = packet.serialized_bytes - 1;
        let capped = memory.recall(query, &opts).unwrap();
        assert!(capped.sources.is_empty());
        assert!(capped
            .dropped
            .iter()
            .any(|drop| drop.source_id == anchor_id && drop.reason == "max_bytes"));
    }
}

#[cfg(feature = "local-models")]
#[test]
fn default_model_stages_accept_one_native_candidate_and_one_retrieved_source() {
    use holographic_memory::core::models::ModelStages;

    let dir = tempfile::tempdir().unwrap();
    let core = HmsCore::new(4096, Some(dir.path().display().to_string()), None).unwrap();
    core.set_model_stages(Some(Arc::new(ModelStages::default())))
        .unwrap();
    let memory = ConversationMemory::new(core).unwrap();
    let original = "The topazcandidate report retains its full original content.";
    memory
        .remember(session(
            "one-candidate",
            "v1",
            vec![turn("reported", ConversationRole::Assistant, original)],
        ))
        .unwrap();
    let opts = ConversationRecallOptions {
        k: 1,
        candidate_limit: 1,
        ..options(vec![ConversationRole::Assistant])
    };
    let packet = memory.recall("topazcandidate", &opts).unwrap();
    assert_eq!(packet.anchors.len(), 1);
    assert_eq!(source_ids(&packet), ["reported"]);
    assert_eq!(packet.sources[0].text, original);
}

#[test]
fn byte_budget_admits_complete_original_sources_at_the_exact_serialized_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    let original = "\n  prismneedle says \"雪\" and retains every qualifier.\t";
    memory
        .remember(session(
            "byte-session",
            "v1",
            vec![turn("byte-source", ConversationRole::Assistant, original)],
        ))
        .unwrap();
    let mut opts = options(vec![ConversationRole::Assistant]);
    let complete = memory.recall("prismneedle", &opts).unwrap();
    let boundary = serde_json::to_vec(&complete.sources).unwrap().len();
    assert_eq!(complete.serialized_bytes, boundary);
    for max_bytes in [boundary - 1, boundary, boundary + 1] {
        opts.max_bytes = max_bytes;
        let packet = memory.recall("prismneedle", &opts).unwrap();
        assert_eq!(
            packet.serialized_bytes,
            serde_json::to_vec(&packet.sources).unwrap().len()
        );
        assert!(packet.serialized_bytes <= max_bytes);
        if max_bytes < boundary {
            assert!(packet.sources.is_empty());
            assert!(packet
                .dropped
                .iter()
                .any(|drop| drop.source_id == "byte-source" && drop.reason == "max_bytes"));
        } else {
            assert_eq!(packet.sources, complete.sources);
            assert_eq!(packet.sources[0].text, original);
        }
    }
}

#[test]
fn source_count_budget_drops_a_followup_instead_of_detaching_its_premises() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    memory
        .remember(session(
            "closure-budget",
            "v1",
            vec![
                turn(
                    "user-a",
                    ConversationRole::User,
                    "I selected the north location.",
                ),
                turn(
                    "user-b",
                    ConversationRole::User,
                    "The reservation is for the evening.",
                ),
                turn(
                    "answer",
                    ConversationRole::Assistant,
                    "The silverneedle reservation uses those details.",
                ),
            ],
        ))
        .unwrap();
    let mut opts = options(vec![ConversationRole::Assistant]);
    for max_sources in [2, 3, 4] {
        opts.max_sources = max_sources;
        let packet = memory.recall("silverneedle", &opts).unwrap();
        if max_sources < 3 {
            assert!(packet.sources.is_empty());
            assert!(packet
                .dropped
                .iter()
                .any(|drop| drop.source_id == "answer" && drop.reason == "max_sources"));
        } else {
            assert_eq!(source_ids(&packet), ["user-a", "user-b", "answer"]);
        }
    }
}

#[test]
fn immutable_retry_conflict_and_restart_preserve_one_complete_observation() {
    let dir = tempfile::tempdir().unwrap();
    let input = session(
        "persistent-thread",
        "revision-one",
        vec![
            turn(
                "user",
                ConversationRole::User,
                "I asked about the instrument.",
            ),
            turn(
                "assistant",
                ConversationRole::Assistant,
                "\n  lodestoneneedle \"α\" is the complete reported value.  \n",
            ),
        ],
    );
    let expected;
    {
        let memory = open(dir.path());
        assert!(memory.remember(input.clone()).unwrap().created);
        assert!(!memory.remember(input.clone()).unwrap().created);
        expected = memory
            .recall(
                "lodestoneneedle",
                &options(vec![ConversationRole::Assistant]),
            )
            .unwrap()
            .sources;
        let mut changed = input.clone();
        changed.turns[1].text = "lodestoneneedle MUTATED_VALUE".into();
        assert!(memory.remember(changed).is_err());
        let after = memory
            .recall(
                "lodestoneneedle",
                &options(vec![ConversationRole::Assistant]),
            )
            .unwrap();
        assert_eq!(after.sources, expected);
    }
    let reopened = open(dir.path());
    assert!(!reopened.remember(input).unwrap().created);
    assert_eq!(
        reopened
            .recall(
                "lodestoneneedle",
                &options(vec![ConversationRole::Assistant])
            )
            .unwrap()
            .sources,
        expected
    );
}

#[test]
fn future_observation_time_does_not_make_reported_memory_not_yet_valid() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    let mut future = turn(
        "future-observation",
        ConversationRole::Assistant,
        "The futurepinnacle report describes an earlier completed survey.",
    );
    future.observation_time = 4_102_444_800_000;
    memory
        .remember(session("future-report", "v1", vec![future.clone()]))
        .unwrap();
    let packet = memory
        .recall(
            "futurepinnacle",
            &options(vec![ConversationRole::Assistant]),
        )
        .unwrap();
    assert_eq!(packet.sources.len(), 1);
    assert_eq!(packet.sources[0].observation_time, future.observation_time);
    assert_eq!(packet.sources[0].text, future.text);
}

#[test]
fn opaque_id_renaming_preserves_unambiguous_retrieval_and_unrelated_growth_does_not_displace_it() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let memories = [open(first.path()), open(second.path())];
    let bodies = [
        "The cobaltneedle midnight calibration has a recorded exact setting.",
        "The cobaltneedle morning calibration has a recorded different setting.",
    ];
    for (index, memory) in memories.iter().enumerate() {
        for (n, body) in bodies.iter().enumerate() {
            memory
                .remember(session(
                    &format!("opaque-session-{index}-{n}"),
                    "v1",
                    vec![turn(
                        &format!("opaque-source-{index}-{n}"),
                        ConversationRole::Assistant,
                        body,
                    )],
                ))
                .unwrap();
        }
    }
    let opts = ConversationRecallOptions {
        k: 2,
        ..options(vec![ConversationRole::Assistant])
    };
    let before = memories[0].recall("cobaltneedle midnight", &opts).unwrap();
    let renamed = memories[1].recall("cobaltneedle midnight", &opts).unwrap();
    let ranked_texts = |packet: &ConversationEvidence| {
        packet
            .anchors
            .iter()
            .map(|anchor| {
                packet
                    .sources
                    .iter()
                    .find(|source| source.record_id == anchor.record_id)
                    .unwrap()
                    .text
                    .clone()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(ranked_texts(&before)[0], bodies[0]);
    assert_eq!(ranked_texts(&before), ranked_texts(&renamed));
    for n in 0..64 {
        memories[0]
            .remember(session(
                &format!("unrelated-session-{n}"),
                "v1",
                vec![turn(
                    "unrelated-source",
                    ConversationRole::User,
                    "A garden log describes beans, rain and soil.",
                )],
            ))
            .unwrap();
    }
    let after = memories[0].recall("cobaltneedle midnight", &opts).unwrap();
    assert_eq!(ranked_texts(&after), ranked_texts(&before));
}

#[test]
fn a_keyword_beyond_the_first_document_chunk_hydrates_the_exact_entire_turn() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    let original = format!(
        "\n  {} terminalazureneedle is after the long prefix.\t\n",
        "ordinary ".repeat(4_200)
    );
    let remembered = memory
        .remember(session(
            "long-session",
            "v1",
            vec![turn("long-source", ConversationRole::Assistant, &original)],
        ))
        .unwrap();
    assert!(remembered.chunk_count > 1);
    let packet = memory
        .recall(
            "terminalazureneedle",
            &options(vec![ConversationRole::Assistant]),
        )
        .unwrap();
    assert_eq!(packet.sources.len(), 1);
    assert_eq!(packet.sources[0].text, original);
    assert!(packet.sources[0].text.starts_with("\n  "));
    assert!(packet.sources[0].text.ends_with("\t\n"));
    assert_eq!(
        packet.serialized_bytes,
        serde_json::to_vec(&packet.sources).unwrap().len()
    );
}

#[test]
fn malformed_scope_source_and_allocation_inputs_never_partially_commit() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    let base = session(
        "atomic-session",
        "v1",
        vec![
            turn(
                "first",
                ConversationRole::User,
                "atomicneedle is the first original source.",
            ),
            turn(
                "second",
                ConversationRole::Assistant,
                "The second source completes the original session.",
            ),
        ],
    );
    let mut invalid = Vec::new();
    let mut bad = base.clone();
    bad.session_id.clear();
    invalid.push(bad);
    let mut bad = base.clone();
    bad.version.clear();
    invalid.push(bad);
    let mut bad = base.clone();
    bad.session_id.push('\0');
    invalid.push(bad);
    let mut bad = base.clone();
    bad.version.push('\n');
    invalid.push(bad);
    let mut bad = base.clone();
    bad.turns[1].source_id.push('\u{1f}');
    invalid.push(bad);
    let mut bad = base.clone();
    bad.turns[1].source_id = bad.turns[0].source_id.clone();
    invalid.push(bad);
    let mut bad = base.clone();
    bad.turns[1].text = "x".repeat(61 * 1024);
    invalid.push(bad);
    let mut bad = base.clone();
    bad.turns.clear();
    invalid.push(bad);
    for input in invalid {
        assert!(memory.remember(input).is_err());
        assert!(memory
            .recall("atomicneedle", &options(vec![ConversationRole::User]))
            .unwrap()
            .sources
            .is_empty());
    }
    let mut malformed = serde_json::to_value(&base).unwrap();
    malformed["turns"][1]["role"] = serde_json::json!("system");
    assert!(serde_json::from_value::<ConversationInput>(malformed).is_err());
    let mut malformed = serde_json::to_value(&base).unwrap();
    malformed["version"] = serde_json::json!(false);
    assert!(serde_json::from_value::<ConversationInput>(malformed).is_err());
    memory.remember(base).unwrap();
    assert_eq!(
        source_ids(
            &memory
                .recall("atomicneedle", &options(vec![ConversationRole::User]))
                .unwrap()
        ),
        ["first"]
    );
}

#[test]
fn invalid_query_role_and_capacity_options_leave_valid_recall_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    memory
        .remember(session(
            "limit-session",
            "v1",
            vec![turn(
                "source",
                ConversationRole::Assistant,
                "limitneedle has an archived answer.",
            )],
        ))
        .unwrap();
    let opts = options(vec![ConversationRole::Assistant]);
    let before = memory.recall("limitneedle", &opts).unwrap().sources;
    let mut invalid = Vec::new();
    let mut bad = opts.clone();
    bad.k = 0;
    invalid.push(bad);
    let mut bad = opts.clone();
    bad.k = 1001;
    bad.candidate_limit = 1001;
    invalid.push(bad);
    let mut bad = opts.clone();
    bad.candidate_limit = 0;
    invalid.push(bad);
    let mut bad = opts.clone();
    bad.candidate_limit = 1001;
    invalid.push(bad);
    let mut bad = opts.clone();
    bad.max_sources = 0;
    invalid.push(bad);
    let mut bad = opts.clone();
    bad.max_sources = 4097;
    invalid.push(bad);
    let mut bad = opts.clone();
    bad.max_bytes = 1;
    invalid.push(bad);
    let mut bad = opts.clone();
    bad.max_bytes = 2 * 1024 * 1024 + 1;
    invalid.push(bad);
    let mut bad = opts.clone();
    bad.roles.clear();
    invalid.push(bad);
    let mut bad = opts.clone();
    bad.roles.push(ConversationRole::Assistant);
    invalid.push(bad);
    for bad in invalid {
        assert!(memory.recall("limitneedle", &bad).is_err());
    }
    for query in ["", " \n\t"] {
        assert!(memory.recall(query, &opts).is_err());
    }
    assert!(memory.recall(&"x".repeat(8193), &opts).is_err());
    assert_eq!(memory.recall("limitneedle", &opts).unwrap().sources, before);
}

#[test]
fn concurrent_conflicting_writes_choose_one_complete_immutable_session() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Arc::new(open(dir.path()));
    let barrier = Arc::new(Barrier::new(2));
    let alternatives = [
        "concurrentneedle has the amber setting.",
        "concurrentneedle has the violet setting.",
    ];
    let threads: Vec<_> = alternatives
        .iter()
        .map(|body| {
            let memory = memory.clone();
            let barrier = barrier.clone();
            let input = session(
                "racing-session",
                "same-version",
                vec![
                    turn(
                        "opening",
                        ConversationRole::User,
                        "The calibration belongs to my field instrument.",
                    ),
                    turn("reported", ConversationRole::Assistant, body),
                ],
            );
            std::thread::spawn(move || {
                barrier.wait();
                memory.remember(input).is_ok()
            })
        })
        .collect();
    let outcomes: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    assert_eq!(outcomes.iter().filter(|success| **success).count(), 1);
    let winner = outcomes.iter().position(|success| *success).unwrap();
    let packet = memory
        .recall(
            "concurrentneedle",
            &options(vec![ConversationRole::Assistant]),
        )
        .unwrap();
    assert_eq!(source_ids(&packet), ["opening", "reported"]);
    let reported = packet
        .sources
        .iter()
        .find(|source| source.source_id == "reported")
        .unwrap();
    assert_eq!(reported.text, alternatives[winner]);
    assert!(!texts(&packet).contains(&alternatives[1 - winner]));
}

#[test]
fn round_budget_counts_complete_premise_closures_at_the_exact_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    memory
        .remember(session(
            "round-budget",
            "v1",
            vec![
                turn(
                    "opening",
                    ConversationRole::User,
                    "The location is the harbor.",
                ),
                turn(
                    "reply",
                    ConversationRole::Assistant,
                    "The location is recorded.",
                ),
                turn(
                    "second",
                    ConversationRole::User,
                    "I selected the evening slot.",
                ),
                turn(
                    "referent",
                    ConversationRole::Assistant,
                    "The booking uses that slot.",
                ),
                turn(
                    "anchor",
                    ConversationRole::User,
                    "roundbudgetneedle confirms it.",
                ),
            ],
        ))
        .unwrap();
    let mut opts = options(vec![ConversationRole::User]);
    let complete = memory.recall("roundbudgetneedle", &opts).unwrap();
    assert_eq!(complete.anchors[0].source_id, "anchor");
    assert_eq!(complete.sources.len(), 4);
    for limit in [2, 3, 4] {
        opts.max_rounds = Some(limit);
        let packet = memory.recall("roundbudgetneedle", &opts).unwrap();
        if limit < 3 {
            assert!(packet.sources.is_empty());
            assert_eq!(packet.dropped[0].reason, "max_rounds");
        } else {
            assert_eq!(packet.sources, complete.sources);
        }
    }
    for limit in [0, 4097] {
        opts.max_rounds = Some(limit);
        assert!(memory.recall("roundbudgetneedle", &opts).is_err());
    }
    for limit in [4095, 4096] {
        opts.max_rounds = Some(limit);
        assert!(memory.recall("roundbudgetneedle", &opts).is_ok());
    }
}

#[test]
fn assistant_weight_preserves_explicit_assistant_queries_and_role_requests() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    for (id, role, text) in [
        (
            "user",
            ConversationRole::User,
            "I asked about roleweightneedle.",
        ),
        (
            "assistant",
            ConversationRole::Assistant,
            "I recommended roleweightneedle.",
        ),
    ] {
        memory
            .remember(session(id, "v1", vec![turn(id, role, text)]))
            .unwrap();
    }
    let mut opts = options(vec![ConversationRole::User, ConversationRole::Assistant]);
    opts.assistant_weight = 0.0;
    let user = memory.recall("roleweightneedle", &opts).unwrap();
    assert!(user
        .sources
        .iter()
        .all(|source| source.role == ConversationRole::User));
    let reply = memory
        .recall("What did you recommend about roleweightneedle?", &opts)
        .unwrap();
    assert!(reply
        .sources
        .iter()
        .any(|source| source.role == ConversationRole::Assistant));
    opts.roles = vec![ConversationRole::Assistant];
    assert_eq!(
        memory.recall("roleweightneedle", &opts).unwrap().sources[0].role,
        ConversationRole::Assistant
    );
    for weight in [0.0, f64::EPSILON, 4.0 - f64::EPSILON * 4.0, 4.0] {
        opts.assistant_weight = weight;
        assert!(memory.recall("roleweightneedle", &opts).is_ok());
    }
    for weight in [-0.01, 4.01, f64::NAN, f64::INFINITY] {
        opts.assistant_weight = weight;
        assert!(memory.recall("roleweightneedle", &opts).is_err());
    }
}

#[test]
fn aggregation_mmr_selects_another_session_before_a_repeated_session() {
    let dir = tempfile::tempdir().unwrap();
    let memory = open(dir.path());
    memory
        .remember(session(
            "repeated",
            "v1",
            vec![
                turn("one", ConversationRole::User, "diversityneedle visit"),
                turn("two", ConversationRole::User, "diversityneedle visit"),
            ],
        ))
        .unwrap();
    memory
        .remember(session(
            "distinct",
            "v1",
            vec![turn(
                "three",
                ConversationRole::User,
                "diversityneedle visit plus additional details",
            )],
        ))
        .unwrap();
    let mut opts = options(vec![ConversationRole::User]);
    opts.k = 3;
    opts.session_diversity = 1.0;
    let diversified = memory
        .recall("How many diversityneedle visits?", &opts)
        .unwrap();
    assert_eq!(diversified.anchors.len(), 3);
    assert_ne!(
        diversified.anchors[0].session_id,
        diversified.anchors[1].session_id
    );
    let ordinary = memory.recall("diversityneedle visit", &opts).unwrap();
    opts.session_diversity = 0.0;
    assert_eq!(
        ordinary,
        memory.recall("diversityneedle visit", &opts).unwrap()
    );
    for strength in [0.0, f64::EPSILON, 1.0 - f64::EPSILON, 1.0] {
        opts.session_diversity = strength;
        assert!(memory.recall("diversityneedle", &opts).is_ok());
    }
    for strength in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
        opts.session_diversity = strength;
        assert!(memory.recall("diversityneedle", &opts).is_err());
    }
}
