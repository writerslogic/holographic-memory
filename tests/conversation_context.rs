// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "conversation-memory")]

use holographic_memory::core::memory_api::{
    ConversationContext, ConversationContextLimits, ConversationEvidence, ConversationInput,
    ConversationMemory, ConversationRecallOptions, ConversationRole, ConversationTurn,
    MAX_CONVERSATION_BYTES, MAX_CONVERSATION_METADATA_BYTES, MAX_CONVERSATION_RECORDS,
};
use holographic_memory::HmsCore;
use serde_json::{json, Value};

fn evidence() -> ConversationEvidence {
    let dir = tempfile::tempdir().unwrap();
    let memory = ConversationMemory::new(
        HmsCore::new(4096, Some(dir.path().display().to_string()), None).unwrap(),
    )
    .unwrap();
    for (session_id, version, suffix) in [("trip", "v1", "open"), ("trip", "v2", "cancelled")] {
        memory
            .remember(ConversationInput {
                session_id: session_id.into(),
                version: version.into(),
                turns: vec![
                    ConversationTurn {
                        source_id: "question".into(),
                        role: ConversationRole::User,
                        text: "  cobaltneedle: How many bookings for Zoë in 東京?\n".into(),
                        observation_time: 1_704_067_200_000,
                    },
                    ConversationTurn {
                        source_id: "answer".into(),
                        role: ConversationRole::Assistant,
                        text: format!("cobaltneedle: Seven; state {suffix}. 🦀\n"),
                        observation_time: 1_704_067_200_001,
                    },
                ],
            })
            .unwrap();
    }
    memory
        .recall("cobaltneedle", &ConversationRecallOptions::default())
        .unwrap()
}

fn decode(value: &Value) -> anyhow::Result<ConversationContext> {
    ConversationContext::from_json(&serde_json::to_vec(value)?)
}

#[test]
fn context_round_trip_preserves_roles_times_conflicts_and_exact_unicode_sources() {
    let packet = evidence();
    assert_eq!(packet.sources.len(), 4);
    let context = packet.pack_context(MAX_CONVERSATION_BYTES).unwrap();
    let bytes = serde_json::to_vec(&context).unwrap();
    let restored = ConversationContext::from_json(&bytes).unwrap();
    assert_eq!(restored.sources().unwrap(), packet.sources);
    assert!(bytes.len() < serde_json::to_vec(&packet.sources).unwrap().len());
    let mut reversed = packet.sources.clone();
    reversed.reverse();
    assert_eq!(
        ConversationContext::from_sources(&reversed, MAX_CONVERSATION_BYTES).unwrap(),
        context
    );
}

#[test]
fn context_byte_budget_is_atomic_at_the_exact_serialized_boundary() {
    let packet = evidence();
    let context = packet.pack_context(MAX_CONVERSATION_BYTES).unwrap();
    let length = serde_json::to_vec(&context).unwrap().len();
    for limit in [length - 1, length, length + 1] {
        assert_eq!(packet.pack_context(limit).is_ok(), limit >= length);
    }
    assert!(packet.pack_context(1).is_err());
    assert!(packet.pack_context(MAX_CONVERSATION_BYTES + 1).is_err());
    assert_eq!(
        packet.pack_context(length).unwrap().sources().unwrap(),
        packet.sources
    );
}

#[test]
fn altered_source_identity_text_role_or_time_fails_the_source_digest() {
    let value =
        serde_json::to_value(evidence().pack_context(MAX_CONVERSATION_BYTES).unwrap()).unwrap();
    for (field, replacement) in [
        (1, json!(3)),
        (2, json!("other-source")),
        (3, json!("assistant")),
        (4, json!(0)),
        (5, json!("rewritten text")),
    ] {
        let mut bad = value.clone();
        bad["turns"][0][field] = replacement;
        assert!(decode(&bad).is_err());
    }
    let mut bad = value.clone();
    bad["unexpected"] = json!(true);
    assert!(decode(&bad).is_err());
    let mut bad = value;
    bad["schemaVersion"] = json!(2);
    assert!(decode(&bad).is_err());
}

#[test]
fn missing_local_forward_duplicate_and_cross_version_premises_are_rejected() {
    let packet = evidence();
    let value = serde_json::to_value(packet.pack_context(MAX_CONVERSATION_BYTES).unwrap()).unwrap();
    for premises in [
        json!([99]),
        json!([1]),
        json!([0, 0]),
        json!([0, 1, 2, 3, 4]),
    ] {
        let mut bad = value.clone();
        bad["turns"][1][6] = premises;
        assert!(decode(&bad).is_err());
    }
    let mut bad = value;
    bad["turns"][3][6] = json!([0]);
    assert!(decode(&bad).is_err());
    let mut incomplete = packet.sources.clone();
    incomplete.remove(0);
    let missing = ConversationContext::from_sources(&incomplete, MAX_CONVERSATION_BYTES).unwrap();
    assert_eq!(missing.sources().unwrap(), incomplete);
    assert_eq!(
        missing.unresolved_premise_ids().unwrap(),
        vec![packet.sources[0].record_id.clone()]
    );
    let mut duplicate = packet.sources.clone();
    duplicate.push(packet.sources[0].clone());
    assert!(ConversationContext::from_sources(&duplicate, MAX_CONVERSATION_BYTES).is_err());
    let mut wrong_identity = packet.sources.clone();
    wrong_identity[0].record_id = "different".into();
    assert!(ConversationContext::from_sources(&wrong_identity, MAX_CONVERSATION_BYTES).is_err());
}

#[test]
fn python_golden_context_reconstructs_the_exact_canonical_rust_sources() {
    let context =
        ConversationContext::from_json(include_bytes!("fixtures/conversation_context_v1.json"))
            .unwrap();
    let sources = context.sources().unwrap();
    assert_eq!(sources[0].text, "  Café 東京 🦀\n");
    assert_eq!(sources[1].text, "Seven.\n");
    assert_eq!(sources[1].role, ConversationRole::Assistant);
    assert_eq!(
        sources[1].premise_record_ids,
        vec![sources[0].record_id.clone()]
    );
    assert_eq!(
        ConversationContext::from_sources(&sources, MAX_CONVERSATION_BYTES).unwrap(),
        context
    );
}

#[test]
fn source_and_frame_allocation_bounds_cover_limit_minus_one_limit_and_plus_one() {
    let packet = evidence();
    for size in [
        MAX_CONVERSATION_METADATA_BYTES - 1,
        MAX_CONVERSATION_METADATA_BYTES,
        MAX_CONVERSATION_METADATA_BYTES + 1,
    ] {
        let mut sources = packet.sources.clone();
        sources[0].text = "x".repeat(size);
        assert_eq!(
            ConversationContext::from_sources(&sources, MAX_CONVERSATION_BYTES).is_ok(),
            size <= MAX_CONVERSATION_METADATA_BYTES
        );
    }
    let context = packet.pack_context(MAX_CONVERSATION_BYTES).unwrap();
    let bytes = serde_json::to_vec(&context).unwrap();
    for size in [
        MAX_CONVERSATION_BYTES - 1,
        MAX_CONVERSATION_BYTES,
        MAX_CONVERSATION_BYTES + 1,
    ] {
        let mut frame = bytes.clone();
        frame.resize(size, b' ');
        assert_eq!(
            ConversationContext::from_json(&frame).is_ok(),
            size <= MAX_CONVERSATION_BYTES
        );
    }
    for count in [
        MAX_CONVERSATION_RECORDS - 1,
        MAX_CONVERSATION_RECORDS,
        MAX_CONVERSATION_RECORDS + 1,
    ] {
        let mut value = serde_json::to_value(&context).unwrap();
        value["turns"] = json!(vec![value["turns"][0].clone(); count]);
        let error = decode(&value).unwrap_err().to_string();
        assert_eq!(
            error == "invalid conversation context metadata",
            count > MAX_CONVERSATION_RECORDS
        );
    }
}

#[test]
fn no_evidence_round_trips_without_inventing_a_source_or_a_count() {
    let context = ConversationContext::from_sources(&[], MAX_CONVERSATION_BYTES).unwrap();
    let restored = ConversationContext::from_json(&serde_json::to_vec(&context).unwrap()).unwrap();
    assert!(restored.sources().unwrap().is_empty());
    let value = serde_json::to_value(restored).unwrap();
    assert_eq!(value["sessions"], json!([]));
    assert_eq!(value["turns"], json!([]));
    assert!(value.get("count").is_none());
    assert_eq!(value["sufficiency"], "unverified");
}

fn census_fixture() -> (
    tempfile::TempDir,
    ConversationMemory,
    ConversationEvidence,
    ConversationEvidence,
) {
    let dir = tempfile::tempdir().unwrap();
    let memory = ConversationMemory::new(
        HmsCore::new(4096, Some(dir.path().display().to_string()), None).unwrap(),
    )
    .unwrap();
    for (index, text) in [
        "cobaltneedle appointment booked",
        "cobaltneedle appointment cancelled",
        "cobaltneedle appointment requested",
    ]
    .iter()
    .enumerate()
    {
        memory
            .remember(ConversationInput {
                session_id: format!("session-{index}"),
                version: "v1".into(),
                turns: vec![ConversationTurn {
                    source_id: "observation".into(),
                    role: ConversationRole::User,
                    text: (*text).into(),
                    observation_time: 1_704_067_200_000 + index as i64,
                }],
            })
            .unwrap();
    }
    let options = ConversationRecallOptions {
        roles: vec![ConversationRole::User],
        k: 1,
        max_rounds: Some(1),
        ..Default::default()
    };
    let seed = memory
        .recall("How many cobaltneedle appointments?", &options)
        .unwrap();
    let broad = memory
        .recall(
            "How many cobaltneedle appointments?",
            &ConversationRecallOptions {
                k: 32,
                max_rounds: None,
                ..options
            },
        )
        .unwrap();
    assert_eq!(seed.sources.len(), 1);
    assert_eq!(broad.sources.len(), 3);
    (dir, memory, seed, broad)
}

#[test]
fn enumeration_keeps_seed_conflicts_and_source_identity_without_claiming_a_count() {
    let (_dir, memory, seed, mut broad) = census_fixture();
    broad.anchors.push(broad.anchors[0].clone());
    let limits = ConversationContextLimits {
        max_bytes: MAX_CONVERSATION_BYTES,
        max_rounds: 3,
        max_sources: 3,
    };
    let context = memory
        .expand_aggregation_context(
            "How many cobaltneedle appointments?",
            &seed,
            &broad,
            &limits,
        )
        .unwrap();
    assert_eq!(context.sources().unwrap(), broad.sources);
    assert_eq!(
        serde_json::to_value(&context).unwrap()["sufficiency"],
        "unverified"
    );
    let repeated = ConversationEvidence {
        sources: context.sources().unwrap(),
        ..seed.clone()
    };
    assert_eq!(
        memory
            .expand_aggregation_context(
                "How many cobaltneedle appointments?",
                &repeated,
                &broad,
                &limits
            )
            .unwrap(),
        context
    );
    assert_eq!(
        memory
            .expand_aggregation_context("Where is cobaltneedle?", &seed, &broad, &limits)
            .unwrap()
            .sources()
            .unwrap(),
        seed.sources
    );
}

#[test]
fn enumeration_admits_whole_sources_at_byte_round_and_source_boundaries() {
    let (_dir, memory, seed, broad) = census_fixture();
    let limits = ConversationContextLimits {
        max_bytes: MAX_CONVERSATION_BYTES,
        max_rounds: 3,
        max_sources: 3,
    };
    let full = memory
        .expand_aggregation_context(
            "How many cobaltneedle appointments?",
            &seed,
            &broad,
            &limits,
        )
        .unwrap();
    let bytes = serde_json::to_vec(&full).unwrap().len();
    for limit in [bytes - 1, bytes, bytes + 1] {
        let result = memory
            .expand_aggregation_context(
                "How many cobaltneedle appointments?",
                &seed,
                &broad,
                &ConversationContextLimits {
                    max_bytes: limit,
                    ..limits.clone()
                },
            )
            .unwrap();
        assert_eq!(result.sources().unwrap().len() == 3, limit >= bytes);
    }
    for limit in [1, 2, 3] {
        for source_budget in [true, false] {
            let bound = if source_budget {
                ConversationContextLimits {
                    max_sources: limit,
                    ..limits.clone()
                }
            } else {
                ConversationContextLimits {
                    max_rounds: limit,
                    ..limits.clone()
                }
            };
            let result = memory
                .expand_aggregation_context(
                    "How many cobaltneedle appointments?",
                    &seed,
                    &broad,
                    &bound,
                )
                .unwrap();
            assert_eq!(result.sources().unwrap().len(), limit);
            assert!(seed
                .sources
                .iter()
                .all(|source| result.sources().unwrap().contains(source)));
        }
    }
}

#[test]
fn enumeration_rejects_fabricated_sources_and_invalid_or_missing_anchors() {
    let (_dir, memory, seed, broad) = census_fixture();
    let limits = ConversationContextLimits::default();
    let mut bad = broad.clone();
    bad.sources[0].text = "fabricated evidence".into();
    assert!(memory
        .expand_aggregation_context("How many appointments?", &seed, &bad, &limits)
        .is_err());
    let mut bad = broad.clone();
    bad.anchors[0].source_id = "wrong".into();
    assert!(memory
        .expand_aggregation_context("How many appointments?", &seed, &bad, &limits)
        .is_err());
    let mut bad = broad.clone();
    bad.sources.clear();
    assert!(memory
        .expand_aggregation_context("How many appointments?", &seed, &bad, &limits)
        .is_err());
    for score in [f64::NAN, f64::INFINITY, -0.1] {
        let mut bad = broad.clone();
        bad.anchors[0].score = score;
        assert!(memory
            .expand_aggregation_context("How many appointments?", &seed, &bad, &limits)
            .is_err());
    }
    let mut zero = broad;
    for anchor in &mut zero.anchors {
        anchor.score = 0.0;
    }
    assert_eq!(
        memory
            .expand_aggregation_context("How many appointments?", &seed, &zero, &limits)
            .unwrap()
            .sources()
            .unwrap(),
        seed.sources
    );
}
