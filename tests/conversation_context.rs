// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "conversation-memory")]

use holographic_memory::core::memory_api::{
    ConversationContext, ConversationEvidence, ConversationInput, ConversationMemory,
    ConversationRecallOptions, ConversationRole, ConversationTurn, MAX_CONVERSATION_BYTES,
    MAX_CONVERSATION_METADATA_BYTES, MAX_CONVERSATION_RECORDS,
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
fn missing_forward_duplicate_and_cross_version_premises_are_rejected() {
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
    assert!(ConversationContext::from_sources(&incomplete, MAX_CONVERSATION_BYTES).is_err());
    let mut duplicate = packet.sources.clone();
    duplicate.push(packet.sources[0].clone());
    assert!(ConversationContext::from_sources(&duplicate, MAX_CONVERSATION_BYTES).is_err());
    let mut wrong_identity = packet.sources.clone();
    wrong_identity[0].record_id = "different".into();
    assert!(ConversationContext::from_sources(&wrong_identity, MAX_CONVERSATION_BYTES).is_err());
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
}
