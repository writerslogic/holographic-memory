// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(feature = "conversation-memory", feature = "local-models"))]

use std::sync::Arc;

use anyhow::{Context, Result};
use holographic_memory::core::memory_api::{
    ConversationInput, ConversationMemory, ConversationRecallOptions, ConversationRole,
    ConversationTurn,
};
use holographic_memory::core::models::{
    default_device, Embedder, EmbeddingQueryTask, ModelSource, ModelStages, PINNED,
};
use holographic_memory::core::HmsConfig;
use holographic_memory::{EmbeddingSpace, HmsCore};

#[test]
fn conversation_sources_reject_user_only_fact_extraction() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let core = HmsCore::new(4096, Some(dir.path().display().to_string()), None)?;
    core.set_model_stages(Some(Arc::new(ModelStages {
        extract_facts: true,
        ..Default::default()
    })))?;
    assert!(ConversationMemory::new(core).is_err());
    Ok(())
}

#[test]
#[ignore = "requires pinned local weights in HMS_CONVERSATION_MODEL_DIR"]
fn conversation_embedding_retrieves_original_assistant_evidence() -> Result<()> {
    let model_dir = std::env::var_os("HMS_CONVERSATION_MODEL_DIR")
        .context("set HMS_CONVERSATION_MODEL_DIR to Qwen3-Embedding-0.6B")?;
    let (model, revision) = PINNED
        .iter()
        .find(|(name, _)| *name == "Qwen/Qwen3-Embedding-0.6B")
        .context("missing pinned embedding model")?;
    let mut embedder = Embedder::load(&ModelSource::new(model_dir, *revision), &default_device()?)?;
    embedder.set_query_task(EmbeddingQueryTask::Conversation);
    let config = HmsConfig {
        embedding_space: Some(EmbeddingSpace {
            model: (*model).into(),
            revision: (*revision).into(),
            dimensions: embedder.dimensions(),
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    };
    let dir = tempfile::tempdir()?;
    let core = HmsCore::new(4096, Some(dir.path().display().to_string()), Some(config))?;
    core.set_model_stages(Some(Arc::new(ModelStages {
        embedder: Some(embedder),
        ..Default::default()
    })))?;
    let memory = ConversationMemory::new(core)?;
    for (session_id, role, text) in [
        (
            "instrument",
            ConversationRole::User,
            "Please check the resonator.",
        ),
        (
            "calibration",
            ConversationRole::Assistant,
            "  The resonator calibration frequency is 37 hertz.\n",
        ),
        (
            "recipe",
            ConversationRole::Assistant,
            "Bake the vegetable tart at 190 degrees for 25 minutes.",
        ),
    ] {
        memory.remember(ConversationInput {
            session_id: session_id.into(),
            version: "v1".into(),
            turns: vec![ConversationTurn {
                source_id: "observation".into(),
                role,
                text: text.into(),
                observation_time: 1_704_067_200_000,
            }],
        })?;
    }
    let packet = memory.recall(
        "What frequency did the assistant report for the resonator?",
        &ConversationRecallOptions {
            roles: vec![ConversationRole::Assistant],
            k: 1,
            ..Default::default()
        },
    )?;
    assert_eq!(packet.anchors.len(), 1);
    assert_eq!(packet.sources.len(), 1);
    assert_eq!(packet.sources[0].session_id, "calibration");
    assert_eq!(packet.sources[0].role, ConversationRole::Assistant);
    assert_eq!(
        packet.sources[0].text,
        "  The resonator calibration frequency is 37 hertz.\n"
    );
    Ok(())
}
