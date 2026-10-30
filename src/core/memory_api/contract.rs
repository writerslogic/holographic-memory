// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! JSON envelopes carried by `Memory::call`; payload schemas reuse the native API types.

pub use super::CapabilitiesRequest;
use schemars::JsonSchema;

use super::{
    Explanation, FactRecord, ForgetOptions, ForgetScope, RecallOptions, Recalled, RememberInput,
    Remembered,
};

#[derive(JsonSchema)]
#[serde(untagged)]
pub enum RememberRequest {
    Fact(Box<RememberInput>),
    Batch(RememberBatch),
}

#[derive(JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RememberBatch {
    #[schemars(length(min = 1))]
    pub facts: Vec<RememberInput>,
}

#[derive(JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecallRequest {
    pub query: String,
    #[serde(flatten)]
    pub options: RecallOptions,
}

#[derive(JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExplainRequest {
    pub query: String,
    pub id: String,
    #[serde(flatten)]
    pub options: RecallOptions,
}

#[derive(JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForgetRequest {
    pub scope: ForgetScope,
    #[serde(flatten)]
    pub options: ForgetOptions,
}

#[derive(JsonSchema)]
/// `get` and `history` retain their legacy acceptance of extra request fields.
pub struct IdRequest {
    pub id: String,
}

#[derive(JsonSchema)]
pub struct RememberReply {
    pub remembered: Vec<Remembered>,
}

#[derive(JsonSchema)]
pub struct RecallReply {
    pub results: Vec<Recalled>,
}

#[derive(JsonSchema)]
pub struct CapabilitiesReply {
    pub capabilities: super::MemoryCapabilities,
}

#[derive(JsonSchema)]
pub struct ExplainReply {
    pub explanation: Explanation,
}

#[derive(JsonSchema)]
pub struct GetReply {
    pub fact: FactOrNull,
}

#[derive(JsonSchema)]
#[serde(untagged)]
pub enum FactOrNull {
    Fact(Box<FactRecord>),
    Missing(()),
}

#[derive(JsonSchema)]
pub struct HistoryReply {
    pub history: Vec<FactRecord>,
}
