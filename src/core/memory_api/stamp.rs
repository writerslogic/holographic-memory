// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The persisted form of a fact's memory fields: one reserved metadata key, `_mem`, on an
//! ordinary document. Nothing else about the store changes, so stores written before the
//! memory API open unchanged, documents without the key are plain facts, and encryption,
//! journaling and replay apply as for any document.
//!
//! Layout of `_mem` (version 1, all fields but `v` and `ingest` optional):
//! `{"v":1,"ingest":ms,"event":ms,"from":ms,"to":ms,"subject":s,"supersedes":[id],
//! "supersededBy":[id],"entities":[s],"hash":hex}`. Readers ignore unknown fields, so later
//! versions may add fields without breaking this one.

use serde_json::{json, Map, Value};

use super::types::Millis;

/// The reserved metadata key.
pub(crate) const KEY: &str = "_mem";
/// The layout version this build writes.
pub(crate) const VERSION: u64 = 1;
/// Prefix of a document `version` string that holds the compact stamp.
pub(crate) const VERSION_PREFIX: &str = "mem1:";
/// The engine's limit on a document version string.
const MAX_VERSION_BYTES: usize = 1024;

/// Where a stamp is persisted.
pub(crate) enum Placement {
    Version(String),
    Metadata(Value),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Stamp {
    pub ingest: Millis,
    pub event: Option<Millis>,
    pub valid_from: Option<Millis>,
    pub valid_to: Option<Millis>,
    pub subject: Option<String>,
    /// Facts this one supersedes.
    pub supersedes: Vec<String>,
    /// Facts known at ingest to supersede this one (an older fact arriving late).
    pub superseded_by: Vec<String>,
    pub entities: Vec<String>,
    /// Hash of the caller's payload, for recognizing an idempotent retry.
    pub hash: Option<String>,
}

impl Stamp {
    /// When the fact takes effect on the valid-time axis.
    pub(crate) fn effective_from(&self) -> Millis {
        self.valid_from.or(self.event).unwrap_or(self.ingest)
    }

    pub(crate) fn windowed(&self) -> bool {
        self.valid_from.is_some() || self.valid_to.is_some()
    }

    pub(crate) fn to_value(&self) -> Value {
        let mut map = Map::new();
        map.insert("v".into(), json!(VERSION));
        map.insert("ingest".into(), json!(self.ingest));
        for (key, value) in [
            ("event", self.event),
            ("from", self.valid_from),
            ("to", self.valid_to),
        ] {
            if let Some(value) = value {
                map.insert(key.into(), json!(value));
            }
        }
        if let Some(subject) = &self.subject {
            map.insert("subject".into(), json!(subject));
        }
        for (key, list) in [
            ("supersedes", &self.supersedes),
            ("supersededBy", &self.superseded_by),
            ("entities", &self.entities),
        ] {
            if !list.is_empty() {
                map.insert(key.into(), json!(list));
            }
        }
        if let Some(hash) = &self.hash {
            map.insert("hash".into(), json!(hash));
        }
        Value::Object(map)
    }

    /// The stamp of a document, or none for a plain document: the compact form in `version`
    /// when present, else the `_mem` metadata object.
    pub(crate) fn from_parts(version: &str, metadata: &Value) -> Option<Self> {
        match version.strip_prefix(VERSION_PREFIX) {
            Some(json) => Self::from_value(&serde_json::from_str(json).ok()?),
            None => Self::from_metadata(metadata),
        }
    }

    /// Where the stamp is stored: the document's version string when it fits (the usual
    /// case; no per-document map allocation), else the `_mem` metadata key.
    pub(crate) fn placement(&self) -> Placement {
        let compact = format!("{VERSION_PREFIX}{}", self.to_value());
        if compact.len() <= MAX_VERSION_BYTES {
            Placement::Version(compact)
        } else {
            Placement::Metadata(self.to_value())
        }
    }

    fn from_metadata(metadata: &Value) -> Option<Self> {
        Self::from_value(metadata.get(KEY)?)
    }

    fn from_value(value: &Value) -> Option<Self> {
        Self::from_object(value.as_object()?)
    }

    fn from_object(map: &serde_json::Map<String, Value>) -> Option<Self> {
        let time = |key: &str| map.get(key).and_then(Value::as_i64);
        let strings = |key: &str| -> Vec<String> {
            map.get(key)
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(str::to_owned))
                        .collect()
                })
                .unwrap_or_default()
        };
        Some(Self {
            ingest: time("ingest")?,
            event: time("event"),
            valid_from: time("from"),
            valid_to: time("to"),
            subject: map
                .get("subject")
                .and_then(Value::as_str)
                .map(str::to_owned),
            supersedes: strings("supersedes"),
            superseded_by: strings("supersededBy"),
            entities: strings("entities"),
            hash: map.get("hash").and_then(Value::as_str).map(str::to_owned),
        })
    }
}

/// The caller's metadata with the reserved key removed.
pub(crate) fn user_metadata(mut metadata: Value) -> Value {
    if let Some(map) = metadata.as_object_mut() {
        map.remove(KEY);
    }
    metadata
}

/// A stable 128-bit FNV-1a over length-prefixed parts, as lowercase hex. It identifies a
/// payload for retries and derives default ids; it is not a security boundary.
pub(crate) fn stable_hash(parts: &[&[u8]]) -> String {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut lanes = [0xcbf2_9ce4_8422_2325_u64, 0x8422_2325_cbf2_9ce4_u64];
    let mut feed = |byte: u8| {
        for (lane, salt) in lanes.iter_mut().zip([0u8, 0x5a]) {
            *lane = (*lane ^ u64::from(byte ^ salt)).wrapping_mul(PRIME);
        }
    };
    for part in parts {
        for byte in (part.len() as u64).to_le_bytes() {
            feed(byte);
        }
        for &byte in *part {
            feed(byte);
        }
    }
    format!("{:016x}{:016x}", lanes[0], lanes[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_round_trips_and_tolerates_unknown_fields() {
        let stamp = Stamp {
            ingest: 5,
            event: Some(1),
            valid_from: Some(2),
            valid_to: Some(9),
            subject: Some("s".into()),
            supersedes: vec!["a".into()],
            superseded_by: vec!["b".into()],
            entities: vec!["e".into()],
            hash: Some("h".into()),
        };
        let mut value = stamp.to_value();
        value["future"] = json!({"x": 1});
        let metadata = json!({ KEY: value });
        assert_eq!(Stamp::from_metadata(&metadata), Some(stamp));
        assert_eq!(Stamp::from_metadata(&json!({"other": 1})), None);
        assert_eq!(Stamp::from_metadata(&json!({ KEY: {"v": 1} })), None);
    }

    #[test]
    fn hash_is_stable_and_separates_part_boundaries() {
        assert_eq!(stable_hash(&[b"ab", b"c"]), stable_hash(&[b"ab", b"c"]),);
        assert_ne!(stable_hash(&[b"ab", b"c"]), stable_hash(&[b"a", b"bc"]));
    }
}
