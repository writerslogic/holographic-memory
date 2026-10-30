// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! JSON dispatch: one request/response contract that the HTTP server and the Node binding
//! both carry, so every surface exposes the same operations with the same shapes. Requests
//! and responses are the camelCase serde forms of the types in [`super::types`]. Unknown
//! request fields are rejected rather than ignored, so a misspelled option fails loudly.

use anyhow::Result;
use serde::de::DeserializeOwned;
use serde_json::{json, Map, Value};

use super::types::*;
use super::{Memory, MemoryError};

/// The operations [`Memory::call`] accepts.
pub const OPERATIONS: &[&str] = &[
    "capabilities",
    "remember",
    "recall",
    "explain",
    "forget",
    "get",
    "history",
    "applyForgetting",
    "consolidate",
];

fn parse<T: DeserializeOwned>(value: Value) -> Result<T> {
    serde_json::from_value(value).map_err(|error| {
        anyhow::Error::new(MemoryError::Invalid(format!("invalid request: {error}")))
    })
}

fn object(value: Value) -> Result<Map<String, Value>> {
    match value {
        Value::Object(map) => Ok(map),
        Value::Null => Ok(Map::new()),
        _ => Err(MemoryError::Invalid("the request must be a JSON object".into()).into()),
    }
}

fn take_string(map: &mut Map<String, Value>, key: &str) -> Result<String> {
    match map.remove(key) {
        Some(Value::String(value)) => Ok(value),
        _ => Err(MemoryError::Invalid(format!("{key} is required and must be a string")).into()),
    }
}

/// JavaScript has one number type, so a time arrives as `1735689600000.0`; integral floats
/// become integers so integer fields accept them.
fn integral_floats(value: &mut Value) {
    match value {
        Value::Number(number) if number.is_f64() => {
            if let Some(float) = number.as_f64() {
                if float.fract() == 0.0 && float.abs() < 9_007_199_254_740_992.0 {
                    *number = (float as i64).into();
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(integral_floats),
        Value::Object(map) => map.values_mut().for_each(integral_floats),
        _ => {}
    }
}

/// Maintenance operations over JSON are dry runs unless the request says `"dryRun": false`.
fn dry_run_by_default(request: Value) -> Result<Value> {
    let mut map = object(request)?;
    map.entry("dryRun").or_insert(Value::Bool(true));
    Ok(Value::Object(map))
}

fn to_value<T: serde::Serialize>(value: &T) -> Result<Value> {
    Ok(serde_json::to_value(value)?)
}

impl Memory {
    pub fn capabilities(&self) -> MemoryCapabilities {
        MemoryCapabilities {
            contract_version: 1,
            profile: "temporal-memory-v1".into(),
            operations: OPERATIONS.iter().map(|op| (*op).to_owned()).collect(),
            temporal_queries: true,
            exact_metadata_filters: true,
            subject_entity_filters: true,
            document_pagination: false,
            committed_error_code: "mutation_committed".into(),
        }
    }

    /// Runs one operation from its JSON request and returns its JSON response.
    ///
    /// | op | request | response |
    /// |---|---|---|
    /// | `remember` | a [`RememberInput`], or `{"facts": [..]}` | `{"remembered": [Remembered]}` |
    /// | `recall` | `{"query", ..RecallOptions}` | `{"results": [Recalled]}` |
    /// | `explain` | `{"query", "id", ..RecallOptions}` | `{"explanation": Explanation}` |
    /// | `forget` | `{"scope": ForgetScope, ..ForgetOptions}` | [`ForgetReport`] |
    /// | `get` | `{"id"}` | `{"fact": FactRecord or null}` |
    /// | `history` | `{"id"}` | `{"history": [FactRecord]}` |
    /// | `applyForgetting` | [`ForgetOptions`] | [`ForgetReport`] |
    /// | `consolidate` | [`ConsolidateOptions`] | [`ConsolidationReport`] |
    ///
    /// Errors that the caller can fix (bad fields, unknown ids, id conflicts) are
    /// [`MemoryError`]s; anything else is an internal failure.
    pub fn call(&self, op: &str, mut request: Value) -> Result<Value> {
        integral_floats(&mut request);
        match op {
            "capabilities" => {
                let request: CapabilitiesRequest = parse(request)?;
                check!(
                    request.contract_version.is_none_or(|version| version == 1),
                    "unsupported memory contractVersion"
                );
                Ok(json!({"capabilities": self.capabilities()}))
            }
            "remember" => {
                let mut map = object(request)?;
                let inputs: Vec<RememberInput> = match map.remove("facts") {
                    Some(facts) if map.is_empty() => parse(facts)?,
                    Some(_) => {
                        return Err(MemoryError::Invalid(
                            "send either a fact or {\"facts\": [..]}, not both".into(),
                        )
                        .into())
                    }
                    None => vec![parse(Value::Object(map))?],
                };
                Ok(json!({ "remembered": self.remember_many(inputs)? }))
            }
            "recall" => {
                let mut map = object(request)?;
                let query = take_string(&mut map, "query")?;
                let options: RecallOptions = parse(Value::Object(map))?;
                Ok(json!({ "results": self.recall(&query, &options)? }))
            }
            "explain" => {
                let mut map = object(request)?;
                let query = take_string(&mut map, "query")?;
                let id = take_string(&mut map, "id")?;
                let options: RecallOptions = parse(Value::Object(map))?;
                Ok(json!({ "explanation": self.explain(&query, &id, &options)? }))
            }
            "forget" => {
                let mut map = object(request)?;
                let scope: ForgetScope = parse(map.remove("scope").unwrap_or(Value::Null))?;
                let options: ForgetOptions = parse(Value::Object(map))?;
                to_value(&self.forget(&scope, &options)?)
            }
            "get" => {
                let id = take_string(&mut object(request)?, "id")?;
                Ok(json!({ "fact": self.get(&id)? }))
            }
            "history" => {
                let id = take_string(&mut object(request)?, "id")?;
                Ok(json!({ "history": self.history(&id)? }))
            }
            "applyForgetting" => {
                let options: ForgetOptions = parse(dry_run_by_default(request)?)?;
                to_value(&self.apply_forgetting(&options)?)
            }
            "consolidate" => {
                let options: ConsolidateOptions = parse(dry_run_by_default(request)?)?;
                to_value(&self.consolidate(&options)?)
            }
            other => Err(MemoryError::Invalid(format!(
                "unknown memory operation {other:?}; expected one of {OPERATIONS:?}"
            ))
            .into()),
        }
    }
}
