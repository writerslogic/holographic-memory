// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use serde_json::{json, Value};
use std::sync::Arc;
use writerslogic_warm::{
    core::memory_api::{ManualClock, Memory, MemoryError},
    WarmCore,
};

fn run_fixture(fixture: &Value, name: &str) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(ManualClock::new(1_000_000));
    let mut checks = 0;
    let mut quoted_sources = 0;
    let mut abstentions = 0;
    let mut failures = Vec::new();
    for (phase, steps) in fixture["phases"].as_array().unwrap().iter().enumerate() {
        let core =
            Arc::new(WarmCore::new(4096, Some(dir.path().display().to_string()), None).unwrap());
        let memory = Memory::new(core.clone()).with_clock(clock.clone());
        for (step, case) in steps.as_array().unwrap().iter().enumerate() {
            if let Some(time) = case["clock"].as_i64() {
                clock.set(time);
            }
            let result = memory.call(case["op"].as_str().unwrap(), case["request"].clone());
            let label = format!("{name} phase {phase} step {step}");
            checks += 1;
            if let Some(expected) = case["error"].as_str() {
                if !matches!(result, Err(ref error) if expected == "conflict" && matches!(error.downcast_ref::<MemoryError>(), Some(MemoryError::Conflict(_))))
                {
                    failures.push(format!("{label}: expected {expected}, got {result:?}"));
                }
                continue;
            }
            let result = match result {
                Ok(result) => result,
                Err(error) => {
                    failures.push(format!("{label}: {error:#}"));
                    continue;
                }
            };
            for (pointer, expected) in case["expect"].as_object().unwrap() {
                if result.pointer(pointer) != Some(expected) {
                    failures.push(format!(
                        "{label}: {pointer}: expected {expected}, got {result}"
                    ));
                }
            }
            if let Some(sources) = result["results"].as_array() {
                if sources.is_empty() && case["expect"]["/results"] == json!([]) {
                    abstentions += 1;
                }
                for source in sources {
                    let original = source["sourceUri"]
                        .as_str()
                        .and_then(|uri| uri.strip_prefix("fixture:transcript/"))
                        .and_then(|id| {
                            fixture["transcript"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .find(|row| row["id"] == id)
                        });
                    if original.is_none_or(|original| source["text"] != original["text"]) {
                        failures.push(format!(
                            "{label}: quote or source boundary differs: {source}"
                        ));
                    } else {
                        quoted_sources += 1;
                    }
                }
            }
        }
        core.flush().unwrap();
    }
    println!(
        "{}",
        json!({"fixture": name, "checks": checks, "quotedSources": quoted_sources,
        "expectedAbstentions": abstentions, "failures": failures, "learnedAnswerQuality": null})
    );
    assert!(
        failures.is_empty(),
        "{} native evidence failures",
        failures.len()
    );
    checks
}

#[test]
fn frozen_native_lifecycle_preserves_sources_and_forgetting_across_restarts() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/native_memory_lifecycle_v1.json")).unwrap();
    assert_eq!(run_fixture(&fixture, "lifecycle-v1"), 19);
}

#[test]
fn frozen_native_temporal_partitions_preserve_fixed_budgets_and_sources() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/native_memory_evidence_v1.json")).unwrap();
    for partition in ["development", "evaluation"] {
        for scenario in fixture["partitions"][partition].as_array().unwrap() {
            run_fixture(
                scenario,
                &format!("{partition}/{}", scenario["id"].as_str().unwrap()),
            );
        }
    }
}

#[test]
fn shared_memory_negotiation_and_validation_contract() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/memory_contract_cases_v1.json")).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let memory = Memory::new(Arc::new(
        WarmCore::new(1024, Some(directory.path().display().to_string()), None).unwrap(),
    ));
    for case in fixture["cases"].as_array().unwrap() {
        let result = memory.call(case["op"].as_str().unwrap(), case["request"].clone());
        if case["error"] == "invalid_request" {
            assert!(
                matches!(
                    result.unwrap_err().downcast_ref::<MemoryError>(),
                    Some(MemoryError::Invalid(_))
                ),
                "{case}"
            );
        } else {
            let result = result.unwrap();
            for (pointer, expected) in case["expect"].as_object().unwrap() {
                assert_eq!(result.pointer(pointer), Some(expected), "{case}");
            }
        }
    }
}
