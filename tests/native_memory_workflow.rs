// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use holographic_memory::{
    core::memory_api::{Memory, MemoryError},
    HmsCore,
};
use serde_json::Value;
use std::sync::Arc;

#[test]
fn frozen_native_lifecycle_preserves_sources_and_forgetting_across_restarts() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/native_memory_lifecycle_v1.json")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let started = std::time::Instant::now();
    let mut checks = 0;
    for (phase, steps) in fixture["phases"].as_array().unwrap().iter().enumerate() {
        let core =
            Arc::new(HmsCore::new(4096, Some(dir.path().display().to_string()), None).unwrap());
        let memory = Memory::new(core.clone());
        for (step, case) in steps.as_array().unwrap().iter().enumerate() {
            let result = memory.call(case["op"].as_str().unwrap(), case["request"].clone());
            if let Some(expected) = case["error"].as_str() {
                let error = result.unwrap_err();
                assert_eq!(expected, "conflict");
                assert!(matches!(
                    error.downcast_ref::<MemoryError>(),
                    Some(MemoryError::Conflict(_))
                ));
            } else {
                let result =
                    result.unwrap_or_else(|error| panic!("phase {phase} step {step}: {error:#}"));
                for (pointer, expected) in case["expect"].as_object().unwrap() {
                    assert_eq!(
                        result.pointer(pointer),
                        Some(expected),
                        "phase {phase} step {step}: {pointer}: {result}"
                    );
                }
                if let Some(sources) = result["results"].as_array() {
                    for source in sources {
                        let id = source["sourceUri"]
                            .as_str()
                            .unwrap()
                            .rsplit('/')
                            .next()
                            .unwrap();
                        let original = fixture["transcript"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|row| row["id"] == id)
                            .unwrap();
                        assert_eq!(
                            source["text"], original["text"],
                            "retrieved quote differs from frozen source"
                        );
                    }
                }
            }
            checks += 1;
        }
        core.flush().unwrap();
    }
    println!("native lifecycle: {checks} cases, 3 opens, all source quotes and expected outcomes verified, elapsed_ms={}", started.elapsed().as_millis());
}
