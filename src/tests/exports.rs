// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "node-api")]
#[test]
fn export_ts_bindings() {
    use ts_rs::TS;

    let cfg = ts_rs::Config::default();
    crate::RetrievalResult::export_all(&cfg).unwrap();
    crate::ConceptCandidate::export_all(&cfg).unwrap();
    crate::TextMetrics::export_all(&cfg).unwrap();
    crate::MemorizeBatchItem::export_all(&cfg).unwrap();
    crate::QueryExplanation::export_all(&cfg).unwrap();
    crate::IndexStatus::export_all(&cfg).unwrap();
    crate::StorageHealth::export_all(&cfg).unwrap();
    crate::HardwareCapabilities::export_all(&cfg).unwrap();
    crate::HmsError::export_all(&cfg).unwrap();
}

#[cfg(feature = "server")]
#[test]
fn export_json_schemas() {
    use schemars::schema_for;
    let dir = std::path::Path::new("schemas");
    std::fs::create_dir_all(dir).unwrap();

    let schemas: Vec<(&str, schemars::Schema)> = vec![
        ("RetrievalResult", schema_for!(crate::RetrievalResult)),
        ("ConceptCandidate", schema_for!(crate::ConceptCandidate)),
        ("TextMetrics", schema_for!(crate::TextMetrics)),
        ("MemorizeBatchItem", schema_for!(crate::MemorizeBatchItem)),
        ("HmsError", schema_for!(crate::HmsError)),
        ("QueryExplanation", schema_for!(crate::QueryExplanation)),
        ("IndexStatus", schema_for!(crate::IndexStatus)),
        ("StorageHealth", schema_for!(crate::StorageHealth)),
        (
            "HardwareCapabilities",
            schema_for!(crate::HardwareCapabilities),
        ),
    ];

    for (name, schema) in schemas {
        let json = serde_json::to_string_pretty(&schema).unwrap();
        std::fs::write(dir.join(format!("{}.json", name)), json).unwrap();
    }
}
