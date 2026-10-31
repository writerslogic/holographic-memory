// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::io::Read;
use writerslogic_warm::{chunk_document, DocumentInput, SearchOptions, WarmCore};

const MAX_INPUT_BYTES: u64 = 8 * 1024 * 1024;

fn required<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value[field]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .with_context(|| format!("requires nonempty string {field}"))
}

fn document_id(namespace: &str, source: &str, id: &str) -> Result<String> {
    for (field, value) in [
        ("namespace", namespace),
        ("source_system", source),
        ("id", id),
    ] {
        ensure!(!value.trim().is_empty(), "{field} must not be empty");
        ensure!(
            !value.chars().any(char::is_control),
            "{field} must not contain control characters"
        );
    }
    let key = serde_json::to_string(&(namespace, source, id))?;
    ensure!(key.len() <= 1024, "namespaced identity exceeds 1024 bytes");
    Ok(key)
}

fn adapt(namespace: &str, value: Value) -> Result<DocumentInput> {
    let object = value.as_object().context("document must be an object")?;
    ensure!(
        !object.contains_key("hms_namespace"),
        "hms_namespace is reserved by the adapter"
    );
    let id = required(&value, "id")?;
    let source = required(&value, "source_system")?;
    let identifier = required(&value, "identifier")?;
    ensure!(identifier.len() <= 1024, "identifier exceeds 1024 bytes");
    let text = value["content"]
        .as_str()
        .context("content must be a string")?;
    let units = value["organizational_unit"]
        .as_array()
        .context("organizational_unit must be an array")?;
    ensure!(
        units.iter().all(Value::is_string),
        "organizational_unit entries must be strings"
    );
    let mut metadata = object.clone();
    metadata.remove("content");
    metadata.insert("hms_namespace".into(), json!(namespace));
    ensure!(
        serde_json::to_vec(&metadata)?.len() <= 64 * 1024,
        "metadata exceeds 64 KiB"
    );
    let source_uri = match value.get("source_url") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(Value::String(s)) => {
            ensure!(s.len() <= 8192, "source_url exceeds 8192 bytes");
            Some(s.clone())
        }
        _ => anyhow::bail!("source_url must be a string"),
    };
    let input = DocumentInput {
        id: document_id(namespace, source, id)?,
        text: text.to_owned(),
        source_uri,
        version: Some(identifier.to_owned()),
        metadata: Some(Value::Object(metadata)),
        chunk_words: Some(500),
        overlap_words: Some(50),
        store_text: Some(true),
        embeddings: None,
        title_weight: None,
    };
    chunk_document(&input)?;
    Ok(input)
}

fn read_document(path: &str) -> Result<Value> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= MAX_INPUT_BYTES, "input exceeds 8 MiB");
    serde_json::from_slice(&bytes).context("invalid JSON document")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "import STORE NAMESPACE FILE.json | search STORE NAMESPACE QUERY | forget STORE NAMESPACE SOURCE_SYSTEM SOURCE_ID [--apply]";
    ensure!(args.len() >= 4, "{usage}");
    let command = args[0].as_str();
    let namespace = &args[2];
    ensure!(!namespace.trim().is_empty(), "namespace must not be empty");
    match command {
        "import" if args.len() == 4 => {
            // Validate before opening a store: malformed input must not create one.
            let input = adapt(namespace, read_document(&args[3])?)?;
            let warm = WarmCore::new(4096, Some(args[1].clone()), None)?;
            let count = warm.memorize_document(input)?;
            warm.flush()?;
            println!("Imported {count} chunks");
        }
        "search" if args.len() == 4 => {
            let warm = WarmCore::new(4096, Some(args[1].clone()), None)?;
            let hits = warm.search_documents(
                &args[3],
                &SearchOptions {
                    filter: Some(json!({"hms_namespace": namespace})),
                    ..Default::default()
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&hits)?);
        }
        "forget" if args.len() == 5 || (args.len() == 6 && args[5] == "--apply") => {
            let id = document_id(namespace, &args[3], &args[4])?;
            if args.len() == 5 {
                println!("Dry run: would remove document {id}; pass --apply to execute");
            } else {
                let warm = WarmCore::new(4096, Some(args[1].clone()), None)?;
                let removed = warm.delete_document(&id)?;
                warm.flush()?;
                println!("Removed: {removed}");
            }
        }
        _ => anyhow::bail!("{usage}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(system: &str, text: &str, revision: &str) -> Value {
        json!({"id":"meeting-1", "source_system":system, "identifier":revision,
            "organizational_unit":["EGWG"], "source_url":"https://example.org/source",
            "content_type":"Transcript", "content":text, "title":"Meeting", "date":"2026-10-28",
            "artifact_id":"sha256:example", "raw_files":["meeting.vtt"],
            "collected_at":"2026-10-28T10:00:00Z"})
    }

    #[test]
    fn sources_versions_offsets_and_forgetting_survive_restart() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().display().to_string();
        let warm = WarmCore::new(4096, Some(path.clone()), None)?;
        let original = format!(
            " \nCafé: {}\nzebraquartz provenance decision.  ",
            "naïve ".repeat(500)
        );
        let mut expected_metadata = source("Zoom", &original, "revision-1");
        let zoom = adapt("toip", expected_metadata.clone())?;
        let id = zoom.id.clone();
        warm.memorize_document(zoom)?;
        warm.memorize_document(adapt(
            "toip",
            source("Confluence", "zebraquartz policy", "revision-1"),
        )?)?;
        warm.memorize_document(adapt(
            "other",
            source("Zoom", "zebraquartz private", "revision-1"),
        )?)?;
        let options = SearchOptions {
            filter: Some(json!({"hms_namespace":"toip"})),
            ..Default::default()
        };
        let hits = warm.search_documents("zebraquartz", &options)?;
        assert_eq!(hits.len(), 2);
        let hit = hits.iter().find(|h| h.document_id == id).unwrap();
        assert!(hit.start_byte > 0);
        assert_eq!(
            hit.text.as_deref(),
            Some(&original[hit.start_byte as usize..hit.end_byte as usize])
        );
        expected_metadata.as_object_mut().unwrap().remove("content");
        expected_metadata["hms_namespace"] = json!("toip");
        assert_eq!(hit.metadata, expected_metadata);
        assert_eq!(
            hit.source_uri.as_deref(),
            Some("https://example.org/source")
        );
        warm.memorize_document(adapt(
            "toip",
            source("Zoom", "amberquartz updated decision", "revision-2"),
        )?)?;
        assert_eq!(warm.search_documents("zebraquartz", &options)?.len(), 1);
        warm.flush()?;
        drop(warm);
        let warm = WarmCore::new(4096, Some(path.clone()), None)?;
        assert_eq!(
            warm.search_documents("amberquartz", &options)?[0].version,
            "revision-2"
        );
        assert!(warm.delete_document(&id)?);
        assert!(!warm.delete_document(&id)?);
        warm.flush()?;
        drop(warm);
        let warm = WarmCore::new(4096, Some(path), None)?;
        assert!(warm.search_documents("amberquartz", &options)?.is_empty());
        assert_eq!(warm.search_documents("zebraquartz", &options)?.len(), 1);
        Ok(())
    }

    #[test]
    fn invalid_identity_and_schema_fail_before_ingestion() {
        for field in [
            "id",
            "source_system",
            "identifier",
            "content",
            "organizational_unit",
        ] {
            let mut value = source("Zoom", "text", "v1");
            value.as_object_mut().unwrap().remove(field);
            assert!(adapt("toip", value).is_err(), "accepted missing {field}");
        }
        let mut value = source("Zoom", "text", "v1");
        value["organizational_unit"] = json!([12]);
        assert!(adapt("toip", value).is_err());
        assert!(adapt(" ", source("Zoom", "text", "v1")).is_err());
        for text in ["", " \n\t", &"x".repeat(64 * 1024 + 1)] {
            assert!(adapt("toip", source("Zoom", text, "v1")).is_err());
        }
        let mut value = source("Zoom", "text", "v1");
        value["hms_namespace"] = json!("upstream-value");
        assert!(adapt("toip", value).is_err());
        for parts in [
            ("toip", "", "id"),
            ("toip", "Zoom", " "),
            ("toip", "Zoom", "id\n"),
        ] {
            assert!(document_id(parts.0, parts.1, parts.2).is_err());
        }
        assert_ne!(
            document_id("a/b", "c", "d").unwrap(),
            document_id("a", "b/c", "d").unwrap()
        );
    }

    #[test]
    fn revision_limit_and_input_limit_are_enforced() -> Result<()> {
        for size in [1023, 1024, 1025] {
            assert_eq!(
                adapt("toip", source("Zoom", "text", &"v".repeat(size))).is_ok(),
                size <= 1024
            );
        }
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("input.json");
        for size in [MAX_INPUT_BYTES - 1, MAX_INPUT_BYTES, MAX_INPUT_BYTES + 1] {
            let mut bytes = vec![b' '; size as usize];
            bytes[..2].copy_from_slice(b"{}");
            std::fs::write(&path, bytes)?;
            assert_eq!(
                read_document(path.to_str().unwrap()).is_ok(),
                size <= MAX_INPUT_BYTES
            );
        }
        Ok(())
    }

    #[test]
    fn identity_metadata_and_uri_limits_are_enforced() -> Result<()> {
        let identity_overhead = document_id("toip", "Zoom", "x")?.len() - 1;
        for size in [1023, 1024, 1025] {
            assert_eq!(
                document_id("toip", "Zoom", &"x".repeat(size - identity_overhead)).is_ok(),
                size <= 1024
            );
        }
        let mut value = source("Zoom", "text", "v1");
        value["extra"] = json!("");
        let overhead =
            serde_json::to_vec(adapt("toip", value.clone())?.metadata.as_ref().unwrap())?.len();
        for size in [64 * 1024 - 1, 64 * 1024, 64 * 1024 + 1] {
            value["extra"] = json!("x".repeat(size - overhead));
            assert_eq!(adapt("toip", value.clone()).is_ok(), size <= 64 * 1024);
        }
        value.as_object_mut().unwrap().remove("extra");
        for size in [8191, 8192, 8193] {
            value["source_url"] = json!(format!("https://example.org/{}", "x".repeat(size - 20)));
            assert_eq!(adapt("toip", value.clone()).is_ok(), size <= 8192);
        }
        Ok(())
    }
}
