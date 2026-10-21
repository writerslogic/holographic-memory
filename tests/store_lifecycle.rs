// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Store lifecycle over a real directory: ingest, compact while readers and mappings are
//! live, reopen, compact again, and delete the directory. Windows refuses to replace, remove,
//! or truncate a file that is still mapped or held open, so this is the surface where an
//! engine that leaves a handle behind fails with "Access is denied (os error 5)".

use holographic_memory::{DocumentInput, HmsCore, SearchOptions};

fn document(id: &str, round: usize) -> DocumentInput {
    let text = (0..400)
        .map(|word| format!("token{round}x{word}"))
        .collect::<Vec<_>>()
        .join(" ");
    serde_json::from_value(serde_json::json!({
        "id": id, "text": text, "sourceUri": format!("{id}.md"), "version": round.to_string(),
        "metadata": {"round": round}, "chunkWords": 40, "overlapWords": 4
    }))
    .unwrap()
}

#[test]
fn compact_reopen_and_delete_with_live_readers() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("store");
    let open = || HmsCore::new(4096, Some(path.display().to_string()), None);

    let hms = open()?;
    for round in 0..3 {
        hms.memorize_documents((0..4).map(|n| document(&format!("d{n}"), round)).collect())?;
        hms.memorize(format!("v{round}"), hms.encode_text("vector payload"))?;
    }
    let before = hms.search_documents("token2x7", &SearchOptions::default())?;
    assert!(!before.is_empty());
    hms.compact()?;
    hms.compact()?;
    let after = hms.search_documents("token2x7", &SearchOptions::default())?;
    assert_eq!(before[0].document_id, after[0].document_id);
    hms.memorize_documents(vec![document("d9", 5)])?;
    hms.maintain_indices()?;
    drop(hms);

    let hms = open()?;
    assert!(!hms
        .search_documents("token5x3", &SearchOptions::default())?
        .is_empty());
    assert!(hms.delete_document("d0")?);
    hms.compact()?;
    drop(hms);

    let hms = open()?;
    assert!(hms
        .search_documents("token2x7", &SearchOptions::default())?
        .iter()
        .all(|hit| hit.document_id != "d0"));
    drop(hms);
    std::fs::remove_dir_all(&path)?;
    assert!(!path.exists());
    Ok(())
}

/// One committed batch larger than the replay chunk (512 mutations) reopens to the same
/// documents and the same ranked results: chunked replay applies the frame in order.
#[test]
fn large_single_batch_reopens_identically() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("store");
    let open = || HmsCore::new(1024, Some(path.display().to_string()), None);
    let documents: Vec<DocumentInput> = (0..1_300)
        .map(|n| {
            serde_json::from_value(serde_json::json!({
                "id": format!("doc{n}"),
                "text": format!("shared topic{} item{n} word{} common", n % 37, n % 101),
            }))
            .unwrap()
        })
        .collect();
    let queries = ["topic5 common", "item1299", "word7 topic11", "shared"];
    let snapshot = |hms: &HmsCore| -> anyhow::Result<Vec<Vec<(String, u64)>>> {
        queries
            .iter()
            .map(|q| {
                let options = SearchOptions {
                    k: Some(50),
                    ..Default::default()
                };
                Ok(hms
                    .search_documents(q, &options)?
                    .into_iter()
                    .map(|r| (r.document_id, r.score.to_bits()))
                    .collect())
            })
            .collect()
    };
    let before = {
        let hms = open()?;
        hms.memorize_documents(documents)?;
        hms.flush()?;
        snapshot(&hms)?
    };
    let hms = open()?;
    assert_eq!(snapshot(&hms)?, before);
    assert!(before.iter().all(|hits| !hits.is_empty()));
    Ok(())
}
