// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! The operations every golden store is built from. Written only against API that exists in
//! every release since 0.6.1 so the same code produces the golden stores of the old tags
//! (`gen_v061.rs`) and of the current tree (`tests/format_golden.rs`).

use anyhow::Result;
use holographic_memory::{DocumentInput, EntangledHVec, HmsCore};

/// Parses a corpus file: a JSON array of `DocumentInput`.
pub fn load_corpus(path: &std::path::Path) -> Result<Vec<DocumentInput>> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

/// Memorizes every corpus document in order (a repeated id replaces the earlier version).
/// With `extras`, then writes three vectors and a relation and deletes a document and a
/// vector, so the journal holds every mutation kind a plain store produces.
pub fn populate(
    core: &HmsCore,
    corpus: Vec<DocumentInput>,
    dim: usize,
    extras: bool,
) -> Result<()> {
    for document in corpus {
        core.memorize_document(document)?;
    }
    if extras {
        core.memorize("vec-a".into(), EntangledHVec::new_deterministic(dim, 7))?;
        core.memorize("vec-b".into(), EntangledHVec::new_deterministic(dim, 8))?;
        core.memorize("vec-gone".into(), EntangledHVec::new_deterministic(dim, 9))?;
        core.add_relation(&serde_json::from_value(serde_json::json!({
            "sourceId": "doc-harbor", "relationType": "mentions", "targetId": "doc-ledger",
            "properties": null, "validFrom": 0, "validTo": 0,
        }))?)?;
        core.delete("vec-gone")?;
        core.delete_document("doc-obsolete")?;
    }
    core.flush()?;
    Ok(())
}
