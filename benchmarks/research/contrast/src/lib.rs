// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Kill test for "counterfactual structural contrast" on a 2,048-bit
//! document-local holographic sketch (research round 2, package `contrast`).
//!
//! # Question
//! Does a bipolar superposition of directional stem pairs, scored against a
//! query sketch minus order-breaking foils, earn a retrieval role next to BM25
//! and next to explicit ordered-pair features?
//!
//! # Modules
//! * [`hv`]: 2,048-bit bipolar primitives (bind = XNOR, permute = rotate,
//!   accumulate / sign, dot products, seeded atoms);
//! * [`sketch`]: document sketch (`sign` bits + int8 ablation), query sketch
//!   with foils, positive / mean-foil / max-foil / cleanup-energy scores, and
//!   the hashed ordered-pair baseline;
//! * [`bm25`]: the root tokenizer + Snowball stemmer copied verbatim, and a
//!   small BM25 index over stem ids;
//! * [`synth`]: the deterministic agent-memory corpus for Part B;
//! * [`eval`]: nDCG@k, ROC AUC, calibrated thresholds, timing;
//! * [`data`]: BEIR jsonl / qrels loading.
//!
//! The binary `hms-contrast` (src/main.rs) runs Part A (NFCorpus rerank) and
//! Part B (synthetic false-accept / abstention / latency) and writes JSON.

pub mod bm25;
pub mod eval;
pub mod hv;
pub mod sketch;
pub mod synth;

/// BEIR-style data loading (`corpus.jsonl`, `queries.jsonl`, `qrels.tsv`).
pub mod data {
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;

    use anyhow::{anyhow, Context, Result};
    use serde::Deserialize;

    /// One `{"id": ..., "text": ...}` row.
    #[derive(Debug, Deserialize)]
    pub struct Row {
        /// Document or query id.
        pub id: String,
        /// Text (NFCorpus documents carry the title as the first line).
        pub text: String,
    }

    /// Reads a jsonl file of [`Row`]s in file order.
    pub fn load_rows(path: &Path) -> Result<Vec<Row>> {
        let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut rows = Vec::new();
        for (n, line) in raw.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let row: Row = serde_json::from_str(line)
                .with_context(|| format!("{}:{}: malformed row", path.display(), n + 1))?;
            rows.push(row);
        }
        Ok(rows)
    }

    /// Reads `query_id \t doc_id \t score` lines (no header) into graded relevance maps.
    pub fn load_qrels(path: &Path) -> Result<HashMap<String, HashMap<String, f64>>> {
        let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut qrels: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for (n, line) in raw.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split('\t');
            let (Some(q), Some(d), Some(s)) = (parts.next(), parts.next(), parts.next()) else {
                if n == 0 {
                    continue;
                }
                return Err(anyhow!("{}:{}: expected three tab-separated fields", path.display(), n + 1));
            };
            let score: f64 = match s.trim().parse() {
                Ok(v) => v,
                Err(_) if n == 0 => continue,
                Err(e) => {
                    return Err(anyhow!("{}:{}: bad score {s:?}: {e}", path.display(), n + 1));
                }
            };
            qrels
                .entry(q.to_string())
                .or_default()
                .insert(d.to_string(), score);
        }
        Ok(qrels)
    }
}
