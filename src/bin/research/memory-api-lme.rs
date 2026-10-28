// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Retrieval-level sanity check of the memory API on LongMemEval_S dev questions of the types
//! that exercise time: knowledge-update and temporal-reasoning. No model, no LLM, lexical
//! retrieval only. Only the ids listed as `dev` in the split file are read; a held-out id is
//! never touched, and the program refuses to run if the split lists an id in both sets.
//!
//! Each question gets a fresh store holding its own haystack, one fact per session (the
//! concatenation of its user turns, as the benchmark's session granularity does) with the
//! session date as the fact's event time and validity start. For each question the program
//! compares, by `recall_any` of the evidence sessions at k:
//!
//! - `document_search`: plain `search_documents`;
//! - `memory_default`: `Memory::recall` with defaults (must rank identically);
//! - `memory_as_of`: `Memory::recall` evaluated as of the question date;
//! - `memory_recency_<days>d`: exponential recency decay on event time with that half-life.
//!
//! It also checks the as-of invariant: no result is valid after the evaluation time, and
//! evaluating as of just before the newest evidence session removes it.

use anyhow::{ensure, Context, Result};
use clap::Parser;
use holographic_memory::core::memory_api::{
    HistoryMode, Memory, MemoryConfig, RecallOptions, RecencyBasis, RecencySpec, RememberInput,
};
use holographic_memory::{HmsCore, SearchOptions};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
struct Args {
    /// longmemeval_s_cleaned.json
    #[arg(long)]
    data: PathBuf,
    /// benchmarks/public/longmemeval_split.json
    #[arg(long)]
    split: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, value_delimiter = ',', default_values_t = [7.0f64, 30.0, 90.0, 365.0])]
    half_life_days: Vec<f64>,
}

#[derive(Deserialize)]
struct Split {
    dev: Vec<String>,
    heldout: Vec<String>,
}

#[derive(Deserialize)]
struct Turn {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct Question {
    question_id: String,
    question_type: String,
    question: String,
    question_date: String,
    haystack_session_ids: Vec<String>,
    haystack_dates: Vec<String>,
    haystack_sessions: Vec<Vec<Turn>>,
    answer_session_ids: Vec<String>,
}

/// "2023/05/20 (Sat) 02:21" as Unix milliseconds (UTC).
fn parse_date(text: &str) -> Result<i64> {
    let mut parts = text.split_whitespace();
    let date = parts.next().context("date")?;
    let time = parts.last().context("time")?;
    let ymd: Vec<i64> = date.split('/').map(str::parse).collect::<Result<_, _>>()?;
    let hm: Vec<i64> = time.split(':').map(str::parse).collect::<Result<_, _>>()?;
    ensure!(ymd.len() == 3 && hm.len() == 2, "unexpected date {text:?}");
    // Days from civil (Howard Hinnant).
    let (y, m, d) = (ymd[0] - i64::from(ymd[1] <= 2), ymd[1], ymd[2]);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Ok(((days * 24 + hm[0]) * 60 + hm[1]) * 60_000)
}

fn recall_any(ranked: &[String], evidence: &BTreeSet<&str>, k: usize) -> bool {
    ranked
        .iter()
        .take(k)
        .any(|id| evidence.contains(id.as_str()))
}

#[derive(Default)]
struct Tally {
    questions: usize,
    hits: BTreeMap<String, [usize; 2]>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let split: Split = serde_json::from_slice(&std::fs::read(&args.split)?)?;
    let dev: BTreeSet<&str> = split.dev.iter().map(String::as_str).collect();
    let heldout: BTreeSet<&str> = split.heldout.iter().map(String::as_str).collect();
    ensure!(dev.is_disjoint(&heldout), "split lists an id in both sets");
    let all: Vec<Question> = serde_json::from_slice(&std::fs::read(&args.data)?)?;
    let questions: Vec<&Question> = all
        .iter()
        .filter(|q| dev.contains(q.question_id.as_str()))
        .filter(|q| !q.question_id.ends_with("_abs"))
        .filter(|q| ["knowledge-update", "temporal-reasoning"].contains(&q.question_type.as_str()))
        .collect();
    ensure!(
        questions
            .iter()
            .all(|q| !heldout.contains(q.question_id.as_str())),
        "a held-out question was selected"
    );

    let mut configs: Vec<(String, Option<f64>)> = vec![
        ("document_search".into(), None),
        ("memory_default".into(), None),
        ("memory_as_of".into(), None),
    ];
    configs.extend(
        args.half_life_days
            .iter()
            .map(|d| (format!("memory_recency_{d}d"), Some(*d))),
    );
    let mut tallies: BTreeMap<String, Tally> = BTreeMap::new();
    let mut identical_default = 0usize;
    let mut as_of_invariant_ok = 0usize;
    let mut as_of_exclusion_ok = 0usize;
    let mut as_of_exclusion_checked = 0usize;

    for q in &questions {
        let dir = tempfile::tempdir()?;
        let core = Arc::new(HmsCore::new(
            1024,
            Some(dir.path().display().to_string()),
            None,
        )?);
        let question_ms = parse_date(&q.question_date)?;
        let mut facts = Vec::new();
        let mut seen = BTreeSet::new();
        for ((id, date), session) in q
            .haystack_session_ids
            .iter()
            .zip(&q.haystack_dates)
            .zip(&q.haystack_sessions)
        {
            let text: Vec<&str> = session
                .iter()
                .filter(|t| t.role == "user")
                .map(|t| t.content.as_str())
                .collect();
            // The haystack can repeat a session id; keep its first occurrence.
            if text.is_empty() || !seen.insert(id.as_str()) {
                continue;
            }
            let at = parse_date(date)?;
            facts.push(RememberInput {
                id: Some(id.clone()),
                text: text.join("\n"),
                event_time: Some(at),
                valid_from: Some(at),
                ..Default::default()
            });
        }
        let evidence: BTreeSet<&str> = q.answer_session_ids.iter().map(String::as_str).collect();
        let mut by_config: BTreeMap<String, Memory> = BTreeMap::new();
        // One store, several memories over it: each carries its own recency policy.
        for (name, half_life) in &configs {
            let config = MemoryConfig {
                recency: half_life.map_or(RecencySpec::None, |days| RecencySpec::Exponential {
                    half_life_ms: (days * 86_400_000.0) as i64,
                    basis: RecencyBasis::Event,
                    floor: 0.0,
                }),
                ..Default::default()
            };
            by_config.insert(name.clone(), Memory::with_config(core.clone(), config)?);
        }
        by_config["memory_default"].remember_many(facts.clone())?;
        let ranked = |name: &str, options: &RecallOptions| -> Result<Vec<String>> {
            let found = by_config[name].recall(&q.question, options)?;
            Ok(found.into_iter().map(|r| r.id).collect())
        };
        let options = RecallOptions {
            k: Some(10),
            as_of: Some(question_ms),
            ..Default::default()
        };
        let raw: Vec<String> = core
            .search_documents(
                &q.question,
                &SearchOptions {
                    k: Some(10),
                    ..Default::default()
                },
            )?
            .into_iter()
            .map(|d| d.document_id)
            .collect();
        let default = ranked(
            "memory_default",
            &RecallOptions {
                k: Some(10),
                ..Default::default()
            },
        )?;
        identical_default += usize::from(raw == default);
        let mut results: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        results.insert("document_search", raw);
        results.insert("memory_default", default);
        results.insert("memory_as_of", ranked("memory_as_of", &options)?);
        for (name, half_life) in &configs {
            if half_life.is_some() {
                let r = ranked(name, &options)?;
                results.insert(name.as_str(), r);
            }
        }
        for (name, ids) in &results {
            let tally = tallies
                .entry(format!("{}::{}", q.question_type, name))
                .or_default();
            tally.questions += 1;
            for (slot, k) in [5usize, 10].into_iter().enumerate() {
                let entry = tally.hits.entry((*name).to_owned()).or_default();
                entry[slot] += usize::from(recall_any(ids, &evidence, k));
            }
        }
        // As-of invariant: nothing returned began after the evaluation time, and evaluating
        // just before the newest evidence session's date removes it from a full-history recall.
        let all = by_config["memory_default"].recall(
            &q.question,
            &RecallOptions {
                k: Some(50),
                history: Some(HistoryMode::Current),
                as_of: Some(question_ms),
                ..Default::default()
            },
        )?;
        as_of_invariant_ok += usize::from(
            all.iter()
                .all(|r| r.valid_from.is_none_or(|t| t <= question_ms)),
        );
        let newest_evidence = facts
            .iter()
            .filter(|f| evidence.contains(f.id.as_deref().unwrap_or("")))
            .filter_map(|f| f.valid_from.zip(f.id.clone()))
            .max();
        if let Some((at, id)) = newest_evidence {
            as_of_exclusion_checked += 1;
            let before = by_config["memory_default"].recall(
                &q.question,
                &RecallOptions {
                    k: Some(1000),
                    as_of: Some(at - 1),
                    ..Default::default()
                },
            )?;
            as_of_exclusion_ok += usize::from(before.iter().all(|r| r.id != id));
        }
    }

    let mut rows = Vec::new();
    for (key, tally) in &tallies {
        let (question_type, config) = key.split_once("::").unwrap();
        let hits = &tally.hits[config];
        rows.push(json!({
            "question_type": question_type, "config": config, "questions": tally.questions,
            "recall_any_at_5": hits[0] as f64 / tally.questions as f64,
            "recall_any_at_10": hits[1] as f64 / tally.questions as f64,
            "hits_at_5": hits[0], "hits_at_10": hits[1],
        }));
    }
    let report: Value = json!({
        "dataset": "LongMemEval_S (longmemeval_s_cleaned.json), dev ids only",
        "protocol": "retrieval only, lexical BM25 through document search; session-level facts of user turns; no LLM, no model; abstention questions excluded",
        "dev_ids_in_split": split.dev.len(), "questions_used": questions.len(),
        "heldout_ids_read": 0,
        "memory_default_ranks_identically_to_document_search": format!("{identical_default}/{}", questions.len()),
        "as_of_no_future_facts": format!("{as_of_invariant_ok}/{}", questions.len()),
        "as_of_before_newest_evidence_excludes_it": format!("{as_of_exclusion_ok}/{as_of_exclusion_checked}"),
        "rows": rows,
        "note": "A sanity check on a small dev sample, not a benchmark claim. In this data some evidence sessions are dated after the question date, so evaluating as of the question date excludes them (see memory_as_of vs document_search on temporal-reasoning). Recency half-lives were tried on these same dev questions; they are not tuned values.",
    });
    std::fs::write(&args.out, serde_json::to_string_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
