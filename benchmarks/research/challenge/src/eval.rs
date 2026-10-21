// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Metrics, timing, the erase check, the verdict and the experiment driver.
//!
//! # Metrics (per method, over every query)
//!
//! - `correctness`: answer-and-evidence correctness at the evidence budget. A resolved gold
//!   needs `status == Resolved`, the gold primary as the answer and every id of
//!   `gold.required_evidence` among the returned items; an unresolved gold needs
//!   `status == UnresolvedConflict` and exactly the gold branches. Returning more than the
//!   budget is wrong. A blended ranking on an unresolved gold is wrong.
//! - `answer_only_accuracy`: the same without the evidence condition.
//! - `stale_answer_rate`: the answer is a claim that is superseded or retracted as of the
//!   query time.
//! - `missed_exception_rate`: over queries whose gold primary is the family's exception, the
//!   fraction not answered with it.
//! - `blended_on_conflict`: unresolved golds answered with a ranking.
//!
//! # Timing
//!
//! Single caller thread. One untimed scoring pass (which also warms caches), `warmup - 1`
//! further untimed passes, then `repeats` timed passes over the whole query set. Per pass the
//! per-query wall-clock durations give nearest-rank p50/p95/p99; the reported figure for
//! each percentile is the median across passes. The engine may parallelise internally.
//!
//! # Verdict
//!
//! PASS when the challenge method's correctness is at least [`REQUIRED_MARGIN_POINTS`]
//! absolute points above the strongest non-challenge baseline and its p95 is at most
//! [`MAX_P95_RATIO`] times the p95 of the faster scope-filtered BM25 variant.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, ensure, Context, Result};
use serde::Serialize;

use crate::baselines::{
    self, Answer, CHALLENGE, GLOBAL_BM25, NON_CHALLENGE, SCOPED_BM25_ENGINE, SCOPED_BM25_LOCAL,
    SCOPED_LOOKUP,
};
use crate::contract::{Defeat, QueryInput, Status, EVIDENCE_BUDGET};
use crate::store::{ArenaStats, ChallengeStore, IngestStats, WipeReport};
use crate::synth::{self, Dataset, GenConfig, Gold, Query};

pub const REQUIRED_MARGIN_POINTS: f64 = 10.0;
pub const MAX_P95_RATIO: f64 = 2.0;
pub const SCHEMA: &str = "hms-research-challenge/results/v1";
pub const SQLITE_BASELINE_NOTE: &str = "skipped: rusqlite is not a dependency of the root crate \
     (Cargo.toml [dependencies], lines 19-58 at the time of writing), and the round forbids new \
     third-party crates";

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Outcome {
    pub correct: bool,
    pub answer_only: bool,
    pub stale: bool,
    pub missed_exception: bool,
    pub blended_on_conflict: bool,
    pub over_budget: bool,
    pub unresolved: bool,
}

fn same_set(a: &[String], b: &[String]) -> bool {
    let mut x = a.to_vec();
    x.sort();
    x.dedup();
    let mut y = b.to_vec();
    y.sort();
    y.dedup();
    x == y
}

/// Scores one answer against its gold.
pub fn score(answer: &Answer, gold: &Gold) -> Outcome {
    let over_budget = answer.items.len() > EVIDENCE_BUDGET;
    let evidence_ok = gold
        .required_evidence
        .iter()
        .all(|id| answer.items.contains(id));
    let answer_only = match gold.status {
        Status::Resolved => answer.status == Status::Resolved && answer.primary == gold.primary,
        Status::UnresolvedConflict => {
            answer.status == Status::UnresolvedConflict && same_set(&answer.branches, &gold.branches)
        }
    };
    Outcome {
        correct: answer_only && evidence_ok && !over_budget,
        answer_only,
        stale: answer
            .primary
            .as_ref()
            .is_some_and(|primary| gold.stale_ids.contains(primary)),
        missed_exception: gold.expects_exception && !answer_only,
        blended_on_conflict: gold.status == Status::UnresolvedConflict
            && answer.status == Status::Resolved,
        over_budget,
        unresolved: answer.status == Status::UnresolvedConflict,
    }
}

/// Nearest-rank percentile of ascending `sorted`.
pub fn percentile_nearest_rank(sorted: &[u64], percent: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((percent / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PassStats {
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    mean_us: f64,
    max_us: f64,
}

fn pass_stats(mut nanos: Vec<u64>) -> PassStats {
    nanos.sort_unstable();
    let to_us = |nanos: u64| nanos as f64 / 1e3;
    let mean = if nanos.is_empty() {
        0.0
    } else {
        nanos.iter().map(|&n| n as f64).sum::<f64>() / nanos.len() as f64 / 1e3
    };
    PassStats {
        p50_us: to_us(percentile_nearest_rank(&nanos, 50.0)),
        p95_us: to_us(percentile_nearest_rank(&nanos, 95.0)),
        p99_us: to_us(percentile_nearest_rank(&nanos, 99.0)),
        mean_us: mean,
        max_us: to_us(nanos.last().copied().unwrap_or(0)),
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Latency {
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub mean_us: f64,
    pub max_us: f64,
    pub repeats: usize,
    pub warmup_passes: usize,
    pub per_pass_p95_us: Vec<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Breakdown {
    pub queries: usize,
    pub correct: usize,
    pub correctness: f64,
    pub answer_only: usize,
    pub stale: usize,
}

impl Breakdown {
    fn add(&mut self, outcome: &Outcome) {
        self.queries += 1;
        self.correct += usize::from(outcome.correct);
        self.answer_only += usize::from(outcome.answer_only);
        self.stale += usize::from(outcome.stale);
    }

    fn finish(&mut self) {
        self.correctness = ratio(self.correct, self.queries);
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MethodReport {
    pub method: String,
    pub queries: usize,
    pub correct: usize,
    pub correctness: f64,
    pub answer_only_accuracy: f64,
    pub stale_answers: usize,
    pub stale_answer_rate: f64,
    pub exception_queries: usize,
    pub missed_exceptions: usize,
    pub missed_exception_rate: f64,
    pub conflict_queries: usize,
    pub blended_on_conflict: usize,
    pub unresolved_returned: usize,
    pub over_budget: usize,
    pub latency: Latency,
    pub by_query_kind: BTreeMap<String, Breakdown>,
    pub by_family_type: BTreeMap<String, Breakdown>,
}

/// Scores and times one method over every query.
pub fn evaluate<F>(
    name: &str,
    queries: &[Query],
    mut method: F,
    repeats: usize,
    warmup: usize,
) -> Result<MethodReport>
where
    F: FnMut(&Query) -> Result<Answer>,
{
    ensure!(repeats >= 1, "repeats must be at least 1");
    let mut outcomes: Vec<Outcome> = Vec::with_capacity(queries.len());
    for query in queries {
        let answer = method(query).with_context(|| format!("{name} failed on {}", query.id))?;
        outcomes.push(score(&answer, &query.gold));
    }
    for _ in 1..warmup.max(1) {
        for query in queries {
            method(query)?;
        }
    }
    let mut passes: Vec<PassStats> = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let mut nanos: Vec<u64> = Vec::with_capacity(queries.len());
        for query in queries {
            let started = Instant::now();
            let answer = method(query)?;
            let elapsed = started.elapsed();
            std::hint::black_box(&answer);
            nanos.push(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX));
        }
        passes.push(pass_stats(nanos));
    }
    let collect = |pick: fn(&PassStats) -> f64| passes.iter().map(pick).collect::<Vec<f64>>();
    let per_pass_p95_us = collect(|p| p.p95_us);
    let latency = Latency {
        p50_us: median(&collect(|p| p.p50_us)),
        p95_us: median(&per_pass_p95_us),
        p99_us: median(&collect(|p| p.p99_us)),
        mean_us: median(&collect(|p| p.mean_us)),
        max_us: median(&collect(|p| p.max_us)),
        repeats,
        warmup_passes: warmup.max(1),
        per_pass_p95_us,
    };

    let mut by_query_kind: BTreeMap<String, Breakdown> = BTreeMap::new();
    let mut by_family_type: BTreeMap<String, Breakdown> = BTreeMap::new();
    let mut correct = 0usize;
    let mut answer_only = 0usize;
    let mut stale = 0usize;
    let mut exception_queries = 0usize;
    let mut missed_exceptions = 0usize;
    let mut conflict_queries = 0usize;
    let mut blended_on_conflict = 0usize;
    let mut unresolved_returned = 0usize;
    let mut over_budget = 0usize;
    for (query, outcome) in queries.iter().zip(&outcomes) {
        correct += usize::from(outcome.correct);
        answer_only += usize::from(outcome.answer_only);
        stale += usize::from(outcome.stale);
        exception_queries += usize::from(query.gold.expects_exception);
        missed_exceptions += usize::from(outcome.missed_exception);
        conflict_queries += usize::from(query.gold.status == Status::UnresolvedConflict);
        blended_on_conflict += usize::from(outcome.blended_on_conflict);
        unresolved_returned += usize::from(outcome.unresolved);
        over_budget += usize::from(outcome.over_budget);
        by_query_kind
            .entry(query.kind.name().to_owned())
            .or_default()
            .add(outcome);
        by_family_type
            .entry(query.family_type.name().to_owned())
            .or_default()
            .add(outcome);
    }
    for breakdown in by_query_kind.values_mut().chain(by_family_type.values_mut()) {
        breakdown.finish();
    }
    Ok(MethodReport {
        method: name.to_owned(),
        queries: queries.len(),
        correct,
        correctness: ratio(correct, queries.len()),
        answer_only_accuracy: ratio(answer_only, queries.len()),
        stale_answers: stale,
        stale_answer_rate: ratio(stale, queries.len()),
        exception_queries,
        missed_exceptions,
        missed_exception_rate: ratio(missed_exceptions, exception_queries),
        conflict_queries,
        blended_on_conflict,
        unresolved_returned,
        over_budget,
        latency,
        by_query_kind,
        by_family_type,
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct Verdict {
    pub strongest_baseline: String,
    pub strongest_baseline_correctness: f64,
    pub challenge_correctness: f64,
    pub margin_points: f64,
    pub required_margin_points: f64,
    pub latency_reference: String,
    pub reference_p95_us: f64,
    pub challenge_p95_us: f64,
    pub p95_ratio: f64,
    pub max_p95_ratio: f64,
    pub accuracy_pass: bool,
    pub latency_pass: bool,
    pub pass: bool,
    pub summary: String,
}

/// The kill-test verdict over the method reports.
pub fn verdict(methods: &BTreeMap<String, MethodReport>) -> Result<Verdict> {
    let challenge = methods
        .get(CHALLENGE)
        .ok_or_else(|| anyhow!("no report for {CHALLENGE}"))?;
    let (strongest, baseline) = NON_CHALLENGE
        .iter()
        .filter_map(|name| methods.get(*name).map(|report| (*name, report)))
        .max_by(|a, b| a.1.correctness.total_cmp(&b.1.correctness))
        .ok_or_else(|| anyhow!("no non-challenge baseline report"))?;
    let (reference, reference_report) = [SCOPED_BM25_ENGINE, SCOPED_BM25_LOCAL]
        .iter()
        .filter_map(|name| methods.get(*name).map(|report| (*name, report)))
        .min_by(|a, b| a.1.latency.p95_us.total_cmp(&b.1.latency.p95_us))
        .ok_or_else(|| anyhow!("no scope-filtered BM25 report"))?;
    let margin_points = (challenge.correctness - baseline.correctness) * 100.0;
    let reference_p95_us = reference_report.latency.p95_us;
    let challenge_p95_us = challenge.latency.p95_us;
    let p95_ratio = if reference_p95_us > 0.0 {
        challenge_p95_us / reference_p95_us
    } else if challenge_p95_us == 0.0 {
        1.0
    } else {
        f64::INFINITY
    };
    let accuracy_pass = margin_points >= REQUIRED_MARGIN_POINTS;
    let latency_pass = p95_ratio <= MAX_P95_RATIO;
    let pass = accuracy_pass && latency_pass;
    let summary = format!(
        "{}: challenge correctness {:.1}% vs strongest non-challenge baseline {} {:.1}% \
         (margin {:+.1} points, need >= {:.0}); challenge p95 {:.1} us vs {} {:.1} us \
         (ratio {:.2}, need <= {:.1})",
        if pass { "PASS" } else { "FAIL" },
        challenge.correctness * 100.0,
        strongest,
        baseline.correctness * 100.0,
        margin_points,
        REQUIRED_MARGIN_POINTS,
        challenge_p95_us,
        reference,
        reference_p95_us,
        p95_ratio,
        MAX_P95_RATIO,
    );
    Ok(Verdict {
        strongest_baseline: strongest.to_owned(),
        strongest_baseline_correctness: baseline.correctness,
        challenge_correctness: challenge.correctness,
        margin_points,
        required_margin_points: REQUIRED_MARGIN_POINTS,
        latency_reference: reference.to_owned(),
        reference_p95_us,
        challenge_p95_us,
        p95_ratio,
        max_p95_ratio: MAX_P95_RATIO,
        accuracy_pass,
        latency_pass,
        pass,
        summary,
    })
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ChallengeSummary {
    pub queries: usize,
    /// Queries where the challenge pass replaced the lexical winner of the support pass.
    pub support_overridden: usize,
    pub support_overridden_rate: f64,
    pub status_counts: BTreeMap<String, usize>,
    pub defeat_counts: BTreeMap<String, usize>,
    pub branches_truncated: usize,
    pub mean_items: f64,
}

fn challenge_summary(store: &ChallengeStore, queries: &[Query]) -> ChallengeSummary {
    let mut summary = ChallengeSummary::default();
    let mut items = 0usize;
    for query in queries {
        let Some(family) = store.arena().family(&query.key) else {
            continue;
        };
        let unit = family.answer(&QueryInput {
            key: &query.key,
            scope: &query.scope,
            as_of: query.as_of,
            text: &query.text,
        });
        summary.queries += 1;
        summary.support_overridden += usize::from(unit.support_overridden);
        summary.branches_truncated += usize::from(unit.branches_truncated);
        items += unit.items.len();
        let status = match unit.status {
            Status::Resolved => "resolved",
            Status::UnresolvedConflict => "unresolved-conflict",
        };
        *summary.status_counts.entry(status.to_owned()).or_insert(0) += 1;
        let defeat = match unit.strongest_defeater.as_ref().map(|d| &d.defeat) {
            None => "none",
            Some(Defeat::Retracted { .. }) => "retracted",
            Some(Defeat::Superseded { .. }) => "superseded",
            Some(Defeat::LessSpecific) => "less-specific",
            Some(Defeat::Outranked) => "outranked",
        };
        *summary.defeat_counts.entry(defeat.to_owned()).or_insert(0) += 1;
    }
    summary.support_overridden_rate = ratio(summary.support_overridden, summary.queries);
    summary.mean_items = ratio(items, summary.queries);
    summary
}

#[derive(Clone, Debug, Serialize)]
pub struct EraseReport {
    pub families_wiped: usize,
    pub assertions_dropped: usize,
    pub facts_forgotten: usize,
    pub wipe_p50_us: f64,
    pub wipe_p95_us: f64,
    /// Every wiped family vanished from the arena and the memory API, its queries find no
    /// family, and the next family's answers are unchanged.
    pub verified: bool,
    pub failures: Vec<String>,
    pub wipes: Vec<WipeReport>,
}

fn erase_check(store: &mut ChallengeStore, dataset: &Dataset, sample: usize) -> Result<EraseReport> {
    let families = dataset.families.len();
    let sample = sample.min(families);
    let mut failures: Vec<String> = Vec::new();
    let mut wipes: Vec<WipeReport> = Vec::new();
    for i in 0..sample {
        let index = i * families / sample;
        let spec = &dataset.families[index];
        let neighbour = &dataset.families[(index + 1) % families];
        let neighbour_queries: Vec<&Query> = dataset
            .queries
            .iter()
            .filter(|q| q.family == neighbour.index)
            .collect();
        let snapshot = |store: &ChallengeStore| -> Vec<Option<Answer>> {
            neighbour_queries
                .iter()
                .map(|q| {
                    store
                        .arena()
                        .family(&q.key)
                        .map(|family| baselines::challenge(family, q))
                })
                .collect()
        };
        let before = snapshot(store);
        let report = store.wipe_family(&spec.key)?;
        if report.arena_assertions_dropped != spec.records.len() {
            failures.push(format!(
                "{}: arena dropped {} assertions, expected {}",
                spec.key,
                report.arena_assertions_dropped,
                spec.records.len()
            ));
        }
        if report.facts_forgotten != spec.records.len() {
            failures.push(format!(
                "{}: memory API forgot {} facts, expected {}",
                spec.key,
                report.facts_forgotten,
                spec.records.len()
            ));
        }
        if store.arena().family(&spec.key).is_some() {
            failures.push(format!("{}: still in the arena after the wipe", spec.key));
        }
        if store.any_stored(&spec.ids())? {
            failures.push(format!("{}: a fact still resolves in the memory API", spec.key));
        }
        if snapshot(store) != before {
            failures.push(format!(
                "{}: wiping it changed the answers of {}",
                spec.key, neighbour.key
            ));
        }
        wipes.push(report);
    }
    let mut nanos: Vec<u64> = wipes
        .iter()
        .map(|w| (w.elapsed_us * 1e3).round() as u64)
        .collect();
    nanos.sort_unstable();
    Ok(EraseReport {
        families_wiped: wipes.len(),
        assertions_dropped: wipes.iter().map(|w| w.arena_assertions_dropped).sum(),
        facts_forgotten: wipes.iter().map(|w| w.facts_forgotten).sum(),
        wipe_p50_us: percentile_nearest_rank(&nanos, 50.0) as f64 / 1e3,
        wipe_p95_us: percentile_nearest_rank(&nanos, 95.0) as f64 / 1e3,
        verified: failures.is_empty() && !wipes.is_empty(),
        failures,
        wipes,
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct RunSummary {
    pub repeats: usize,
    pub warmup_passes: usize,
    pub dimensions: u32,
    pub store_dir: Option<String>,
    pub single_threaded_timing: bool,
    pub evidence_budget: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct DatasetSummary {
    pub families: usize,
    pub conflict_families: usize,
    pub records: usize,
    pub queries: usize,
    pub query_mix: BTreeMap<String, usize>,
    pub templates: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct Results {
    pub schema: &'static str,
    pub generated_at_unix_ms: i64,
    pub split: &'static str,
    pub seed: u64,
    pub run: RunSummary,
    pub dataset: DatasetSummary,
    pub ingest: IngestStats,
    pub arena: ArenaStats,
    pub methods: BTreeMap<String, MethodReport>,
    pub challenge: ChallengeSummary,
    pub erase: EraseReport,
    pub sqlite_baseline: &'static str,
    pub verdict: Verdict,
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub gen: GenConfig,
    pub repeats: usize,
    pub warmup: usize,
    pub store_dir: Option<PathBuf>,
    pub dimensions: u32,
    pub wipe_sample: usize,
}

fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX))
}

/// Generates, ingests, indexes, evaluates every method, checks erasure and judges.
pub fn run_experiment(options: &RunOptions) -> Result<Results> {
    ensure!(options.gen.families >= 1, "families must be at least 1");
    ensure!(options.repeats >= 1, "repeats must be at least 1");
    let dataset = synth::generate(&options.gen);
    let mut store = ChallengeStore::open(options.store_dir.clone(), options.dimensions)?;
    let ingest = store.ingest(&dataset.batches())?;
    let arena = store.build_arena(&dataset.ids_by_family())?;
    ensure!(
        arena.assertions == dataset.records(),
        "arena holds {} assertions, dataset has {}",
        arena.assertions,
        dataset.records()
    );
    let queries = &dataset.queries;
    let (repeats, warmup) = (options.repeats, options.warmup);
    let mut methods: BTreeMap<String, MethodReport> = BTreeMap::new();
    {
        let core = store.core().clone();
        let report = evaluate(
            GLOBAL_BM25,
            queries,
            |q| baselines::global_bm25(&core, q),
            repeats,
            warmup,
        )?;
        methods.insert(GLOBAL_BM25.to_owned(), report);
    }
    {
        let memory = store.memory();
        let report = evaluate(
            SCOPED_BM25_ENGINE,
            queries,
            |q| baselines::scoped_bm25_engine(memory, q),
            repeats,
            warmup,
        )?;
        methods.insert(SCOPED_BM25_ENGINE.to_owned(), report);
    }
    {
        let arena = store.arena();
        let report = evaluate(
            SCOPED_BM25_LOCAL,
            queries,
            |q| {
                Ok(arena
                    .family(&q.key)
                    .map_or_else(Answer::absent, |family| baselines::scoped_bm25_local(family, q)))
            },
            repeats,
            warmup,
        )?;
        methods.insert(SCOPED_BM25_LOCAL.to_owned(), report);
        let report = evaluate(
            SCOPED_LOOKUP,
            queries,
            |q| {
                Ok(arena.family(&q.key).map_or_else(Answer::absent, |family| {
                    baselines::scoped_lookup_no_challenge(family, q)
                }))
            },
            repeats,
            warmup,
        )?;
        methods.insert(SCOPED_LOOKUP.to_owned(), report);
        let report = evaluate(
            CHALLENGE,
            queries,
            |q| {
                Ok(arena
                    .family(&q.key)
                    .map_or_else(Answer::absent, |family| baselines::challenge(family, q)))
            },
            repeats,
            warmup,
        )?;
        methods.insert(CHALLENGE.to_owned(), report);
    }
    let challenge = challenge_summary(&store, queries);
    let erase = erase_check(&mut store, &dataset, options.wipe_sample)?;
    let verdict = verdict(&methods)?;
    Ok(Results {
        schema: SCHEMA,
        generated_at_unix_ms: now_unix_ms(),
        split: options.gen.split.name(),
        seed: options.gen.seed,
        run: RunSummary {
            repeats,
            warmup_passes: warmup.max(1),
            dimensions: options.dimensions,
            store_dir: options.store_dir.as_ref().map(|p| p.display().to_string()),
            single_threaded_timing: true,
            evidence_budget: EVIDENCE_BUDGET,
        },
        dataset: DatasetSummary {
            families: dataset.families.len(),
            conflict_families: dataset.conflict_families(),
            records: dataset.records(),
            queries: dataset.queries.len(),
            query_mix: dataset.query_mix(),
            templates: options.gen.split.name(),
        },
        ingest,
        arena,
        methods,
        challenge,
        erase,
        sqlite_baseline: SQLITE_BASELINE_NOTE,
        verdict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentiles_at_the_boundaries() {
        let sorted: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile_nearest_rank(&sorted, 95.0), 95);
        assert_eq!(percentile_nearest_rank(&sorted, 100.0), 100);
        assert_eq!(percentile_nearest_rank(&sorted, 0.0), 1);
        assert_eq!(percentile_nearest_rank(&[7], 95.0), 7);
        assert_eq!(percentile_nearest_rank(&[], 95.0), 0);
    }

    #[test]
    fn blended_ranking_on_a_conflict_is_wrong() {
        let gold = Gold {
            status: Status::UnresolvedConflict,
            primary: None,
            branches: vec!["a".into(), "b".into()],
            required_evidence: vec!["a".into(), "b".into()],
            stale_ids: vec![],
            expects_exception: false,
        };
        let blended = Answer::blended(vec![
            ("a".into(), crate::contract::AssertionKind::Claim),
            ("b".into(), crate::contract::AssertionKind::Claim),
        ]);
        let outcome = score(&blended, &gold);
        assert!(!outcome.correct && outcome.blended_on_conflict);
        let honest = Answer {
            status: Status::UnresolvedConflict,
            primary: None,
            branches: vec!["b".into(), "a".into()],
            items: vec!["b".into(), "a".into()],
        };
        assert!(score(&honest, &gold).correct);
    }
}
