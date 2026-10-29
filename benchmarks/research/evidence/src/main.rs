// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, ensure, Context, Result};
use hms_research_evidence::contract::{
    validate_query, Claim, Operation, Query, Source, MAX_TOTAL_BYTES,
};
use hms_research_evidence::execute::execute;
use hms_research_evidence::proof::{
    plan_forgetting, ProofCutResult, ProofCutStatus, ProofProblem, SearchLimits,
};
use hms_research_evidence::retrieval::{answer_with_repair, RetrievalLimits, RetrievalReport};
use hms_research_evidence::store::{validate_bundle, EvidenceStore};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

const USAGE: &str = "usage: hms-research-evidence demo | run FILE.json STORE | proof FILE.json";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RunInput {
    namespace: String,
    sources: Vec<Source>,
    claims: Vec<Claim>,
    terms: String,
    query: Query,
    limits: RetrievalLimits,
}

impl RunInput {
    fn validate(&self) -> Result<()> {
        validate_bundle(&self.namespace, &self.sources, &self.claims)?;
        validate_query(&self.query)?;
        self.limits.validate()?;
        ensure!(
            !self.terms.trim().is_empty() && self.terms.len() <= 8192,
            "search terms require 1..=8192 bytes"
        );
        execute(&self.query, &self.claims, &self.sources)?;
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProofInput {
    problem: ProofProblem,
    limits: SearchLimits,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ForgettingFixture {
    namespace: String,
    sources: Vec<Source>,
    claims: Vec<Claim>,
    problem: ProofProblem,
    limits: SearchLimits,
    protected_source_ids: Vec<String>,
}

#[derive(Serialize)]
struct DemoCase {
    name: &'static str,
    report: RetrievalReport,
}

#[derive(Serialize)]
struct SourcePin {
    id: String,
    version: String,
}

#[derive(Serialize)]
struct DemoForgetting {
    mapping: &'static str,
    pinned_sources: Vec<SourcePin>,
    cut: ProofCutResult,
    forgotten_sources_absent_after_reopen: Vec<String>,
    protected_sources_present: Vec<String>,
    same_version_replay_rejected: bool,
    fresh_version_relearned: Vec<SourcePin>,
}

#[derive(Serialize)]
struct Demo {
    research_prototype: bool,
    supplied_annotations: &'static str,
    cases: Vec<DemoCase>,
    forgetting: DemoForgetting,
}

fn main() {
    if let Err(error) = command() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn command() -> Result<()> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let verb = arguments.first().and_then(|value| value.to_str());
    match (verb, arguments.len()) {
        (Some("demo"), 1) => write_json(&demo()?),
        (Some("run"), 3) => {
            let input: RunInput = read_json(Path::new(&arguments[1]))?;
            input.validate()?;
            let mut store = EvidenceStore::open(Path::new(&arguments[2]), &input.namespace)?;
            store.put(&input.sources, &input.claims)?;
            write_json(&answer_with_repair(
                &store,
                &input.terms,
                &input.query,
                input.limits,
            )?)
        }
        (Some("proof"), 2) => {
            let input: ProofInput = read_json(Path::new(&arguments[1]))?;
            write_json(&plan_forgetting(&input.problem, input.limits)?)
        }
        _ => bail!(USAGE),
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).context("cannot open JSON input")?;
    let metadata = file.metadata().context("cannot inspect JSON input")?;
    ensure!(metadata.is_file(), "JSON input must be a regular file");
    ensure!(
        metadata.len() <= MAX_TOTAL_BYTES as u64,
        "JSON input exceeds 8 MiB"
    );
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take((MAX_TOTAL_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .context("cannot read JSON input")?;
    ensure!(bytes.len() <= MAX_TOTAL_BYTES, "JSON input exceeds 8 MiB");
    serde_json::from_slice(&bytes).context("invalid JSON input")
}

fn fixture<T: DeserializeOwned>(text: &str) -> Result<T> {
    ensure!(text.len() <= MAX_TOTAL_BYTES, "fixture exceeds 8 MiB");
    serde_json::from_str(text).context("invalid generated fixture")
}

fn write_json(value: &impl Serialize) -> Result<()> {
    let mut output = io::BufWriter::new(io::stdout().lock());
    serde_json::to_writer_pretty(&mut output, value)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}

fn demo() -> Result<Demo> {
    let payments: RunInput = fixture(include_str!("../fixtures/payments.json"))?;
    let identity: RunInput = fixture(include_str!("../fixtures/identity.json"))?;
    payments.validate()?;
    identity.validate()?;
    let directory = tempfile::tempdir()?;
    let mut payment_store =
        EvidenceStore::open(directory.path().join("payments"), &payments.namespace)?;
    payment_store.put(&payments.sources, &payments.claims)?;
    let mut cases = vec![DemoCase {
        name: "payment_sum_with_source_repair",
        report: answer_with_repair(
            &payment_store,
            &payments.terms,
            &payments.query,
            payments.limits,
        )?,
    }];
    let mut count = payments.query.clone();
    count.operation = Operation::CountDistinct;
    cases.push(DemoCase {
        name: "distinct_actual_payments",
        report: answer_with_repair(&payment_store, &payments.terms, &count, payments.limits)?,
    });
    let mut identity_store =
        EvidenceStore::open(directory.path().join("identity"), &identity.namespace)?;
    identity_store.put(&identity.sources, &identity.claims)?;
    cases.push(DemoCase {
        name: "identity_range",
        report: answer_with_repair(
            &identity_store,
            &identity.terms,
            &identity.query,
            identity.limits,
        )?,
    });
    let mut limited = identity.query.clone();
    limited.max_worlds = 1;
    cases.push(DemoCase {
        name: "identity_budget_abstention",
        report: answer_with_repair(&identity_store, &identity.terms, &limited, identity.limits)?,
    });
    Ok(Demo {
        research_prototype: true,
        supplied_annotations: "oracle-extraction control; no extraction or model calls",
        cases,
        forgetting: forgetting_demo(directory.path().join("forgetting").as_path())?,
    })
}

fn forgetting_demo(path: &Path) -> Result<DemoForgetting> {
    let input: ForgettingFixture = fixture(include_str!("../fixtures/forgetting-demo.json"))?;
    validate_bundle(&input.namespace, &input.sources, &input.claims)?;
    let cut = plan_forgetting(&input.problem, input.limits)?;
    ensure!(
        cut.status == ProofCutStatus::Solved,
        "demo cut was not solved"
    );
    let plan = cut.plan.as_ref().context("demo cut has no plan")?;
    let removed: BTreeSet<_> = plan.removed_sources.iter().cloned().collect();
    let declared: BTreeSet<_> = input
        .sources
        .iter()
        .map(|source| source.id.as_str())
        .collect();
    let primitive: BTreeSet<_> = input
        .problem
        .sources
        .iter()
        .map(|source| source.id.as_str())
        .collect();
    ensure!(declared == primitive, "demo source-to-root mapping differs");
    ensure!(!removed.is_empty(), "demo requires a nonempty cut");
    for id in &input.protected_source_ids {
        ensure!(
            declared.contains(id.as_str()) && !removed.contains(id),
            "demo protection differs"
        );
    }
    let pinned_sources = pins(&input.sources);
    let replay_sources: Vec<_> = input
        .sources
        .iter()
        .filter(|source| removed.contains(&source.id))
        .cloned()
        .collect();
    let replay_claims: Vec<_> = input
        .claims
        .iter()
        .filter(|claim| removed.contains(&claim.evidence[0].source_id))
        .cloned()
        .collect();
    {
        let mut store = EvidenceStore::open(path, &input.namespace)?;
        store.put(&input.sources, &input.claims)?;
        let (sources, _) = store.snapshot()?;
        ensure!(
            source_versions(&sources) == source_versions(&input.sources),
            "demo version pins differ"
        );
        let forgotten = store.forget_sources(&plan.removed_sources)?;
        ensure!(forgotten == removed.len(), "demo deletion count differs");
    }
    let mut store = EvidenceStore::open(path, &input.namespace)?;
    let (sources, _) = store.snapshot()?;
    ensure!(
        sources.iter().all(|source| !removed.contains(&source.id)),
        "forgotten demo source reappeared"
    );
    for id in &input.protected_source_ids {
        ensure!(
            sources.iter().any(|source| &source.id == id),
            "protected demo source was removed"
        );
    }
    let same_version_replay_rejected = store.put(&replay_sources, &replay_claims).is_err();
    ensure!(
        same_version_replay_rejected,
        "same-version demo replay was accepted"
    );
    ensure!(
        store.snapshot()?.0 == sources,
        "rejected demo replay changed storage"
    );
    let mut fresh_sources = replay_sources;
    for source in &mut fresh_sources {
        source.version = "demo-v2".into();
    }
    let mut fresh_claims = replay_claims;
    for claim in &mut fresh_claims {
        for span in &mut claim.evidence {
            if removed.contains(&span.source_id) {
                span.source_version = "demo-v2".into();
            }
        }
    }
    store.put(&fresh_sources, &fresh_claims)?;
    drop(store);
    let reopened = EvidenceStore::open(path, &input.namespace)?;
    let (relearned, _) = reopened.snapshot()?;
    let versions = source_versions(&relearned);
    for source in &fresh_sources {
        ensure!(
            versions.get(&source.id) == Some(&source.version),
            "fresh demo version was not retained"
        );
    }
    for id in &input.protected_source_ids {
        ensure!(
            versions.get(id) == source_versions(&input.sources).get(id),
            "protected demo version changed"
        );
    }
    Ok(DemoForgetting {
        mapping: "isolated demo only: primitive IDs equal exact pinned source IDs",
        pinned_sources,
        cut,
        forgotten_sources_absent_after_reopen: removed.into_iter().collect(),
        protected_sources_present: input.protected_source_ids,
        same_version_replay_rejected,
        fresh_version_relearned: pins(&fresh_sources),
    })
}

fn source_versions(sources: &[Source]) -> BTreeMap<String, String> {
    sources
        .iter()
        .map(|source| (source.id.clone(), source.version.clone()))
        .collect()
}

fn pins(sources: &[Source]) -> Vec<SourcePin> {
    source_versions(sources)
        .into_iter()
        .map(|(id, version)| SourcePin { id, version })
        .collect()
}
