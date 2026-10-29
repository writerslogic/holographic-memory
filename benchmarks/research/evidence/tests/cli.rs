// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use hms_research_evidence::contract::{validate_claims, Gap, MAX_TOTAL_BYTES};
use hms_research_evidence::retrieval::RetrievalReport;
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_hms-research-evidence"))
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn output_json(output: Output) -> Value {
    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("stdout must be one JSON value")
}

#[test]
fn demo_runs_real_retrieval_uncertainty_and_pinned_forgetting_checks() {
    let output = output_json(command().arg("demo").output().unwrap());
    assert_eq!(output["research_prototype"], true);
    let cases = output["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    let case =
        |name: &str| cases.iter().find(|case| case["name"] == name).unwrap()["report"].clone();
    let sum: RetrievalReport =
        serde_json::from_value(case("payment_sum_with_source_repair")).unwrap();
    assert_eq!(sum.answer.lower, Some(700));
    assert_eq!(sum.answer.upper, Some(700));
    assert_eq!(sum.steps[0].admitted, ["anchor"]);
    assert!(sum.steps.iter().skip(1).any(|step| {
        step.phase == "scope_repair"
            && step.gaps_before.contains(&Gap::MissingValue {
                claim_id: "unknown-amount".into(),
            })
    }));
    assert!(!sum
        .answer
        .claim_ids
        .iter()
        .any(|id| id == "planned-payment" || id == "attendance-only"));
    assert!(sum
        .sources
        .iter()
        .any(|source| source.text.contains("Café λ")));
    assert_eq!(case("distinct_actual_payments")["answer"]["lower"], 2);
    assert_eq!(case("identity_range")["answer"]["status"], "bounded");
    assert_eq!(case("identity_range")["answer"]["lower"], 1);
    assert_eq!(case("identity_range")["answer"]["upper"], 2);
    assert_eq!(
        case("identity_budget_abstention")["answer"]["status"],
        "unresolved"
    );
    assert!(case("identity_budget_abstention")["answer"]["lower"].is_null());
    for case in cases {
        let report: RetrievalReport = serde_json::from_value(case["report"].clone()).unwrap();
        validate_claims(&report.claims, &report.sources).unwrap();
        assert!(report.answer.observed_only);
        assert!(report.packet_bytes <= report.limits.max_packet_bytes);
        assert!(report.source_bytes <= report.limits.max_source_bytes);
    }
    let forgetting = &output["forgetting"];
    assert_eq!(forgetting["cut"]["status"], "solved");
    assert_eq!(
        forgetting["cut"]["plan"]["removed_sources"],
        json!(["source-a", "source-b"])
    );
    assert_eq!(
        forgetting["forgotten_sources_absent_after_reopen"],
        json!(["source-a", "source-b"])
    );
    assert_eq!(
        forgetting["protected_sources_present"],
        json!(["source-protected"])
    );
    assert_eq!(forgetting["same_version_replay_rejected"], true);
    assert_eq!(
        forgetting["fresh_version_relearned"],
        json!([
            {"id":"source-a", "version":"demo-v2"},
            {"id":"source-b", "version":"demo-v2"}
        ])
    );
    assert_eq!(
        forgetting["cut"]["plan"]["evidence"]["targets"][0]["derivable"],
        false
    );
    assert_eq!(
        forgetting["cut"]["plan"]["evidence"]["protected"][0]["derivable"],
        true
    );
}

#[test]
fn run_and_proof_accept_the_documented_fixtures() {
    let directory = tempfile::tempdir().unwrap();
    let store = directory.path().join("store");
    let run = output_json(
        command()
            .arg("run")
            .arg(fixture("payments.json"))
            .arg(&store)
            .output()
            .unwrap(),
    );
    assert!(store.is_dir());
    assert_eq!(run["answer"]["status"], "resolved");
    assert_eq!(run["answer"]["lower"], 700);
    let proof = output_json(
        command()
            .arg("proof")
            .arg(fixture("proof.json"))
            .output()
            .unwrap(),
    );
    assert_eq!(proof["status"], "solved");
    assert_eq!(proof["plan"]["cost"], 3);
    assert_eq!(
        proof["plan"]["removed_sources"],
        json!(["source-a", "source-b"])
    );
}

#[test]
fn invalid_input_is_rejected_before_creating_the_destination_store() {
    let directory = tempfile::tempdir().unwrap();
    let base: Value = serde_json::from_slice(&fs::read(fixture("payments.json")).unwrap()).unwrap();
    for mutation in 0..8 {
        let mut input = base.clone();
        match mutation {
            0 => input["unexpected"] = json!(true),
            1 => input["claims"][0]["evidence"][0]["quote"] = json!("forged quote"),
            2 => input["query"]["max_worlds"] = json!(0),
            3 => input["limits"]["max_sources"] = json!(0),
            4 => input["terms"] = json!("x".repeat(8193)),
            5 => input["claims"][0]["supersedes"] = json!(["absent-claim"]),
            6 => input["claims"][0]["evidence"][0]["source_version"] = json!("stale"),
            7 => input["namespace"] = json!("invalid\nnamespace"),
            _ => unreachable!(),
        }
        let path = directory.path().join(format!("invalid-{mutation}.json"));
        let store = directory.path().join(format!("store-{mutation}"));
        fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
        let output = command()
            .arg("run")
            .arg(&path)
            .arg(&store)
            .output()
            .unwrap();
        assert!(!output.status.success(), "mutation {mutation} was accepted");
        assert!(output.stdout.is_empty());
        assert!(!store.exists(), "mutation {mutation} created a store");
    }
}

#[test]
fn per_source_metadata_limit_is_checked_before_creating_a_store() {
    let directory = tempfile::tempdir().unwrap();
    let mut input: Value =
        serde_json::from_slice(&fs::read(fixture("payments.json")).unwrap()).unwrap();
    let prototype = input["claims"][0].clone();
    input["claims"] = Value::Array(
        (0..1200)
            .map(|index| {
                let mut claim = prototype.clone();
                claim["id"] = json!(format!("claim-{index}"));
                claim
            })
            .collect(),
    );
    let path = directory.path().join("metadata-too-large.json");
    let store = directory.path().join("store");
    let encoded = serde_json::to_vec(&input).unwrap();
    assert!(encoded.len() < MAX_TOTAL_BYTES);
    fs::write(&path, encoded).unwrap();
    let output = command().arg("run").arg(path).arg(&store).output().unwrap();
    assert!(!output.status.success());
    assert!(!store.exists());
}

#[test]
fn json_input_limit_accepts_minus_one_and_exactly_and_rejects_plus_one() {
    let directory = tempfile::tempdir().unwrap();
    let base = fs::read(fixture("payments.json")).unwrap();
    for size in [MAX_TOTAL_BYTES - 1, MAX_TOTAL_BYTES, MAX_TOTAL_BYTES + 1] {
        let path = directory.path().join(format!("input-{size}.json"));
        let store = directory.path().join(format!("store-{size}"));
        if size <= MAX_TOTAL_BYTES {
            let mut input = base.clone();
            input.resize(size, b' ');
            fs::write(&path, input).unwrap();
        } else {
            fs::File::create(&path)
                .unwrap()
                .set_len(size as u64)
                .unwrap();
        }
        let output = command().arg("run").arg(path).arg(&store).output().unwrap();
        if size <= MAX_TOTAL_BYTES {
            assert_eq!(output_json(output)["answer"]["lower"], 700);
        } else {
            assert!(!output.status.success());
            assert!(!store.exists());
        }
    }
}

#[test]
fn proof_input_rejects_unknown_fields_and_reports_search_exhaustion() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("proof.json");
    let mut input: Value =
        serde_json::from_slice(&fs::read(fixture("proof.json")).unwrap()).unwrap();
    input["limits"]["max_steps"] = json!(0);
    fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    let exhausted = output_json(command().arg("proof").arg(&path).output().unwrap());
    assert_eq!(exhausted["status"], "budget_exhausted");
    assert!(exhausted["plan"].is_null());
    input["apply"] = json!(true);
    fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    let output = command().arg("proof").arg(path).output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}
