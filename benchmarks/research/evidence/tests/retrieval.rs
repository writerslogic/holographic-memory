// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use hms_research_evidence::contract::{
    validate_claims, Amount, AnswerStatus, Claim, Gap, Interval, Modality, Operation, Query,
    Selection, Source, Span, MAX_SOURCES, MAX_TOTAL_BYTES, MAX_WORLDS,
};
use hms_research_evidence::retrieval::{
    answer_with_repair, RetrievalLimits, RetrievalReport, EMPTY_PACKET_BYTES,
};
use hms_research_evidence::store::EvidenceStore;
use std::collections::BTreeSet;

fn source(id: &str, text: &str) -> Source {
    Source {
        id: id.into(),
        version: "v1".into(),
        text: text.into(),
        observed_at: 10,
    }
}

fn span(source: &Source) -> Span {
    Span {
        source_id: source.id.clone(),
        source_version: source.version.clone(),
        start: 0,
        end: source.text.len(),
        quote: source.text.clone(),
    }
}

fn claim(id: &str, event: &str, amount: Option<i64>, source: &Source) -> Claim {
    Claim {
        id: id.into(),
        subject: "person:alice".into(),
        predicate: "paid_for".into(),
        event_key: event.into(),
        event_candidates: vec![event.into()],
        value: amount.map(|minor| Amount {
            minor,
            unit: "USD_cent".into(),
        }),
        event_time: Some(Interval { start: 10, end: 20 }),
        valid_from: 0,
        valid_to: None,
        positive: true,
        modality: Modality::Actual,
        evidence: vec![span(source)],
        supersedes: vec![],
    }
}

fn query(operation: Operation) -> Query {
    Query {
        selection: Selection {
            subject: "person:alice".into(),
            predicate: "paid_for".into(),
            modality: Modality::Actual,
            event_window: None,
        },
        operation,
        as_of: 100,
        known_at: None,
        max_worlds: MAX_WORLDS,
    }
}

fn sum_query() -> Query {
    query(Operation::Sum {
        unit: "USD_cent".into(),
    })
}

fn limits() -> RetrievalLimits {
    RetrievalLimits {
        initial_k: 1,
        repair_k: 8,
        max_rounds: 3,
        ..RetrievalLimits::default()
    }
}

fn persistent_store(sources: &[Source], claims: &[Claim]) -> (tempfile::TempDir, EvidenceStore) {
    let directory = tempfile::tempdir().unwrap();
    {
        let mut store = EvidenceStore::open(directory.path(), "retrieval-test").unwrap();
        store.put(sources, claims).unwrap();
    }
    let store = EvidenceStore::open(directory.path(), "retrieval-test").unwrap();
    (directory, store)
}

fn check_packet(report: &RetrievalReport) {
    validate_claims(&report.claims, &report.sources).unwrap();
    assert_eq!(
        report.source_bytes,
        report
            .sources
            .iter()
            .map(|source| source.text.len())
            .sum::<usize>()
    );
    assert!(report.source_bytes <= report.limits.max_source_bytes);
    assert_eq!(
        report.packet_bytes,
        serde_json::to_vec(
            &serde_json::json!({"sources": &report.sources, "claims": &report.claims})
        )
        .unwrap()
        .len()
    );
    assert!(report.packet_bytes <= report.limits.max_packet_bytes);
    assert!(report.sources.len() <= report.limits.max_sources);
    assert!(report.steps.len() <= report.limits.max_rounds);
    let ids: BTreeSet<_> = report
        .sources
        .iter()
        .map(|source| source.id.as_str())
        .collect();
    assert_eq!(ids.len(), report.sources.len());
    for claim in &report.claims {
        assert!(claim
            .evidence
            .iter()
            .all(|span| ids.contains(span.source_id.as_str())));
    }
}

#[test]
fn scope_repair_adds_distributed_events_and_deduplicates_mentions_for_sums_and_counts() {
    let sources = vec![
        source("anchor", "anchorneedle first receipt"),
        source("second", "Another receipt from the second visit."),
        source("echo", "The first receipt is mentioned again."),
    ];
    let claims = vec![
        claim("first", "visit-one", Some(100), &sources[0]),
        claim("second", "visit-two", Some(250), &sources[1]),
        claim("echo", "visit-one", Some(100), &sources[2]),
    ];
    let (_directory, store) = persistent_store(&sources, &claims);
    for (operation, expected) in [
        (
            Operation::Sum {
                unit: "USD_cent".into(),
            },
            350,
        ),
        (Operation::CountDistinct, 2),
    ] {
        let report =
            answer_with_repair(&store, "anchorneedle", &query(operation), limits()).unwrap();
        assert_eq!(report.steps[0].admitted.len(), 1);
        assert_eq!(report.sources.len(), 3);
        assert!(report.steps.len() > 1);
        assert!(report
            .steps
            .iter()
            .skip(1)
            .any(|step| !step.admitted.is_empty()));
        assert_eq!(report.answer.status, AnswerStatus::Resolved);
        assert_eq!(
            (report.answer.lower, report.answer.upper),
            (Some(expected), Some(expected))
        );
        assert_eq!(report.answer.claim_ids.len(), 3);
        assert!(report.answer.observed_only);
        assert!(report.answer.gaps.is_empty());
        check_packet(&report);
    }
}

#[test]
fn compatible_source_repairs_unknown_amount_without_counting_the_event_twice() {
    let sources = vec![
        source("anchor", "anchorneedle a receipt with no total"),
        source("amount", "The receipt total was recorded separately."),
    ];
    let claims = vec![
        claim("unknown", "same-event", None, &sources[0]),
        claim("known", "same-event", Some(425), &sources[1]),
    ];
    let (_directory, store) = persistent_store(&sources, &claims);
    let report = answer_with_repair(&store, "anchorneedle", &sum_query(), limits()).unwrap();
    assert_eq!(report.steps[0].admitted.len(), 1);
    assert!(report
        .steps
        .iter()
        .skip(1)
        .any(|step| step.gaps_before.contains(&Gap::MissingValue {
            claim_id: "unknown".into()
        })));
    assert_eq!(report.answer.status, AnswerStatus::Resolved);
    assert_eq!(
        (report.answer.lower, report.answer.upper),
        (Some(425), Some(425))
    );
    assert!(!report
        .answer
        .gaps
        .iter()
        .any(|gap| matches!(gap, Gap::MissingValue { .. })));
    check_packet(&report);
}

#[test]
fn repairing_negative_counterevidence_prevents_a_confident_positive_total() {
    let sources = vec![
        source("anchor", "anchorneedle the transaction completed"),
        source("denial", "The transaction did not complete."),
    ];
    let positive = claim("positive", "transaction", Some(300), &sources[0]);
    let mut negative = claim("negative", "transaction", None, &sources[1]);
    negative.positive = false;
    let (_directory, store) = persistent_store(&sources, &[positive, negative]);
    let report = answer_with_repair(&store, "anchorneedle", &sum_query(), limits()).unwrap();
    assert_eq!(report.steps[0].admitted.len(), 1);
    assert_eq!(report.sources.len(), 2);
    assert_eq!(report.answer.status, AnswerStatus::Unresolved);
    assert_eq!((report.answer.lower, report.answer.upper), (None, None));
    assert!(report.answer.gaps.contains(&Gap::CounterEvidence {
        claim_id: "negative".into()
    }));
    assert!(report.answer.claim_ids.contains(&"negative".into()));
    check_packet(&report);
}

fn three_events() -> (Vec<Source>, Vec<Claim>) {
    let sources = vec![
        source("anchor", "anchorneedle α"),
        source("second", "second receipt β"),
        source("third", "third receipt γ"),
    ];
    let claims = sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            claim(
                &format!("claim-{index}"),
                &format!("event-{index}"),
                Some((index + 1) as i64),
                source,
            )
        })
        .collect();
    (sources, claims)
}

#[test]
fn source_count_budget_below_at_and_above_the_complete_packet_is_enforced() {
    let (sources, claims) = three_events();
    let (_directory, store) = persistent_store(&sources, &claims);
    for max_sources in [2, 3, 4] {
        let report = answer_with_repair(
            &store,
            "anchorneedle",
            &sum_query(),
            RetrievalLimits {
                max_sources,
                ..limits()
            },
        )
        .unwrap();
        check_packet(&report);
        if max_sources < sources.len() {
            assert_eq!(report.answer.status, AnswerStatus::Unresolved);
            assert_eq!((report.answer.lower, report.answer.upper), (None, None));
            assert!(report.answer.gaps.contains(&Gap::EvidenceBudget));
            assert!(report
                .steps
                .iter()
                .any(|step| !step.skipped_for_budget.is_empty()));
        } else {
            assert_eq!(report.sources.len(), sources.len());
            assert_eq!(report.answer.status, AnswerStatus::Resolved);
            assert_eq!(
                (report.answer.lower, report.answer.upper),
                (Some(6), Some(6))
            );
        }
    }
}

#[test]
fn utf8_source_byte_budget_below_at_and_above_the_complete_packet_is_enforced() {
    let (sources, claims) = three_events();
    let exact: usize = sources.iter().map(|source| source.text.len()).sum();
    let (_directory, store) = persistent_store(&sources, &claims);
    for max_source_bytes in [exact - 1, exact, exact + 1] {
        let report = answer_with_repair(
            &store,
            "anchorneedle",
            &sum_query(),
            RetrievalLimits {
                max_source_bytes,
                ..limits()
            },
        )
        .unwrap();
        check_packet(&report);
        if max_source_bytes < exact {
            assert_eq!(report.answer.status, AnswerStatus::Unresolved);
            assert!(report.answer.gaps.contains(&Gap::EvidenceBudget));
            assert!(report
                .steps
                .iter()
                .any(|step| !step.skipped_for_budget.is_empty()));
        } else {
            assert_eq!(report.source_bytes, exact);
            assert_eq!(report.answer.status, AnswerStatus::Resolved);
            assert_eq!(
                (report.answer.lower, report.answer.upper),
                (Some(6), Some(6))
            );
        }
    }
}

#[test]
fn serialized_packet_budget_includes_claims_and_has_exact_boundaries() {
    let (sources, claims) = three_events();
    let (_directory, store) = persistent_store(&sources, &claims);
    let reference = answer_with_repair(&store, "anchorneedle", &sum_query(), limits()).unwrap();
    let exact = reference.packet_bytes;
    assert!(exact > reference.source_bytes);
    for max_packet_bytes in [exact - 1, exact, exact + 1] {
        let report = answer_with_repair(
            &store,
            "anchorneedle",
            &sum_query(),
            RetrievalLimits {
                max_packet_bytes,
                ..limits()
            },
        )
        .unwrap();
        check_packet(&report);
        if max_packet_bytes < exact {
            assert_eq!(report.answer.status, AnswerStatus::Unresolved);
            assert_eq!((report.answer.lower, report.answer.upper), (None, None));
            assert!(report.answer.gaps.contains(&Gap::EvidenceBudget));
            assert!(report
                .steps
                .iter()
                .any(|step| !step.skipped_for_budget.is_empty()));
        } else {
            assert_eq!(report.packet_bytes, exact);
            assert_eq!(report.answer.status, AnswerStatus::Resolved);
            assert_eq!(
                (report.answer.lower, report.answer.upper),
                (Some(6), Some(6))
            );
        }
    }
}

#[test]
fn round_limit_bounds_progress_and_incomplete_repair_never_returns_a_total() {
    let mut sources = vec![source("anchor", "anchorneedle first event")];
    sources.extend((1..5).map(|i| {
        source(
            &format!("other-{i}"),
            &format!("Receipt for separate event {i}."),
        )
    }));
    let claims: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(i, source)| {
            claim(
                &format!("c{i}"),
                &format!("e{i}"),
                Some((i + 1) as i64),
                source,
            )
        })
        .collect();
    let (_directory, store) = persistent_store(&sources, &claims);
    for max_rounds in [1, 2, 4, 5, 6, 8] {
        let report = answer_with_repair(
            &store,
            "anchorneedle",
            &sum_query(),
            RetrievalLimits {
                max_rounds,
                repair_k: 1,
                ..limits()
            },
        )
        .unwrap();
        check_packet(&report);
        assert_eq!(report.steps.len(), max_rounds.min(5));
        assert_eq!(report.sources.len(), max_rounds.min(5));
        if max_rounds < 5 {
            assert_eq!(report.answer.status, AnswerStatus::Unresolved);
            assert_eq!((report.answer.lower, report.answer.upper), (None, None));
            assert!(report.answer.gaps.contains(&Gap::EvidenceBudget));
        } else if max_rounds >= 5 {
            assert_eq!(report.answer.status, AnswerStatus::Resolved);
            assert_eq!(
                (report.answer.lower, report.answer.upper),
                (Some(15), Some(15))
            );
        }
    }
}

#[test]
fn evidence_and_supersession_dependency_group_is_admitted_atomically() {
    let sources = vec![
        source("owner", "anchorneedle corrected receipt"),
        source("premise", "The venue confirmed the corrected amount."),
        source("previous", "The original receipt amount."),
    ];
    let previous = claim("old", "same-event", Some(100), &sources[2]);
    let mut corrected = claim("new", "same-event", Some(150), &sources[0]);
    corrected.evidence.push(span(&sources[1]));
    corrected.supersedes.push(previous.id.clone());
    let exact: usize = sources.iter().map(|source| source.text.len()).sum();
    let (_directory, store) = persistent_store(&sources, &[previous, corrected]);
    let serialized_exact = answer_with_repair(
        &store,
        "anchorneedle",
        &sum_query(),
        RetrievalLimits {
            max_rounds: 1,
            ..limits()
        },
    )
    .unwrap()
    .packet_bytes;
    for (max_sources, max_source_bytes, max_packet_bytes, fits) in [
        (2, exact, serialized_exact, false),
        (3, exact - 1, serialized_exact, false),
        (3, exact, serialized_exact - 1, false),
        (3, exact, serialized_exact, true),
        (4, exact + 1, serialized_exact + 1, true),
    ] {
        let report = answer_with_repair(
            &store,
            "anchorneedle",
            &sum_query(),
            RetrievalLimits {
                max_sources,
                max_source_bytes,
                max_packet_bytes,
                max_rounds: 1,
                ..limits()
            },
        )
        .unwrap();
        check_packet(&report);
        if fits {
            assert_eq!(report.sources.len(), 3);
            assert_eq!(report.claims.len(), 2);
            assert_eq!(report.steps[0].admitted.len(), 3);
            assert_eq!(report.answer.status, AnswerStatus::Resolved);
            assert_eq!(
                (report.answer.lower, report.answer.upper),
                (Some(150), Some(150))
            );
        } else {
            assert!(report.sources.is_empty());
            assert!(report.claims.is_empty());
            assert!(report.steps[0].admitted.is_empty());
            assert!(report.steps[0].skipped_for_budget.contains(&"owner".into()));
            assert_eq!(report.answer.status, AnswerStatus::Unresolved);
            assert!(report.answer.gaps.contains(&Gap::EvidenceBudget));
        }
    }
}

#[test]
fn hydrated_context_still_obeys_subject_predicate_modality_and_both_time_axes() {
    let anchor = source(
        "anchor",
        "anchorneedle statements about several people and dates",
    );
    let mut late = source("late", "A later observed receipt.");
    late.observed_at = 30;
    let good = claim("good", "good-event", Some(100), &anchor);
    let mut wrong_role = claim("assistant", "assistant-event", Some(1000), &anchor);
    wrong_role.subject = "assistant:alice".into();
    let mut homonym = claim("homonym", "homonym-event", Some(1000), &anchor);
    homonym.subject = "person:bob".into();
    let mut wrong_predicate = claim("attended", "conference", Some(1000), &anchor);
    wrong_predicate.predicate = "attended".into();
    let mut planned = claim("planned", "planned-event", Some(1000), &anchor);
    planned.modality = Modality::Planned;
    let mut outside = claim("outside", "outside-event", Some(1000), &anchor);
    outside.event_time = Some(Interval { start: 20, end: 30 });
    let mut expired = claim("expired", "expired-event", Some(1000), &anchor);
    expired.valid_to = Some(100);
    let mut future = claim("future", "future-event", Some(1000), &anchor);
    future.valid_from = 101;
    let late_claim = claim("late-knowledge", "late-event", Some(1000), &late);
    let claims = vec![
        good,
        wrong_role,
        homonym,
        wrong_predicate,
        planned,
        outside,
        expired,
        future,
        late_claim,
    ];
    let (_directory, store) = persistent_store(&[anchor, late], &claims);
    let mut query = sum_query();
    query.selection.event_window = Some(Interval { start: 10, end: 20 });
    query.known_at = Some(20);
    let report = answer_with_repair(&store, "anchorneedle", &query, limits()).unwrap();
    assert_eq!(report.sources.len(), 2);
    assert_eq!(report.claims.len(), claims.len());
    assert_eq!(report.answer.status, AnswerStatus::Resolved);
    assert_eq!(
        (report.answer.lower, report.answer.upper),
        (Some(100), Some(100))
    );
    assert!(report.answer.claim_ids.contains(&"good".into()));
    for excluded in [
        "assistant",
        "homonym",
        "attended",
        "planned",
        "expired",
        "future",
        "late-knowledge",
    ] {
        assert!(!report.answer.claim_ids.iter().any(|id| id == excluded));
    }
    check_packet(&report);
}

#[test]
fn namespaced_retrieval_keeps_identical_source_and_claim_ids_separate_on_reopen() {
    let directory = tempfile::tempdir().unwrap();
    for (namespace, amount, text) in [
        ("tenant:a", 11, "anchorneedle private first receipt"),
        ("tenant:a:b", 999, "anchorneedle private second receipt"),
    ] {
        let source = source("same-source", text);
        let mut store = EvidenceStore::open(directory.path(), namespace).unwrap();
        store
            .put(
                std::slice::from_ref(&source),
                &[claim("same-claim", "same-event", Some(amount), &source)],
            )
            .unwrap();
    }
    for (namespace, amount, text) in [
        ("tenant:a", 11, "anchorneedle private first receipt"),
        ("tenant:a:b", 999, "anchorneedle private second receipt"),
    ] {
        let store = EvidenceStore::open(directory.path(), namespace).unwrap();
        let report = answer_with_repair(&store, "anchorneedle", &sum_query(), limits()).unwrap();
        assert_eq!(report.sources.len(), 1);
        assert_eq!(report.sources[0].text, text);
        assert_eq!(report.claims.len(), 1);
        assert_eq!(report.answer.status, AnswerStatus::Resolved);
        assert_eq!(
            (report.answer.lower, report.answer.upper),
            (Some(amount), Some(amount))
        );
        check_packet(&report);
    }
}

#[test]
fn invalid_retrieval_limits_and_terms_fail_before_search() {
    let (sources, claims) = three_events();
    let (_directory, store) = persistent_store(&sources, &claims);
    for invalid in [
        RetrievalLimits {
            max_sources: 0,
            ..limits()
        },
        RetrievalLimits {
            max_sources: MAX_SOURCES + 1,
            ..limits()
        },
        RetrievalLimits {
            max_source_bytes: 0,
            ..limits()
        },
        RetrievalLimits {
            max_source_bytes: MAX_TOTAL_BYTES + 1,
            ..limits()
        },
        RetrievalLimits {
            max_packet_bytes: 0,
            ..limits()
        },
        RetrievalLimits {
            max_packet_bytes: MAX_TOTAL_BYTES + 1,
            ..limits()
        },
        RetrievalLimits {
            max_rounds: 0,
            ..limits()
        },
        RetrievalLimits {
            max_rounds: 9,
            ..limits()
        },
        RetrievalLimits {
            initial_k: 0,
            ..limits()
        },
        RetrievalLimits {
            initial_k: 1001,
            ..limits()
        },
        RetrievalLimits {
            repair_k: 0,
            ..limits()
        },
        RetrievalLimits {
            repair_k: MAX_SOURCES + 1,
            ..limits()
        },
    ] {
        assert!(answer_with_repair(&store, "anchorneedle", &sum_query(), invalid).is_err());
    }
    for terms in ["".to_owned(), " \n\t".to_owned(), "x".repeat(8193)] {
        assert!(answer_with_repair(&store, &terms, &sum_query(), limits()).is_err());
    }
}

#[test]
fn serialized_budget_always_accounts_for_the_empty_packet_envelope() {
    let (sources, claims) = three_events();
    let (_directory, store) = persistent_store(&sources, &claims);
    for bytes in [
        EMPTY_PACKET_BYTES - 1,
        EMPTY_PACKET_BYTES,
        EMPTY_PACKET_BYTES + 1,
    ] {
        let result = answer_with_repair(
            &store,
            "anchorneedle",
            &sum_query(),
            RetrievalLimits {
                max_packet_bytes: bytes,
                ..limits()
            },
        );
        if bytes < EMPTY_PACKET_BYTES {
            assert!(result.is_err());
        } else {
            let report = result.unwrap();
            assert!(report.sources.is_empty());
            assert_eq!(report.packet_bytes, EMPTY_PACKET_BYTES);
            assert_eq!(report.answer.status, AnswerStatus::Unresolved);
            assert!(report.answer.gaps.contains(&Gap::EvidenceBudget));
            check_packet(&report);
        }
    }
}
