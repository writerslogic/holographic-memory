// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use warm_research_evidence::contract::{
    Amount, AnswerStatus, Claim, Gap, Interval, Modality, Operation, Query, Selection, Source,
    Span, MAX_CLAIMS, MAX_SOURCES, MAX_SOURCE_BYTES, MAX_TOTAL_BYTES, MAX_WORLDS,
};
use warm_research_evidence::execute::execute;

fn source(id: &str, observed_at: i64) -> Source {
    Source {
        id: id.into(),
        version: "v1".into(),
        text: "assertion λ".into(),
        observed_at,
    }
}

fn claim(id: &str, event: &str, source: &Source) -> Claim {
    Claim {
        id: id.into(),
        subject: "person:alex:1".into(),
        predicate: "paid_for".into(),
        event_key: event.into(),
        event_candidates: vec![event.into()],
        value: Some(Amount {
            minor: 100,
            unit: "USD_cent".into(),
        }),
        event_time: Some(Interval { start: 10, end: 20 }),
        valid_from: 0,
        valid_to: None,
        positive: true,
        modality: Modality::Actual,
        evidence: vec![Span {
            source_id: source.id.clone(),
            source_version: source.version.clone(),
            start: 0,
            end: source.text.len(),
            quote: source.text.clone(),
        }],
        supersedes: Vec::new(),
    }
}

fn query(operation: Operation) -> Query {
    Query {
        selection: Selection {
            subject: "person:alex:1".into(),
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

#[test]
fn selection_binds_entity_relation_and_modality_before_aggregation() {
    let source = source("record", 1000);
    let actual = claim("actual", "purchase", &source);
    let mut homonym = claim("homonym", "other-purchase", &source);
    homonym.subject = "person:alex:2".into();
    let mut suggestion = claim("suggestion", "suggested-purchase", &source);
    suggestion.subject = "assistant:alex:1".into();
    let mut attended = claim("attended", "conference", &source);
    attended.predicate = "attended".into();
    let mut planned = claim("planned", "future-purchase", &source);
    planned.modality = Modality::Planned;
    let claims = vec![actual, homonym, suggestion, attended, planned];
    let answer = execute(&sum_query(), &claims, &[source]).unwrap();
    assert_eq!(answer.status, AnswerStatus::Resolved);
    assert_eq!((answer.lower, answer.upper), (Some(100), Some(100)));
    assert_eq!(answer.claim_ids, ["actual"]);
    assert!(answer.observed_only);
}

#[test]
fn repeated_mentions_and_explicit_aliases_are_counted_and_summed_once() {
    let source = source("record", 0);
    let first = claim("first", "same-event", &source);
    let duplicate = claim("duplicate", "same-event", &source);
    let mut alias = claim("alias", "different-mention-key", &source);
    alias.event_candidates = vec!["same-event".into()];
    let mut unvalued = claim("unvalued", "same-event", &source);
    unvalued.value = None;
    let claims = vec![first, duplicate, alias, unvalued];
    let count = execute(
        &query(Operation::CountDistinct),
        &claims,
        std::slice::from_ref(&source),
    )
    .unwrap();
    let sum = execute(&sum_query(), &claims, &[source]).unwrap();
    assert_eq!(count.lower, Some(1));
    assert_eq!(sum.lower, Some(100));
    assert_eq!(sum.status, AnswerStatus::Resolved);
    assert_eq!(sum.claim_ids.len(), 4);
    assert_eq!(sum.evidence.len(), 1);
}

#[test]
fn missing_values_are_unknown_while_an_explicit_zero_is_resolved() {
    let source = source("record", 0);
    let mut missing = claim("missing", "event", &source);
    missing.value = None;
    let answer = execute(
        &sum_query(),
        &[missing.clone()],
        std::slice::from_ref(&source),
    )
    .unwrap();
    assert_eq!(answer.status, AnswerStatus::Unresolved);
    assert_eq!((answer.lower, answer.upper), (None, None));
    assert!(answer.gaps.contains(&Gap::MissingValue {
        claim_id: "missing".into()
    }));
    assert_eq!(
        execute(
            &query(Operation::CountDistinct),
            &[missing],
            std::slice::from_ref(&source)
        )
        .unwrap()
        .lower,
        Some(1)
    );
    let mut zero = claim("zero", "event", &source);
    zero.value.as_mut().unwrap().minor = 0;
    let answer = execute(&sum_query(), &[zero], &[source]).unwrap();
    assert_eq!(answer.status, AnswerStatus::Resolved);
    assert_eq!((answer.lower, answer.upper), (Some(0), Some(0)));
}

#[test]
fn conflicting_amounts_and_incompatible_units_do_not_produce_a_total() {
    let source = source("record", 0);
    let first = claim("first", "event", &source);
    for value in [
        Amount {
            minor: 101,
            unit: "USD_cent".into(),
        },
        Amount {
            minor: 100,
            unit: "EUR_cent".into(),
        },
    ] {
        let mut other = claim("other", "event", &source);
        other.value = Some(value);
        let answer = execute(
            &sum_query(),
            &[first.clone(), other],
            std::slice::from_ref(&source),
        )
        .unwrap();
        assert_eq!(answer.status, AnswerStatus::Unresolved);
        assert_eq!(answer.lower, None);
        assert!(answer.gaps.contains(&Gap::ConflictingValues {
            event_id: "event".into()
        }));
    }
    let mut wrong_unit = first;
    wrong_unit.value.as_mut().unwrap().unit = "EUR_cent".into();
    let answer = execute(&sum_query(), &[wrong_unit], &[source]).unwrap();
    assert_eq!(answer.status, AnswerStatus::Unresolved);
    assert!(answer.gaps.contains(&Gap::MissingValue {
        claim_id: "first".into()
    }));
}

#[test]
fn identity_alternatives_bound_counts_without_forcing_an_alias() {
    let source = source("record", 0);
    let first = claim("first", "event-a", &source);
    let mut ambiguous = claim("ambiguous", "mention-b", &source);
    ambiguous.event_candidates = vec!["event-a".into(), "event-b".into()];
    let claims = vec![first, ambiguous];
    for (operation, expected) in [
        (Operation::CountDistinct, (1, 2)),
        (
            Operation::Sum {
                unit: "USD_cent".into(),
            },
            (100, 200),
        ),
    ] {
        let answer = execute(&query(operation), &claims, std::slice::from_ref(&source)).unwrap();
        assert_eq!(answer.status, AnswerStatus::Bounded);
        assert_eq!(
            (answer.lower, answer.upper),
            (Some(expected.0), Some(expected.1))
        );
        assert_eq!(answer.worlds, 2);
    }
}

#[test]
fn shared_mention_keys_share_one_identity_choice() {
    let source = source("record", 0);
    let mut first = claim("first", "mention", &source);
    first.event_candidates = vec!["a".into(), "b".into()];
    let mut repeated = first.clone();
    repeated.id = "repeated".into();
    repeated.event_candidates.reverse();
    let answer = execute(
        &query(Operation::CountDistinct),
        &[first, repeated],
        &[source],
    )
    .unwrap();
    assert_eq!(answer.status, AnswerStatus::Resolved);
    assert_eq!((answer.lower, answer.upper), (Some(1), Some(1)));
    assert_eq!(answer.worlds, 2);
}

#[test]
fn all_identity_worlds_are_required_even_if_the_partial_results_agree() {
    let source = source("record", 0);
    let mut claims = Vec::new();
    for (index, alternatives) in [8, 8, 4].into_iter().enumerate() {
        let mut claim = claim(&format!("claim-{index}"), &format!("key-{index}"), &source);
        claim.event_candidates = (0..alternatives)
            .map(|candidate| format!("event-{index}-{candidate}"))
            .collect();
        claims.push(claim);
    }
    for limit in [MAX_WORLDS - 1, MAX_WORLDS] {
        let mut query = query(Operation::CountDistinct);
        query.max_worlds = limit;
        let answer = execute(&query, &claims, std::slice::from_ref(&source)).unwrap();
        assert_eq!(answer.worlds, limit);
        if limit == MAX_WORLDS {
            assert_eq!(answer.status, AnswerStatus::Resolved);
            assert_eq!(answer.lower, Some(3));
        } else {
            assert_eq!(answer.status, AnswerStatus::Unresolved);
            assert_eq!((answer.lower, answer.upper), (None, None));
            assert!(answer.gaps.contains(&Gap::IdentityBudget));
        }
    }
    for limit in [0, MAX_WORLDS + 1] {
        let mut query = query(Operation::CountDistinct);
        query.max_worlds = limit;
        assert!(execute(&query, &claims, std::slice::from_ref(&source)).is_err());
    }
}

#[test]
fn observation_cutoff_and_validity_use_different_clocks_and_exact_boundaries() {
    let early = source("early", 100);
    let late = source("late", 200);
    let mut old = claim("old", "event", &early);
    old.valid_from = 50;
    old.valid_to = Some(150);
    let mut new = claim("new", "event", &late);
    new.valid_from = 150;
    new.value.as_mut().unwrap().minor = 200;
    let claims = vec![old, new];
    let sources = [early, late];
    for (as_of, expected) in [
        (49, None),
        (50, Some(100)),
        (149, Some(100)),
        (150, Some(200)),
        (151, Some(200)),
    ] {
        let mut query = sum_query();
        query.as_of = as_of;
        assert_eq!(execute(&query, &claims, &sources).unwrap().lower, expected);
    }
    for (cutoff, expected) in [(199, None), (200, Some(200)), (201, Some(200))] {
        let mut query = sum_query();
        query.as_of = 150;
        query.known_at = Some(cutoff);
        assert_eq!(execute(&query, &claims, &sources).unwrap().lower, expected);
    }
    let mut query = sum_query();
    query.as_of = 150;
    query.known_at = None;
    assert_eq!(
        execute(&query, &claims, &sources).unwrap().claim_ids,
        ["new"]
    );
}

#[test]
fn a_multi_source_assertion_is_hidden_until_every_source_is_visible() {
    let early = source("early", 10);
    let late = source("late", 20);
    let mut assertion = claim("assertion", "event", &early);
    assertion
        .evidence
        .extend(claim("other", "event", &late).evidence);
    let mut query = sum_query();
    query.known_at = Some(19);
    let hidden = execute(&query, &[assertion.clone()], &[early.clone(), late.clone()]).unwrap();
    assert_eq!(hidden.status, AnswerStatus::Unresolved);
    assert!(hidden.claim_ids.is_empty());
    query.known_at = Some(20);
    assert_eq!(
        execute(&query, &[assertion], &[early, late]).unwrap().lower,
        Some(100)
    );
}

#[test]
fn event_windows_exclude_disjoint_events_and_bound_partial_or_unknown_time() {
    let source = source("record", 1000);
    let mut query = query(Operation::CountDistinct);
    query.selection.event_window = Some(Interval { start: 10, end: 20 });
    for (time, status, lower, upper, worlds) in [
        (
            Some(Interval { start: 9, end: 10 }),
            AnswerStatus::Unresolved,
            None,
            None,
            1,
        ),
        (
            Some(Interval { start: 10, end: 11 }),
            AnswerStatus::Resolved,
            Some(1),
            Some(1),
            1,
        ),
        (
            Some(Interval { start: 10, end: 20 }),
            AnswerStatus::Resolved,
            Some(1),
            Some(1),
            1,
        ),
        (
            Some(Interval { start: 19, end: 20 }),
            AnswerStatus::Resolved,
            Some(1),
            Some(1),
            1,
        ),
        (
            Some(Interval { start: 20, end: 21 }),
            AnswerStatus::Unresolved,
            None,
            None,
            1,
        ),
        (
            Some(Interval { start: 9, end: 11 }),
            AnswerStatus::Bounded,
            Some(0),
            Some(1),
            2,
        ),
        (
            Some(Interval { start: 19, end: 21 }),
            AnswerStatus::Bounded,
            Some(0),
            Some(1),
            2,
        ),
        (None, AnswerStatus::Bounded, Some(0), Some(1), 2),
    ] {
        let mut assertion = claim("assertion", "event", &source);
        assertion.event_time = time;
        let answer = execute(&query, &[assertion], std::slice::from_ref(&source)).unwrap();
        assert_eq!(
            (answer.status, answer.lower, answer.upper, answer.worlds),
            (status, lower, upper, worlds)
        );
        if worlds == 2 {
            assert!(answer.gaps.contains(&Gap::UnknownTime {
                claim_id: "assertion".into()
            }));
        }
    }
    let mut unknown = claim("unknown", "event", &source);
    unknown.event_time = None;
    query.max_worlds = 1;
    let answer = execute(&query, &[unknown], &[source]).unwrap();
    assert_eq!(answer.status, AnswerStatus::Unresolved);
    assert_eq!(answer.lower, None);
    assert!(answer.gaps.contains(&Gap::IdentityBudget));
}

#[test]
fn incompatible_times_for_one_assigned_event_cannot_create_an_exact_count() {
    let source = source("record", 0);
    let inside = claim("inside", "event", &source);
    let mut outside = claim("outside", "event", &source);
    outside.event_time = Some(Interval { start: 30, end: 40 });
    let mut query = query(Operation::CountDistinct);
    query.selection.event_window = Some(Interval { start: 10, end: 20 });
    let answer = execute(&query, &[inside, outside], &[source]).unwrap();
    assert_eq!(answer.status, AnswerStatus::Unresolved);
    assert_eq!(answer.lower, None);
    assert!(answer.gaps.contains(&Gap::UnknownTime {
        claim_id: "inside".into()
    }));
}

#[test]
fn identity_and_temporal_choices_share_one_world_budget() {
    let source = source("record", 0);
    let first = claim("first", "a", &source);
    let mut other = claim("other", "mention", &source);
    other.event_candidates = vec!["b".into(), "c".into()];
    other.event_time = None;
    let claims = [first, other];
    for limit in [3, 4] {
        let mut query = query(Operation::CountDistinct);
        query.selection.event_window = Some(Interval { start: 10, end: 20 });
        query.max_worlds = limit;
        let answer = execute(&query, &claims, std::slice::from_ref(&source)).unwrap();
        assert_eq!(answer.worlds, limit);
        if limit == 4 {
            assert_eq!(answer.status, AnswerStatus::Bounded);
            assert_eq!((answer.lower, answer.upper), (Some(1), Some(2)));
        } else {
            assert_eq!(answer.status, AnswerStatus::Unresolved);
            assert_eq!(answer.lower, None);
            assert!(answer.gaps.contains(&Gap::IdentityBudget));
        }
    }
}

#[test]
fn scoped_negation_does_not_erase_other_events_or_become_a_global_zero() {
    let source = source("record", 0);
    let positive = claim("positive", "event-a", &source);
    let mut negative = claim("negative", "event-b", &source);
    negative.positive = false;
    let answer = execute(
        &sum_query(),
        &[positive.clone(), negative.clone()],
        std::slice::from_ref(&source),
    )
    .unwrap();
    assert_eq!(answer.status, AnswerStatus::Resolved);
    assert_eq!(answer.lower, Some(100));
    assert!(answer.gaps.contains(&Gap::CounterEvidence {
        claim_id: "negative".into()
    }));
    let answer = execute(
        &query(Operation::CountDistinct),
        &[negative.clone()],
        std::slice::from_ref(&source),
    )
    .unwrap();
    assert_eq!(answer.status, AnswerStatus::Unresolved);
    assert_eq!(answer.lower, None);
    negative.event_key = "event-a".into();
    negative.event_candidates = vec!["event-a".into()];
    let answer = execute(&sum_query(), &[positive, negative], &[source]).unwrap();
    assert_eq!(answer.status, AnswerStatus::Unresolved);
    assert_eq!(answer.lower, None);
}

#[test]
fn explicit_supersession_is_visible_valid_scoped_and_transitive() {
    let early = source("early", 10);
    let late = source("late", 30);
    let old = claim("old", "event", &early);
    let mut middle = claim("middle", "event", &early);
    middle.valid_from = 10;
    middle.valid_to = Some(20);
    middle.value.as_mut().unwrap().minor = 200;
    middle.supersedes = vec!["old".into()];
    let mut new = claim("new", "new-mention", &late);
    new.event_candidates = vec!["event".into()];
    new.valid_from = 20;
    new.value.as_mut().unwrap().minor = 300;
    new.supersedes = vec!["middle".into()];
    let claims = [old, middle, new];
    let sources = [early, late];
    for (as_of, expected) in [(9, 100), (10, 200), (19, 200), (20, 300), (21, 300)] {
        let mut query = sum_query();
        query.as_of = as_of;
        let answer = execute(&query, &claims, &sources).unwrap();
        assert_eq!(answer.lower, Some(expected));
    }
    let mut query = sum_query();
    query.as_of = 20;
    query.known_at = Some(29);
    assert_eq!(execute(&query, &claims, &sources).unwrap().lower, Some(100));
    query.known_at = Some(30);
    let answer = execute(&query, &claims, &sources).unwrap();
    assert_eq!(answer.lower, Some(300));
    assert_eq!(answer.claim_ids, ["middle", "new", "old"]);
}

#[test]
fn malformed_or_ambiguous_supersession_is_rejected_before_execution() {
    let source = source("record", 0);
    let first = claim("first", "event", &source);
    let mut second = claim("second", "event", &source);
    second.supersedes = vec!["first".into()];
    for change in 0..6 {
        let mut first = first.clone();
        let mut second = second.clone();
        match change {
            0 => first.supersedes = vec!["second".into()],
            1 => second.subject = "another-person".into(),
            2 => second.predicate = "attended".into(),
            3 => second.modality = Modality::Planned,
            4 => second.supersedes = vec!["absent".into()],
            5 => {
                second.event_key = "ambiguous-key".into();
                second.event_candidates = vec!["event".into(), "other-event".into()];
            }
            _ => unreachable!(),
        }
        assert!(execute(
            &sum_query(),
            &[first, second],
            std::slice::from_ref(&source)
        )
        .is_err());
    }
}

#[test]
fn a_negative_explicit_correction_does_not_assert_a_global_zero() {
    let source = source("record", 0);
    let positive = claim("positive", "event", &source);
    let mut correction = claim("correction", "event", &source);
    correction.positive = false;
    correction.supersedes = vec!["positive".into()];
    let answer = execute(
        &query(Operation::CountDistinct),
        &[positive, correction],
        &[source],
    )
    .unwrap();
    assert_eq!(answer.status, AnswerStatus::Unresolved);
    assert_eq!(answer.lower, None);
    assert_eq!(answer.claim_ids, ["correction", "positive"]);
}

#[test]
fn comparison_uses_named_event_operands_and_returns_ordering_without_subtraction() {
    let source = source("record", 0);
    for (left, right, ordering) in [
        (i64::MIN, i64::MAX, -1),
        (100, 100, 0),
        (i64::MAX, i64::MIN, 1),
    ] {
        let mut a = claim("left", "a", &source);
        a.value.as_mut().unwrap().minor = left;
        let mut b = claim("right", "b", &source);
        b.value.as_mut().unwrap().minor = right;
        let query = query(Operation::Compare {
            left_event: "a".into(),
            right_event: "b".into(),
            unit: "USD_cent".into(),
        });
        let answer = execute(&query, &[a.clone(), b], std::slice::from_ref(&source)).unwrap();
        assert_eq!(answer.status, AnswerStatus::Resolved);
        assert_eq!(
            (answer.lower, answer.upper),
            (Some(ordering), Some(ordering))
        );
        assert_eq!(answer.unit, "ordering");
        let answer = execute(&query, &[a], std::slice::from_ref(&source)).unwrap();
        assert_eq!(answer.status, AnswerStatus::Unresolved);
        assert!(answer.gaps.contains(&Gap::MissingEvent {
            event_id: "b".into()
        }));
    }
}

#[test]
fn missing_comparison_identity_in_one_world_prevents_an_exact_ordering() {
    let source = source("record", 0);
    let first = claim("first", "a", &source);
    let mut other = claim("other", "mention-b", &source);
    other.event_candidates = vec!["a".into(), "b".into()];
    let query = query(Operation::Compare {
        left_event: "a".into(),
        right_event: "b".into(),
        unit: "USD_cent".into(),
    });
    let answer = execute(&query, &[first, other], &[source]).unwrap();
    assert_eq!(answer.status, AnswerStatus::Unresolved);
    assert_eq!(answer.lower, None);
    assert!(answer.gaps.contains(&Gap::MissingEvent {
        event_id: "b".into()
    }));
}

#[test]
fn unresolved_aliases_do_not_block_an_ordering_that_is_invariant_across_worlds() {
    let source = source("record", 0);
    let first = claim("first", "a", &source);
    let mut second = claim("second", "b", &source);
    second.value.as_mut().unwrap().minor = 200;
    let mut ambiguous = claim("ambiguous", "mention", &source);
    ambiguous.event_candidates = vec!["a".into(), "unrelated".into()];
    let query = query(Operation::Compare {
        left_event: "a".into(),
        right_event: "b".into(),
        unit: "USD_cent".into(),
    });
    let answer = execute(&query, &[first, second, ambiguous], &[source]).unwrap();
    assert_eq!(answer.status, AnswerStatus::Resolved);
    assert_eq!(answer.lower, Some(-1));
    assert_eq!(answer.worlds, 2);
}

#[test]
fn sums_check_representable_minor_units_but_allow_exact_cancellation() {
    let source = source("record", 0);
    for (numbers, expected) in [
        (vec![i64::MAX - 1, 1], Some(i64::MAX)),
        (vec![i64::MAX, 1], None),
        (vec![i64::MIN + 1, -1], Some(i64::MIN)),
        (vec![i64::MIN, -1], None),
        (vec![i64::MAX, 1, -1], Some(i64::MAX)),
    ] {
        let claims: Vec<_> = numbers
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let mut claim = claim(
                    &format!("claim-{index}"),
                    &format!("event-{index}"),
                    &source,
                );
                claim.value.as_mut().unwrap().minor = *value;
                claim
            })
            .collect();
        let answer = execute(&sum_query(), &claims, std::slice::from_ref(&source));
        match expected {
            Some(expected) => assert_eq!(answer.unwrap().lower, Some(expected)),
            None => assert!(answer.is_err()),
        }
    }
}

#[test]
fn generated_reordering_duplication_and_irrelevant_growth_preserve_results() {
    let source = source("record", 0);
    for count in 1..=24 {
        let mut claims: Vec<_> = (0..count)
            .map(|index| {
                let mut claim = claim(
                    &format!("claim-{index}"),
                    &format!("event-{index}"),
                    &source,
                );
                claim.value.as_mut().unwrap().minor = index * 17 - 91;
                claim
            })
            .collect();
        let expected = execute(&sum_query(), &claims, std::slice::from_ref(&source)).unwrap();
        claims.reverse();
        assert_eq!(
            execute(&sum_query(), &claims, std::slice::from_ref(&source)).unwrap(),
            expected
        );
        let mut duplicate = claims[0].clone();
        duplicate.id = "repeated-observation".into();
        claims.push(duplicate);
        let mut irrelevant = claim("irrelevant", "unrelated-event", &source);
        irrelevant.subject = "other-person".into();
        claims.push(irrelevant);
        let actual = execute(&sum_query(), &claims, std::slice::from_ref(&source)).unwrap();
        assert_eq!(
            (actual.status, actual.lower, actual.upper, actual.worlds),
            (
                expected.status,
                expected.lower,
                expected.upper,
                expected.worlds
            )
        );
        assert!(!actual.claim_ids.contains(&"irrelevant".to_owned()));
    }
}

#[test]
fn empty_evidence_never_claims_a_zero_count_or_sum() {
    for operation in [
        Operation::CountDistinct,
        Operation::Sum {
            unit: "USD_cent".into(),
        },
    ] {
        let answer = execute(&query(operation), &[], &[]).unwrap();
        assert_eq!(answer.status, AnswerStatus::Unresolved);
        assert_eq!((answer.lower, answer.upper), (None, None));
        assert!(answer.gaps.contains(&Gap::NoMatchingEvidence));
        assert!(answer.observed_only);
    }
}

#[test]
fn source_and_claim_limits_are_checked_at_minus_one_exactly_and_plus_one() {
    let source = source("record", 0);
    for count in [MAX_CLAIMS - 1, MAX_CLAIMS, MAX_CLAIMS + 1] {
        let claims: Vec<_> = (0..count)
            .map(|index| {
                claim(
                    &format!("claim-{index}"),
                    &format!("event-{index}"),
                    &source,
                )
            })
            .collect();
        let answer = execute(
            &query(Operation::CountDistinct),
            &claims,
            std::slice::from_ref(&source),
        );
        if count <= MAX_CLAIMS {
            assert_eq!(answer.unwrap().lower, Some(count as i64));
        } else {
            assert!(answer.is_err());
        }
    }
    for count in [MAX_SOURCES - 1, MAX_SOURCES, MAX_SOURCES + 1] {
        let sources: Vec<_> = (0..count).map(|index| source_fixture(index, 1)).collect();
        let assertion = claim("assertion", "event", &sources[0]);
        assert_eq!(
            execute(&sum_query(), &[assertion], &sources).is_ok(),
            count <= MAX_SOURCES
        );
    }
    for bytes in [MAX_SOURCE_BYTES - 1, MAX_SOURCE_BYTES, MAX_SOURCE_BYTES + 1] {
        let source = source_fixture(0, bytes);
        let assertion = claim("assertion", "event", &source);
        assert_eq!(
            execute(&sum_query(), &[assertion], &[source]).is_ok(),
            bytes <= MAX_SOURCE_BYTES
        );
    }
    for bytes in [MAX_TOTAL_BYTES - 1, MAX_TOTAL_BYTES, MAX_TOTAL_BYTES + 1] {
        let sources: Vec<_> = (0..bytes.div_ceil(MAX_SOURCE_BYTES))
            .map(|index| {
                source_fixture(
                    index,
                    (bytes - index * MAX_SOURCE_BYTES).min(MAX_SOURCE_BYTES),
                )
            })
            .collect();
        let assertion = claim("assertion", "event", &sources[0]);
        assert_eq!(
            execute(&sum_query(), &[assertion], &sources).is_ok(),
            bytes <= MAX_TOTAL_BYTES
        );
    }
}

fn source_fixture(index: usize, bytes: usize) -> Source {
    Source {
        id: format!("source-{index}"),
        version: "v1".into(),
        text: "x".repeat(bytes),
        observed_at: 0,
    }
}

#[test]
fn forged_stale_or_non_character_boundary_spans_are_rejected() {
    let source = source("record", 0);
    for change in 0..4 {
        let mut assertion = claim("assertion", "event", &source);
        match change {
            0 => assertion.evidence[0].quote.push('!'),
            1 => assertion.evidence[0].source_version = "v0".into(),
            2 => assertion.evidence[0].source_id = "absent".into(),
            3 => assertion.evidence[0].end -= 1,
            _ => unreachable!(),
        }
        assert!(execute(&sum_query(), &[assertion], std::slice::from_ref(&source)).is_err());
    }
}
