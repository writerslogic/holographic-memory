// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::contract::{
    validate_claims, validate_query, Answer, AnswerStatus, Claim, Gap, Operation, Query, Source,
};
use anyhow::{ensure, Context, Result};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Membership {
    Included,
    Excluded,
    Uncertain,
}

struct Event<'a> {
    id: &'a str,
    claims: Vec<&'a Claim>,
    membership: Membership,
    inconsistent_time: bool,
}

pub fn execute(query: &Query, claims: &[Claim], sources: &[Source]) -> Result<Answer> {
    validate_query(query)?;
    validate_claims(claims, sources)?;
    let links = supersession_links(claims)?;
    let observed: BTreeMap<_, _> = sources
        .iter()
        .map(|source| (source.id.as_str(), source.observed_at))
        .collect();
    let visible: Vec<_> = claims
        .iter()
        .map(|claim| {
            query.known_at.is_none_or(|cutoff| {
                claim
                    .evidence
                    .iter()
                    .all(|span| observed[span.source_id.as_str()] <= cutoff)
            })
        })
        .collect();
    let active: Vec<_> = claims
        .iter()
        .zip(&visible)
        .map(|(claim, visible)| {
            *visible
                && claim.valid_from <= query.as_of
                && claim.valid_to.is_none_or(|end| query.as_of < end)
        })
        .collect();
    let relevant = |claim: &Claim| {
        claim.subject == query.selection.subject
            && claim.predicate == query.selection.predicate
            && claim.modality == query.selection.modality
    };
    let mut suppressed = vec![false; claims.len()];
    let mut traversed = vec![false; claims.len()];
    let mut pending: Vec<_> = (0..claims.len()).filter(|index| active[*index]).collect();
    while let Some(index) = pending.pop() {
        if std::mem::replace(&mut traversed[index], true) {
            continue;
        }
        for target in &links[index] {
            if visible[*target] {
                suppressed[*target] = true;
                pending.push(*target);
            }
        }
    }
    let considered: Vec<_> = claims
        .iter()
        .enumerate()
        .filter(|(index, claim)| relevant(claim) && visible[*index] && traversed[*index])
        .map(|(_, claim)| claim)
        .collect();
    let mut answer = Answer {
        status: AnswerStatus::Unresolved,
        lower: None,
        upper: None,
        unit: match &query.operation {
            Operation::CountDistinct => "count".into(),
            Operation::Sum { unit } => unit.clone(),
            Operation::Compare { .. } => "ordering".into(),
        },
        observed_only: true,
        worlds: 0,
        claim_ids: considered
            .iter()
            .map(|claim| claim.id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        evidence: considered
            .iter()
            .flat_map(|claim| claim.evidence.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        gaps: Vec::new(),
    };
    let selected: Vec<_> = claims
        .iter()
        .enumerate()
        .filter(|(index, claim)| relevant(claim) && active[*index] && !suppressed[*index])
        .map(|(_, claim)| claim)
        .collect();
    let mut identities = BTreeMap::new();
    for claim in &selected {
        identities
            .entry(claim.event_key.as_str())
            .or_insert_with(|| {
                claim
                    .event_candidates
                    .iter()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
            });
    }
    let identities: Vec<_> = identities.into_iter().collect();
    let mut assignment = vec![0; identities.len()];
    let mut gaps = BTreeSet::new();
    let mut values = BTreeSet::new();
    let mut insufficient = false;
    let mut exhausted = false;
    loop {
        let assigned: BTreeMap<_, _> = identities
            .iter()
            .zip(&assignment)
            .map(|((key, candidates), index)| (*key, candidates[*index]))
            .collect();
        let mut groups: BTreeMap<&str, Vec<&Claim>> = BTreeMap::new();
        for claim in &selected {
            groups
                .entry(assigned[claim.event_key.as_str()])
                .or_default()
                .push(claim);
        }
        let events: Vec<_> = groups
            .into_iter()
            .map(|(id, mut group)| {
                group.sort_by(|left, right| left.id.cmp(&right.id));
                event(query, id, group, &mut gaps)
            })
            .collect();
        let uncertain: Vec<_> = events
            .iter()
            .enumerate()
            .filter(|(_, event)| event.membership == Membership::Uncertain)
            .map(|(index, _)| index)
            .collect();
        let mut inclusion = vec![0; uncertain.len()];
        loop {
            if answer.worlds == query.max_worlds {
                exhausted = true;
                break;
            }
            answer.worlds += 1;
            let uncertain_inclusion: BTreeMap<_, _> = uncertain
                .iter()
                .copied()
                .zip(inclusion.iter().map(|value| *value == 1))
                .collect();
            let included: Vec<_> = events
                .iter()
                .enumerate()
                .filter(|(index, event)| match event.membership {
                    Membership::Included => true,
                    Membership::Excluded => false,
                    Membership::Uncertain => uncertain_inclusion[index],
                })
                .map(|(_, event)| event)
                .collect();
            let possible_positive = events.iter().any(|event| {
                event.membership != Membership::Excluded
                    && event.claims.iter().any(|claim| claim.positive)
            });
            match evaluate(query, &included, possible_positive, &mut gaps)? {
                Some(value) => {
                    values.insert(value);
                }
                None => insufficient = true,
            }
            if !advance(&mut inclusion, &vec![2; uncertain.len()]) {
                break;
            }
        }
        if exhausted
            || !advance(
                &mut assignment,
                &identities
                    .iter()
                    .map(|(_, candidates)| candidates.len())
                    .collect::<Vec<_>>(),
            )
        {
            break;
        }
    }
    if exhausted {
        gaps.insert(Gap::IdentityBudget);
    }
    if !exhausted && !insufficient && !values.is_empty() {
        answer.lower = values.first().copied();
        answer.upper = values.last().copied();
        answer.status = if answer.lower == answer.upper {
            AnswerStatus::Resolved
        } else {
            AnswerStatus::Bounded
        };
    }
    answer.gaps = gaps.into_iter().collect();
    Ok(answer)
}

pub(crate) fn supersession_links(claims: &[Claim]) -> Result<Vec<Vec<usize>>> {
    let indices: BTreeMap<_, _> = claims
        .iter()
        .enumerate()
        .map(|(index, claim)| (claim.id.as_str(), index))
        .collect();
    let mut links = vec![Vec::new(); claims.len()];
    let mut indegrees = vec![0usize; claims.len()];
    for (index, claim) in claims.iter().enumerate() {
        for target in &claim.supersedes {
            let target_index = *indices
                .get(target.as_str())
                .context("superseded claim is absent")?;
            let previous = &claims[target_index];
            ensure!(
                claim.subject == previous.subject
                    && claim.predicate == previous.predicate
                    && claim.modality == previous.modality,
                "supersession crosses subject, predicate or modality scope"
            );
            ensure!(
                claim.event_key == previous.event_key
                    || (claim.event_candidates.len() == 1
                        && previous.event_candidates.len() == 1
                        && claim.event_candidates[0] == previous.event_candidates[0]),
                "supersession has incompatible event identity scope"
            );
            links[index].push(target_index);
            indegrees[target_index] += 1;
        }
    }
    let mut roots: BTreeSet<_> = indegrees
        .iter()
        .enumerate()
        .filter(|(_, degree)| **degree == 0)
        .map(|(index, _)| index)
        .collect();
    let mut visited = 0;
    while let Some(index) = roots.pop_first() {
        visited += 1;
        for target in &links[index] {
            indegrees[*target] -= 1;
            if indegrees[*target] == 0 {
                roots.insert(*target);
            }
        }
    }
    ensure!(visited == claims.len(), "supersession cycle");
    Ok(links)
}

fn event<'a>(
    query: &Query,
    id: &'a str,
    claims: Vec<&'a Claim>,
    gaps: &mut BTreeSet<Gap>,
) -> Event<'a> {
    let Some(window) = query.selection.event_window else {
        return Event {
            id,
            claims,
            membership: Membership::Included,
            inconsistent_time: false,
        };
    };
    let mut inside = false;
    let mut outside = false;
    let mut partial = false;
    for claim in &claims {
        match claim.event_time {
            Some(time) if window.contains(time) => inside = true,
            Some(time) if !window.overlaps(time) => outside = true,
            Some(_) => partial = true,
            None => {}
        }
    }
    let inconsistent_time = inside && outside;
    let membership = if inconsistent_time || partial || (!inside && !outside) {
        for claim in &claims {
            gaps.insert(Gap::UnknownTime {
                claim_id: claim.id.clone(),
            });
        }
        Membership::Uncertain
    } else if inside {
        Membership::Included
    } else {
        Membership::Excluded
    };
    Event {
        id,
        claims,
        membership,
        inconsistent_time,
    }
}

fn evaluate(
    query: &Query,
    events: &[&Event<'_>],
    possible_positive: bool,
    gaps: &mut BTreeSet<Gap>,
) -> Result<Option<i64>> {
    let mut invalid = false;
    let mut positive = BTreeMap::new();
    for event in events {
        let affirmations: Vec<_> = event.claims.iter().filter(|claim| claim.positive).collect();
        let negatives: Vec<_> = event
            .claims
            .iter()
            .filter(|claim| !claim.positive)
            .collect();
        for claim in &negatives {
            gaps.insert(Gap::CounterEvidence {
                claim_id: claim.id.clone(),
            });
        }
        if affirmations.is_empty() {
            continue;
        }
        invalid |= !negatives.is_empty() || event.inconsistent_time;
        positive.insert(event.id, affirmations);
    }
    if !possible_positive {
        if !matches!(query.operation, Operation::Compare { .. }) {
            gaps.insert(Gap::NoMatchingEvidence);
        }
        invalid = true;
    }
    let value = match &query.operation {
        Operation::CountDistinct => Some(i64::try_from(positive.len()).context("count overflow")?),
        Operation::Sum { unit } => {
            let mut sum = 0i128;
            for (id, claims) in &positive {
                if let Some(value) = amount(id, claims, unit, gaps) {
                    sum = sum.checked_add(i128::from(value)).context("sum overflow")?;
                } else {
                    invalid = true;
                }
            }
            Some(i64::try_from(sum).context("sum exceeds signed 64-bit minor units")?)
        }
        Operation::Compare {
            left_event,
            right_event,
            unit,
        } => {
            let mut operand = |id: &str| match positive.get(id) {
                Some(claims) => amount(id, claims, unit, gaps),
                None => {
                    gaps.insert(Gap::MissingEvent {
                        event_id: id.to_owned(),
                    });
                    None
                }
            };
            let left = operand(left_event);
            let right = operand(right_event);
            match (left, right) {
                (Some(left), Some(right)) => Some(match left.cmp(&right) {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                }),
                _ => None,
            }
        }
    };
    Ok(if invalid { None } else { value })
}

fn amount(event_id: &str, claims: &[&&Claim], unit: &str, gaps: &mut BTreeSet<Gap>) -> Option<i64> {
    let values: BTreeSet<_> = claims
        .iter()
        .filter_map(|claim| claim.value.as_ref())
        .map(|value| (value.unit.as_str(), value.minor))
        .collect();
    if values.len() > 1 {
        gaps.insert(Gap::ConflictingValues {
            event_id: event_id.to_owned(),
        });
        return None;
    }
    if let Some((_, value)) = values
        .first()
        .copied()
        .filter(|(value_unit, _)| *value_unit == unit)
    {
        return Some(value);
    }
    for claim in claims {
        gaps.insert(Gap::MissingValue {
            claim_id: claim.id.clone(),
        });
    }
    None
}

fn advance(assignment: &mut [usize], sizes: &[usize]) -> bool {
    for (index, value) in assignment.iter_mut().enumerate().rev() {
        *value += 1;
        if *value < sizes[index] {
            return true;
        }
        *value = 0;
    }
    false
}
