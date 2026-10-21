// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! The retrieval contract: answer-and-challenge over one `(entity, attribute)` family.
//!
//! # Mechanism
//!
//! A query names a family through `(entity, attribute)`, a [`Scope`] (qualifier map), an
//! `as_of` time on the valid-time axis and free text. The unit of retrieval is a
//! [`RetrievalUnit`]: a primary claim, supporting evidence, the strongest defeater and a
//! [`Status`] that is either `Resolved` or `UnresolvedConflict`. Two passes produce it:
//!
//! 1. Support pass: lexical ranking (BM25-lite, see [`Family::lexical_score`]) of the
//!    family's current claims whose scope covers the query scope. Its top is the lexical
//!    winner.
//! 2. Challenge pass: four similarity-independent exact lookups on the same family.
//!    (a) A newer, or more specific, current claim covering the query replaces the lexical
//!        winner (precedence below).
//!    (b) A retraction targeting a claim voids that claim entirely, including the explicit
//!        supersession links it carried, so what it had replaced stands again.
//!    (c) A current narrower-scope claim that is compatible with the query scope but not
//!        covered by it cannot be ruled in or out; it becomes a branch.
//!    (d) Current claims at the winner's specificity and effective time with no supersession
//!        between them are concurrent branches.
//!
//! More than one branch means `UnresolvedConflict` and every branch is returned. A blended
//! ranking is never produced.
//!
//! # Precedence among current claims covering the query
//!
//! Most specific scope first, then latest effective time (`valid_from`), then highest
//! sequence number. A wider claim never overrides a narrower one by recency alone; only an
//! explicit supersession marker (`supersedes`) does.
//!
//! # Data layout
//!
//! A [`Family`] is one contiguous block: a `Vec<Assertion>` in chronological order, the
//! per-assertion token sets, a document-frequency table and an id -> offset map. Every lookup
//! of both passes is a scan of this block, so the cost per query is O(|family| * |query
//! tokens|) and nothing global is touched. Wiping the family drops the block in one operation.
//!
//! # Invariants
//!
//! - `assertions` is sorted by `(effective, seq)`; `seq` is unique within a family.
//! - A retraction's `target` names an id of the same family; an unknown target is inert.
//! - `items` of a unit never exceeds [`EVIDENCE_BUDGET`] entries, contains no duplicates and
//!   starts with the primary claim (or with the branches, for a conflict).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Unix time in milliseconds on the valid-time axis.
pub type Millis = i64;

/// The evidence budget of one retrieval unit: how many records the caller receives at most.
pub const EVIDENCE_BUDGET: usize = 4;

/// The identity of a family: what the assertion is about and which attribute it states.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FamilyKey {
    pub entity: String,
    pub attribute: String,
}

impl FamilyKey {
    pub fn new(entity: impl Into<String>, attribute: impl Into<String>) -> Self {
        Self {
            entity: entity.into(),
            attribute: attribute.into(),
        }
    }

    /// The memory API `subject` under which the family's facts are stored, so that
    /// `ForgetScope::Subject` erases exactly one family.
    pub fn subject(&self) -> String {
        format!("{}/{}", self.entity, self.attribute)
    }

    pub fn parse_subject(subject: &str) -> Option<Self> {
        let (entity, attribute) = subject.split_once('/')?;
        (!entity.is_empty() && !attribute.is_empty()).then(|| Self::new(entity, attribute))
    }
}

impl fmt::Display for FamilyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.entity, self.attribute)
    }
}

/// A set of qualifiers such as `{region: eu, env: prod}`. The empty scope is global. Scope
/// `A` covers scope `B` when every qualifier of `A` appears in `B` with the same value, so a
/// claim at `A` applies to a query at `B`. Two scopes are compatible when no shared qualifier
/// disagrees.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Scope {
    pub qualifiers: BTreeMap<String, String>,
}

impl Scope {
    pub fn global() -> Self {
        Self::default()
    }

    pub fn single(dimension: &str, value: &str) -> Self {
        Self::global().and(dimension, value)
    }

    pub fn and(mut self, dimension: &str, value: &str) -> Self {
        self.qualifiers
            .insert(dimension.to_owned(), value.to_owned());
        self
    }

    /// How many qualifiers the scope pins; higher is narrower.
    pub fn specificity(&self) -> usize {
        self.qualifiers.len()
    }

    pub fn is_global(&self) -> bool {
        self.qualifiers.is_empty()
    }

    /// Whether a claim at this scope applies to a query at `query`.
    pub fn covers(&self, query: &Scope) -> bool {
        self.qualifiers
            .iter()
            .all(|(dimension, value)| query.qualifiers.get(dimension) == Some(value))
    }

    /// Whether nothing in this scope contradicts `other`.
    pub fn compatible(&self, other: &Scope) -> bool {
        self.qualifiers.iter().all(|(dimension, value)| {
            other
                .qualifiers
                .get(dimension)
                .is_none_or(|theirs| theirs == value)
        })
    }

    pub fn to_json(&self) -> Value {
        Value::Object(
            self.qualifiers
                .iter()
                .map(|(dimension, value)| (dimension.clone(), Value::String(value.clone())))
                .collect::<Map<String, Value>>(),
        )
    }

    /// Reads a scope from a JSON object of string values; anything else is not a scope.
    pub fn from_json(value: &Value) -> Option<Self> {
        let map = value.as_object()?;
        let mut qualifiers = BTreeMap::new();
        for (dimension, value) in map {
            qualifiers.insert(dimension.clone(), value.as_str()?.to_owned());
        }
        Some(Self { qualifiers })
    }

    /// `global` or `dimension=value,...` in dimension order.
    pub fn describe(&self) -> String {
        if self.is_global() {
            return "global".to_owned();
        }
        self.qualifiers
            .iter()
            .map(|(dimension, value)| format!("{dimension}={value}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AssertionKind {
    /// States a value for the family's attribute at a scope.
    Claim,
    /// Withdraws one earlier assertion (`target`) entirely.
    Retraction,
}

/// One observation stored in a family. `seq` is the family-local ingest order and breaks
/// every tie; `supersedes` carries the caller's explicit supersession markers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Assertion {
    pub id: String,
    pub seq: u32,
    pub kind: AssertionKind,
    pub scope: Scope,
    pub value: String,
    pub text: String,
    pub valid_from: Option<Millis>,
    pub valid_to: Option<Millis>,
    pub supersedes: Vec<String>,
    /// The retracted assertion; only for [`AssertionKind::Retraction`].
    pub target: Option<String>,
}

impl Assertion {
    /// Whether the valid-time window contains `at` (`valid_to` is exclusive).
    pub fn valid_at(&self, at: Millis) -> bool {
        self.valid_from.is_none_or(|from| from <= at) && self.valid_to.is_none_or(|to| at < to)
    }

    /// When the assertion takes effect on the valid-time axis; unbounded is the earliest.
    pub fn effective(&self) -> Millis {
        self.valid_from.unwrap_or(Millis::MIN)
    }

    pub fn is_claim(&self) -> bool {
        self.kind == AssertionKind::Claim
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Status {
    /// Exactly one claim applies; `primary_claim` holds it (or nothing applies at all).
    Resolved,
    /// Several claims could apply and the query cannot tell them apart; `branches` lists
    /// every one of them.
    UnresolvedConflict,
}

/// Why the strongest alternative to the primary claim does not stand.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "reason")]
pub enum Defeat {
    /// A retraction (`by`) withdrew it.
    Retracted { by: String },
    /// A live claim (`by`) explicitly supersedes it.
    Superseded { by: String },
    /// It covers the query but a narrower claim takes precedence.
    LessSpecific,
    /// It is current and covers the query but a claim of the same specificity is later.
    Outranked,
}

/// The strongest record that challenged the primary claim, with why it lost. It is the
/// latest valid claim compatible with the query when the two passes are ignored, which is
/// what a recency-first retriever would have answered.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Defeater {
    pub id: String,
    #[serde(flatten)]
    pub defeat: Defeat,
}

/// What one query returns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalUnit {
    pub status: Status,
    /// The claim that applies, when `status` is `Resolved` and one exists.
    pub primary_claim: Option<String>,
    /// Every claim that could apply, when `status` is `UnresolvedConflict`; otherwise the
    /// primary claim alone, or empty.
    pub branches: Vec<String>,
    /// Records returned to justify the unit beyond the primary and the defeater: the
    /// record that defeated the defeater, and what the primary explicitly replaced.
    pub supporting_evidence: Vec<String>,
    pub strongest_defeater: Option<Defeater>,
    /// Every record id the unit returns, in presentation order; at most [`EVIDENCE_BUDGET`].
    pub items: Vec<String>,
    /// The challenge pass replaced the lexical winner of the support pass.
    pub support_overridden: bool,
    /// More branches exist than the budget could carry.
    pub branches_truncated: bool,
}

impl RetrievalUnit {
    /// The unit for a family that does not exist (or was wiped).
    pub fn absent() -> Self {
        Self {
            status: Status::Resolved,
            primary_claim: None,
            branches: Vec::new(),
            supporting_evidence: Vec::new(),
            strongest_defeater: None,
            items: Vec::new(),
            support_overridden: false,
            branches_truncated: false,
        }
    }
}

/// The caller-supplied side of a query.
#[derive(Clone, Copy, Debug)]
pub struct QueryInput<'a> {
    pub key: &'a FamilyKey,
    pub scope: &'a Scope,
    pub as_of: Millis,
    pub text: &'a str,
}

/// Lowercase alphanumeric runs of `text`, deduplicated.
pub fn tokenize(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn push_unique(items: &mut Vec<String>, id: &str) {
    if items.len() < EVIDENCE_BUDGET && !items.iter().any(|existing| existing == id) {
        items.push(id.to_owned());
    }
}

/// The standing of every assertion of a family at one `as_of` time: which are retracted and
/// which are explicitly superseded by a live claim. Built once per query in O(|family|).
pub struct Standing<'a> {
    family: &'a Family,
    as_of: Millis,
    /// Retracted id -> the retraction that withdrew it (the earliest, by sequence).
    retracted_by: HashMap<&'a str, &'a str>,
    /// Superseded id -> the latest live claim that explicitly supersedes it.
    superseded_by: HashMap<&'a str, &'a str>,
}

impl<'a> Standing<'a> {
    pub fn new(family: &'a Family, as_of: Millis) -> Self {
        let mut retracted_by: HashMap<&str, &str> = HashMap::new();
        for assertion in &family.assertions {
            if assertion.kind == AssertionKind::Retraction {
                if let Some(target) = &assertion.target {
                    retracted_by
                        .entry(target.as_str())
                        .or_insert(assertion.id.as_str());
                }
            }
        }
        let mut standing = Self {
            family,
            as_of,
            retracted_by,
            superseded_by: HashMap::new(),
        };
        let mut superseded_by: HashMap<&str, &str> = HashMap::new();
        for assertion in &family.assertions {
            if standing.live_assertion(assertion) {
                for old in &assertion.supersedes {
                    superseded_by.insert(old.as_str(), assertion.id.as_str());
                }
            }
        }
        standing.superseded_by = superseded_by;
        standing
    }

    fn live_assertion(&self, assertion: &Assertion) -> bool {
        assertion.is_claim()
            && assertion.valid_at(self.as_of)
            && !self.retracted_by.contains_key(assertion.id.as_str())
    }

    /// A claim that is valid at `as_of` and not retracted. Live claims keep their
    /// supersession authority even when they are themselves superseded.
    pub fn live(&self, index: usize) -> bool {
        self.live_assertion(&self.family.assertions[index])
    }

    /// A live claim that no live claim explicitly supersedes.
    pub fn current(&self, index: usize) -> bool {
        self.live(index)
            && !self
                .superseded_by
                .contains_key(self.family.assertions[index].id.as_str())
    }

    pub fn retracted_by(&self, id: &str) -> Option<&'a str> {
        self.retracted_by.get(id).copied()
    }

    pub fn superseded_by(&self, id: &str) -> Option<&'a str> {
        self.superseded_by.get(id).copied()
    }
}

/// One `(entity, attribute)` family as a contiguous block. See the module documentation for
/// the layout and invariants.
#[derive(Clone, Debug)]
pub struct Family {
    key: FamilyKey,
    assertions: Vec<Assertion>,
    tokens: Vec<BTreeSet<String>>,
    document_frequency: HashMap<String, usize>,
    by_id: HashMap<String, usize>,
}

impl Family {
    pub fn new(key: FamilyKey, mut assertions: Vec<Assertion>) -> Self {
        assertions.sort_by(|a, b| (a.effective(), a.seq).cmp(&(b.effective(), b.seq)));
        let tokens: Vec<BTreeSet<String>> =
            assertions.iter().map(|a| tokenize(&a.text)).collect();
        let mut document_frequency: HashMap<String, usize> = HashMap::new();
        for set in &tokens {
            for token in set {
                *document_frequency.entry(token.clone()).or_insert(0) += 1;
            }
        }
        let by_id = assertions
            .iter()
            .enumerate()
            .map(|(index, a)| (a.id.clone(), index))
            .collect();
        Self {
            key,
            assertions,
            tokens,
            document_frequency,
            by_id,
        }
    }

    pub fn key(&self) -> &FamilyKey {
        &self.key
    }

    /// The block, in chronological order.
    pub fn assertions(&self) -> &[Assertion] {
        &self.assertions
    }

    pub fn len(&self) -> usize {
        self.assertions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assertions.is_empty()
    }

    pub fn get(&self, id: &str) -> Option<&Assertion> {
        self.by_id.get(id).map(|&index| &self.assertions[index])
    }

    pub fn id_at(&self, index: usize) -> &str {
        &self.assertions[index].id
    }

    /// The precedence key: specificity, effective time, sequence; greater wins.
    pub fn rank_key(&self, index: usize) -> (usize, Millis, u32) {
        let a = &self.assertions[index];
        (a.scope.specificity(), a.effective(), a.seq)
    }

    /// BM25-lite over the family: the sum, over the distinct query tokens the assertion
    /// contains, of `ln(1 + (N - df + 0.5) / (df + 0.5))`. Term frequency and length
    /// normalisation are omitted: inside a family every record states the same attribute of
    /// the same entity, so the lexical pass only orders near-duplicates and the challenge
    /// pass decides.
    pub fn lexical_score(&self, index: usize, query_tokens: &BTreeSet<String>) -> f64 {
        let n = self.assertions.len() as f64;
        let mut score = 0.0;
        for token in query_tokens {
            if self.tokens[index].contains(token) {
                let df = self.document_frequency.get(token).copied().unwrap_or(0) as f64;
                score += ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            }
        }
        score
    }

    /// Orders `candidates` by lexical score, then by the precedence key, both descending.
    pub fn lexical_rank(&self, query_tokens: &BTreeSet<String>, candidates: &mut [usize]) {
        candidates.sort_by(|&a, &b| {
            self.lexical_score(b, query_tokens)
                .total_cmp(&self.lexical_score(a, query_tokens))
                .then_with(|| self.rank_key(b).cmp(&self.rank_key(a)))
        });
    }

    /// Keeps, per distinct scope, the latest of `candidates` and anything concurrent with
    /// it (same effective time, neither superseding the other).
    fn latest_per_scope(&self, candidates: &[usize]) -> Vec<usize> {
        let mut best: BTreeMap<&Scope, Vec<usize>> = BTreeMap::new();
        for &index in candidates {
            let scope = &self.assertions[index].scope;
            let entry = best.entry(scope).or_default();
            let leader = entry.first().copied();
            match leader {
                None => entry.push(index),
                Some(leader) => {
                    let (lead_effective, this_effective) = (
                        self.assertions[leader].effective(),
                        self.assertions[index].effective(),
                    );
                    if this_effective > lead_effective {
                        entry.clear();
                        entry.push(index);
                    } else if this_effective == lead_effective {
                        entry.push(index);
                    }
                }
            }
        }
        best.into_values().flatten().collect()
    }

    /// The support pass followed by the challenge pass. See the module documentation.
    pub fn answer(&self, query: &QueryInput<'_>) -> RetrievalUnit {
        let standing = Standing::new(self, query.as_of);
        let query_tokens = tokenize(query.text);
        let n = self.assertions.len();

        // Support pass: lexical ranking of current claims whose scope covers the query.
        let mut support: Vec<usize> = (0..n)
            .filter(|&i| standing.current(i) && self.assertions[i].scope.covers(query.scope))
            .collect();
        self.lexical_rank(&query_tokens, &mut support);
        let lexical_winner = support.first().copied();

        // Challenge (a): the precedence winner among the same candidates, by exact keys.
        let winner = support.iter().copied().max_by_key(|&i| self.rank_key(i));

        // Challenge (d): concurrent claims at the winner's specificity and effective time.
        // They are current, so no supersession runs between them.
        let ties: Vec<usize> = match winner {
            Some(w) => {
                let (specificity, effective, _) = self.rank_key(w);
                support
                    .iter()
                    .copied()
                    .filter(|&i| {
                        i != w
                            && self.assertions[i].scope.specificity() == specificity
                            && self.assertions[i].effective() == effective
                    })
                    .collect()
            }
            None => Vec::new(),
        };

        // Challenge (c): current claims the query scope neither covers nor contradicts.
        let ambiguous_all: Vec<usize> = (0..n)
            .filter(|&i| {
                let scope = &self.assertions[i].scope;
                standing.current(i) && !scope.covers(query.scope) && scope.compatible(query.scope)
            })
            .collect();
        let ambiguous = self.latest_per_scope(&ambiguous_all);

        let mut branches: Vec<usize> = Vec::new();
        if let Some(w) = winner {
            branches.push(w);
        }
        branches.extend(ties);
        branches.extend(ambiguous);
        let status = if branches.len() > 1 || (winner.is_none() && !branches.is_empty()) {
            Status::UnresolvedConflict
        } else {
            Status::Resolved
        };

        // What a recency-first retriever would have returned: the latest valid compatible
        // claim, ignoring retraction, supersession and specificity.
        let naive = (0..n)
            .filter(|&i| {
                let a = &self.assertions[i];
                a.is_claim() && a.valid_at(query.as_of) && a.scope.compatible(query.scope)
            })
            .max_by_key(|&i| (self.assertions[i].effective(), self.assertions[i].seq));
        let strongest_defeater = naive
            .filter(|i| !branches.contains(i))
            .map(|i| {
                let a = &self.assertions[i];
                let defeat = if let Some(by) = standing.retracted_by(&a.id) {
                    Defeat::Retracted { by: by.to_owned() }
                } else if let Some(by) = standing.superseded_by(&a.id) {
                    Defeat::Superseded { by: by.to_owned() }
                } else if a.scope.covers(query.scope) {
                    match winner {
                        Some(w) if self.rank_key(w).0 > a.scope.specificity() => {
                            Defeat::LessSpecific
                        }
                        _ => Defeat::Outranked,
                    }
                } else {
                    Defeat::Outranked
                };
                Defeater {
                    id: a.id.clone(),
                    defeat,
                }
            });

        // Evidence within the budget: branches (or the primary), the defeater, what defeated
        // it, then what the primary explicitly replaced.
        let mut items: Vec<String> = Vec::with_capacity(EVIDENCE_BUDGET);
        for &i in &branches {
            push_unique(&mut items, &self.assertions[i].id);
        }
        let branches_truncated = branches.len() > EVIDENCE_BUDGET;
        if let Some(defeater) = &strongest_defeater {
            push_unique(&mut items, &defeater.id);
            match &defeater.defeat {
                Defeat::Retracted { by } | Defeat::Superseded { by } => push_unique(&mut items, by),
                Defeat::LessSpecific | Defeat::Outranked => {}
            }
        }
        if let (Status::Resolved, Some(w)) = (status, winner) {
            for old in &self.assertions[w].supersedes {
                if self.by_id.contains_key(old) {
                    push_unique(&mut items, old);
                }
            }
        }

        let primary_claim = match status {
            Status::Resolved => winner.map(|w| self.assertions[w].id.clone()),
            Status::UnresolvedConflict => None,
        };
        let defeater_id = strongest_defeater.as_ref().map(|d| d.id.as_str());
        let supporting_evidence: Vec<String> = items
            .iter()
            .filter(|id| {
                primary_claim.as_deref() != Some(id.as_str()) && defeater_id != Some(id.as_str())
            })
            .cloned()
            .collect();

        RetrievalUnit {
            status,
            primary_claim,
            branches: branches
                .iter()
                .map(|&i| self.assertions[i].id.clone())
                .collect(),
            supporting_evidence,
            strongest_defeater,
            items,
            support_overridden: lexical_winner != winner,
            branches_truncated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: Millis = 86_400_000;

    fn claim(seq: u32, scope: Scope, value: &str, from: Millis, supersedes: &[&str]) -> Assertion {
        Assertion {
            id: format!("a#{seq}"),
            seq,
            kind: AssertionKind::Claim,
            scope,
            value: value.to_owned(),
            text: format!("acct1 payment terms is {value}"),
            valid_from: Some(from),
            valid_to: None,
            supersedes: supersedes.iter().map(|s| (*s).to_owned()).collect(),
            target: None,
        }
    }

    fn retraction(seq: u32, target: &str) -> Assertion {
        Assertion {
            id: format!("a#{seq}"),
            seq,
            kind: AssertionKind::Retraction,
            scope: Scope::global(),
            value: String::new(),
            text: "retraction: disregard the earlier note".to_owned(),
            valid_from: None,
            valid_to: None,
            supersedes: vec![target.to_owned()],
            target: Some(target.to_owned()),
        }
    }

    fn family() -> Family {
        // D1 <- D2 <- D3 (global), E at region=eu, D4 supersedes D3, R retracts D4.
        Family::new(
            FamilyKey::new("acct1", "payment_terms"),
            vec![
                claim(1, Scope::global(), "net 30", DAY, &[]),
                claim(2, Scope::global(), "net 45", 2 * DAY, &["a#1"]),
                claim(3, Scope::global(), "net 60", 3 * DAY, &["a#2"]),
                claim(4, Scope::single("region", "eu"), "net 15", 4 * DAY, &[]),
                claim(5, Scope::global(), "net 90", 5 * DAY, &["a#3"]),
                retraction(6, "a#5"),
            ],
        )
    }

    fn query<'a>(key: &'a FamilyKey, scope: &'a Scope, as_of: Millis) -> QueryInput<'a> {
        QueryInput {
            key,
            scope,
            as_of,
            text: "what is acct1 payment terms",
        }
    }

    #[test]
    fn retraction_restores_the_superseded_default() {
        let family = family();
        let scope = Scope::single("region", "us");
        let unit = family.answer(&query(family.key(), &scope, 7 * DAY));
        assert_eq!(unit.status, Status::Resolved);
        assert_eq!(unit.primary_claim.as_deref(), Some("a#3"));
        let defeater = unit.strongest_defeater.expect("the retracted claim is the defeater");
        assert_eq!(defeater.id, "a#5");
        assert_eq!(
            defeater.defeat,
            Defeat::Retracted {
                by: "a#6".to_owned()
            }
        );
        assert_eq!(unit.items, vec!["a#3", "a#5", "a#6", "a#2"]);
    }

    #[test]
    fn narrower_scope_wins_when_covered_and_branches_when_not() {
        let family = family();
        let eu = Scope::single("region", "eu").and("env", "prod");
        let unit = family.answer(&query(family.key(), &eu, 7 * DAY));
        assert_eq!(unit.primary_claim.as_deref(), Some("a#4"));
        let none = Scope::global();
        let unit = family.answer(&query(family.key(), &none, 7 * DAY));
        assert_eq!(unit.status, Status::UnresolvedConflict);
        assert_eq!(unit.branches, vec!["a#3", "a#4"]);
        assert!(unit.items.len() <= EVIDENCE_BUDGET);
    }

    #[test]
    fn as_of_before_the_exception_resolves_to_the_default_of_that_time() {
        let family = family();
        let none = Scope::global();
        let unit = family.answer(&query(family.key(), &none, 3 * DAY + 1));
        assert_eq!(unit.status, Status::Resolved);
        assert_eq!(unit.primary_claim.as_deref(), Some("a#3"));
        assert!(unit.strongest_defeater.is_none());
    }

    #[test]
    fn concurrent_claims_are_an_unresolved_conflict() {
        let family = Family::new(
            FamilyKey::new("acct2", "tier"),
            vec![
                claim(1, Scope::global(), "gold", DAY, &[]),
                claim(2, Scope::global(), "silver", 2 * DAY, &["a#1"]),
                claim(3, Scope::global(), "platinum", 2 * DAY, &["a#1"]),
            ],
        );
        let none = Scope::global();
        let unit = family.answer(&query(family.key(), &none, 3 * DAY));
        assert_eq!(unit.status, Status::UnresolvedConflict);
        assert_eq!(unit.branches, vec!["a#3", "a#2"]);
    }
}
