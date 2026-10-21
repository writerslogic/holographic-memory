// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic dataset generator with gold labels derived from the construction, not from
//! the contract implementation.
//!
//! # Mechanism
//!
//! `families` families, each `(entity, attribute)` with ten chronological observations at
//! daily slots from a per-family origin `t0`. Two family layouts:
//!
//! | slot | normal family                    | conflict family                               |
//! |------|----------------------------------|-----------------------------------------------|
//! | 1-6  | D1..D6, global, chained          | D1..D6, global, chained                       |
//! | 7    | D7, global, supersedes D6        | E, exception at `{dim: val}`                  |
//! | 8    | E, exception at `{dim: val}`     | D7 and D8, global, both supersede D6, same    |
//! |      |                                  | `valid_from` (concurrent, seq 8 and 9)        |
//! | 9    | D8, global, supersedes D7        | (empty)                                       |
//! | 10   | R, retracts D8                   | R, retracts D6                                |
//!
//! Every family therefore holds 8 defaults (7 stale in the normal layout: D1..D6 superseded
//! and D8 retracted), 1 active narrower-scope exception and 1 explicit retraction. A
//! retraction voids its target entirely, including the target's supersession of what came
//! before it.
//!
//! Four queries per family: `in_scope` (query scope covered by the exception, half the time
//! with an extra unrelated qualifier), `out_of_scope` (the exception's dimension at another
//! value), `no_scope` (no qualifier, or only an unrelated dimension: the exception can
//! neither be ruled in nor out) and `as_of_mid` (as of 7.5 days, before slot 8; scope
//! alternates in/out by family parity). The first three are evaluated as of day 11.
//!
//! # Determinism and hold-out
//!
//! One SplitMix64 stream seeded with [`FROZEN_SEED`] drives every choice. The record and
//! query phrasing templates are split into a `tune` and a `test` pool with no shared
//! template; the structural draw (values, dimensions, conflict flags, scopes) is identical
//! for both splits because it is consumed before any template choice, so the two splits
//! differ only in wording.
//!
//! # Invariants
//!
//! - Record ids are `entity/attribute#seq` with `seq` 1..=10 unique per family.
//! - D7's value differs from D8's and from E's, so a stale or wrong answer carries a wrong
//!   value, not just a wrong id.
//! - `Gold::required_evidence` fits the evidence budget.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::contract::{Assertion, AssertionKind, FamilyKey, Millis, Scope, Status};
use crate::store::RecordInput;

/// Frozen seed of the structural draw.
pub const FROZEN_SEED: u64 = 0x5EED_C0DE_2026_0001;
pub const DAY: Millis = 86_400_000;
pub const HOUR: Millis = 3_600_000;
/// 2023-11-14T22:13:20Z; family `f` starts `f` hours later.
const BASE_TIME: Millis = 1_700_000_000_000;
/// Ingest times are after every family's timeline; they sit on the knowledge-time axis,
/// which no query in this crate evaluates.
const INGEST_BASE: Millis = BASE_TIME + 60 * DAY;
const SLOTS: u32 = 10;
const ATTRIBUTES_PER_ENTITY: usize = 4;

/// SplitMix64. Small, deterministic, no dependency.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`; `n == 0` yields 0.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    pub fn chance(&mut self, probability: f64) -> bool {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        unit < probability
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Split {
    Tune,
    Test,
}

impl Split {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "tune" => Some(Self::Tune),
            "test" => Some(Self::Test),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Tune => "tune",
            Self::Test => "test",
        }
    }

    fn templates(self) -> &'static Templates {
        match self {
            Self::Tune => &TUNE,
            Self::Test => &TEST,
        }
    }
}

/// Phrasing templates. Placeholders: `{entity}`, `{attr}`, `{value}`, `{dim}`, `{val}`.
pub struct Templates {
    pub claim: &'static [&'static str],
    pub exception: &'static [&'static str],
    pub retraction: &'static [&'static str],
    pub query: &'static [&'static str],
    pub scope_hint: &'static [&'static str],
}

static TUNE: Templates = Templates {
    claim: &[
        "{entity} {attr} is {value}.",
        "Noted: {entity} {attr} = {value}.",
        "{entity}'s {attr} is set to {value}.",
        "Per the latest check, {entity} {attr} is {value}.",
    ],
    exception: &[
        "For {dim} {val}, {entity} {attr} is {value}.",
        "Exception: when {dim} is {val}, {entity} {attr} is {value}.",
    ],
    retraction: &[
        "Retraction: the note that {entity} {attr} is {value} was wrong; disregard it.",
        "Withdrawn: {entity} {attr} is not {value}; that observation was an error.",
    ],
    query: &[
        "What is {entity}'s {attr}?",
        "{entity} {attr}",
        "Tell me the {attr} for {entity}.",
        "Current {attr} of {entity}",
        "{attr} {entity} latest",
    ],
    scope_hint: &[" ({dim} {val})", " in {dim} {val}"],
};

static TEST: Templates = Templates {
    claim: &[
        "{entity} has {attr} {value}.",
        "Update: {attr} for {entity} is now {value}.",
        "Recorded {attr} for {entity}: {value}.",
        "{entity}, {attr}: {value}.",
    ],
    exception: &[
        "Within {dim} {val} only, {entity} {attr} is {value}.",
        "{entity} {attr} is {value} for {dim} {val}; other scopes unchanged.",
    ],
    retraction: &[
        "Correction: {entity} {attr} was never {value}; withdraw that statement.",
        "Disregard the earlier claim that {entity} {attr} is {value}.",
    ],
    query: &[
        "Which {attr} applies to {entity}?",
        "{entity}: {attr}?",
        "Look up {attr} for {entity}",
        "What {attr} does {entity} have now?",
        "{entity} {attr} value",
    ],
    scope_hint: &[" for {dim} {val}", ", {dim}={val}"],
};

/// `(attribute key, attribute phrase, value pool)`.
static ATTRIBUTES: [(&str, &str, [&str; 8]); 10] = [
    (
        "payment_terms",
        "payment terms",
        ["net 7", "net 15", "net 30", "net 45", "net 60", "net 90", "prepaid", "due on receipt"],
    ),
    (
        "support_tier",
        "support tier",
        ["bronze", "silver", "gold", "platinum", "basic", "standard", "premium", "enterprise"],
    ),
    (
        "billing_cycle",
        "billing cycle",
        ["monthly", "quarterly", "yearly", "biennial", "weekly", "usage based", "semiannual", "one time"],
    ),
    (
        "data_region",
        "data region",
        ["frankfurt", "dublin", "virginia", "oregon", "singapore", "sydney", "tokyo", "sao paulo"],
    ),
    (
        "api_version",
        "api version",
        ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"],
    ),
    (
        "rate_limit",
        "rate limit",
        ["50 rps", "100 rps", "250 rps", "500 rps", "1000 rps", "2000 rps", "5000 rps", "10000 rps"],
    ),
    (
        "contact_channel",
        "contact channel",
        ["email", "slack", "phone", "pager", "sms", "webhook", "ticket", "teams"],
    ),
    (
        "default_branch",
        "default branch",
        ["main", "master", "trunk", "develop", "release", "stable", "mainline", "canary"],
    ),
    (
        "timezone",
        "timezone",
        ["utc", "cet", "est", "pst", "ist", "jst", "aest", "gmt"],
    ),
    (
        "seat_count",
        "seat count",
        ["5 seats", "10 seats", "25 seats", "50 seats", "100 seats", "250 seats", "500 seats", "1000 seats"],
    ),
];

/// Scope dimensions and their values. Chosen not to collide with attribute values.
static DIMENSIONS: [(&str, &[&str]); 5] = [
    ("region", &["eu", "us", "apac", "latam"]),
    ("env", &["prod", "staging", "dev"]),
    ("tier", &["free", "pro", "enterprise"]),
    ("channel", &["web", "mobile", "api"]),
    ("plan", &["starter", "growth", "scale"]),
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenConfig {
    pub families: usize,
    pub conflict_fraction: f64,
    pub split: Split,
    pub seed: u64,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            families: 1000,
            conflict_fraction: 0.2,
            split: Split::Test,
            seed: FROZEN_SEED,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryKind {
    InScope,
    OutOfScope,
    NoScope,
    AsOfMid,
}

impl QueryKind {
    pub const ALL: [QueryKind; 4] = [
        QueryKind::InScope,
        QueryKind::OutOfScope,
        QueryKind::NoScope,
        QueryKind::AsOfMid,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::InScope => "in_scope",
            Self::OutOfScope => "out_of_scope",
            Self::NoScope => "no_scope",
            Self::AsOfMid => "as_of_mid",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FamilyType {
    Normal,
    Conflict,
}

impl FamilyType {
    pub fn name(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Conflict => "conflict",
        }
    }
}

/// The expected unit. `branches` is the primary alone for a resolved query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gold {
    pub status: Status,
    pub primary: Option<String>,
    pub branches: Vec<String>,
    /// Ids the returned evidence must contain for the answer to count as justified.
    pub required_evidence: Vec<String>,
    /// Claims that are superseded or retracted as of the query time: returning one of them
    /// as the answer is a stale answer.
    pub stale_ids: Vec<String>,
    /// The expected primary is the family's exception.
    pub expects_exception: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Query {
    pub id: String,
    pub family: usize,
    pub family_type: FamilyType,
    pub key: FamilyKey,
    pub scope: Scope,
    pub as_of: Millis,
    pub text: String,
    pub kind: QueryKind,
    pub gold: Gold,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlotRecord {
    pub slot: u32,
    pub assertion: Assertion,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FamilySpec {
    pub index: usize,
    pub key: FamilyKey,
    pub family_type: FamilyType,
    /// The exception's qualifier.
    pub dimension: String,
    pub qualifier: String,
    pub t0: Millis,
    pub records: Vec<SlotRecord>,
}

impl FamilySpec {
    pub fn ids(&self) -> Vec<String> {
        self.records.iter().map(|r| r.assertion.id.clone()).collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dataset {
    pub config: GenConfig,
    pub families: Vec<FamilySpec>,
    pub queries: Vec<Query>,
}

impl Dataset {
    pub fn records(&self) -> usize {
        self.families.iter().map(|f| f.records.len()).sum()
    }

    pub fn conflict_families(&self) -> usize {
        self.families
            .iter()
            .filter(|f| f.family_type == FamilyType::Conflict)
            .count()
    }

    /// Ingest batches in slot order: every family's slot-`s` records together, at one
    /// ingest time per slot.
    pub fn batches(&self) -> Vec<(Millis, Vec<RecordInput>)> {
        (1..=SLOTS)
            .map(|slot| {
                let records = self
                    .families
                    .iter()
                    .flat_map(|family| {
                        family
                            .records
                            .iter()
                            .filter(move |r| r.slot == slot)
                            .map(move |r| RecordInput {
                                key: family.key.clone(),
                                assertion: r.assertion.clone(),
                            })
                    })
                    .collect();
                (INGEST_BASE + Millis::from(slot) * DAY, records)
            })
            .collect()
    }

    pub fn ids_by_family(&self) -> Vec<(FamilyKey, Vec<String>)> {
        self.families
            .iter()
            .map(|family| (family.key.clone(), family.ids()))
            .collect()
    }

    pub fn query_mix(&self) -> BTreeMap<String, usize> {
        let mut mix = BTreeMap::new();
        for query in &self.queries {
            *mix.entry(query.kind.name().to_owned()).or_insert(0) += 1;
        }
        mix
    }
}

fn fill(template: &str, entity: &str, attr: &str, value: &str, dim: &str, val: &str) -> String {
    template
        .replace("{entity}", entity)
        .replace("{attr}", attr)
        .replace("{value}", value)
        .replace("{dim}", dim)
        .replace("{val}", val)
}

fn pick_different<'a>(rng: &mut Rng, pool: &[&'a str], avoid: &[&str]) -> &'a str {
    loop {
        let candidate = pool[rng.below(pool.len())];
        if !avoid.contains(&candidate) {
            return candidate;
        }
    }
}

fn ids(d: &[String]) -> Vec<String> {
    d.to_vec()
}

struct Draft {
    index: usize,
    key: FamilyKey,
    attr_phrase: &'static str,
    family_type: FamilyType,
    dim: &'static str,
    val: &'static str,
    other_val: &'static str,
    other_dim: &'static str,
    other_dim_val: &'static str,
    t0: Millis,
    defaults: [&'static str; 8],
    exception_value: &'static str,
    in_scope_extra: bool,
    no_scope_other_dim: bool,
    hint_in: bool,
    hint_out: bool,
    hint_none: bool,
    hint_mid: bool,
}

/// The structural draw of one family: everything but wording.
fn draw(rng: &mut Rng, index: usize, config: &GenConfig) -> Draft {
    let entity_index = index / ATTRIBUTES_PER_ENTITY;
    let attr_slot = index % ATTRIBUTES_PER_ENTITY;
    let (attr_key, attr_phrase, pool) = &ATTRIBUTES[(entity_index + attr_slot * 3) % ATTRIBUTES.len()];
    let attr_phrase: &'static str = attr_phrase;
    let pool: &'static [&'static str] = pool;
    let key = FamilyKey::new(format!("acct{entity_index:03}"), *attr_key);
    let family_type = if rng.chance(config.conflict_fraction) {
        FamilyType::Conflict
    } else {
        FamilyType::Normal
    };
    let dim_index = rng.below(DIMENSIONS.len());
    let (dim, dim_values) = DIMENSIONS[dim_index];
    let val_index = rng.below(dim_values.len());
    let val = dim_values[val_index];
    let other_val = dim_values[(val_index + 1 + rng.below(dim_values.len() - 1)) % dim_values.len()];
    let other_dim_index = (dim_index + 1 + rng.below(DIMENSIONS.len() - 1)) % DIMENSIONS.len();
    let (other_dim, other_dim_values) = DIMENSIONS[other_dim_index];
    let other_dim_val = other_dim_values[rng.below(other_dim_values.len())];

    let mut defaults: [&'static str; 8] = [""; 8];
    defaults[0] = pool[rng.below(pool.len())];
    for i in 1..6 {
        defaults[i] = if rng.chance(0.4) {
            defaults[i - 1]
        } else {
            pool[rng.below(pool.len())]
        };
    }
    defaults[6] = pool[rng.below(pool.len())];
    defaults[7] = pick_different(rng, pool, &[defaults[6]]);
    let exception_value = pick_different(rng, pool, &[defaults[6], defaults[7]]);

    Draft {
        index,
        key,
        attr_phrase,
        family_type,
        dim,
        val,
        other_val,
        other_dim,
        other_dim_val,
        t0: BASE_TIME + Millis::try_from(index).unwrap_or(0) * HOUR,
        defaults,
        exception_value,
        in_scope_extra: rng.chance(0.5),
        no_scope_other_dim: rng.chance(0.5),
        hint_in: rng.chance(0.5),
        hint_out: rng.chance(0.5),
        hint_none: rng.chance(0.5),
        hint_mid: rng.chance(0.5),
    }
}

fn claim(
    draft: &Draft,
    seq: u32,
    scope: Scope,
    value: &str,
    text: String,
    valid_from: Millis,
    supersedes: Vec<String>,
) -> Assertion {
    Assertion {
        id: format!("{}#{seq:02}", draft.key.subject()),
        seq,
        kind: AssertionKind::Claim,
        scope,
        value: value.to_owned(),
        text,
        valid_from: Some(valid_from),
        valid_to: None,
        supersedes,
        target: None,
    }
}

fn retraction(draft: &Draft, seq: u32, target: &Assertion, text: String) -> Assertion {
    Assertion {
        id: format!("{}#{seq:02}", draft.key.subject()),
        seq,
        kind: AssertionKind::Retraction,
        scope: Scope::global(),
        value: target.value.clone(),
        text,
        valid_from: None,
        valid_to: None,
        supersedes: Vec::new(),
        target: Some(target.id.clone()),
    }
}

/// Wording and gold for one drafted family.
fn realize(rng: &mut Rng, draft: Draft, templates: &Templates) -> (FamilySpec, Vec<Query>) {
    let entity = draft.key.entity.as_str();
    let attr = draft.attr_phrase;
    let (dim, val) = (draft.dim, draft.val);
    let slot_time = |slot: u32| draft.t0 + Millis::from(slot) * DAY;
    let claim_text = |rng: &mut Rng, value: &str| {
        fill(
            templates.claim[rng.below(templates.claim.len())],
            entity,
            attr,
            value,
            dim,
            val,
        )
    };

    let mut records: Vec<SlotRecord> = Vec::with_capacity(SLOTS as usize);
    let mut previous: Option<String> = None;
    // D1..D6 at slots 1..=6, chained.
    for i in 0..6u32 {
        let seq = i + 1;
        let text = claim_text(rng, draft.defaults[i as usize]);
        let supersedes = previous.iter().cloned().collect();
        let assertion = claim(
            &draft,
            seq,
            Scope::global(),
            draft.defaults[i as usize],
            text,
            slot_time(seq),
            supersedes,
        );
        previous = Some(assertion.id.clone());
        records.push(SlotRecord {
            slot: seq,
            assertion,
        });
    }
    let d6_id = previous.clone().unwrap_or_default();
    let exception_text = fill(
        templates.exception[rng.below(templates.exception.len())],
        entity,
        attr,
        draft.exception_value,
        dim,
        val,
    );
    let retraction_text = |rng: &mut Rng, value: &str| {
        fill(
            templates.retraction[rng.below(templates.retraction.len())],
            entity,
            attr,
            value,
            dim,
            val,
        )
    };

    let (d7, d8, e, r) = match draft.family_type {
        FamilyType::Normal => {
            let d7_text = claim_text(rng, draft.defaults[6]);
            let d7 = claim(&draft, 7, Scope::global(), draft.defaults[6], d7_text, slot_time(7), vec![d6_id.clone()]);
            let e = claim(&draft, 8, Scope::single(dim, val), draft.exception_value, exception_text, slot_time(8), Vec::new());
            let d8_text = claim_text(rng, draft.defaults[7]);
            let d8 = claim(&draft, 9, Scope::global(), draft.defaults[7], d8_text, slot_time(9), vec![d7.id.clone()]);
            let r_text = retraction_text(rng, draft.defaults[7]);
            let r = retraction(&draft, 10, &d8, r_text);
            records.push(SlotRecord { slot: 7, assertion: d7.clone() });
            records.push(SlotRecord { slot: 8, assertion: e.clone() });
            records.push(SlotRecord { slot: 9, assertion: d8.clone() });
            records.push(SlotRecord { slot: 10, assertion: r.clone() });
            (d7, d8, e, r)
        }
        FamilyType::Conflict => {
            let e = claim(&draft, 7, Scope::single(dim, val), draft.exception_value, exception_text, slot_time(7), Vec::new());
            let d7_text = claim_text(rng, draft.defaults[6]);
            let d7 = claim(&draft, 8, Scope::global(), draft.defaults[6], d7_text, slot_time(8), vec![d6_id.clone()]);
            let d8_text = claim_text(rng, draft.defaults[7]);
            let d8 = claim(&draft, 9, Scope::global(), draft.defaults[7], d8_text, slot_time(8), vec![d6_id.clone()]);
            let d6_value = records[5].assertion.value.clone();
            let r_text = retraction_text(rng, &d6_value);
            let r = retraction(&draft, 10, &records[5].assertion, r_text);
            records.push(SlotRecord { slot: 7, assertion: e.clone() });
            records.push(SlotRecord { slot: 8, assertion: d7.clone() });
            records.push(SlotRecord { slot: 8, assertion: d8.clone() });
            records.push(SlotRecord { slot: 10, assertion: r.clone() });
            (d7, d8, e, r)
        }
    };

    // Gold, from the construction.
    let d: Vec<String> = records
        .iter()
        .filter(|r| r.assertion.is_claim() && r.assertion.scope.is_global())
        .map(|r| r.assertion.id.clone())
        .collect();
    debug_assert_eq!(d.len(), 8, "eight defaults per family");
    let (d5, d6) = (d[4].clone(), d[5].clone());
    let resolved = |primary: &String, required: Vec<String>, stale: Vec<String>, exception: bool| Gold {
        status: Status::Resolved,
        primary: Some(primary.clone()),
        branches: vec![primary.clone()],
        required_evidence: required,
        stale_ids: stale,
        expects_exception: exception,
    };
    let unresolved = |branches: Vec<String>, stale: Vec<String>| Gold {
        status: Status::UnresolvedConflict,
        primary: None,
        branches: branches.clone(),
        required_evidence: branches,
        stale_ids: stale,
        expects_exception: false,
    };
    let mid_in = draft.index % 2 == 0;
    let gold_for = |kind: QueryKind| -> Gold {
        match (draft.family_type, kind) {
            (FamilyType::Normal, QueryKind::InScope) => resolved(
                &e.id,
                vec![e.id.clone(), d8.id.clone(), r.id.clone()],
                [ids(&d[..6]), vec![d8.id.clone()]].concat(),
                true,
            ),
            (FamilyType::Normal, QueryKind::OutOfScope) => resolved(
                &d7.id,
                vec![d7.id.clone(), d8.id.clone(), r.id.clone()],
                [ids(&d[..6]), vec![d8.id.clone()]].concat(),
                false,
            ),
            (FamilyType::Normal, QueryKind::NoScope) => unresolved(
                vec![d7.id.clone(), e.id.clone()],
                [ids(&d[..6]), vec![d8.id.clone()]].concat(),
            ),
            (FamilyType::Normal, QueryKind::AsOfMid) => {
                resolved(&d7.id, vec![d7.id.clone()], ids(&d[..6]), false)
            }
            (FamilyType::Conflict, QueryKind::InScope) => resolved(
                &e.id,
                vec![e.id.clone(), d8.id.clone()],
                ids(&d[..6]),
                true,
            ),
            (FamilyType::Conflict, QueryKind::OutOfScope) => {
                unresolved(vec![d7.id.clone(), d8.id.clone()], ids(&d[..6]))
            }
            (FamilyType::Conflict, QueryKind::NoScope) => unresolved(
                vec![d7.id.clone(), d8.id.clone(), e.id.clone()],
                ids(&d[..6]),
            ),
            (FamilyType::Conflict, QueryKind::AsOfMid) => {
                let stale = [ids(&d[..4]), vec![d6.clone()]].concat();
                if mid_in {
                    resolved(&e.id, vec![e.id.clone()], stale, true)
                } else {
                    resolved(&d5, vec![d5.clone(), d6.clone(), r.id.clone()], stale, false)
                }
            }
        }
    };

    let in_scope = {
        let scope = Scope::single(dim, val);
        if draft.in_scope_extra {
            scope.and(draft.other_dim, draft.other_dim_val)
        } else {
            scope
        }
    };
    let out_scope = Scope::single(dim, draft.other_val);
    let no_scope = if draft.no_scope_other_dim {
        Scope::single(draft.other_dim, draft.other_dim_val)
    } else {
        Scope::global()
    };
    let mid_scope = if mid_in { in_scope.clone() } else { out_scope.clone() };
    let t_end = draft.t0 + 11 * DAY;
    let t_mid = draft.t0 + 7 * DAY + DAY / 2;

    let mut queries = Vec::with_capacity(QueryKind::ALL.len());
    for kind in QueryKind::ALL {
        let (scope, as_of, hint) = match kind {
            QueryKind::InScope => (in_scope.clone(), t_end, draft.hint_in),
            QueryKind::OutOfScope => (out_scope.clone(), t_end, draft.hint_out),
            QueryKind::NoScope => (no_scope.clone(), t_end, draft.hint_none),
            QueryKind::AsOfMid => (mid_scope.clone(), t_mid, draft.hint_mid),
        };
        let mut text = fill(
            templates.query[rng.below(templates.query.len())],
            entity,
            attr,
            "",
            dim,
            val,
        );
        if hint {
            if let Some((hint_dim, hint_val)) = scope.qualifiers.iter().next() {
                let hint_template = templates.scope_hint[rng.below(templates.scope_hint.len())];
                let trimmed = text.trim_end_matches(['?', '.']).to_owned();
                text = format!("{trimmed}{}", fill(hint_template, entity, attr, "", hint_dim, hint_val));
            }
        }
        queries.push(Query {
            id: format!("q{:04}-{}", draft.index, kind.name()),
            family: draft.index,
            family_type: draft.family_type,
            key: draft.key.clone(),
            scope,
            as_of,
            text,
            kind,
            gold: gold_for(kind),
        });
    }
    (
        FamilySpec {
            index: draft.index,
            key: draft.key.clone(),
            family_type: draft.family_type,
            dimension: dim.to_owned(),
            qualifier: val.to_owned(),
            t0: draft.t0,
            records,
        },
        queries,
    )
}

/// Generates the dataset for `config`. Deterministic in `config`.
pub fn generate(config: &GenConfig) -> Dataset {
    let mut rng = Rng::new(config.seed);
    let templates = config.split.templates();
    let mut families = Vec::with_capacity(config.families);
    let mut queries = Vec::with_capacity(config.families * QueryKind::ALL.len());
    for index in 0..config.families {
        // Structural draw first, from the shared stream; wording from a per-family stream
        // so that the structure is identical across splits.
        let draft = draw(&mut rng, index, config);
        let mut wording = Rng::new(
            config.seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5,
        );
        let (spec, family_queries) = realize(&mut wording, draft, templates);
        families.push(spec);
        queries.extend(family_queries);
    }
    Dataset {
        config: config.clone(),
        families,
        queries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic_and_structurally_identical_across_splits() {
        let config = GenConfig {
            families: 24,
            ..GenConfig::default()
        };
        let a = generate(&config);
        let b = generate(&config);
        assert_eq!(serde_json::to_string(&a).ok(), serde_json::to_string(&b).ok());
        let tune = generate(&GenConfig {
            split: Split::Tune,
            ..config.clone()
        });
        for (x, y) in a.families.iter().zip(&tune.families) {
            assert_eq!(x.key, y.key);
            assert_eq!(x.family_type, y.family_type);
            assert_eq!(x.records.len(), 10);
            for (p, q) in x.records.iter().zip(&y.records) {
                assert_eq!(p.assertion.id, q.assertion.id);
                assert_eq!(p.assertion.value, q.assertion.value);
                assert_eq!(p.assertion.scope, q.assertion.scope);
                assert_eq!(p.assertion.supersedes, q.assertion.supersedes);
            }
        }
        for (x, y) in a.queries.iter().zip(&tune.queries) {
            assert_eq!(x.gold, y.gold);
            assert_eq!(x.scope, y.scope);
        }
        assert!(a.queries.iter().all(|q| q.gold.required_evidence.len() <= 4));
    }
}
