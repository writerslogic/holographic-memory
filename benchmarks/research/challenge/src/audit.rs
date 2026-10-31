// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Real-trace audit: what fraction of an agent's queries carry a usable `(entity,
//! attribute)` key, through caller metadata or a small template rule set.
//!
//! # Trace format (JSONL, one object per line; unknown fields are ignored)
//!
//! ```json
//! {"id": "q-17", "text": "What is acct017's payment terms?",
//!  "entity_id": "acct017", "attribute_key": "payment_terms",
//!  "scope": {"region": "eu"}, "as_of": 1700950400000}
//! ```
//!
//! `text` is required; everything else is optional. A query is keyed by metadata when both
//! `entity_id` and `attribute_key` are present and well formed; otherwise the rules below
//! run on the lowercased text, in order, and the first match keys it:
//!
//! | rule | pattern |
//! |------|---------|
//! | `kv` | `entity=X attribute=Y` (also `entity_id`, `subject`; `attr`, `attribute_key`, `field`; `:` or `=`) |
//! | `possessive` | `... X's <attribute>` |
//! | `attribute-of-entity` | `<attribute> of X ...` |
//! | `attribute-for-entity` | `<attribute> for X ...` |
//! | `entity-colon-attribute` | `X: <attribute>` |
//! | `identifier-then-attribute` | `X <attribute>` where `X` contains a digit |
//!
//! An entity is one token of `[A-Za-z0-9_.-]{1,64}`; an attribute is one to four words of
//! letters or underscores after leading question stems (`what is the`, `tell me`, ...) and
//! trailing fillers (`now`, `please`, ...) are removed. The attribute key is the words joined
//! by `_`.
//!
//! The synthetic examples written by [`synthetic_examples`] exercise every rule and exist
//! only so the pipeline runs end to end. Their usable fraction says nothing about real
//! agents: a real trace must replace them before the threshold means anything.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::synth::Rng;

/// The audit passes when at least this fraction of queries is keyed.
pub const THRESHOLD: f64 = 0.5;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TraceQuery {
    pub id: Option<String>,
    pub text: String,
    pub entity_id: Option<String>,
    pub attribute_key: Option<String>,
    pub scope: Option<Value>,
    pub as_of: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "source", content = "rule")]
pub enum KeySource {
    Metadata,
    Rule(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedKey {
    pub entity: String,
    pub attribute: String,
    #[serde(flatten)]
    pub source: KeySource,
}

const STEMS: [&str; 22] = [
    "what is the ",
    "what's the ",
    "whats the ",
    "what is ",
    "what's ",
    "whats ",
    "which ",
    "tell me the ",
    "tell me ",
    "look up the ",
    "look up ",
    "lookup the ",
    "lookup ",
    "get the ",
    "get ",
    "show me the ",
    "show me ",
    "find the ",
    "find ",
    "current ",
    "latest ",
    "the ",
];

const TAILS: [&str; 9] = [
    " right now",
    " at the moment",
    " now",
    " today",
    " currently",
    " please",
    " value",
    " setting",
    " again",
];

fn normalize(text: &str) -> String {
    text.trim()
        .trim_end_matches(['?', '.', '!'])
        .trim()
        .to_lowercase()
}

fn strip_stems(mut text: &str) -> &str {
    loop {
        let before = text;
        for stem in STEMS {
            if let Some(rest) = text.strip_prefix(stem) {
                text = rest.trim_start();
            }
        }
        if text == before {
            return text;
        }
    }
}

fn strip_tails(mut text: &str) -> &str {
    loop {
        let before = text;
        for tail in TAILS {
            if let Some(rest) = text.strip_suffix(tail) {
                text = rest.trim_end();
            }
        }
        if text == before {
            return text;
        }
    }
}

fn ident_ok(token: &str) -> bool {
    !token.is_empty()
        && token.len() <= 64
        && token
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
}

fn attr_ok(phrase: &str) -> bool {
    let words: Vec<&str> = phrase.split_whitespace().collect();
    (1..=4).contains(&words.len())
        && words
            .iter()
            .all(|word| word.chars().all(|c| c.is_alphabetic() || c == '_'))
}

fn attr_key(phrase: &str) -> String {
    phrase.split_whitespace().collect::<Vec<_>>().join("_")
}

fn clean_attr(text: &str) -> Option<String> {
    let text = strip_tails(strip_stems(text.trim()))
        .trim_matches(|c: char| c == ',' || c == ':' || c == '-' || c == ' ');
    attr_ok(text).then(|| attr_key(text))
}

fn clean_entity(token: &str) -> Option<String> {
    let token = token.trim_matches(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-' || c == '.'));
    ident_ok(token).then(|| token.to_owned())
}

fn rule_kv(text: &str) -> Option<(String, String, &'static str)> {
    let mut entity = None;
    let mut attribute = None;
    for token in text.split(|c: char| c.is_whitespace() || c == ',' || c == ';') {
        let Some((name, value)) = token.split_once(['=', ':']) else {
            continue;
        };
        match name {
            "entity" | "entity_id" | "subject" => entity = clean_entity(value),
            "attribute" | "attribute_key" | "attr" | "field" => {
                attribute = clean_attr(&value.replace('_', " "));
            }
            _ => {}
        }
    }
    Some((entity?, attribute?, "kv"))
}

fn rule_possessive(text: &str) -> Option<(String, String, &'static str)> {
    let (before, after) = text.split_once("'s ")?;
    let entity = clean_entity(before.rsplit(' ').next()?)?;
    let attribute = clean_attr(after)?;
    Some((entity, attribute, "possessive"))
}

fn rule_attribute_then_entity<'a>(
    text: &str,
    separator: &str,
    name: &'a str,
) -> Option<(String, String, &'a str)> {
    let (before, after) = text.split_once(separator)?;
    let attribute = clean_attr(before)?;
    let entity = clean_entity(strip_stems(after.trim()).split(' ').next()?)?;
    Some((entity, attribute, name))
}

fn rule_colon(text: &str) -> Option<(String, String, &'static str)> {
    let (before, after) = text.split_once(':')?;
    if before.contains(' ') {
        return None;
    }
    let entity = clean_entity(before)?;
    let attribute = clean_attr(after)?;
    Some((entity, attribute, "entity-colon-attribute"))
}

fn rule_lead_identifier(text: &str) -> Option<(String, String, &'static str)> {
    let (first, rest) = text.split_once(' ')?;
    if !first.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    let entity = clean_entity(first)?;
    let attribute = clean_attr(rest)?;
    Some((entity, attribute, "identifier-then-attribute"))
}

/// The key a query carries, if any.
pub fn extract_key(query: &TraceQuery) -> Option<ExtractedKey> {
    if let (Some(entity), Some(attribute)) = (&query.entity_id, &query.attribute_key) {
        if ident_ok(entity.trim()) && !attribute.trim().is_empty() {
            return Some(ExtractedKey {
                entity: entity.trim().to_owned(),
                attribute: attribute.trim().to_owned(),
                source: KeySource::Metadata,
            });
        }
    }
    let text = normalize(&query.text);
    let found = rule_kv(&text)
        .or_else(|| rule_possessive(&text))
        .or_else(|| rule_attribute_then_entity(&text, " of ", "attribute-of-entity"))
        .or_else(|| rule_attribute_then_entity(&text, " for ", "attribute-for-entity"))
        .or_else(|| rule_colon(&text))
        .or_else(|| rule_lead_identifier(&text))?;
    let (entity, attribute, rule) = found;
    Some(ExtractedKey {
        entity,
        attribute,
        source: KeySource::Rule(rule.to_owned()),
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct AuditSample {
    pub text: String,
    pub key: Option<ExtractedKey>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AuditReport {
    pub schema: &'static str,
    pub trace: String,
    /// The trace was generated by this crate; the fraction is not evidence about agents.
    pub synthetic: bool,
    pub queries: usize,
    pub malformed_lines: usize,
    pub keyed_by_metadata: usize,
    pub keyed_by_rule: usize,
    pub unkeyed: usize,
    pub usable_fraction: f64,
    pub threshold: f64,
    pub pass: bool,
    pub by_rule: BTreeMap<String, usize>,
    /// The first ten rule-keyed and the first ten unkeyed queries, for a reader to judge the
    /// rules.
    pub samples_keyed_by_rule: Vec<AuditSample>,
    pub samples_unkeyed: Vec<AuditSample>,
    pub note: String,
}

/// Audits the JSONL trace at `path`.
pub fn audit_file(path: &Path, synthetic: bool) -> Result<AuditReport> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("cannot open trace {}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut queries = Vec::new();
    let mut malformed_lines = 0usize;
    for (number, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("cannot read line {} of {}", number + 1, path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<TraceQuery>(&line) {
            Ok(query) if !query.text.trim().is_empty() => queries.push(query),
            _ => malformed_lines += 1,
        }
    }
    Ok(audit(&queries, &path.display().to_string(), synthetic, malformed_lines))
}

/// Audits parsed queries.
pub fn audit(queries: &[TraceQuery], trace: &str, synthetic: bool, malformed_lines: usize) -> AuditReport {
    let mut keyed_by_metadata = 0usize;
    let mut keyed_by_rule = 0usize;
    let mut by_rule: BTreeMap<String, usize> = BTreeMap::new();
    let mut samples_keyed_by_rule = Vec::new();
    let mut samples_unkeyed = Vec::new();
    for query in queries {
        match extract_key(query) {
            Some(key) => match &key.source {
                KeySource::Metadata => keyed_by_metadata += 1,
                KeySource::Rule(rule) => {
                    keyed_by_rule += 1;
                    *by_rule.entry(rule.clone()).or_insert(0) += 1;
                    if samples_keyed_by_rule.len() < 10 {
                        samples_keyed_by_rule.push(AuditSample {
                            text: query.text.clone(),
                            key: Some(key.clone()),
                        });
                    }
                }
            },
            None => {
                if samples_unkeyed.len() < 10 {
                    samples_unkeyed.push(AuditSample {
                        text: query.text.clone(),
                        key: None,
                    });
                }
            }
        }
    }
    let total = queries.len();
    let unkeyed = total - keyed_by_metadata - keyed_by_rule;
    let usable_fraction = if total == 0 {
        0.0
    } else {
        (keyed_by_metadata + keyed_by_rule) as f64 / total as f64
    };
    let note = if synthetic {
        "SYNTHETIC TRACE: these queries were generated by warm-research-challenge to exercise \
         the rules. The fraction is not evidence about any agent. Replace the trace with a \
         real agent query log (see the module documentation for the format) before reading \
         pass/fail."
            .to_owned()
    } else {
        "Fraction of queries that carry a usable (entity, attribute) key through metadata or \
         the template rules; the threshold is the panel's 50%."
            .to_owned()
    };
    AuditReport {
        schema: "warm-research-challenge/audit/v1",
        trace: trace.to_owned(),
        synthetic,
        queries: total,
        malformed_lines,
        keyed_by_metadata,
        keyed_by_rule,
        unkeyed,
        usable_fraction,
        threshold: THRESHOLD,
        pass: total > 0 && usable_fraction >= THRESHOLD,
        by_rule,
        samples_keyed_by_rule,
        samples_unkeyed,
        note,
    }
}

const ENTITIES: [&str; 8] = [
    "acct017", "acct204", "user_42", "proj-alpha", "svc9", "order7731", "team.core", "acct999",
];

const ATTRIBUTES: [&str; 10] = [
    "payment terms",
    "support tier",
    "billing cycle",
    "data region",
    "timezone",
    "default branch",
    "api version",
    "rate limit",
    "contact channel",
    "seat count",
];

const FREEFORM: [&str; 10] = [
    "what did we decide about the launch last week",
    "remind me how the migration went",
    "summarize the open issues with the vendor",
    "anything new on the roadmap?",
    "how is the on-call rotation set up",
    "what happened in the retro",
    "find the notes from the kickoff",
    "who owns the billing service",
    "status of the audit findings",
    "what were the action items",
];

/// One hundred synthetic queries covering metadata-keyed, every rule, and free-form text.
pub fn synthetic_examples(seed: u64) -> Vec<TraceQuery> {
    let mut rng = Rng::new(seed);
    (0..100)
        .map(|i| {
            let entity = ENTITIES[rng.below(ENTITIES.len())];
            let attribute = ATTRIBUTES[rng.below(ATTRIBUTES.len())];
            let id = Some(format!("synthetic-{i:03}"));
            let as_of = Some(1_700_000_000_000 + i64::from(rng.below(90) as u32) * 86_400_000);
            let scope = if rng.chance(0.5) {
                let region = ["eu", "us", "apac"][rng.below(3)];
                Some(serde_json::json!({ "region": region }))
            } else {
                None
            };
            let (text, entity_id, attribute_key) = match i % 10 {
                0..=2 => (
                    format!("what is {entity} {attribute}"),
                    Some(entity.to_owned()),
                    Some(attr_key(attribute)),
                ),
                3 => (format!("What is {entity}'s {attribute}?"), None, None),
                4 => (format!("Current {attribute} of {entity}"), None, None),
                5 => (format!("Tell me the {attribute} for {entity} please."), None, None),
                6 => (format!("{entity}: {attribute}?"), None, None),
                7 => (
                    format!("lookup entity={entity} attribute={}", attr_key(attribute)),
                    None,
                    None,
                ),
                _ => (FREEFORM[rng.below(FREEFORM.len())].to_owned(), None, None),
            };
            TraceQuery {
                id,
                text,
                entity_id,
                attribute_key,
                scope,
                as_of,
            }
        })
        .collect()
}

/// Writes queries as JSONL.
pub fn write_jsonl(path: &Path, queries: &[TraceQuery]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let file = std::fs::File::create(path)
        .with_context(|| format!("cannot create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    for query in queries {
        let line = serde_json::to_string(query).context("cannot serialize a trace query")?;
        writeln!(writer, "{line}").with_context(|| format!("cannot write {}", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("cannot flush {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(text: &str) -> TraceQuery {
        TraceQuery {
            text: text.to_owned(),
            ..TraceQuery::default()
        }
    }

    #[test]
    fn rules_key_the_template_forms_and_not_free_text() {
        let cases = [
            ("What is acct017's payment terms?", "acct017", "payment_terms"),
            ("Current support tier of user_42", "user_42", "support_tier"),
            ("Tell me the billing cycle for proj-alpha.", "proj-alpha", "billing_cycle"),
            ("svc9: data region?", "svc9", "data_region"),
            ("lookup entity=acct204 attribute=rate_limit", "acct204", "rate_limit"),
            ("order7731 contact channel now", "order7731", "contact_channel"),
        ];
        for (input, entity, attribute) in cases {
            let key = extract_key(&text(input)).unwrap_or_else(|| panic!("{input} should key"));
            assert_eq!(key.entity, entity, "{input}");
            assert_eq!(key.attribute, attribute, "{input}");
        }
        assert!(extract_key(&text("what did we decide about the launch last week")).is_none());
    }

    #[test]
    fn metadata_wins_over_rules() {
        let query = TraceQuery {
            entity_id: Some("acct1".into()),
            attribute_key: Some("timezone".into()),
            ..text("anything at all")
        };
        assert_eq!(
            extract_key(&query).map(|k| k.source),
            Some(KeySource::Metadata)
        );
    }
}
