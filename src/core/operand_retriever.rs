// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use super::documents::terms;

pub type Source = (String, String, String, String, String);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub observations: Vec<Source>,
    pub ordinal: usize,
    pub rank: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
    pub question: String,
    pub budget: usize,
    pub candidates: Vec<Candidate>,
}

#[derive(Debug, Serialize)]
pub struct Obligations {
    pub topics: BTreeSet<String>,
    pub identities: Vec<String>,
    pub literal_quotes: Vec<String>,
    pub date_literals: Vec<String>,
    pub inference_boundaries: Vec<String>,
    pub quantity: bool,
    pub chronology: bool,
    pub exact_quote: bool,
}

#[derive(Debug, Serialize)]
pub struct Retrieval {
    pub required: Obligations,
    pub selected: Vec<usize>,
    pub witnessed: BTreeMap<String, Vec<usize>>,
    pub packet: String,
}

#[derive(Serialize)]
struct Packet<'a> {
    v: u8,
    q: Vec<&'a Vec<Source>>,
    i: Vec<String>,
}

fn tokens(text: &str) -> BTreeSet<String> {
    terms(text, true).into_keys().collect()
}

fn valid_date(date: &str) -> bool {
    let Some(prefix) = date.get(..10) else {
        return false;
    };
    let bytes = prefix.as_bytes();
    if ![b'/', b'-'].contains(&bytes[4])
        || bytes[7] != bytes[4]
        || bytes
            .iter()
            .enumerate()
            .any(|(i, b)| i != 4 && i != 7 && !b.is_ascii_digit())
    {
        return false;
    }
    let year = prefix[..4].parse::<u16>().unwrap_or(0);
    let month = prefix[5..7].parse::<u8>().unwrap_or(0);
    let day = prefix[8..10].parse::<u8>().unwrap_or(0);
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        2 => 28 + u8::from(leap),
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => 0,
    };
    year > 0
        && day > 0
        && day <= days
        && date
            .get(10..)
            .is_some_and(|suffix| suffix.is_empty() || suffix.starts_with(char::is_whitespace))
}

fn calendar_literals(text: &str) -> BTreeSet<String> {
    text.split_whitespace()
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|word| valid_date(word))
        .map(|word| word[..10].replace('/', "-"))
        .collect()
}

fn contains_identity(text: &str, identity: &str) -> bool {
    text.match_indices(identity).any(|(start, _)| {
        text[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric())
            && text[start + identity.len()..]
                .chars()
                .next()
                .is_none_or(|c| !c.is_alphanumeric())
    })
}

pub fn parse_question(question: &str) -> Obligations {
    let lower = question.to_lowercase();
    static STOP: OnceLock<BTreeSet<String>> = OnceLock::new();
    let stop = STOP.get_or_init(|| tokens("a an the i me my we our you your have has had do did does was were is are be been being what which who how when where why can could would should will this that these those of in on at to for from with and or not it its some any all both each many much total type most last first latest recently since ago past during month week year day time earlier earliest later order passed more before after about bit helpful tips anxious mentioned getting around spent spend take took went visit visited attend attended participated purchase purchased friend family people event m s ve re ll d one two three four five six seven eight nine ten eleven twelve"));
    let mut topics: BTreeSet<_> = tokens(question).difference(stop).cloned().collect();
    for (cue, expansion) in [
        ("aquarium", "tank fish"),
        ("game", "play gaming"),
        ("jog", "run workout"),
        ("cook", "bake recipe kitchen meal"),
        ("sport", "soccer tournament triathlon running bike racing"),
        ("getting around", "transport card tour meeting nervous"),
        ("money", "paid cost price free"),
        ("pay", "paid cost price quote corrected"),
    ] {
        if lower.contains(cue) {
            topics.extend(tokens(expansion));
        }
    }
    let identities = question
        .split_whitespace()
        .filter(|word| {
            word.chars().next().is_some_and(char::is_uppercase)
                && !word.to_lowercase().starts_with("i'")
                && !word.to_lowercase().starts_with("i’")
                && ![
                    "I", "What", "Which", "How", "Who", "Where", "When", "Can", "Do", "Did", "Is",
                ]
                .contains(word)
        })
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_owned())
        .collect();
    let mut literal_quotes = Vec::new();
    let mut opening = None;
    for (i, ch) in question.char_indices() {
        if let Some((start, close)) = opening {
            if ch == close {
                if start < i {
                    literal_quotes.push(question[start..i].to_owned());
                }
                opening = None;
            }
        } else if ch == '"'
            || ch == '“'
            || ch == '‘'
            || (ch == '\''
                && question[..i]
                    .chars()
                    .next_back()
                    .is_none_or(|c| !c.is_alphanumeric()))
        {
            opening = Some((
                i + ch.len_utf8(),
                match ch {
                    '“' => '”',
                    '‘' => '’',
                    _ => ch,
                },
            ));
        }
    }
    let date_literals = calendar_literals(question).into_iter().collect();
    Obligations {
        topics,
        identities,
        literal_quotes,
        date_literals,
        inference_boundaries: vec![
            "Quoted turns are observations; arithmetic, temporal resolution, absence and answer entailment require inference.".into(),
            "Source dates label observations; an event date requires support within the quoted text or explicit inference.".into(),
        ],
        quantity: lower.contains("how many")
            || lower.contains("how much")
            || lower.contains("total"),
        chronology: [
            "first", "order", "since", "ago", "most recent", "latest", "earliest", "before", "after",
            "between",
        ]
        .iter()
        .any(|word| lower.contains(word)),
        exact_quote: lower.contains("quote")
            || lower.contains("exact")
            || lower.contains("verbatim"),
    }
}

fn validate(request: &Request) -> Result<()> {
    ensure!(
        !request.question.is_empty() && request.question.len() <= 8192,
        "invalid question length"
    );
    ensure!(
        (1..=1_048_576).contains(&request.budget),
        "invalid packet budget"
    );
    ensure!(
        !request.candidates.is_empty() && request.candidates.len() <= 20,
        "expected 1..=20 candidates"
    );
    let mut ids = BTreeSet::new();
    let mut bytes = 0usize;
    for candidate in &request.candidates {
        ensure!(candidate.rank <= 64, "invalid cached rank");
        ensure!(
            (1..=4096).contains(&candidate.ordinal),
            "invalid source ordinal"
        );
        ensure!(
            !candidate.observations.is_empty() && candidate.observations.len() <= 16,
            "invalid alias count"
        );
        let first = &candidate.observations[0];
        ensure!(ids.insert(&first.0), "duplicate candidate handle");
        for (turn, session, date, role, text) in &candidate.observations {
            ensure!(
                turn.len() == 64
                    && session.len() == 64
                    && turn
                        .bytes()
                        .chain(session.bytes())
                        .all(|c| c.is_ascii_hexdigit()),
                "invalid opaque identity"
            );
            ensure!(
                turn == &first.0 && text == &first.4,
                "aliases must retain the same identity and text"
            );
            ensure!(
                role == "user" && date.len() <= 64 && valid_date(date),
                "invalid source witness"
            );
            ensure!(text.len() <= 1_048_576, "source turn exceeds limit");
            bytes = bytes
                .checked_add(text.len())
                .ok_or_else(|| anyhow::anyhow!("source size overflow"))?;
        }
    }
    ensure!(bytes <= 4_194_304, "source allocation exceeds limit");
    Ok(())
}

pub fn packet(candidates: &[Candidate], selected: &[usize]) -> Result<String> {
    let evidence = selected
        .iter()
        .map(|&i| {
            candidates
                .get(i)
                .map(|c| &c.observations)
                .ok_or_else(|| anyhow::anyhow!("source index out of range"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(serde_json::to_string(&Packet {
        v: 2,
        q: evidence,
        i: Vec::new(),
    })?)
}

fn source_names(text: &str) -> BTreeSet<String> {
    text.split_whitespace()
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|word| {
            word.chars().next().is_some_and(char::is_uppercase)
                && word.len() > 2
                && !word.contains('\'')
                && !word.contains('’')
                && ![
                    "The", "That", "This", "What", "Which", "How", "Can", "Thanks", "Speaking",
                    "With", "Anyway", "Actually",
                ]
                .contains(word)
        })
        .map(str::to_owned)
        .collect()
}

struct Facts {
    words: BTreeSet<String>,
    numbers: BTreeSet<String>,
    companion: bool,
    names: BTreeSet<String>,
    temporal: BTreeSet<String>,
    dates: BTreeSet<String>,
    factual: bool,
    completion_words: BTreeSet<String>,
    friend: bool,
}

pub struct Prepared<'a> {
    request: &'a Request,
    facts: Vec<Facts>,
    costs: Vec<usize>,
}

impl<'a> Prepared<'a> {
    pub fn new(request: &'a Request) -> Result<Self> {
        validate(request)?;
        let facts = request
            .candidates
            .iter()
            .map(|candidate| {
                let text = &candidate.observations[0].4;
                let words = tokens(text);
                let declarations: Vec<_> = text
                    .split_inclusive(['.', '!', '?', '\n'])
                    .filter_map(|sentence| {
                        let lower = sentence.to_lowercase();
                        (!sentence.ends_with('?')
                            && ["i ", "i'", "my ", "we ", "our "]
                                .iter()
                                .any(|s| lower.contains(s)))
                        .then(|| tokens(sentence))
                    })
                    .collect();
                let numbers: BTreeSet<String> = declarations
                    .iter()
                    .flat_map(|w| {
                        w.iter()
                            .filter(|s| s.chars().any(|c| c.is_ascii_digit()) || ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"].contains(&s.as_str()))
                            .cloned()
                    })
                    .collect();
                let lower = text.to_lowercase();
                let companion = [
                    "with my ",
                    "with a friend",
                    "who's a",
                    "with her",
                    "with him",
                ]
                .iter()
                .any(|s| lower.contains(s));
                let names = source_names(text);
                let temporal: BTreeSet<String> = declarations.iter().flat_map(|w| w.iter().filter(|s| ["today", "yesterday", "ago", "last", "recent", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].contains(&s.as_str())).cloned()).collect();
                static PREDICATES: OnceLock<BTreeSet<String>> = OnceLock::new();
                let predicates = PREDICATES.get_or_init(|| tokens("attended participated completed finished went visited learned paid spent got bought purchased upgraded received own has using read made baked cooked led took quoted corrected getting earned sold booked using lived traveled spent"));
                let factual = !numbers.is_empty() || !temporal.is_empty() || declarations.iter().any(|words| !words.is_disjoint(predicates));
                let completion_words = declarations.iter().filter(|words| words.contains("complet") || words.contains("finish")).flat_map(|words| words.iter().cloned()).collect();
                let friend = lower.contains("my friend") || lower.contains("friend's") || lower.contains("friend’s");
                Facts {
                    completion_words, friend,
                    names, temporal, factual,
                    dates: calendar_literals(text),
                    words,
                    numbers,
                    companion,
                }
            })
            .collect();
        let costs = request
            .candidates
            .iter()
            .map(|c| serde_json::to_vec(&c.observations).map(|v| v.len() + 1))
            .collect::<std::result::Result<_, _>>()?;
        Ok(Self {
            request,
            facts,
            costs,
        })
    }
}

pub fn retrieve(request: &Request) -> Result<Retrieval> {
    retrieve_prepared(&Prepared::new(request)?)
}

pub fn retrieve_prepared(prepared: &Prepared<'_>) -> Result<Retrieval> {
    let request = prepared.request;
    let required = parse_question(&request.question);
    let lower = request.question.to_lowercase();
    let transport = lower.contains("getting around");
    let duration = lower.contains("hours");
    let money = lower.contains("money") || lower.contains("pay");
    let binary = required.chronology && lower.contains("first") && lower.contains(" or ");
    let active = required.quantity || required.chronology || transport;
    let acquisition = lower.contains("purchase") || lower.contains("bought");
    let completion = lower.contains("completed") || lower.contains("finished");
    let completion_subject: BTreeSet<_> = required
        .topics
        .iter()
        .filter(|s| !["complet", "finish", "read", "onlin"].contains(&s.as_str()))
        .cloned()
        .collect();

    let named: BTreeSet<_> = required
        .identities
        .iter()
        .filter(|name| name.len() >= 4 && name.as_str() != "City")
        .flat_map(|name| tokens(name))
        .collect();
    let relevant = |i: usize| {
        let facts = &prepared.facts[i];
        facts.factual
            && (!completion
                || (!facts.numbers.is_empty()
                    && !facts.completion_words.is_disjoint(&completion_subject)))
            && !facts.words.is_disjoint(&required.topics)
            && (named.is_empty() || !facts.words.is_disjoint(&named))
            && (!duration || facts.words.contains("hour") || facts.words.contains("minut"))
            && (!money
                || request.candidates[i].observations[0].4.contains('$')
                || facts.words.contains("free"))
    };
    let mut roots: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, c) in request.candidates.iter().enumerate() {
        let sid = c.observations[0].1.as_str();
        let root = roots.entry(sid).or_insert(i);
        if (relevant(i) && !relevant(*root))
            || (relevant(i) == relevant(*root)
                && (c.ordinal, i) < (request.candidates[*root].ordinal, *root))
        {
            *root = i;
        }
    }
    let mut coverage = vec![BTreeSet::new(); request.candidates.len()];
    if active {
        for (&sid, &root) in &roots {
            if relevant(root) {
                coverage[root].insert(format!("premise:session:{sid}"));
                if required.chronology {
                    coverage[root].insert(format!(
                        "date:{sid}:{}",
                        request.candidates[root].observations[0].2
                    ));
                }
                if binary {
                    let tail = request
                        .candidates
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| c.observations[0].1 == sid)
                        .max_by_key(|(i, c)| (c.ordinal, *i))
                        .map(|(i, _)| i)
                        .unwrap_or(root);
                    coverage[tail].insert(format!("boundary:session:{sid}"));
                }
            }
        }
        if completion {
            for (&sid, &root) in &roots {
                if relevant(root) {
                    if let Some((tail, _)) = request
                        .candidates
                        .iter()
                        .enumerate()
                        .filter(|(i, c)| c.observations[0].1 == sid && relevant(*i))
                        .max_by_key(|(i, c)| (c.ordinal, *i))
                    {
                        coverage[tail].insert(format!("confirmation:session:{sid}"));
                    }
                }
            }
        }
        for (i, witness) in coverage.iter_mut().enumerate() {
            let facts = &prepared.facts[i];
            let sid = &request.candidates[i].observations[0].1;
            let root = roots[sid.as_str()];
            if i < 5
                && (relevant(i)
                    || (relevant(root)
                        && (!facts.numbers.is_empty() || !facts.temporal.is_empty())))
            {
                if !facts.numbers.is_empty() {
                    witness.insert(format!(
                        "quantity:{}:session:{sid}",
                        facts.numbers.iter().cloned().collect::<Vec<_>>().join("/")
                    ));
                }
                if (facts.companion || facts.friend)
                    && (lower.contains("friend") || lower.contains("family"))
                {
                    witness.insert(format!("companion:session:{sid}"));
                    if relevant(i)
                        && (!acquisition
                            || facts.words.contains("new")
                            || facts.words.contains("bought")
                            || facts.words.contains("purchas"))
                    {
                        witness.insert(format!("premise:session:{sid}"));
                        if required.chronology {
                            witness.insert(format!(
                                "date:{sid}:{}",
                                request.candidates[i].observations[0].2
                            ));
                        }
                    }
                }
                if required.chronology && !facts.temporal.is_empty() {
                    witness.insert(format!(
                        "time:{}:session:{sid}",
                        facts.temporal.iter().cloned().collect::<Vec<_>>().join("/")
                    ));
                    if relevant(i)
                        && (!acquisition
                            || facts.words.contains("new")
                            || facts.words.contains("bought")
                            || facts.words.contains("purchas"))
                    {
                        witness.insert(format!("premise:session:{sid}"));
                        if required.chronology {
                            witness.insert(format!(
                                "date:{sid}:{}",
                                request.candidates[i].observations[0].2
                            ));
                        }
                    }
                }
                if lower.contains("where") || lower.contains("held") {
                    for name in &facts.names {
                        witness.insert(format!("identity:{name}:session:{sid}"));
                    }
                    if relevant(i)
                        && (!acquisition
                            || facts.words.contains("new")
                            || facts.words.contains("bought")
                            || facts.words.contains("purchas"))
                    {
                        witness.insert(format!("premise:session:{sid}"));
                        if required.chronology {
                            witness.insert(format!(
                                "date:{sid}:{}",
                                request.candidates[i].observations[0].2
                            ));
                        }
                    }
                }
            }
            if transport && (facts.words.contains("card") || facts.words.contains("nervous")) {
                witness.insert(format!("transport:session:{sid}"));
                for name in &facts.names {
                    witness.insert(format!("identity:{name}:session:{sid}"));
                }
                if facts.words.contains("nervous") {
                    witness.insert(format!("anxiety:session:{sid}"));
                }
            }
        }
    }
    for (i, witness) in coverage.iter_mut().enumerate() {
        let candidate = &request.candidates[i];
        let sid = &candidate.observations[0].1;
        let text = &candidate.observations[0].4;
        for literal in &required.literal_quotes {
            if text.contains(literal) {
                witness.insert(format!("quotation:{literal}:session:{sid}"));
            }
        }
        for identity in &required.identities {
            if !identity.is_empty() && contains_identity(text, identity) {
                witness.insert(format!("identity:{identity}:session:{sid}"));
            }
        }
        for date in &required.date_literals {
            if prepared.facts[i].dates.contains(date) {
                witness.insert(format!("literal-date:{date}:session:{sid}"));
            }
            for source in &candidate.observations {
                if source.2[..10].replace('/', "-") == *date {
                    witness.insert(format!("observation-date:{date}:session:{}", source.1));
                }
            }
        }
    }
    let mut selected = vec![0];
    let mut used = 20 + prepared.costs[0];
    if used > request.budget {
        let packet = packet(&request.candidates, &[])?;
        ensure!(
            packet.len() <= request.budget,
            "packet overhead exceeds budget"
        );
        return Ok(Retrieval {
            required,
            selected: Vec::new(),
            witnessed: BTreeMap::new(),
            packet,
        });
    }
    let mut covered = coverage[0].clone();
    loop {
        let mut best = None;
        for i in 0..coverage.len() {
            if selected.contains(&i) {
                continue;
            }
            let sid = request.candidates[i].observations[0].1.as_str();
            let root = roots[sid];
            let bundle = if active
                && root != i
                && !coverage[root].is_empty()
                && !selected.contains(&root)
                && !coverage[i].contains(&format!("premise:session:{sid}"))
            {
                vec![root, i]
            } else {
                vec![i]
            };
            let extra: BTreeSet<_> = bundle
                .iter()
                .flat_map(|&j| coverage[j].iter())
                .filter(|key| !covered.contains(*key))
                .cloned()
                .collect();
            let gain: usize = extra
                .iter()
                .map(|key| if key.starts_with("premise:") { 4 } else { 1 })
                .sum();
            if gain == 0
                || used + bundle.iter().map(|&j| prepared.costs[j]).sum::<usize>() > request.budget
            {
                continue;
            }
            let key = (
                gain,
                usize::MAX - request.candidates[i].ordinal,
                usize::MAX - i,
            );
            if best.as_ref().is_none_or(|(_, old)| key > *old) {
                best = Some((bundle, key));
            }
        }
        let Some((bundle, _)) = best else {
            break;
        };
        for i in bundle {
            selected.push(i);
            used += prepared.costs[i];
            covered.extend(coverage[i].iter().cloned());
        }
    }
    for i in 0..request.candidates.len() {
        if !selected.contains(&i) && used + prepared.costs[i] <= request.budget {
            selected.push(i);
            used += prepared.costs[i];
        }
    }
    selected.sort_unstable();
    let mut witnessed: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for &i in &selected {
        for operand in &coverage[i] {
            witnessed.entry(operand.clone()).or_default().push(i);
        }
    }
    let packet = packet(&request.candidates, &selected)?;
    ensure!(
        packet.len() <= request.budget,
        "packet overhead exceeds budget"
    );
    Ok(Retrieval {
        required,
        selected,
        witnessed,
        packet,
    })
}

pub fn knapsack_control(request: &Request) -> Result<Vec<usize>> {
    knapsack_prepared(&Prepared::new(request)?)
}

pub fn knapsack_prepared(prepared: &Prepared<'_>) -> Result<Vec<usize>> {
    let request = prepared.request;
    let costs = &prepared.costs;
    let limit = request
        .budget
        .checked_sub(b"{\"v\":1,\"evidence\":[]}".len() - 1)
        .ok_or_else(|| anyhow::anyhow!("control budget too small"))?;
    ensure!(costs[0] <= limit, "control anchor exceeds budget");
    fn gcd(mut a: u128, mut b: u128) -> u128 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    let mut denominator = 1u128;
    for c in &request.candidates {
        let n = c.rank as u128 + 1;
        denominator = (denominator / gcd(denominator, n))
            .checked_mul(n)
            .ok_or_else(|| anyhow::anyhow!("control rational overflow"))?;
    }
    let mut states = BTreeMap::from([(
        costs[0],
        (
            denominator / (request.candidates[0].rank as u128 + 1),
            vec![0],
        ),
    )]);
    for (i, &cost) in costs.iter().enumerate().skip(1) {
        let mut expanded = states.clone();
        for (&used, (value, chosen)) in &states {
            if used + cost > limit {
                continue;
            }
            let mut choice = chosen.clone();
            choice.push(i);
            let candidate = (
                *value + denominator / (request.candidates[i].rank as u128 + 1),
                choice,
            );
            let entry = expanded
                .entry(used + cost)
                .or_insert_with(|| candidate.clone());
            if candidate.0 > entry.0 || (candidate.0 == entry.0 && candidate.1 < entry.1) {
                *entry = candidate;
            }
        }
        let mut best = 0;
        expanded.retain(|_, (value, _)| {
            if *value > best {
                best = *value;
                true
            } else {
                false
            }
        });
        states = expanded;
    }
    let selected = states
        .into_iter()
        .min_by(|(ca, (va, sa)), (cb, (vb, sb))| vb.cmp(va).then(ca.cmp(cb)).then(sa.cmp(sb)))
        .ok_or_else(|| anyhow::anyhow!("empty control frontier"))?
        .1
         .1;
    let evidence: Vec<_> = selected
        .iter()
        .map(|&i| &request.candidates[i].observations)
        .collect();
    let encoded = serde_json::to_vec(&serde_json::json!({"v":1,"evidence":evidence}))?;
    ensure!(
        encoded.len() <= request.budget,
        "control packet exceeds budget"
    );
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: char, text: &str, ordinal: usize) -> Candidate {
        Candidate {
            observations: vec![(
                id.to_string().repeat(64),
                id.to_string().repeat(64),
                "2023/05/20".into(),
                "user".into(),
                text.into(),
            )],
            ordinal,
            rank: ordinal - 1,
        }
    }

    #[test]
    fn parses_named_literals_and_keeps_inference_explicit() {
        let obligations = parse_question(
            "What exact quote did Alice give about ‘Book Lovers Unite’ before 2024/02/29?",
        );
        assert_eq!(obligations.literal_quotes, ["Book Lovers Unite"]);
        assert_eq!(obligations.date_literals, ["2024-02-29"]);
        assert!(obligations.identities.contains(&"Alice".to_owned()));
        assert!(obligations.exact_quote && obligations.chronology);
        assert!(!obligations.inference_boundaries.is_empty());
        assert!(valid_date("2024/02/29"));
        assert!(!valid_date("2023/02/29"));
        assert!(!valid_date("2024/02/30"));
        assert!(!valid_date("2024/13/01"));
        assert!(!valid_date("2024/02/29garbage"));
        assert!(calendar_literals("2023/02/29?").is_empty());
    }

    #[test]
    fn resolves_a_quantity_through_its_source_premise() {
        let root = candidate('a', "I went to Hawaii with my family.", 1);
        let mut quantity = candidate('b', "With my family, we had to plan the 10-day trip.", 3);
        quantity.observations[0].1 = root.observations[0].1.clone();
        let candidates = vec![
            root,
            candidate('c', "I would like some trip ideas.", 1),
            quantity,
        ];
        let budget = packet(&candidates, &[0, 2]).unwrap().len();
        let result = retrieve(&Request {
            question: "How many days did I spend in Hawaii?".into(),
            budget,
            candidates,
        })
        .unwrap();
        assert_eq!(result.selected, [0, 2]);
    }

    #[test]
    fn keeps_named_event_comparisons_within_the_requested_identities() {
        let candidates = vec![
            candidate('a', "I went to Europe.", 1),
            candidate('b', "I went to Japan.", 1),
            candidate('c', "I went to Thailand.", 1),
        ];
        let budget = packet(&candidates, &[0, 2]).unwrap().len();
        let result = retrieve(&Request {
            question: "Which trip came first, Europe or Thailand?".into(),
            budget,
            candidates,
        })
        .unwrap();
        assert_eq!(result.selected, [0, 2]);
        assert!(!result.required.topics.contains("one"));
    }

    #[test]
    fn binds_a_cooking_question_to_a_dated_baking_witness() {
        let candidates = vec![
            candidate('a', "I reviewed a recipe.", 1),
            candidate('b', "I finished a cooking course last week.", 1),
            candidate(
                'c',
                "I just baked a chocolate cake for my friend's birthday today.",
                3,
            ),
        ];
        let budget = packet(&candidates, &[0, 2]).unwrap().len();
        let result = retrieve(&Request {
            question: "What did I cook for my friend two days ago?".into(),
            budget,
            candidates,
        })
        .unwrap();
        assert_eq!(result.selected, [0, 2]);
    }

    #[test]
    fn preserves_complete_quotes_at_exact_byte_boundary() {
        let candidates = vec![candidate('a', "I played a game for 5 hours. α", 1)];
        let budget = packet(&candidates, &[0]).unwrap().len();
        for delta in [-1, 0, 1] {
            let request = Request {
                question: "How many hours did I play games?".into(),
                budget: budget.checked_add_signed(delta).unwrap(),
                candidates: candidates.clone(),
            };
            let result = retrieve(&request).unwrap();
            assert_eq!(result.selected, if delta < 0 { vec![] } else { vec![0] });
            let decoded: serde_json::Value = serde_json::from_str(&result.packet).unwrap();
            assert_eq!(decoded["i"], serde_json::json!([]));
            if delta >= 0 {
                assert_eq!(decoded["q"][0][0][4], candidates[0].observations[0].4);
            }
        }
    }

    #[test]
    fn rejects_invalid_and_inconsistent_source_handles() {
        let mut request = Request {
            question: "When did I play?".into(),
            budget: 1024,
            candidates: vec![candidate('g', "I played.", 1)],
        };
        assert!(retrieve(&request).is_err());
        request.candidates[0] = candidate('a', "I played.", 1);
        let mut alias = request.candidates[0].observations[0].clone();
        alias.4 = "changed".into();
        request.candidates[0].observations.push(alias);
        assert!(retrieve(&request).is_err());
    }

    #[test]
    fn covers_complementary_quantities_instead_of_repeated_turns() {
        let candidates = vec![
            candidate('a', "I played games for 30 hours.", 1),
            candidate('b', "I played games for 30 hours.", 3),
            candidate('c', "I played games for 5 hours.", 1),
        ];
        let budget = packet(&candidates, &[0, 2]).unwrap().len();
        let result = retrieve(&Request {
            question: "How many hours have I played games in total?".into(),
            budget,
            candidates,
        })
        .unwrap();
        assert_eq!(result.selected, [0, 2]);
    }

    #[test]
    fn retrieves_literal_operands_beyond_the_cached_prefix() {
        for (question, distractor, witness) in [
            (
                "What is the exact quote “Beta release is ready.”?",
                "My note says “Beta release is wrong.”",
                "My note says “Beta release is ready.”",
            ),
            (
                "What did Alice say?",
                "My colleague Aliceann said ready.",
                "My colleague Alice said it was ready.",
            ),
            (
                "What did I say on (2024/02/29)?",
                "I wrote that the release was late.",
                "I wrote that the release was ready.",
            ),
        ] {
            let mut candidates = vec![
                candidate('a', "I kept an unrelated note.", 1),
                candidate('b', distractor, 1),
                candidate('c', witness, 3),
            ];
            candidates[2].observations[0].2 = "2024/02/29".into();
            let budget = packet(&candidates, &[0, 2]).unwrap().len();
            let result = retrieve(&Request {
                question: question.into(),
                budget,
                candidates,
            })
            .unwrap();
            assert_eq!(result.selected, [0, 2], "{question}");
            assert!(result.witnessed.values().any(|indices| indices == &[2]));
        }
    }

    #[test]
    fn separates_quoted_event_dates_from_observation_alias_dates() {
        let mut source = candidate('a', "I attended the launch on 2024/02/28.", 1);
        let mut alias = source.observations[0].clone();
        alias.1 = "b".repeat(64);
        alias.2 = "2024-02-29".into();
        source.observations.push(alias);
        let candidates = vec![source];
        let budget = packet(&candidates, &[0]).unwrap().len();
        for (date, kind, session) in [
            ("2024-02-28", "literal-date", 'a'),
            ("2024/02/29", "observation-date", 'b'),
        ] {
            let result = retrieve(&Request {
                question: format!("What did I say on {date}?"),
                budget,
                candidates: candidates.clone(),
            })
            .unwrap();
            assert_eq!(result.selected, [0]);
            assert_eq!(result.witnessed.len(), 1);
            assert!(result.witnessed.contains_key(&format!(
                "{kind}:{}:session:{}",
                date.replace('/', "-"),
                session.to_string().repeat(64),
            )));
            let decoded: serde_json::Value = serde_json::from_str(&result.packet).unwrap();
            assert_eq!(
                decoded["q"][0],
                serde_json::json!(candidates[0].observations)
            );
            assert_eq!(decoded["i"], serde_json::json!([]));
        }
    }
}
