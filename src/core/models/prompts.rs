// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Prompts and output parsing of the tuned LongMemEval pipeline
//! (`benchmarks/public/longmemeval_pipeline.py`). The text and the parsing rules are part of the
//! tuned result and must stay byte-identical to the Python reference.

use serde_json::Value;

/// (year, month, day).
pub type Date = (i32, u32, u32);
/// (turn index, fact); `None` when the model named no valid turn.
pub type Facts = Vec<(Option<usize>, String)>;

pub const EMBED_TASK: &str = "Given a question a user asks about their earlier conversations with an AI assistant, retrieve the user messages that contain the information needed to answer it";
pub const RERANK_TASK: &str = "Given a question a user asks about their earlier conversations with an AI assistant, judge whether the conversation excerpt contains information needed to answer it";

pub const RERANK_PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
pub const RERANK_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

const FACT_PROMPT: &str = r#"Below are the user's messages from one conversation with an AI assistant, numbered.

List every piece of personal information the user reveals: facts about themselves, people they know, possessions, purchases, places, events, plans, preferences, opinions, habits, numbers and dates. Write each as one short self-contained sentence in the third person ("The user ...") that keeps the specifics (names, items, quantities, and time expressions as the user stated them). Skip generic requests that reveal nothing about the user.

Answer with JSON only: {"facts": [{"turn": <message number>, "fact": "<sentence>"}]}. Use {"facts": []} if there are none.

"#;

const QUERY_PROMPT_HEAD: &str = "A user is asking an AI assistant a question about their earlier conversations with it. Today is ";
const QUERY_PROMPT_MID: &str = ".\n\nQuestion: ";
const QUERY_PROMPT_TAIL: &str = r#"

Answer with JSON only, with these fields:
"rewrite": the question restated as a search query that names the key entities, with likely synonyms;
"subqueries": a list of up to 4 short search queries, one per separate fact the answer needs (for example each event to compare, each item to count, the old and the new value of something that changed); an empty list if one search suffices;
"time_range": ["YYYY/MM/DD", "YYYY/MM/DD"] if the question restricts when the relevant conversations happened (for example "last week", "in March", "two months ago", "the first time"... only when a calendar range is implied), resolved against today; otherwise null."#;

/// Python `s[:n]`: the first `n` Unicode scalar values.
fn take_chars(s: &str, n: usize) -> &str {
    s.char_indices().nth(n).map_or(s, |(i, _)| &s[..i])
}

/// `Session.fact_input`: numbered user turns, each cut to 1500 characters, the whole to 24000.
pub fn fact_input<S: AsRef<str>>(user_turns: &[S]) -> String {
    let joined = user_turns
        .iter()
        .enumerate()
        .map(|(i, t)| format!("[{}] {}", i + 1, take_chars(t.as_ref(), 1500)))
        .collect::<Vec<_>>()
        .join("\n");
    take_chars(&joined, 24000).to_string()
}

pub fn fact_prompt<S: AsRef<str>>(user_turns: &[S]) -> String {
    format!("{FACT_PROMPT}{}", fact_input(user_turns))
}

/// `QUERY_PROMPT.format(today=..., question=...)`; a single pass, as `str.format` does.
pub fn query_prompt(today: &str, question: &str) -> String {
    format!("{QUERY_PROMPT_HEAD}{today}{QUERY_PROMPT_MID}{question}{QUERY_PROMPT_TAIL}")
}

pub fn embed_query(query: &str) -> String {
    format!("Instruct: {EMBED_TASK}\nQuery:{query}")
}

pub fn rerank_body(query: &str, document: &str) -> String {
    format!("<Instruct>: {RERANK_TASK}\n<Query>: {query}\n<Document>: {document}")
}

/// Qwen3 chat template for one user message with `add_generation_prompt=True` and
/// `enable_thinking=False` (the Instruct-2507 template emits no think block).
pub fn chat(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

/// `_parse_json`: the outermost `{...}` span of the model output, if it is a JSON object.
pub fn parse_json(text: &str) -> Option<serde_json::Map<String, Value>> {
    let a = text.find('{')?;
    let b = text.rfind('}')?;
    if b <= a {
        return None;
    }
    match serde_json::from_str::<Value>(&text[a..=b]) {
        Ok(Value::Object(m)) => Some(m),
        _ => None,
    }
}

/// `session_facts`: (turn index, fact); a missing or out-of-range turn maps to `None`.
pub fn parse_facts(parsed: Option<&serde_json::Map<String, Value>>, n_turns: usize) -> Facts {
    let Some(Value::Array(items)) = parsed.and_then(|m| m.get("facts")) else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|f| {
            let fact = f.get("fact")?.as_str()?.trim();
            if fact.is_empty() {
                return None;
            }
            // Python: isinstance(t, int) (bool included) or a str of digits; int(t) - 1.
            let i: Option<i128> = match f.get("turn") {
                Some(Value::Bool(b)) => Some(i128::from(*b) - 1),
                Some(Value::Number(n)) if n.is_i64() || n.is_u64() => {
                    n.as_i64().map(|v| i128::from(v) - 1).or(Some(i128::MAX))
                }
                Some(Value::String(s))
                    if !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()) =>
                {
                    Some(s.parse::<i128>().map_or(i128::MAX, |v| v - 1))
                }
                _ => None,
            };
            let turn = i
                .filter(|&i| i >= 0 && (i as usize) < n_turns)
                .map(|i| i as usize);
            Some((turn, fact.to_string()))
        })
        .collect()
}

/// A parsed query rewrite: `query_variants` and `time_range` of the reference pipeline.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct QueryPlan {
    pub rewrite: Option<String>,
    pub subqueries: Vec<String>,
    /// Inclusive (start, end) as (year, month, day), ordered.
    pub time_range: Option<(Date, Date)>,
}

pub fn parse_query_plan(parsed: Option<&serde_json::Map<String, Value>>) -> QueryPlan {
    let Some(m) = parsed else {
        return QueryPlan::default();
    };
    let rewrite = m
        .get("rewrite")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from);
    let raw: Vec<Value> = match m.get("subqueries") {
        Some(Value::Array(a)) => a.clone(),
        // Python iterates a string's characters.
        Some(Value::String(s)) => s.chars().map(|c| Value::String(c.into())).collect(),
        _ => Vec::new(),
    };
    let subqueries = raw
        .iter()
        .take(4)
        .filter_map(|v| v.as_str().map(str::trim).filter(|s| !s.is_empty()))
        .map(String::from)
        .collect();
    let time_range = match m.get("time_range") {
        Some(Value::Array(a)) if a.len() == 2 => {
            let text = |v: &Value| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            match (parse_date(&text(&a[0])), parse_date(&text(&a[1]))) {
                (Some(x), Some(y)) => Some((x.min(y), x.max(y))),
                _ => None,
            }
        }
        _ => None,
    };
    QueryPlan {
        rewrite,
        subqueries,
        time_range,
    }
}

/// `parse_date`: `\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})` at the start, a valid calendar date.
pub fn parse_date(s: &str) -> Option<Date> {
    let s = s.trim_start_matches(|c: char| c.is_whitespace());
    let b = s.as_bytes();
    let digits = |from: usize, max: usize| {
        let n = b[from..]
            .iter()
            .take(max)
            .take_while(|c| c.is_ascii_digit())
            .count();
        (n > 0).then(|| (s[from..from + n].parse::<u32>().ok(), from + n))
    };
    if b.len() < 4 || !b[..4].iter().all(u8::is_ascii_digit) {
        return None;
    }
    let y: i32 = s[..4].parse().ok()?;
    let sep = |i: usize| b.get(i).is_some_and(|c| *c == b'/' || *c == b'-');
    if !sep(4) {
        return None;
    }
    let (m, i) = digits(5, 2)?;
    if !sep(i) {
        return None;
    }
    let (d, _) = digits(i + 1, 2)?;
    let (m, d) = (m?, d?);
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let days = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return None,
    };
    (y >= 1 && (1..=days).contains(&d)).then_some((y, m, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_input_cuts_by_characters() {
        let long = "é".repeat(2000);
        let s = fact_input(&[long.as_str(), "b"]);
        assert_eq!(s.chars().count(), 4 + 1500 + 1 + 5);
        assert!(s.ends_with("\n[2] b"));
        let many: Vec<String> = (0..40).map(|_| "x".repeat(1500)).collect();
        assert_eq!(fact_input(&many).chars().count(), 24000);
    }

    #[test]
    fn prompts_render_like_python_format() {
        let p = query_prompt("2023/05/30 (Tue) 23:40", "What {today} did I buy?");
        assert!(p.starts_with("A user is asking an AI assistant a question about their earlier conversations with it. Today is 2023/05/30 (Tue) 23:40.\n\nQuestion: What {today} did I buy?\n\nAnswer with JSON only"));
        assert!(p.ends_with("resolved against today; otherwise null."));
        let f = fact_prompt(&["hi"]);
        assert!(f.contains("Answer with JSON only: {\"facts\": [{\"turn\": <message number>, \"fact\": \"<sentence>\"}]}. Use {\"facts\": []} if there are none.\n\n[1] hi"));
        assert_eq!(embed_query("q"), format!("Instruct: {EMBED_TASK}\nQuery:q"));
    }

    #[test]
    fn facts_follow_session_facts_rules() {
        let out = r#"noise {"facts": [{"turn": 1, "fact": " A "}, {"turn": "2", "fact": "B"},
            {"turn": 3, "fact": "C"}, {"turn": 0, "fact": "D"}, {"turn": 1.0, "fact": "E"},
            {"turn": true, "fact": "F"}, {"fact": ""}, {"turn": 1}, "x"]} tail"#;
        let parsed = parse_json(out);
        let facts = parse_facts(parsed.as_ref(), 2);
        let want: Vec<(Option<usize>, String)> = vec![
            (Some(0), "A".into()),
            (Some(1), "B".into()),
            (None, "C".into()),
            (None, "D".into()),
            (None, "E".into()),
            (Some(0), "F".into()),
        ];
        assert_eq!(facts, want);
        assert!(parse_facts(parse_json("no json").as_ref(), 2).is_empty());
        assert!(parse_json("} {").is_none());
        assert!(parse_json("[1] {").is_none());
    }

    #[test]
    fn query_plan_follows_query_variants_rules() {
        let out = r#"{"rewrite": "  r ", "subqueries": ["a", 5, " ", "b", "c", "d"],
            "time_range": ["2023/05/30", "2023-5-1"]}"#;
        let plan = parse_query_plan(parse_json(out).as_ref());
        assert_eq!(plan.rewrite.as_deref(), Some("r"));
        assert_eq!(plan.subqueries, vec!["a", "b"]);
        assert_eq!(plan.time_range, Some(((2023, 5, 1), (2023, 5, 30))));
        let bad = parse_query_plan(
            parse_json(r#"{"time_range": ["2023/02/30", "2023/03/01"]}"#).as_ref(),
        );
        assert_eq!(bad, QueryPlan::default());
    }

    #[test]
    fn parse_date_matches_python_regex() {
        assert_eq!(parse_date(" 2024/02/29 (Thu)"), Some((2024, 2, 29)));
        assert_eq!(parse_date("2023/02/29"), None);
        assert_eq!(parse_date("2023/13/01"), None);
        assert_eq!(parse_date("23/01/01"), None);
        assert_eq!(parse_date("2023/1/123"), Some((2023, 1, 12)));
        assert_eq!(parse_date("0000/01/01"), None);
    }
}
