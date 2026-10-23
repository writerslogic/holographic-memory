// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! The memory model and request validation the tool handlers are written against.

use serde::Serialize;
use serde_json::{json, Map, Value};

/// Structured failure with a stable machine-readable code.
#[derive(Debug, Clone, Serialize)]
pub struct Fail {
    pub code: &'static str,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

pub type Res<T> = Result<T, Fail>;

impl Fail {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            details: None,
        }
    }
    pub fn with(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new("invalid_argument", message)
    }
    pub fn not_found(id: &str) -> Self {
        Self::new("not_found", format!("no memory with id {id:?}")).with(json!({"id": id}))
    }
    /// An unexpected failure: the cause goes to stderr, the caller gets a fixed message.
    pub fn internal(cause: impl std::fmt::Display) -> Self {
        crate::log::error(&format!("internal error: {cause:#}"));
        Self::new("internal", "internal error; see the server log")
    }
}

#[derive(Debug, Clone)]
pub struct Limits {
    pub max_text_bytes: usize,
    pub max_metadata_bytes: usize,
    pub max_tags: usize,
    pub max_top_k: usize,
    pub max_page: usize,
    pub max_export_bytes: usize,
    pub max_forget: usize,
}

pub const MAX_ID_BYTES: usize = 256;
pub const MAX_TITLE_CHARS: usize = 256;
pub const MAX_TAG_CHARS: usize = 64;
pub const MAX_QUERY_BYTES: usize = 4096;
/// Prefix of metadata keys the server owns; callers cannot set them.
pub const RESERVED: &str = "hms.";

#[derive(Debug, Clone, Serialize)]
pub struct Memory {
    pub id: String,
    pub text: String,
    /// Length of `text` in UTF-8 bytes (stays exact when a listing truncates `text`).
    pub bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub tags: Vec<String>,
    pub metadata: Map<String, Value>,
    /// Milliseconds since the Unix epoch.
    pub created_at: i64,
    pub updated_at: i64,
    /// Changes on every write; pass it as `if_version` for compare-and-set.
    pub version: String,
}

#[derive(Debug, Clone)]
pub struct RememberReq {
    pub text: String,
    pub id: Option<String>,
    pub title: Option<String>,
    pub tags: Vec<String>,
    pub metadata: Map<String, Value>,
    pub if_version: Option<String>,
    pub create_only: bool,
    pub allow_empty: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RememberOut {
    pub id: String,
    pub version: String,
    pub created: bool,
    pub bytes: usize,
    pub namespace: String,
}

#[derive(Debug, Clone)]
pub struct SearchReq {
    pub query: String,
    pub k: usize,
    pub tags: Vec<String>,
    pub filter: Map<String, Value>,
    pub min_score: Option<f64>,
    pub lexical_weight: Option<f64>,
    pub semantic_weight: Option<f64>,
    pub fusion: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub memory: Memory,
    /// Relative rank score in 0..=1 (top-normalized lexical and semantic blend), not a probability.
    pub score: f64,
    pub lexical_score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_score: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct ListReq {
    pub prefix: Option<String>,
    pub tag: Option<String>,
    /// Id of the last memory of the previous page.
    pub cursor: Option<String>,
    pub limit: usize,
}

#[derive(Debug, Clone)]
pub struct Page {
    pub memories: Vec<Memory>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ForgetReq {
    pub ids: Vec<String>,
    pub prefix: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ForgetOut {
    pub deleted: usize,
    pub missing: Vec<String>,
}

pub fn validate_id(id: &str) -> Res<()> {
    if id.is_empty() || id.len() > MAX_ID_BYTES || id.chars().any(char::is_control) {
        return Err(Fail::invalid(format!(
            "id must be 1..={MAX_ID_BYTES} bytes without control characters"
        )));
    }
    if id.starts_with("hms:") {
        return Err(Fail::invalid("ids starting with \"hms:\" are reserved"));
    }
    Ok(())
}

pub fn validate_tag(tag: &str) -> Res<()> {
    if tag.is_empty() || tag.chars().count() > MAX_TAG_CHARS || tag.chars().any(char::is_control) {
        return Err(Fail::invalid(format!(
            "a tag is 1..={MAX_TAG_CHARS} characters without control characters"
        )));
    }
    Ok(())
}

pub fn validate_metadata(meta: &Map<String, Value>, limits: &Limits) -> Res<()> {
    if let Some(key) = meta
        .keys()
        .find(|k| k.starts_with(RESERVED) || k.is_empty())
    {
        return Err(Fail::invalid(format!(
            "metadata key {key:?} is empty or uses the reserved prefix {RESERVED:?}"
        )));
    }
    let size = serde_json::to_vec(meta).map_or(usize::MAX, |v| v.len());
    if size > limits.max_metadata_bytes {
        return Err(Fail::new(
            "too_large",
            format!("metadata exceeds {} bytes", limits.max_metadata_bytes),
        ));
    }
    Ok(())
}

impl RememberReq {
    pub fn validate(&self, limits: &Limits) -> Res<()> {
        if self.text.len() > limits.max_text_bytes {
            return Err(Fail::new(
                "too_large",
                format!(
                    "text is {} bytes; the limit is {}. Split it into several memories.",
                    self.text.len(),
                    limits.max_text_bytes
                ),
            ));
        }
        if self.text.trim().is_empty() && !self.allow_empty {
            return Err(Fail::invalid(
                "text must not be blank (pass allow_empty for file-like use)",
            ));
        }
        if self.text.split_whitespace().count() > 4096 {
            return Err(Fail::new(
                "too_large",
                "text has more than 4096 words; split it into several memories",
            ));
        }
        if let Some(id) = &self.id {
            validate_id(id)?;
        }
        if let Some(title) = &self.title {
            if title.chars().count() > MAX_TITLE_CHARS || title.chars().any(char::is_control) {
                return Err(Fail::invalid(format!(
                    "title is at most {MAX_TITLE_CHARS} characters without control characters"
                )));
            }
        }
        if self.tags.len() > limits.max_tags {
            return Err(Fail::invalid(format!(
                "at most {} tags per memory",
                limits.max_tags
            )));
        }
        self.tags.iter().try_for_each(|t| validate_tag(t))?;
        validate_metadata(&self.metadata, limits)?;
        if self.create_only && self.if_version.is_some() {
            return Err(Fail::invalid(
                "create_only and if_version cannot be combined",
            ));
        }
        Ok(())
    }
}

impl SearchReq {
    pub fn validate(&self, limits: &Limits) -> Res<()> {
        if self.query.trim().is_empty() || self.query.len() > MAX_QUERY_BYTES {
            return Err(Fail::invalid(format!(
                "query must be non-blank and at most {MAX_QUERY_BYTES} bytes"
            )));
        }
        if !(1..=limits.max_top_k).contains(&self.k) {
            return Err(Fail::invalid(format!("k must be 1..={}", limits.max_top_k)));
        }
        if self.tags.len() > limits.max_tags {
            return Err(Fail::invalid("too many tags"));
        }
        self.tags.iter().try_for_each(|t| validate_tag(t))?;
        if let Some(key) = self.filter.keys().find(|k| k.starts_with(RESERVED)) {
            return Err(Fail::invalid(format!(
                "filter key {key:?} uses the reserved prefix {RESERVED:?}; use tags"
            )));
        }
        if self.filter.len() > 16 {
            return Err(Fail::invalid("filter has at most 16 keys"));
        }
        if let Some(s) = self.min_score {
            if !(0.0..=1.0).contains(&s) {
                return Err(Fail::invalid("min_score must be within 0..=1"));
            }
        }
        for w in [self.lexical_weight, self.semantic_weight]
            .into_iter()
            .flatten()
        {
            if !w.is_finite() || !(0.0..=100.0).contains(&w) {
                return Err(Fail::invalid("weights must be within 0..=100"));
            }
        }
        if self
            .fusion
            .as_deref()
            .is_some_and(|f| f != "blend" && f != "rrf")
        {
            return Err(Fail::invalid("fusion must be \"blend\" or \"rrf\""));
        }
        Ok(())
    }
}

impl ListReq {
    pub fn validate(&self, limits: &Limits) -> Res<()> {
        if !(1..=limits.max_page).contains(&self.limit) {
            return Err(Fail::invalid(format!(
                "limit must be 1..={}",
                limits.max_page
            )));
        }
        if let Some(tag) = &self.tag {
            validate_tag(tag)?;
        }
        for s in [&self.prefix, &self.cursor].into_iter().flatten() {
            if s.len() > MAX_ID_BYTES || s.contains('\0') {
                return Err(Fail::invalid("prefix and cursor are at most 256 bytes"));
            }
        }
        Ok(())
    }
}

impl ForgetReq {
    pub fn validate(&self, limits: &Limits) -> Res<()> {
        if self.ids.is_empty() == self.prefix.is_none() {
            return Err(Fail::invalid("pass either ids or prefix"));
        }
        if self.ids.len() > limits.max_forget {
            return Err(Fail::invalid(format!(
                "at most {} ids per call",
                limits.max_forget
            )));
        }
        self.ids.iter().try_for_each(|id| validate_id(id))?;
        if let Some(p) = &self.prefix {
            if p.is_empty() || p.len() > MAX_ID_BYTES || p.contains('\0') {
                return Err(Fail::invalid("prefix must be 1..=256 bytes"));
            }
        }
        Ok(())
    }
}

/// Truncate to at most `max` characters on a character boundary.
pub fn preview(text: &str, max: usize) -> (String, bool) {
    match text.char_indices().nth(max) {
        Some((at, _)) => (text[..at].to_string(), true),
        None => (text.to_string(), false),
    }
}

#[cfg(test)]
pub fn test_limits() -> Limits {
    Limits {
        max_text_bytes: 100,
        max_metadata_bytes: 64,
        max_tags: 2,
        max_top_k: 5,
        max_page: 10,
        max_export_bytes: 4096,
        max_forget: 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remember(text: &str) -> RememberReq {
        RememberReq {
            text: text.into(),
            id: None,
            title: None,
            tags: vec![],
            metadata: Map::new(),
            if_version: None,
            create_only: false,
            allow_empty: false,
        }
    }

    #[test]
    fn text_limit_is_inclusive() {
        let limits = test_limits();
        assert!(remember(&"a".repeat(100)).validate(&limits).is_ok());
        let err = remember(&"a".repeat(101)).validate(&limits).unwrap_err();
        assert_eq!(err.code, "too_large");
        assert_eq!(
            remember("   ").validate(&limits).unwrap_err().code,
            "invalid_argument"
        );
    }

    fn code(result: Res<()>) -> Option<&'static str> {
        result.err().map(|e| e.code)
    }

    #[test]
    fn remember_bounds() {
        let limits = test_limits();
        let with_id = |id: &str| RememberReq {
            id: Some(id.into()),
            ..remember("x")
        };
        let tags = |n: usize| RememberReq {
            tags: (0..n).map(|i| format!("t{i}")).collect(),
            ..remember("x")
        };
        // {"k":"<v>"} is 8 bytes plus the value.
        let meta = |value: &str| {
            let mut m = Map::new();
            m.insert("k".into(), Value::String(value.into()));
            RememberReq {
                metadata: m,
                ..remember("x")
            }
        };
        let reserved_key = RememberReq {
            metadata: Map::from_iter([("hms.created".to_string(), Value::Null)]),
            ..remember("x")
        };
        for (name, request, expected) in [
            ("id at limit", with_id(&"i".repeat(MAX_ID_BYTES)), None),
            (
                "id over limit",
                with_id(&"i".repeat(MAX_ID_BYTES + 1)),
                Some("invalid_argument"),
            ),
            ("empty id", with_id(""), Some("invalid_argument")),
            (
                "reserved id",
                with_id("hms:chunk:1"),
                Some("invalid_argument"),
            ),
            (
                "control character in id",
                with_id("a\nb"),
                Some("invalid_argument"),
            ),
            ("tags at limit", tags(2), None),
            ("tags over limit", tags(3), Some("invalid_argument")),
            ("metadata at limit", meta(&"v".repeat(56)), None),
            (
                "metadata over limit",
                meta(&"v".repeat(57)),
                Some("too_large"),
            ),
            (
                "reserved metadata key",
                reserved_key,
                Some("invalid_argument"),
            ),
        ] {
            assert_eq!(code(request.validate(&limits)), expected, "{name}");
        }
    }

    #[test]
    fn search_list_and_forget_bounds() {
        let limits = test_limits();
        let search = |k: usize| SearchReq {
            query: "q".into(),
            k,
            tags: vec![],
            filter: Map::new(),
            min_score: None,
            lexical_weight: None,
            semantic_weight: None,
            fusion: None,
        };
        let list = |limit: usize| ListReq {
            limit,
            ..ListReq::default()
        };
        let forget = |n: usize| ForgetReq {
            ids: (0..n).map(|i| format!("id{i}")).collect(),
            prefix: None,
        };
        for (name, result, expected) in [
            (
                "search k below range",
                search(0).validate(&limits),
                Some("invalid_argument"),
            ),
            ("search k at limit", search(5).validate(&limits), None),
            (
                "search k over limit",
                search(6).validate(&limits),
                Some("invalid_argument"),
            ),
            (
                "list limit below range",
                list(0).validate(&limits),
                Some("invalid_argument"),
            ),
            ("list limit at limit", list(10).validate(&limits), None),
            (
                "list limit over limit",
                list(11).validate(&limits),
                Some("invalid_argument"),
            ),
            (
                "forget without ids or prefix",
                forget(0).validate(&limits),
                Some("invalid_argument"),
            ),
            ("forget ids at limit", forget(3).validate(&limits), None),
            (
                "forget ids over limit",
                forget(4).validate(&limits),
                Some("invalid_argument"),
            ),
        ] {
            assert_eq!(code(result), expected, "{name}");
        }
    }

    #[test]
    fn preview_cuts_on_a_character_boundary() {
        assert_eq!(preview("héllo wörld", 5), ("héllo".to_string(), true));
        assert_eq!(preview("abc", 3), ("abc".to_string(), false));
    }
}
