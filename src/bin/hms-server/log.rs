// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Structured logging to stderr: one JSON object (or `key=value` line) per event.
//!
//! Callers pass only identifiers, counts and timings. Request bodies, query text, vectors,
//! metadata and credentials never reach a log line; handlers do not hand them to this module.

use std::io::Write;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value};

use crate::config::{LogFormat, LogLevel};

tokio::task_local! {
    /// The id of the request being served, set by the observe middleware.
    pub static REQUEST_ID: String;
}

static SETTINGS: OnceLock<(LogFormat, LogLevel)> = OnceLock::new();

pub fn init(format: LogFormat, level: LogLevel) {
    let _ = SETTINGS.set((format, level));
}

pub fn request_id() -> Option<String> {
    REQUEST_ID.try_with(Clone::clone).ok()
}

fn level_name(level: LogLevel) -> &'static str {
    match level {
        LogLevel::Error => "error",
        LogLevel::Warn => "warn",
        LogLevel::Info => "info",
        LogLevel::Debug => "debug",
    }
}

/// RFC 3339 UTC timestamp with millisecond precision.
pub fn rfc3339(now: SystemTime) -> String {
    let since = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = since.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil-from-days (Howard Hinnant), valid for the whole i64 day range used here.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
        since.subsec_millis()
    )
}

pub fn emit(level: LogLevel, event: &str, fields: &[(&str, Value)]) {
    let (format, minimum) = SETTINGS
        .get()
        .copied()
        .unwrap_or((LogFormat::Json, LogLevel::Info));
    if level > minimum {
        return;
    }
    let ts = rfc3339(SystemTime::now());
    let line = match format {
        LogFormat::Json => {
            let mut map = Map::new();
            map.insert("ts".into(), ts.into());
            map.insert("level".into(), level_name(level).into());
            map.insert("event".into(), event.into());
            if let Some(id) = request_id() {
                map.insert("request_id".into(), id.into());
            }
            for (key, value) in fields {
                map.insert((*key).into(), value.clone());
            }
            Value::Object(map).to_string()
        }
        LogFormat::Text => {
            let mut line = format!("{ts} {} {event}", level_name(level));
            if let Some(id) = request_id() {
                line.push_str(&format!(" request_id={id}"));
            }
            for (key, value) in fields {
                match value {
                    Value::String(s) => line.push_str(&format!(" {key}={s:?}")),
                    other => line.push_str(&format!(" {key}={other}")),
                }
            }
            line
        }
    };
    // A closed stderr must not take the server down.
    let _ = writeln!(std::io::stderr().lock(), "{line}");
}

#[macro_export]
macro_rules! log_event {
    ($level:ident, $event:expr $(, $key:literal => $value:expr)* $(,)?) => {
        $crate::log::emit(
            $crate::config::LogLevel::$level,
            $event,
            &[$(($key, ::serde_json::json!($value))),*],
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn rfc3339_matches_known_instants() {
        assert_eq!(rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
        // 2024-02-29T12:34:56.789Z, a leap day.
        let t = UNIX_EPOCH + Duration::new(1_709_210_096, 789_000_000);
        assert_eq!(rfc3339(t), "2024-02-29T12:34:56.789Z");
        let t = UNIX_EPOCH + Duration::from_secs(4_102_444_799);
        assert_eq!(rfc3339(t), "2099-12-31T23:59:59.000Z");
    }
}
