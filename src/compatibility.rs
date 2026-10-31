// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::env::VarError;
use std::ffi::OsString;

/// Reads a WARM setting, falling back to its HMS name only when the new name is absent.
pub fn var_os(name: &str) -> Option<OsString> {
    std::env::var_os(name).or_else(|| {
        name.strip_prefix("WARM_")
            .and_then(|suffix| std::env::var_os(format!("HMS_{suffix}")))
    })
}

pub fn var(name: &str) -> Result<String, VarError> {
    var_os(name)
        .ok_or(VarError::NotPresent)?
        .into_string()
        .map_err(VarError::NotUnicode)
}

/// Selects the environment name before clap captures its value, without changing the process.
pub fn env_name(current: &'static str, legacy: &'static str) -> &'static str {
    if std::env::var_os(current).is_none() && std::env::var_os(legacy).is_some() {
        legacy
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CURRENT: &str = "WARM_COMPATIBILITY_TEST_VALUE";
    const LEGACY: &str = "HMS_COMPATIBILITY_TEST_VALUE";

    #[test]
    fn environment_precedence() {
        for case in ["absent", "legacy", "current", "both", "empty", "unicode"] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child
                .args(["--exact", "compatibility::tests::environment_probe"])
                .env("WARM_COMPATIBILITY_TEST_CASE", case)
                .env_remove(CURRENT)
                .env_remove(LEGACY);
            match case {
                "legacy" => {
                    child.env(LEGACY, "17");
                }
                "current" => {
                    child.env(CURRENT, "23");
                }
                "both" => {
                    child.env(LEGACY, "17").env(CURRENT, "23");
                }
                "empty" => {
                    child.env(LEGACY, "17").env(CURRENT, "");
                }
                "unicode" => {
                    child.env(LEGACY, "17").env(CURRENT, "記憶");
                }
                _ => {}
            }
            assert!(child.status().unwrap().success(), "case {case}");
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "compatibility::tests::environment_probe"])
                .env("WARM_COMPATIBILITY_TEST_CASE", "nonunicode")
                .env(LEGACY, "17")
                .env(CURRENT, OsString::from_vec(vec![0xff]))
                .status()
                .unwrap();
            assert!(status.success());
        }
    }

    #[test]
    fn environment_probe() {
        let Ok(case) = std::env::var("WARM_COMPATIBILITY_TEST_CASE") else {
            return;
        };
        if case == "nonunicode" {
            assert!(matches!(var(CURRENT), Err(VarError::NotUnicode(_))));
            assert_eq!(env_name(CURRENT, LEGACY), CURRENT);
            return;
        }
        let expected = match case.as_str() {
            "absent" => None,
            "legacy" => Some("17"),
            "current" | "both" => Some("23"),
            "empty" => Some(""),
            "unicode" => Some("記憶"),
            _ => panic!("unknown test case"),
        };
        assert_eq!(var(CURRENT).ok().as_deref(), expected);
        assert_eq!(
            env_name(CURRENT, LEGACY),
            if case == "legacy" { LEGACY } else { CURRENT }
        );
        #[cfg(feature = "cli")]
        {
            use clap::Parser;
            #[derive(Parser)]
            struct Settings {
                #[arg(long, env = env_name(CURRENT, LEGACY))]
                value: Option<String>,
            }
            assert_eq!(
                Settings::try_parse_from(["test"]).unwrap().value.as_deref(),
                expected
            );
            assert_eq!(
                Settings::try_parse_from(["test", "--value", "31"])
                    .unwrap()
                    .value
                    .as_deref(),
                Some("31")
            );
        }
    }
}
