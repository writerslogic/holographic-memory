// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#[test]
fn audit2_live_nul_prefix_is_not_a_scrub_marker() {
    let dir = tempfile::tempdir().unwrap();
    let segment = create(&dir.path().join("text"));
    for prefix in [15, 16, 17] {
        let text = format!("{}live searchable text", "\0".repeat(prefix));
        let (at, len) = segment.append(&text).unwrap();
        assert_eq!(segment.read(at, len).unwrap(), text);
        assert!(
            !segment.is_scrubbed(at, len),
            "live prefix of {prefix} NUL bytes triggers the deleted-record path"
        );
    }
}
