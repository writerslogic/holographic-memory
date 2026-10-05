// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

fn main() {
    #[cfg(feature = "node-api")]
    napi_build::setup();
}
