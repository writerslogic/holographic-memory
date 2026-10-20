// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

fn main() {
    #[cfg(feature = "node-api")]
    napi_build::setup();
}
