// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Runs the Rust wasm-bindgen-test suite (bindings/wasm/tests) in headless browsers.
//   node scripts/wasm-test.mjs [chrome] [firefox]
// Needs chromedriver / geckodriver on PATH or downloadable by wasm-pack, and the browsers
// themselves (Firefox: set MOZ_FIREFOX_BINARY-style path via WARM_FIREFOX_DIR on PATH).

import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const crate = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const browsers = process.argv.slice(2).length ? process.argv.slice(2) : ["chrome", "firefox"];
const env = {
  ...process.env,
  CARGO_TARGET_DIR: process.env.WARM_WASM_TARGET_DIR ? `${process.env.WARM_WASM_TARGET_DIR}-test` : "./.warm-target-wasm-test",
  RUSTFLAGS: '--cfg getrandom_backend="wasm_js"',
};
let failed = false;
for (const browser of browsers) {
  console.log(`== wasm-bindgen-test in headless ${browser}`);
  const r = spawnSync("wasm-pack", ["test", "--headless", `--${browser}`, crate], { env, stdio: "inherit" });
  if (r.status !== 0) failed = true;
}
process.exit(failed ? 1 : 0);
