// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// One corpus size under Node (or Bun/Deno with the same entry): prints one JSON line.
//   node bench/wasm-one.mjs <documents> <queries> [scalar|simd]
import { benchOne } from "./bench-core.mjs";

const [documents, queries, variant = "simd"] = process.argv.slice(2);
const api = await import("../dist/node.js");
const started = performance.now();
const result = await benchOne(api, {
  documents: Number(documents),
  queries: Number(queries),
  simd: variant === "simd",
});
result.runtime = typeof Bun === "undefined" ? `node ${process.version}` : `bun ${Bun.version}`;
result.variant = variant;
result.wallMs = performance.now() - started;
result.rssPeakBytes = process.resourceUsage().maxRSS * 1024;
console.log(JSON.stringify(result));
