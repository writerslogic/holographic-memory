// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Node and Bun entry. wasm-bindgen's nodejs target is CommonJS and loads its .wasm
// synchronously from disk, so it is required rather than imported.
import { createRequire } from "node:module";
import { bind } from "./common.js";
import type { WasmApi } from "./types.js";

export * from "./common.js";

const require = createRequire(import.meta.url);

export const { openStore, loadWasm } = bind(async (simd) =>
  require(simd ? "./wasm/node/simd/warm_wasm.cjs" : "./wasm/node/scalar/warm_wasm.cjs") as WasmApi,
);
