// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import { HolographicStore } from "./store.js";
import type { OpenOptions, WasmApi, WasmLoader } from "./types.js";

export { HolographicStore } from "./store.js";
export { MemoryStorage } from "./storage.js";
export { detectSimd } from "./simd.js";
export { isEncrypted } from "./crypto.js";
export type * from "./types.js";

/** Builds the `openStore` / `loadWasm` pair for one runtime's module loader. */
export function bind(loader: WasmLoader) {
  return {
    /**
     * Opens a store. With `storage` it loads the saved snapshot and autosaves changes;
     * without, it is in memory.
     */
    openStore: (options?: OpenOptions): Promise<HolographicStore> => HolographicStore.open(loader, options),
    /** Loads the raw module (for `argon2idKey`, `simdBuild`, `version`). */
    loadWasm: (simd?: boolean): Promise<WasmApi> => loader(simd ?? false),
  };
}
