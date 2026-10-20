// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// A module whose only function returns a v128 constant after an i8x16 operation: it validates
// exactly when the engine implements the fixed-width SIMD proposal.
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253,
  15, 253, 98, 11,
]);

let detected: boolean | undefined;

/** Whether this runtime can run the SIMD build. */
export async function detectSimd(): Promise<boolean> {
  detected ??= (() => {
    try {
      return typeof WebAssembly === "object" && WebAssembly.validate(SIMD_PROBE);
    } catch {
      return false;
    }
  })();
  return detected;
}
