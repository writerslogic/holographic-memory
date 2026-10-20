// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import type { StorageAdapter } from "./types.js";

/** Holds snapshots in memory only; the default for tests and ephemeral stores. */
export class MemoryStorage implements StorageAdapter {
  readonly #map = new Map<string, Uint8Array>();

  async get(key: string): Promise<Uint8Array | null> {
    const value = this.#map.get(key);
    return value ? value.slice() : null;
  }

  async set(key: string, value: Uint8Array): Promise<void> {
    this.#map.set(key, value.slice());
  }

  async delete(key: string): Promise<void> {
    this.#map.delete(key);
  }

  async keys(): Promise<string[]> {
    return [...this.#map.keys()].sort();
  }
}

/** Keys become file names, so they are limited to a safe alphabet. */
export function checkKey(key: string): string {
  if (!/^[A-Za-z0-9._-]{1,128}$/.test(key) || key === "." || key === "..") {
    throw new Error("storage keys are 1..128 characters from A-Z a-z 0-9 . _ -");
  }
  return key;
}
