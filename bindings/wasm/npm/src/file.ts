// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import { mkdir, open, readdir, readFile, rename, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { checkKey } from "./storage.js";
import type { StorageAdapter } from "./types.js";

const SUFFIX = ".hms";

async function syncDirectory(path: string): Promise<void> {
  if (process.platform === "win32") return;
  const handle = await open(path, "r");
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
}

/**
 * File storage for Node, Deno and Bun: one `<key>.hms` file per key. A save writes a
 * temporary file, syncs it, then renames it over the old one, so a crash leaves either
 * the old snapshot or the new one.
 */
export class FileStorage implements StorageAdapter {
  readonly #directory: string;

  constructor(directory: string) {
    if (!directory) throw new Error("a directory is required");
    this.#directory = directory;
  }

  #path(key: string): string {
    return join(this.#directory, checkKey(key) + SUFFIX);
  }

  async get(key: string): Promise<Uint8Array | null> {
    try {
      return new Uint8Array(await readFile(this.#path(key)));
    } catch (error) {
      if ((error as { code?: string }).code === "ENOENT") return null;
      throw error;
    }
  }

  async set(key: string, value: Uint8Array): Promise<void> {
    const created = await mkdir(this.#directory, { recursive: true });
    const target = this.#path(key);
    const temporary = `${target}.${crypto.randomUUID()}.tmp`;
    try {
      const handle = await open(temporary, "w", 0o600);
      try {
        await handle.writeFile(value);
        await handle.sync();
      } finally {
        await handle.close();
      }
      await rename(temporary, target);
      await syncDirectory(this.#directory);
      if (created) {
        const stop = dirname(resolve(created));
        let parent = dirname(resolve(this.#directory));
        for (;;) {
          await syncDirectory(parent);
          if (parent === stop) break;
          parent = dirname(parent);
        }
      }
    } catch (error) {
      await rm(temporary, { force: true });
      throw error;
    }
  }

  async delete(key: string): Promise<void> {
    await rm(this.#path(key), { force: true });
    try {
      await syncDirectory(this.#directory);
    } catch (error) {
      if ((error as { code?: string }).code !== "ENOENT") throw error;
    }
  }

  async keys(): Promise<string[]> {
    try {
      const names = await readdir(this.#directory);
      return names.filter((n) => n.endsWith(SUFFIX)).map((n) => n.slice(0, -SUFFIX.length)).sort();
    } catch (error) {
      if ((error as { code?: string }).code === "ENOENT") return [];
      throw error;
    }
  }
}
