// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import { checkKey } from "./storage.js";
import type { StorageAdapter } from "./types.js";

export interface OpfsStorageOptions {
  /** Directory under the origin-private root (default `"holographic-memory"`). */
  directory?: string;
  /** A directory handle to use instead of the origin-private file system root. */
  root?: FileSystemDirectoryHandle;
}

interface SyncAccessHandle {
  truncate(size: number): void;
  write(data: Uint8Array, options?: { at?: number }): number;
  flush(): void;
  close(): void;
}

type FileHandleWithSync = FileSystemFileHandle & {
  createSyncAccessHandle?: () => Promise<SyncAccessHandle>;
};

function isNotFound(error: unknown): boolean {
  return error instanceof DOMException && error.name === "NotFoundError";
}

/**
 * Browser storage in the origin private file system. `createWritable()` commits through a swap
 * file, so a save is atomic; Workers without it fall back to a synchronous access handle.
 */
export class OpfsStorage implements StorageAdapter {
  readonly #directory: string;
  readonly #root: FileSystemDirectoryHandle | undefined;
  #dir: Promise<FileSystemDirectoryHandle> | undefined;

  constructor(options: OpfsStorageOptions = {}) {
    this.#directory = options.directory ?? "holographic-memory";
    this.#root = options.root;
    if (!options.root && !globalThis.navigator?.storage?.getDirectory) {
      throw new Error("the origin private file system is unavailable in this runtime");
    }
  }

  #open(): Promise<FileSystemDirectoryHandle> {
    this.#dir ??= (async () => {
      const root = this.#root ?? (await navigator.storage.getDirectory());
      return root.getDirectoryHandle(this.#directory, { create: true });
    })();
    this.#dir.catch(() => {
      this.#dir = undefined;
    });
    return this.#dir;
  }

  async get(key: string): Promise<Uint8Array | null> {
    const dir = await this.#open();
    try {
      const file = await (await dir.getFileHandle(checkKey(key))).getFile();
      return new Uint8Array(await file.arrayBuffer());
    } catch (error) {
      if (isNotFound(error)) return null;
      throw error;
    }
  }

  async set(key: string, value: Uint8Array): Promise<void> {
    const dir = await this.#open();
    const handle: FileHandleWithSync = await dir.getFileHandle(checkKey(key), { create: true });
    if (typeof (handle as { createWritable?: unknown }).createWritable === "function") {
      const writable = await handle.createWritable();
      try {
        await writable.write(value as Uint8Array<ArrayBuffer>);
      } catch (error) {
        await writable.abort().catch(() => undefined);
        throw error;
      }
      await writable.close();
      return;
    }
    if (!handle.createSyncAccessHandle) throw new Error("this runtime cannot write to the origin private file system");
    const access = await handle.createSyncAccessHandle();
    try {
      access.truncate(0);
      access.write(value, { at: 0 });
      access.flush();
    } finally {
      access.close();
    }
  }

  async delete(key: string): Promise<void> {
    const dir = await this.#open();
    try {
      await dir.removeEntry(checkKey(key));
    } catch (error) {
      if (!isNotFound(error)) throw error;
    }
  }

  async keys(): Promise<string[]> {
    const dir = await this.#open();
    const names: string[] = [];
    for await (const name of (dir as unknown as { keys(): AsyncIterable<string> }).keys()) names.push(name);
    return names.sort();
  }
}
