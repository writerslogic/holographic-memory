// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import { checkKey } from "./storage.js";
import type { StorageAdapter } from "./types.js";

export interface IndexedDbStorageOptions {
  /** Database name (default `"holographic-memory"`). */
  database?: string;
  /** Object store name (default `"snapshots"`). */
  store?: string;
  /** An IndexedDB factory, for runtimes that do not expose `indexedDB` globally. */
  factory?: IDBFactory;
}

function request<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB request failed"));
  });
}

function transaction(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("IndexedDB transaction failed"));
    tx.onabort = () => reject(tx.error ?? new Error("IndexedDB transaction aborted"));
  });
}

/** Browser storage in IndexedDB. One put is one transaction, so a save is atomic. */
export class IndexedDbStorage implements StorageAdapter {
  readonly #database: string;
  readonly #store: string;
  readonly #factory: IDBFactory | undefined;
  #db: Promise<IDBDatabase> | undefined;

  constructor(options: IndexedDbStorageOptions = {}) {
    this.#database = options.database ?? "holographic-memory";
    this.#store = options.store ?? "snapshots";
    this.#factory = options.factory ?? globalThis.indexedDB;
    if (!this.#factory) throw new Error("IndexedDB is unavailable in this runtime");
  }

  #open(): Promise<IDBDatabase> {
    this.#db ??= new Promise<IDBDatabase>((resolve, reject) => {
      const req = (this.#factory as IDBFactory).open(this.#database, 1);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(this.#store)) req.result.createObjectStore(this.#store);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => {
        this.#db = undefined;
        reject(req.error ?? new Error("cannot open IndexedDB"));
      };
    });
    return this.#db;
  }

  async get(key: string): Promise<Uint8Array | null> {
    const db = await this.#open();
    const value = await request(db.transaction(this.#store, "readonly").objectStore(this.#store).get(checkKey(key)));
    if (value === undefined) return null;
    return value instanceof Uint8Array ? value : new Uint8Array(value as ArrayBuffer);
  }

  async set(key: string, value: Uint8Array): Promise<void> {
    const db = await this.#open();
    const tx = db.transaction(this.#store, "readwrite");
    tx.objectStore(this.#store).put(value.slice(), checkKey(key));
    await transaction(tx);
  }

  async delete(key: string): Promise<void> {
    const db = await this.#open();
    const tx = db.transaction(this.#store, "readwrite");
    tx.objectStore(this.#store).delete(checkKey(key));
    await transaction(tx);
  }

  async keys(): Promise<string[]> {
    const db = await this.#open();
    const keys = await request(db.transaction(this.#store, "readonly").objectStore(this.#store).getAllKeys());
    return keys.map(String).sort();
  }

  /** Closes the connection; the adapter reopens it on the next call. */
  async close(): Promise<void> {
    const pending = this.#db;
    this.#db = undefined;
    if (pending) (await pending).close();
  }
}
