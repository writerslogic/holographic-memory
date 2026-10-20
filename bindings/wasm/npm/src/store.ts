// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import {
  cipherFor,
  isEncrypted,
  newCipher,
  open as openEnvelope,
  seal,
  type CipherState,
} from "./crypto.js";
import { detectSimd } from "./simd.js";
import type {
  AutosaveOptions,
  DocumentChunk,
  DocumentInput,
  DocumentResult,
  EncryptionOptions,
  OpenOptions,
  SearchOptions,
  StorageAdapter,
  StoreStats,
  WasmApi,
  WasmLoader,
  WasmStore,
} from "./types.js";

const DEFAULT_KEY = "hms-store";
const DEFAULT_DEBOUNCE_MS = 500;

/** Typed arrays become plain arrays, which is what the module's deserializer accepts. */
function plainEmbeddings<T extends { embeddings?: ArrayLike<number>[] | null; embedding?: ArrayLike<number> | null }>(
  value: T,
): T {
  if (!value.embeddings && !value.embedding) return value;
  const copy = { ...value };
  if (copy.embeddings) copy.embeddings = copy.embeddings.map((e) => Array.from(e));
  if (copy.embedding) copy.embedding = Array.from(copy.embedding);
  return copy;
}

/**
 * A local document store running in WebAssembly. Ingest, BM25 search, delete and stats run
 * synchronously inside the module (wrapped in promises to match the Node binding); the
 * serialized store persists through a {@link StorageAdapter}, optionally encrypted.
 *
 * Calls run on the thread that makes them; for large corpora use the store from a Worker.
 */
export class HolographicStore {
  readonly #wasm: WasmApi;
  readonly #getWasm: () => Promise<WasmApi>;
  #inner: WasmStore | null;
  readonly #storage: StorageAdapter | undefined;
  readonly #key: string;
  #cipher: CipherState | null;
  readonly #debounceMs: number | null;
  #timer: ReturnType<typeof setTimeout> | undefined;
  #dirty = false;
  #saving: Promise<void> = Promise.resolve();
  #autosaveError: unknown;

  private constructor(
    wasm: WasmApi,
    getWasm: () => Promise<WasmApi>,
    inner: WasmStore,
    options: OpenOptions,
    cipher: CipherState | null,
  ) {
    this.#wasm = wasm;
    this.#getWasm = getWasm;
    this.#inner = inner;
    this.#storage = options.storage;
    this.#key = options.key ?? DEFAULT_KEY;
    this.#cipher = cipher;
    const autosave = options.autosave ?? (options.storage ? true : false);
    this.#debounceMs =
      autosave === false ? null : autosave === true ? DEFAULT_DEBOUNCE_MS : ((autosave as AutosaveOptions).debounceMs ?? DEFAULT_DEBOUNCE_MS);
    if (this.#debounceMs !== null && !this.#storage) {
      throw new Error("autosave needs a storage adapter");
    }
  }

  /** Opens a store, loading the saved snapshot (or `options.snapshot`) when there is one. */
  static async open(loader: WasmLoader, options: OpenOptions = {}): Promise<HolographicStore> {
    const simd = options.simd === undefined || options.simd === "auto" ? await detectSimd() : options.simd;
    let cached: Promise<WasmApi> | undefined;
    const getWasm = () => (cached ??= loader(simd));
    const wasm = await getWasm();

    let bytes: Uint8Array | null = options.snapshot ?? null;
    if (!bytes && options.storage) bytes = await options.storage.get(options.key ?? DEFAULT_KEY);

    let cipher: CipherState | null = null;
    let inner: WasmStore;
    if (bytes) {
      let plain = bytes;
      if (isEncrypted(bytes)) {
        if (!options.encryption) throw new Error("the snapshot is encrypted: provide `encryption.passphrase`");
        cipher = await cipherFor(bytes, options.encryption.passphrase, getWasm);
        plain = await openEnvelope(cipher, bytes);
      } else if (options.encryption) {
        throw new Error(
          "the snapshot is not encrypted but `encryption` was given; open it without encryption and call setEncryption()",
        );
      }
      inner = wasm.HmsStore.importBytes(plain);
      for (const [name, want, have] of [
        ["dimensions", options.dimensions, inner.dimensions],
        ["embeddingDimensions", options.embeddingDimensions, inner.stats().embeddingDimensions],
      ] as const) {
        if (want !== undefined && want !== have) {
          inner.free();
          throw new Error(`the snapshot has ${name} ${String(have)}, not the requested ${want}`);
        }
      }
    } else {
      inner = new wasm.HmsStore({
        dimensions: options.dimensions,
        embeddingDimensions: options.embeddingDimensions,
      });
      if (options.encryption) cipher = await newCipher(options.encryption, getWasm);
    }
    return new HolographicStore(wasm, getWasm, inner, options, cipher);
  }

  get #store(): WasmStore {
    if (!this.#inner) throw new Error("the store is closed");
    return this.#inner;
  }

  /** Stores or replaces a document; resolves to its chunk count. */
  async memorizeDocument(input: DocumentInput): Promise<number> {
    const count = this.#store.memorizeDocument(plainEmbeddings(input));
    this.#changed();
    return count;
  }

  /** Stores or replaces several documents atomically; resolves to each one's chunk count. */
  async memorizeDocuments(inputs: DocumentInput[]): Promise<number[]> {
    const counts = Array.from(this.#store.memorizeDocuments(inputs.map(plainEmbeddings)));
    this.#changed();
    return counts;
  }

  /** Splits a document into chunks without storing it. */
  async chunkDocument(input: DocumentInput): Promise<DocumentChunk[]> {
    return this.#store.chunkDocument(input);
  }

  /** BM25 over the stored passages, optionally fused with dense embeddings. */
  async searchDocuments(text: string, options?: SearchOptions): Promise<DocumentResult[]> {
    return this.#store.searchDocuments(text, options ? plainEmbeddings(options) : undefined);
  }

  /** Removes a document; resolves to whether it existed. */
  async deleteDocument(id: string): Promise<boolean> {
    const existed = this.#store.deleteDocument(id);
    if (existed) this.#changed();
    return existed;
  }

  async hasDocument(id: string): Promise<boolean> {
    return this.#store.hasDocument(id);
  }

  async documentIds(): Promise<string[]> {
    return this.#store.documentIds();
  }

  async stats(): Promise<StoreStats> {
    return this.#store.stats();
  }

  /** Builds the dense index over stored embeddings; resolves to whether one is current. */
  async trainDocumentIndex(): Promise<boolean> {
    return this.#store.trainDocumentIndex();
  }

  get documentIndexTrained(): boolean {
    return this.#store.documentIndexTrained;
  }

  get dimensions(): number {
    return this.#store.dimensions;
  }

  /** Size of the module's linear memory in bytes (a high-water mark: it never shrinks). */
  get memoryBytes(): number {
    return this.#wasm.memoryBytes();
  }

  /** Whether the WebAssembly build in use is the SIMD one. */
  get simd(): boolean {
    return this.#wasm.simdBuild();
  }

  /** The serialized store: the exact bytes `save()` writes (encrypted when encryption is on). */
  async exportSnapshot(): Promise<Uint8Array> {
    const plain = this.#store.exportBytes();
    return this.#cipher ? seal(this.#cipher, plain) : plain;
  }

  /** Turns encryption on, changes it, or (with `null`) off; the next save uses it. */
  async setEncryption(options: EncryptionOptions | null): Promise<void> {
    this.#cipher = options ? await newCipher(options, this.#getWasm) : null;
    this.#changed();
  }

  /** Writes the snapshot to the storage adapter now. Concurrent saves run in order. */
  async save(): Promise<void> {
    const storage = this.#storage;
    if (!storage) throw new Error("no storage adapter was given to open()");
    clearTimeout(this.#timer);
    this.#timer = undefined;
    const run = this.#saving.then(async () => {
      this.#dirty = false;
      try {
        await storage.set(this.#key, await this.exportSnapshot());
      } catch (error) {
        this.#dirty = true;
        throw error;
      }
    });
    this.#saving = run.catch(() => undefined);
    return run;
  }

  /** Saves any pending change and waits for in-flight saves. */
  async flush(): Promise<void> {
    if (this.#storage && (this.#dirty || this.#timer !== undefined)) await this.save();
    await this.#saving;
    if (this.#autosaveError !== undefined) {
      const error = this.#autosaveError;
      this.#autosaveError = undefined;
      throw error;
    }
  }

  /** Flushes, then releases the module's memory. The store cannot be used afterwards. */
  async close(): Promise<void> {
    if (!this.#inner) return;
    try {
      await this.flush();
    } finally {
      clearTimeout(this.#timer);
      this.#inner.free();
      this.#inner = null;
    }
  }

  #changed(): void {
    this.#dirty = true;
    if (this.#debounceMs === null) return;
    clearTimeout(this.#timer);
    this.#timer = setTimeout(() => {
      this.#timer = undefined;
      this.save().catch((error) => {
        this.#autosaveError = error;
      });
    }, this.#debounceMs);
    // Do not keep a Node/Bun process alive for a pending autosave; close() or flush() saves it.
    (this.#timer as { unref?: () => void }).unref?.();
  }
}
