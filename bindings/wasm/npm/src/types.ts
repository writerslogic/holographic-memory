// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

/** A document to ingest. Field names match the Node binding's `DocumentInput`. */
export interface DocumentInput {
  /** 1..4096 bytes, no NUL. Ingesting an existing id replaces that document. */
  id: string;
  text: string;
  sourceUri?: string | null;
  version?: string | null;
  /** A JSON object of scalar-valued fields; exact-match filterable at search time. */
  metadata?: Record<string, unknown> | null;
  /** Words per chunk, 1..4096 (default 256). */
  chunkWords?: number;
  /** Words shared between neighbouring chunks, smaller than chunkWords (default 32). */
  overlapWords?: number;
  /** Keep passage text in the store so results can return it (default true). */
  storeText?: boolean;
  /** One dense embedding per chunk; needs `embeddingDimensions` on the store. */
  embeddings?: ArrayLike<number>[] | null;
  /** Counts the first line's terms this many times in the first chunk, 1..16 (default 1). */
  titleWeight?: number | null;
}

export interface DocumentChunk {
  id: string;
  text: string;
  startByte: number;
  endByte: number;
}

export interface SearchOptions {
  /** Results to return, 1..1000 (default 10). */
  k?: number;
  /** Candidates scored before the cut, k..4096 (default max(100, k)). */
  candidateLimit?: number;
  /** Exact metadata matches, e.g. `{ kind: "note" }`. */
  filter?: Record<string, unknown> | null;
  sourceUri?: string | null;
  documentIds?: string[] | null;
  /** Dense query embedding; needs `embeddingDimensions` on the store. */
  embedding?: ArrayLike<number> | null;
  lexicalWeight?: number;
  semanticWeight?: number;
  minSemanticScore?: number;
  /** `"blend"` (default) adds normalized scores; `"rrf"` is reciprocal rank fusion, k = 60. */
  fusion?: "blend" | "rrf";
  /** BM25 term-frequency saturation, finite and >= 0 (default 1.2). */
  bm25K1?: number;
  /** BM25 length normalization in [0, 1] (default 0.75). */
  bm25B?: number;
}

export interface DocumentResult {
  id: string;
  documentId: string;
  sourceUri: string | null;
  version: string;
  metadata: Record<string, unknown>;
  text: string | null;
  startByte: number;
  endByte: number;
  score: number;
  lexicalScore: number;
  semanticScore: number | null;
}

export interface StoreStats {
  documents: number;
  chunks: number;
  embeddedChunks: number;
  dimensions: number;
  embeddingDimensions: number | null;
  indexTrained: boolean;
}

/** Where snapshots live. Implement this to plug in any store; keys are short ASCII names. */
export interface StorageAdapter {
  get(key: string): Promise<Uint8Array | null>;
  /** Must replace the value atomically: a crash leaves the old or the new bytes, never a mix. */
  set(key: string, value: Uint8Array): Promise<void>;
  delete(key: string): Promise<void>;
  /** Optional; lists the keys the adapter holds. */
  keys?(): Promise<string[]>;
}

export type Kdf = "pbkdf2" | "argon2id";

export interface EncryptionOptions {
  /** A passphrase string, or raw bytes. */
  passphrase: string | Uint8Array;
  /**
   * `"pbkdf2"` (default) derives with WebCrypto PBKDF2-HMAC-SHA256; `"argon2id"` runs the
   * memory-hard KDF inside the WebAssembly module. AES-256-GCM is always WebCrypto.
   */
  kdf?: Kdf;
  /** PBKDF2 iterations (default 600000). */
  iterations?: number;
  /** Argon2id memory in KiB (default 19456), passes (default 2) and lanes (default 1). */
  memoryKiB?: number;
  passes?: number;
  lanes?: number;
}

export interface AutosaveOptions {
  /** Quiet period after the last change before a save, in milliseconds (default 500). */
  debounceMs?: number;
}

export interface OpenOptions {
  /** Sparse text-vector dimension for a new store (default 16384). */
  dimensions?: number;
  /** Enables dense embeddings of this size for a new store. */
  embeddingDimensions?: number;
  /** Where to load from and save to. Without one the store lives only in memory. */
  storage?: StorageAdapter;
  /** The storage key (default `"hms-store"`). */
  key?: string;
  /** Encrypt snapshots at rest. */
  encryption?: EncryptionOptions;
  /** Save after changes. Default: on (500 ms) when `storage` is given, off otherwise. */
  autosave?: boolean | AutosaveOptions;
  /** Start from these snapshot bytes (from `exportSnapshot`) instead of the storage. */
  snapshot?: Uint8Array;
  /** `"auto"` (default) uses the SIMD build when the runtime validates it. */
  simd?: "auto" | boolean;
}

/** The surface of the wasm-bindgen module the wrapper relies on. */
export interface WasmStore {
  memorizeDocument(input: unknown): number;
  memorizeDocuments(inputs: unknown): Uint32Array | number[];
  chunkDocument(input: unknown): DocumentChunk[];
  searchDocuments(text: string, options?: unknown): DocumentResult[];
  deleteDocument(id: string): boolean;
  hasDocument(id: string): boolean;
  documentIds(): string[];
  stats(): StoreStats;
  trainDocumentIndex(): boolean;
  readonly documentIndexTrained: boolean;
  readonly dimensions: number;
  exportBytes(): Uint8Array;
  free(): void;
}

export interface WasmApi {
  HmsStore: {
    new (options?: { dimensions?: number; embeddingDimensions?: number }): WasmStore;
    importBytes(bytes: Uint8Array): WasmStore;
  };
  argon2idKey(
    passphrase: Uint8Array,
    salt: Uint8Array,
    memoryKiB: number,
    passes: number,
    lanes: number,
  ): Uint8Array;
  simdBuild(): boolean;
  memoryBytes(): number;
  version(): string;
}

/** Loads the compiled module; `simd` picks the build. Supplied by each runtime entry point. */
export type WasmLoader = (simd: boolean) => Promise<WasmApi>;
