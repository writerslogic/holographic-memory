# @holographic-memory/wasm

Local-first document search in WebAssembly. Ingest text, search it with BM25 (optionally fused with
your own dense embeddings), filter on metadata, and keep the store on the device in IndexedDB, OPFS or a
file, optionally encrypted. Nothing leaves the machine. It is the same document engine as the native
and Node bindings; result lists match them exactly.

Runs in browsers, Node 20+, Deno 2 and Bun.

```js
import { openStore } from "@holographic-memory/wasm";
import { OpfsStorage } from "@holographic-memory/wasm/storage/opfs";

const store = await openStore({
  storage: new OpfsStorage(),            // loaded on open, autosaved after changes
  encryption: { passphrase: "..." },     // optional, AES-256-GCM at rest
});

await store.memorizeDocument({ id: "note-1", text: "Okapi BM25 ranks passages by term frequency." });
const hits = await store.searchDocuments("bm25 frequency", { k: 5, bm25K1: 1.2, bm25B: 0.75 });
await store.close();                     // flushes pending saves
```

## Entry points

| Import | For |
| --- | --- |
| `@holographic-memory/wasm` | picks `web`, `node` or `deno` by runtime (Bun uses `node`) |
| `@holographic-memory/wasm/web` | browsers and any ES-module host with `fetch` / `new URL` assets (Vite, Rollup, esbuild, webpack 5) |
| `@holographic-memory/wasm/bundler` | bundlers with WebAssembly ESM integration (webpack 5 `asyncWebAssembly`, Vite + `vite-plugin-wasm`) |
| `@holographic-memory/wasm/node` | Node and Bun (CommonJS wasm-bindgen glue) |
| `@holographic-memory/wasm/deno` | Deno (`--allow-read`) |
| `@holographic-memory/wasm/storage/indexeddb` | `IndexedDbStorage` (browser) |
| `@holographic-memory/wasm/storage/opfs` | `OpfsStorage` (browser, origin private file system) |
| `@holographic-memory/wasm/storage/file` | `FileStorage` (Node, Deno, Bun) |
| `@holographic-memory/wasm/wasm/*` | the raw wasm-bindgen packages, one directory per target and `scalar` / `simd` build |

A SIMD and a scalar build ship together; `openStore` validates SIMD support at run time and falls back
to scalar (`simd: false` forces scalar). Threads are not used: the document path is single-threaded, so
no cross-origin isolation is needed.

## Storage

A store is saved as one checksummed, compressed snapshot under a key. Implement `StorageAdapter`
(`get`, `set`, `delete`, optional `keys`) to use any backend; `set` must replace the value atomically.
`store.exportSnapshot()` returns the same bytes, and `openStore({ snapshot })` restores them.
`close()` waits for in-flight saves and encryption changes and rejects new mutations while
closing. If persistence fails, it leaves the store open with unsaved data so you can retry
`save()` or `close()`. File storage syncs the replacement file and, on Unix, its directory entries;
Windows retains atomic rename without a claim of directory-sync power-loss durability.
A failed encryption change blocks export/save/close until a successful retry or an explicit
`setEncryption(null)`; unsaved data remains available in memory.

## Encryption

`encryption: { passphrase, kdf?: "pbkdf2" | "argon2id" }` encrypts snapshots with AES-256-GCM through
WebCrypto. PBKDF2-HMAC-SHA256 (600000 iterations) is the default; `"argon2id"` runs the memory-hard KDF
inside the module. The key is derived once per open; every save uses a fresh random nonce. Header
parameters are authenticated, and a forged header cannot request unbounded key stretching.

See the [documentation](https://github.com/writerslogic/holographic-memory/tree/main/docs/runtimes/wasm).

Apache-2.0.
