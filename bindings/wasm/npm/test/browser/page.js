// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Runs inside a real browser page; test/browser/run.mjs drives it and reads window.__results.
import { openStore, detectSimd, isEncrypted } from "/pkg/web.js";
import { IndexedDbStorage } from "/pkg/indexeddb.js";
import { OpfsStorage } from "/pkg/opfs.js";

const results = [];
window.__results = results;
const params = new URLSearchParams(location.search);
const run = params.get("run") ?? crypto.randomUUID();

function assert(condition, message) {
  if (!condition) throw new Error(message ?? "assertion failed");
}
async function rejects(promise, pattern) {
  try {
    await promise;
  } catch (error) {
    assert(!pattern || pattern.test(error.message), `unexpected error: ${error.message}`);
    return;
  }
  throw new Error("expected a rejection");
}
async function test(name, fn) {
  const started = performance.now();
  try {
    await fn();
    results.push({ name, ok: true, ms: Math.round(performance.now() - started) });
  } catch (error) {
    results.push({ name, ok: false, error: String(error?.stack ?? error) });
  }
}

const DOCS = [
  { id: "a", text: "Okapi BM25 ranks passages by term frequency saturation" },
  { id: "b", text: "holographic binding superposes vectors into one trace" },
  { id: "c", text: "plant tomatoes after the last frost in spring" },
];

async function roundTrip(storage, label, extra = {}) {
  const key = `${label}-${run}`;
  const store = await openStore({ dimensions: 1024, storage, key, autosave: false, ...extra });
  await store.memorizeDocuments(DOCS);
  await store.save();
  await store.close();
  assert((await storage.keys()).includes(key), "the key is listed");
  const again = await openStore({ storage, key, autosave: false, ...extra });
  assert((await again.stats()).documents === 3);
  assert((await again.searchDocuments("holographic"))[0].documentId === "b");
  await again.close();
  await storage.delete(key);
  assert((await storage.get(key)) === null, "deleted");
}

await test("runs in a browser window with WebAssembly SIMD detection", async () => {
  assert(typeof document === "object" && typeof navigator.userAgent === "string");
  const simd = await detectSimd();
  const store = await openStore({ dimensions: 512 });
  assert(store.simd === simd, "the loaded build matches detection");
  const forced = await openStore({ dimensions: 512, simd: false });
  assert(forced.simd === false);
  window.__simd = simd;
  await Promise.all([store.close(), forced.close()]);
});

await test("ingest, search, filters, delete and export in the page", async () => {
  const store = await openStore({ dimensions: 1024 });
  await store.memorizeDocuments(DOCS.map((d) => ({ ...d, metadata: { n: d.id } })));
  assert((await store.searchDocuments("bm25 frequency"))[0].documentId === "a");
  assert((await store.searchDocuments("the", { filter: { n: "c" } })).length === 1);
  const copy = await openStore({ snapshot: await store.exportSnapshot() });
  assert((await copy.stats()).documents === 3);
  assert(await store.deleteDocument("a"));
  await rejects(store.searchDocuments("x", { bm25B: 2 }), /bm25/);
  await Promise.all([store.close(), copy.close()]);
});

await test("IndexedDB adapter round trip", async () => {
  await roundTrip(new IndexedDbStorage({ database: `hms-test-${run}` }), "idb");
});

await test("OPFS adapter round trip", async () => {
  await roundTrip(new OpfsStorage({ directory: `hms-test-${run}` }), "opfs");
});

for (const kdf of ["pbkdf2", "argon2id"]) {
  await test(`encrypted IndexedDB persistence with ${kdf}`, async () => {
    const storage = new IndexedDbStorage({ database: `hms-enc-${run}-${kdf}` });
    const encryption = { passphrase: "pässword ✓", kdf, iterations: 2000, memoryKiB: 64, passes: 1 };
    const key = "enc";
    const store = await openStore({ dimensions: 512, storage, key, encryption, autosave: false });
    await store.memorizeDocuments(DOCS);
    await store.save();
    await store.close();
    assert(isEncrypted(await storage.get(key)));
    const again = await openStore({ storage, key, encryption, autosave: false });
    assert((await again.searchDocuments("tomatoes"))[0].documentId === "c");
    await again.close();
    await rejects(openStore({ storage, key, encryption: { ...encryption, passphrase: "nope" } }), /decryption failed/);
    await rejects(openStore({ storage, key }), /encrypted/);
  });
}

await test("autosave writes through to IndexedDB without an explicit save", async () => {
  const storage = new IndexedDbStorage({ database: `hms-auto-${run}` });
  const store = await openStore({ dimensions: 512, storage, key: "auto", autosave: { debounceMs: 30 } });
  await store.memorizeDocument({ id: "x", text: "written by autosave" });
  await new Promise((resolve) => setTimeout(resolve, 300));
  const bytes = await storage.get("auto");
  assert(bytes && bytes.length > 20, "snapshot present");
  await store.close();
});

document.getElementById("out").textContent = JSON.stringify(results, null, 2);
window.__done = true;
