// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// One suite, run unchanged by node --test, deno test and bun test. Each runner supplies its
// `test(name, fn)`, the package under test, a scratch directory and the FileStorage adapter.

import assert from "node:assert/strict";
import { Buffer } from "node:buffer";

const CORPUS = [
  ["rust", "Rust ownership rules prevent data races at compile time without a garbage collector and keep memory use predictable"],
  ["hrr", "The holographic reduced representation binds vectors with circular convolution and superposes many bindings into one trace"],
  ["bm25", "Okapi BM25 ranks passages by term frequency saturation and length normalization over an inverse document frequency weight"],
  ["garden", "Gardening tips for spring include pruning roses and planting tomatoes after the last frost has passed in your region"],
  ["unicode", "Café déjà vu naïve résumé façade jalapeño — 日本語のテキスト and emoji 🧠 survive the round trip"],
];

const ids = (hits) => hits.map((h) => h.documentId);

async function filled(api, options = {}) {
  const store = await api.openStore({ dimensions: 2048, ...options });
  for (const [id, text] of CORPUS) {
    await store.memorizeDocument({ id, text, metadata: { group: id === "garden" ? "outdoor" : "tech" }, sourceUri: `mem://${id}` });
  }
  return store;
}

export function defineSuite({ test, api, FileStorage, scratch }) {
  test("ingest, search, delete and stats", async () => {
    const store = await filled(api);
    assert.deepEqual(await store.stats(), {
      documents: 5,
      chunks: 5,
      embeddedChunks: 0,
      dimensions: 2048,
      embeddingDimensions: null,
      indexTrained: false,
    });
    const hits = await store.searchDocuments("BM25 term frequency");
    assert.equal(hits[0].documentId, "bm25");
    assert.equal(hits[0].sourceUri, "mem://bm25");
    assert.match(hits[0].text, /Okapi/);
    assert.equal(await store.deleteDocument("bm25"), true);
    assert.equal(await store.deleteDocument("bm25"), false);
    assert.ok(!ids(await store.searchDocuments("BM25 term frequency")).includes("bm25"));
    assert.equal((await store.stats()).documents, 4);
    assert.deepEqual(await store.documentIds(), ["garden", "hrr", "rust", "unicode"]);
    await store.close();
  });

  test("replacing a document by id, and atomic batches", async () => {
    const store = await filled(api);
    await store.memorizeDocument({ id: "garden", text: "now about astronomy and telescopes" });
    assert.equal((await store.stats()).documents, 5);
    assert.deepEqual(ids(await store.searchDocuments("telescopes")), ["garden"]);
    assert.deepEqual(await store.searchDocuments("tomatoes frost"), []);
    await assert.rejects(store.memorizeDocuments([{ id: "x1", text: "fine" }, { id: "", text: "bad id" }]));
    assert.equal(await store.hasDocument("x1"), false);
    assert.deepEqual(await store.memorizeDocuments([{ id: "y1", text: "one" }, { id: "y2", text: "two" }]), [1, 1]);
    await store.close();
  });

  test("bm25 options, titleWeight, filters and fusion knobs", async () => {
    const store = await filled(api);
    const base = await store.searchDocuments("memory");
    const tuned = await store.searchDocuments("memory", { bm25K1: 2, bm25B: 0.2 });
    assert.notDeepEqual(base.map((h) => h.lexicalScore), tuned.map((h) => h.lexicalScore));
    await assert.rejects(store.searchDocuments("memory", { bm25B: 1.5 }), /bm25/);
    await assert.rejects(store.searchDocuments("memory", { k: 0 }), /k <= candidateLimit/);

    const filler = "alpha beta gamma delta epsilon theta iota kappa lambda sigma";
    const plain = await api.openStore({ dimensions: 512 });
    const weighted = await api.openStore({ dimensions: 512 });
    for (const [store, titleWeight] of [[plain, undefined], [weighted, 8]]) {
      await store.memorizeDocument({ id: "titled", text: `Zebra\n${filler}`, titleWeight });
      await store.memorizeDocument({ id: "body", text: `${filler} zebra` });
      await store.memorizeDocument({ id: "other", text: filler });
    }
    const p = await plain.searchDocuments("zebra");
    const w = await weighted.searchDocuments("zebra");
    assert.equal(w[0].documentId, "titled");
    assert.ok(w.find((h) => h.documentId === "titled").lexicalScore > p.find((h) => h.documentId === "titled").lexicalScore);
    await assert.rejects(weighted.memorizeDocument({ id: "z", text: "a b", titleWeight: 99 }), /titleWeight/);

    assert.deepEqual(ids(await store.searchDocuments("the", { filter: { group: "outdoor" } })), ["garden"]);
    assert.deepEqual(ids(await store.searchDocuments("the", { documentIds: ["hrr"] })), ["hrr"]);
    assert.deepEqual(ids(await store.searchDocuments("rust", { sourceUri: "mem://rust" })), ["rust"]);
    assert.equal((await store.searchDocuments("the", { k: 2 })).length, 2);
    assert.ok((await store.chunkDocument({ id: "c", text: "w ".repeat(100), chunkWords: 30, overlapWords: 5 })).length > 3);
    await Promise.all([store.close(), plain.close(), weighted.close()]);
  });

  test("scalar and SIMD builds return identical results", async () => {
    const scalar = await filled(api, { simd: false });
    const auto = await filled(api, { simd: "auto" });
    assert.equal(scalar.simd, false);
    for (const query of ["memory threads", "holographic trace", "frost tomatoes", "café 日本語"]) {
      assert.deepEqual(await scalar.searchDocuments(query), await auto.searchDocuments(query), query);
    }
    assert.deepEqual(await scalar.exportSnapshot(), await auto.exportSnapshot());
    await Promise.all([scalar.close(), auto.close()]);
  });

  test("export and import round-trip exactly and reject damage", async () => {
    const store = await filled(api);
    const bytes = await store.exportSnapshot();
    const copy = await api.openStore({ snapshot: bytes });
    for (const query of ["rust ownership", "holographic", "BM25", "日本語"]) {
      assert.deepEqual(await copy.searchDocuments(query), await store.searchDocuments(query));
    }
    assert.deepEqual(await copy.exportSnapshot(), bytes);
    const flipped = bytes.slice();
    flipped[flipped.length - 3] ^= 0x40;
    await assert.rejects(api.openStore({ snapshot: flipped }));
    await assert.rejects(api.openStore({ snapshot: bytes.slice(0, 20) }));
    await assert.rejects(api.openStore({ snapshot: bytes, dimensions: 4096 }), /dimensions/);
    await Promise.all([store.close(), copy.close()]);
  });

  test("file persistence: save, reopen, autosave on close, atomic replace", async () => {
    const dir = await scratch();
    const storage = new FileStorage(dir);
    const first = await filled(api, { storage, key: "notes", autosave: false });
    await first.save();
    assert.deepEqual(await storage.keys(), ["notes"]);
    await first.close();
    const second = await api.openStore({ storage, key: "notes", autosave: { debounceMs: 20 } });
    assert.equal((await second.stats()).documents, 5);
    assert.equal((await second.searchDocuments("holographic"))[0].documentId, "hrr");
    await second.memorizeDocument({ id: "late", text: "added after reopening" });
    await second.close();
    const third = await api.openStore({ storage, key: "notes" });
    assert.equal(await third.hasDocument("late"), true);
    await third.deleteDocument("late");
    await new Promise((resolve) => setTimeout(resolve, 80));
    await third.flush();
    await third.close();
    const fourth = await api.openStore({ storage, key: "notes" });
    assert.equal(await fourth.hasDocument("late"), false);
    await fourth.close();
    const { readdir } = await import("node:fs/promises");
    assert.deepEqual((await readdir(dir)).sort(), ["notes.warm"]);
    await assert.rejects(storage.set("../escape", new Uint8Array(1)), /keys are/);
  });

  test("a user-supplied storage adapter is used for every operation", async () => {
    const calls = [];
    const map = new Map();
    const adapter = {
      async get(key) { calls.push(["get", key]); return map.get(key) ?? null; },
      async set(key, value) { calls.push(["set", key]); map.set(key, value.slice()); },
      async delete(key) { calls.push(["delete", key]); map.delete(key); },
    };
    const store = await api.openStore({ dimensions: 512, storage: adapter, key: "mine", autosave: false });
    await store.memorizeDocument({ id: "a", text: "plug in your own storage" });
    await store.save();
    await store.close();
    assert.deepEqual(calls, [["get", "mine"], ["set", "mine"]]);
    const again = await api.openStore({ storage: adapter, key: "mine", autosave: false });
    assert.equal(await again.hasDocument("a"), true);
    await assert.rejects(api.openStore({ autosave: true }), /storage adapter/);
    await again.close();
  });

  test("a failed close retains unsaved data for retry", async () => {
    let fail = true;
    let bytes;
    const storage = {
      async get() { return bytes ?? null; },
      async set(_key, value) {
        if (fail) throw new Error("storage unavailable");
        bytes = value.slice();
      },
      async delete() {},
    };
    const store = await filled(api, { storage, autosave: false });
    await assert.rejects(store.close(), /storage unavailable/);
    assert.equal(await store.hasDocument("rust"), true);
    await store.memorizeDocument({ id: "after-error", text: "retry keeps every acknowledged write" });
    fail = false;
    await store.close();
    const copy = await api.openStore({ storage, autosave: false });
    assert.equal((await copy.stats()).documents, 6);
    assert.equal(await copy.hasDocument("after-error"), true);
    await copy.close();
  });

  test("concurrent close waits for persistence and blocks new mutations", async () => {
    let release;
    let began;
    let bytes;
    const started = new Promise((resolve) => { began = resolve; });
    const pending = new Promise((resolve) => { release = resolve; });
    const storage = {
      async get() { return null; },
      async set(_key, value) {
        began();
        await pending;
        bytes = value.slice();
      },
      async delete() {},
    };
    const store = await filled(api, { storage, autosave: false });
    const first = store.close();
    const second = store.close();
    assert.equal(first, second);
    await started;
    await assert.rejects(store.memorizeDocument({ id: "late", text: "must reject" }), /closing or closed/);
    await assert.rejects(store.deleteDocument("rust"), /closing or closed/);
    await assert.rejects(store.setEncryption(null), /closing or closed/);
    release();
    await Promise.all([first, second]);
    await store.close();
    const copy = await api.openStore({ snapshot: bytes });
    assert.equal((await copy.stats()).documents, 5);
    await copy.close();
  });

  test("close includes an in-flight encryption change", async () => {
    let bytes;
    const storage = {
      async get() { return null; },
      async set(_key, value) { bytes = value.slice(); },
      async delete() {},
    };
    const store = await filled(api, { storage, autosave: false });
    const encryption = { passphrase: "close waits for this key", iterations: 1000 };
    const change = store.setEncryption(encryption);
    await Promise.all([change, store.close()]);
    assert.deepEqual([...bytes.slice(0, 4)], [0x48, 0x4d, 0x53, 0x45]);
    const copy = await api.openStore({ snapshot: bytes, encryption });
    assert.equal((await copy.stats()).documents, 5);
    await copy.close();
  });

  test("a failed encryption change cannot close or export in the previous plaintext mode", async () => {
    let bytes;
    const storage = {
      async get() { return null; },
      async set(_key, value) { bytes = value.slice(); },
      async delete() {},
    };
    const store = await filled(api, { storage, autosave: false });
    const change = store.setEncryption({ passphrase: "pw", kdf: "unknown" });
    const outcomes = await Promise.allSettled([change, store.close()]);
    assert.deepEqual(outcomes.map((outcome) => outcome.status), ["rejected", "rejected"]);
    assert.equal(bytes, undefined);
    assert.equal(await store.hasDocument("rust"), true);
    await assert.rejects(store.save(), /unknown kdf/);
    await assert.rejects(store.exportSnapshot(), /unknown kdf/);
    const encryption = { passphrase: "retry encryption", iterations: 1000 };
    await store.setEncryption(encryption);
    await store.close();
    const copy = await api.openStore({ snapshot: bytes, encryption });
    assert.equal((await copy.stats()).documents, 5);
    await copy.close();
  });

  for (const kdf of ["pbkdf2", "argon2id"]) {
    test(`encrypted persistence with ${kdf}`, async () => {
      const dir = await scratch();
      const storage = new FileStorage(dir);
      const encryption = { passphrase: "correct horse battery staple", kdf, iterations: 1000, memoryKiB: 64, passes: 1 };
      const store = await filled(api, { storage, encryption, autosave: false });
      await store.save();
      await store.close();
      const raw = await storage.get("warm-store");
      assert.deepEqual([...raw.slice(0, 4)], [0x48, 0x4d, 0x53, 0x45]);
      assert.ok(!Buffer.from(raw).includes("holographic"), "plaintext leaked into the envelope");
      assert.ok(!Buffer.from(raw).includes("HMSD"), "snapshot magic leaked");

      const reopened = await api.openStore({ storage, encryption, autosave: false });
      assert.equal((await reopened.searchDocuments("holographic"))[0].documentId, "hrr");
      await reopened.memorizeDocument({ id: "more", text: "second save reuses the key with a new nonce" });
      await reopened.save();
      const second = await storage.get("warm-store");
      assert.notDeepEqual(second.slice(34, 46), raw.slice(34, 46), "nonce reused");
      assert.deepEqual(second.slice(18, 34), raw.slice(18, 34), "salt is per store, not per save");
      await reopened.close();

      await assert.rejects(api.openStore({ storage }), /encrypted/);
      await assert.rejects(api.openStore({ storage, encryption: { ...encryption, passphrase: "wrong" } }), /decryption failed/);
      for (const at of [5, 6, 20, 40, second.length - 1]) {
        const forged = second.slice();
        forged[at] ^= 1;
        await storage.set("forged", forged);
        await assert.rejects(api.openStore({ storage, key: "forged", encryption }), undefined, `byte ${at}`);
      }
      const unencrypted = await filled(api);
      await storage.set("plain", await unencrypted.exportSnapshot());
      await assert.rejects(api.openStore({ storage, key: "plain", encryption }), /not encrypted/);
      await unencrypted.close();
    });
  }

  test("forged key-derivation parameters are bounded", async () => {
    const storage = new FileStorage(await scratch());
    const encryption = { passphrase: "pw", iterations: 1000 };
    const store = await filled(api, { storage, encryption, autosave: false });
    await store.save();
    await store.close();
    const raw = await storage.get("warm-store");
    const bomb = raw.slice();
    new DataView(bomb.buffer).setUint32(6, 0xffffffff);
    await storage.set("bomb", bomb);
    const started = Date.now();
    await assert.rejects(api.openStore({ storage, key: "bomb", encryption }), /invalid pbkdf2 parameters/);
    assert.ok(Date.now() - started < 1000);
  });

  test("dense embeddings: typed arrays, fusion and the trained index", async () => {
    const dims = 8;
    const store = await api.openStore({ dimensions: 1024, embeddingDimensions: dims });
    const vector = (i) => Float32Array.from({ length: dims }, (_, d) => Math.sin(i * 1.7 + d) + 0.01 * d);
    const docs = Array.from({ length: 160 }, (_, i) => ({ id: `d${i}`, text: `document number ${i} about subject ${i % 7}`, embeddings: [vector(i)] }));
    assert.equal((await store.memorizeDocuments(docs)).length, 160);
    assert.equal((await store.stats()).embeddedChunks, 160);
    assert.equal(store.documentIndexTrained, false);
    const before = await store.searchDocuments("subject", { embedding: vector(42), k: 20, lexicalWeight: 0 });
    assert.ok(before.some((h) => h.documentId === "d42" && h.semanticScore > 0.999));
    assert.equal(await store.trainDocumentIndex(), true);
    assert.equal(store.documentIndexTrained, true);
    const after = await store.searchDocuments("subject", { embedding: vector(42), k: 20, fusion: "rrf" });
    assert.ok(after.some((h) => h.documentId === "d42" && h.semanticScore > 0.999));
    
    await assert.rejects(store.searchDocuments("x", { embedding: [1, 2, 3] }), /dimensions/);
    const restored = await api.openStore({ snapshot: await store.exportSnapshot() });
    assert.equal((await restored.stats()).embeddedChunks, 160);
    await Promise.all([store.close(), restored.close()]);
  });

  test("a store without an embedding space rejects dense queries; closed stores reject use", async () => {
    const store = await filled(api);
    await assert.rejects(store.searchDocuments("x", { embedding: [1, 0] }), /embedding space/);
    await store.close();
    await assert.rejects(store.stats(), /closed/);
  });

  test("a larger corpus: ingest 1500 documents and search", async () => {
    const store = await api.openStore({ dimensions: 1024 });
    const words = Array.from({ length: 400 }, (_, i) => `w${(i * 2654435761) % 100003}`);
    const batch = Array.from({ length: 1500 }, (_, i) => ({
      id: `doc${i}`,
      text: Array.from({ length: 60 }, (_, j) => words[(i * 7 + j * 13) % 400]).join(" ") + ` unique${i}`,
    }));
    for (let i = 0; i < batch.length; i += 500) await store.memorizeDocuments(batch.slice(i, i + 500));
    assert.equal((await store.stats()).documents, 1500);
    assert.equal((await store.searchDocuments("unique1234"))[0].documentId, "doc1234");
    const snapshot = await store.exportSnapshot();
    const copy = await api.openStore({ snapshot });
    assert.equal((await copy.searchDocuments("unique1499"))[0].documentId, "doc1499");
    await Promise.all([store.close(), copy.close()]);
  });
}
