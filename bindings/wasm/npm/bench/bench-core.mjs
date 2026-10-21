// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Runtime-neutral wasm benchmark: used by bench/run.mjs under Node and by the browser page.
import { FNV_OFFSET, WORDS_PER_DOC, documentText, fnv, percentile, queryText } from "./corpus.mjs";

export const DIMENSIONS = 16384;
const K = 10;
const BATCH = 500;
const WARMUP = 20;

/**
 * Runs one corpus size. `api` is a package entry (`openStore`, `loadWasm`). Memory is the wasm
 * linear memory size, which only grows, so after a step it is that step's high-water mark.
 */
export async function benchOne(api, { documents, queries, simd }) {
  const out = { documents, queries, wordsPerDocument: WORDS_PER_DOC, k: K, dimensions: DIMENSIONS };
  const now = () => performance.now();

  let t = now();
  const store = await api.openStore({ dimensions: DIMENSIONS, simd });
  out.openMs = now() - t;
  out.simdBuild = store.simd;
  out.memoryAfterOpenBytes = store.memoryBytes;

  t = now();
  for (let start = 0; start < documents; start += BATCH) {
    const batch = [];
    for (let i = start; i < Math.min(documents, start + BATCH); i++) batch.push({ id: `doc${i}`, text: documentText(i) });
    await store.memorizeDocuments(batch);
  }
  out.ingestMs = now() - t;
  out.ingestDocsPerSecond = documents / (out.ingestMs / 1000);
  out.memoryAfterIngestBytes = store.memoryBytes;

  const texts = Array.from({ length: queries + WARMUP }, (_, i) => queryText(documents, i));
  for (let i = 0; i < WARMUP; i++) await store.searchDocuments(texts[i], { k: K });
  const latencies = [];
  let digest = FNV_OFFSET;
  let prefix = FNV_OFFSET;
  let scoreSum = 0;
  t = now();
  for (let i = WARMUP; i < texts.length; i++) {
    const q0 = now();
    const hits = await store.searchDocuments(texts[i], { k: K });
    latencies.push(now() - q0);
    for (const h of hits) {
      digest = fnv(digest, h.documentId);
      if (i < WARMUP + 20) prefix = fnv(prefix, h.documentId);
      scoreSum += h.score;
    }
  }
  const total = now() - t;
  latencies.sort((a, b) => a - b);
  out.search = {
    meanMs: total / queries,
    p50Ms: percentile(latencies, 50),
    p95Ms: percentile(latencies, 95),
    p99Ms: percentile(latencies, 99),
    maxMs: latencies[latencies.length - 1],
    qps: queries / (total / 1000),
  };
  out.topKDigest = digest;
  out.scoreSum = scoreSum;
  out.memoryAfterSearchBytes = store.memoryBytes;

  t = now();
  const snapshot = await store.exportSnapshot();
  out.exportMs = now() - t;
  out.snapshotBytes = snapshot.length;
  out.memoryAfterExportBytes = store.memoryBytes;
  await store.close();

  t = now();
  const copy = await api.openStore({ snapshot, simd });
  out.importMs = now() - t;
  out.memoryAfterImportBytes = copy.memoryBytes;
  let check = FNV_OFFSET;
  for (let i = WARMUP; i < WARMUP + 20; i++) {
    for (const h of await copy.searchDocuments(texts[i], { k: K })) check = fnv(check, h.documentId);
  }
  out.importedDigestMatches = check === prefix;
  await copy.close();
  return out;
}
