// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// The benchmark corpus generator. bindings/wasm/examples/native_bench.rs implements the same
// generator with the same integer arithmetic, so the wasm and native runs ingest byte-identical
// documents and run byte-identical queries.

export const VOCABULARY = 20_000;
export const WORDS_PER_DOC = 100;
export const QUERY_TERMS = 3;
export const SEED = 0x5eed1234;

/** mulberry32: a 32-bit generator expressible identically with wrapping u32 arithmetic. */
export function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0);
  };
}

/** A Zipf-like draw over the vocabulary: u^3 concentrates mass on low word ids. */
function word(next) {
  const u = next() / 4294967296;
  return Math.floor(VOCABULARY * u * u * u);
}

export function documentText(index) {
  const next = rng((SEED ^ Math.imul(index + 1, 0x9e3779b1)) >>> 0);
  const words = new Array(WORDS_PER_DOC);
  for (let i = 0; i < WORDS_PER_DOC; i++) words[i] = `t${word(next)}`;
  return words.join(" ");
}

/** Queries: `QUERY_TERMS` words sampled from a pseudo-randomly chosen document. */
export function queryText(count, queryIndex) {
  const next = rng((SEED ^ Math.imul(queryIndex + 7, 0x85ebca6b)) >>> 0);
  const doc = next() % count;
  const words = documentText(doc).split(" ");
  const terms = [];
  for (let i = 0; i < QUERY_TERMS; i++) terms.push(words[next() % WORDS_PER_DOC]);
  return terms.join(" ");
}

/** 32-bit FNV-1a over an ASCII string, so digests match across languages. */
export function fnv(hash, text) {
  for (let i = 0; i < text.length; i++) hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193) >>> 0;
  return hash;
}

export const FNV_OFFSET = 0x811c9dc5;

export function percentile(sorted, p) {
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
}
