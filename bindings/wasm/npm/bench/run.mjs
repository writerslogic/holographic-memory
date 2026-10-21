// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Orchestrates the wasm-versus-native sweep and writes benchmarks/results/wasm_v1.json.
//   node bench/run.mjs [--sizes 1000,10000,...] [--browser-max N] [--out path] [--crypto path-to-json]
// Each point runs in its own process so memory numbers are not contaminated by earlier points.

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { cpus, platform, release, totalmem } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = resolve(here, "..");
const crate = resolve(pkg, "..");
const repo = resolve(crate, "../..");
const args = process.argv.slice(2);
const option = (name, fallback) => (args.includes(name) ? args[args.indexOf(name)+1] : fallback);
const sizes = option("--sizes", "1000,10000,50000,100000,200000,400000").split(",").map(Number);
const browserMax = Number(option("--browser-max", "50000"));
const out = option("--out", join(repo, "benchmarks/results/wasm_v1.json"));
const cryptoFile = option("--crypto", "");
const QUERIES = 200;
const HMS_ENGINE_MAX = Number(option("--hms-max", "25000"));
const SCALAR_MAX = Number(option("--scalar-max", "25000"));
const timeoutMs = 20 * 60 * 1000;
const chrome = process.env.HMS_CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const firefox = process.env.HMS_FIREFOX;

function json(cmd, cmdArgs, options = {}) {
  const r = spawnSync(cmd, cmdArgs, { encoding: "utf8", timeout: timeoutMs, maxBuffer: 1 << 28, ...options });
  if (r.status !== 0) return { failed: true, status: r.status, signal: r.signal, stderr: (r.stderr ?? "").slice(-400) };
  return JSON.parse(r.stdout.trim().split("\n").pop());
}

// Native binary: opt-level 3 (the wasm profile is size-oriented and must not handicap the baseline).
const build = spawnSync(
  "cargo",
  ["build", "--release", "--example", "native_bench", "--manifest-path", join(crate, "Cargo.toml"), "--config", "profile.release.opt-level=3"],
  { env: { ...process.env, CARGO_TARGET_DIR: "./.hms-target-wasm-native" }, stdio: ["ignore", "ignore", "inherit"] },
);
if (build.status !== 0) throw new Error("native bench build failed");
const nativeBin = "./.hms-target-wasm-native/release/examples/native_bench";

const points = [];
let nodeFailed = false;
for (const documents of sizes) {
  const point = { documents };
  point.nativeMemoryStore = json(nativeBin, ["--engine", "memory", "--documents", documents, "--queries", QUERIES]);
  if (documents <= HMS_ENGINE_MAX) point.nativeEngineHmsCore = json(nativeBin, ["--engine", "hms", "--documents", documents, "--queries", QUERIES]);
  for (const variant of ["simd", "scalar"]) {
    if (variant === "scalar" && documents > SCALAR_MAX) continue;
    point[`nodeWasm_${variant}`] = json(process.execPath, [join(here, "wasm-one.mjs"), documents, QUERIES, variant]);
  }
  if (point.nodeWasm_simd?.failed) nodeFailed = true;
  const native = point.nativeMemoryStore;
  for (const [name, run] of Object.entries(point)) {
    if (name.startsWith("nodeWasm_") && !run.failed && !native.failed) {
      run.topKDigestMatchesNativeMemoryStore = run.topKDigest === native.topKDigest;
      if (point.nativeEngineHmsCore && !point.nativeEngineHmsCore.failed) {
        run.topKDigestMatchesNativeHmsCore = run.topKDigest === point.nativeEngineHmsCore.topKDigest;
      }
    }
  }
  points.push(point);
  writeFileSync(`${out}.partial.json`, JSON.stringify(points));
  console.error(`done ${documents}`);
  if (nodeFailed) break;
}

// Browser runs: the same code in headless Chrome and Firefox, served over localhost.
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm" };
const server = await new Promise((res) =>
  createServer((req, resp) => {
    const path = new URL(req.url, "http://x").pathname;
    const file = path.startsWith("/pkg/") ? join(pkg, "dist", path.slice(5)) : join(here, path.replace(/^\//, ""));
    try {
      const body = readFileSync(file);
      resp.writeHead(200, { "content-type": types[extname(file)] ?? "application/octet-stream" }).end(body);
    } catch {
      resp.writeHead(404).end();
    }
  }).listen(0, "127.0.0.1", function () { res(this); }),
);
const browsers = [];
for (const [name, spec] of [["chrome", { browser: "chrome", executablePath: chrome }], ["firefox", { browser: "firefox", executablePath: firefox }]]) {
  if (!spec.executablePath || !existsSync(spec.executablePath)) continue;
  const browser = await puppeteer.launch({ ...spec, headless: true, args: name === "chrome" ? ["--no-sandbox"] : [] });
  const entry = { browser: name, version: await browser.version(), points: [] };
  for (const documents of sizes.filter((n) => n <= browserMax)) {
    const page = await browser.newPage();
    try {
      await page.goto(`http://localhost:${server.address().port}/page.html?documents=${documents}&queries=${QUERIES}`);
      await page.waitForFunction("window.__result !== undefined", { timeout: timeoutMs });
      const result = await page.evaluate("window.__result");
      const native = points.find((p) => p.documents === documents)?.nativeMemoryStore;
      if (native && !result.failed) result.topKDigestMatchesNativeMemoryStore = result.topKDigest === native.topKDigest;
      entry.points.push(result);
    } catch (error) {
      entry.points.push({ documents, failed: true, error: String(error) });
    }
    await page.close();
  }
  await browser.close();
  browsers.push(entry);
}
server.close();

const sizesJson = JSON.parse(readFileSync(join(pkg, "dist/wasm-sizes.json"), "utf8"));
const report = {
  schema: "wasm_v1",
  generatedAt: new Date().toISOString(),
  machine: { platform: platform(), release: release(), cpu: cpus()[0].model, cores: cpus().length, memoryBytes: totalmem() },
  corpus: {
    kind: "synthetic Zipf-like, see npm/bench/corpus.mjs; identical generator in examples/native_bench.rs",
    vocabulary: 20000,
    wordsPerDocument: 100,
    queryTerms: 3,
    queries: QUERIES,
    k: 10,
    dimensions: 16384,
  },
  notes: [
    "Native baselines: nativeMemoryStore is the same MemoryDocumentStore the wasm module runs; nativeEngineHmsCore is the full journaled engine (HmsCore) in a temporary directory, run up to " + HMS_ENGINE_MAX + " documents.",
    "Native is built at opt-level 3; wasm at opt-level s with wasm-opt -Oz.",
    "wasm memory figures are the linear memory size (it never shrinks, so it is a high-water mark); native figures are peak RSS (ru_maxrss). They are different metrics.",
    "Latency is per query including result conversion to JavaScript objects. A sweep stops at the first size where the Node wasm run fails.",
  ],
  wasmSizes: sizesJson,
  points,
  browsers,
  crypto: cryptoFile && existsSync(cryptoFile) ? JSON.parse(readFileSync(cryptoFile, "utf8")) : undefined,
};
writeFileSync(out, JSON.stringify(report, null, 2) + "\n");
console.error(`wrote ${out}`);
