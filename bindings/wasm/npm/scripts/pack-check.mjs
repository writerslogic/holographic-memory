// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Verifies the publishable artifact, not the source tree: `npm pack`, install the tarball into a
// fresh directory, and import it from Node, Deno and Bun (and check types resolve with tsc).
//   node scripts/pack-check.mjs [scratch-dir]
// Deno and Bun are used when found on PATH (or HMS_DENO / HMS_BUN).

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const pkgRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const scratch = process.argv[2] ?? mkdtempSync(join(tmpdir(), "hms-pack-"));
mkdirSync(scratch, { recursive: true });

function sh(command, args, options = {}) {
  const r = spawnSync(command, args, { encoding: "utf8", ...options });
  if (r.status !== 0) throw new Error(`${command} ${args.join(" ")}\n${r.stdout}\n${r.stderr}`);
  return r.stdout;
}

const manifest = JSON.parse(readFileSync(join(pkgRoot, "package.json"), "utf8"));
const dry = JSON.parse(sh("npm", ["pack", "--dry-run", "--json"], { cwd: pkgRoot }))[0];
const files = dry.files.map((f) => f.path);
for (const required of [
  "package.json",
  "README.md",
  "LICENSE",
  "dist/web.js",
  "dist/web.d.ts",
  "dist/node.js",
  "dist/deno.js",
  "dist/bundler.js",
  "dist/file.js",
  "dist/opfs.js",
  "dist/indexeddb.js",
  "dist/wasm/web/simd/hms_wasm_bg.wasm",
  "dist/wasm/web/scalar/hms_wasm_bg.wasm",
  "dist/wasm/bundler/simd/hms_wasm_bg.wasm",
  "dist/wasm/node/scalar/hms_wasm.cjs",
  "dist/wasm/deno/simd/hms_wasm.js",
]) {
  assert.ok(files.includes(required), `tarball is missing ${required}`);
}
assert.ok(!files.some((f) => f.startsWith("src/") || f.startsWith("test/") || f.startsWith("bench/") || f.startsWith("scripts/")), "tarball holds sources");
console.log(`pack: ${dry.filename}, ${files.length} files, ${dry.size} bytes packed, ${dry.unpackedSize} unpacked`);

// Every export target named in the exports map must exist in the tarball.
const targets = new Set();
(function walk(value) {
  if (typeof value === "string") targets.add(value.replace(/^\.\//, ""));
  else if (value && typeof value === "object") Object.values(value).forEach(walk);
})(manifest.exports);
for (const target of targets) {
  if (target.includes("*") || target === "package.json") continue;
  assert.ok(files.includes(target), `exports map points at ${target}, which the tarball lacks`);
}

const tarball = sh("npm", ["pack", "--json", "--pack-destination", scratch], { cwd: pkgRoot });
const tarName = JSON.parse(tarball)[0].filename;
const consumer = join(scratch, "consumer");
mkdirSync(consumer, { recursive: true });
writeFileSync(join(consumer, "package.json"), JSON.stringify({ name: "consumer", type: "module", private: true }));
sh("npm", ["install", "--no-audit", "--no-fund", join(scratch, tarName), "typescript@5", "@types/node@22"], { cwd: consumer });

const program = `
import { openStore } from "@holographic-memory/wasm";
import { FileStorage } from "@holographic-memory/wasm/storage/file";
const dir = await (await import("node:fs/promises")).mkdtemp((await import("node:os")).tmpdir() + "/hms-consumer-");
const store = await openStore({ dimensions: 512, storage: new FileStorage(dir), encryption: { passphrase: "p", iterations: 1000 } });
await store.memorizeDocument({ id: "a", text: "installed from the tarball" });
await store.close();
const again = await openStore({ dimensions: 512, storage: new FileStorage(dir), encryption: { passphrase: "p", iterations: 1000 } });
const hits = await again.searchDocuments("tarball");
if (hits[0]?.documentId !== "a") throw new Error("search failed");
await again.close();
console.log("ok " + (again.simd ? "simd" : "scalar"));
`;
writeFileSync(join(consumer, "check.mjs"), program);
const runners = [
  ["node", [process.execPath, ["check.mjs"]]],
  ["deno", [process.env.HMS_DENO ?? "deno", ["run", "--allow-read", "--allow-write", "--allow-env", "check.mjs"]]],
  ["bun", [process.env.HMS_BUN ?? "bun", ["check.mjs"]]],
];
for (const [name, [command, runnerArgs]] of runners) {
  const r = spawnSync(command, runnerArgs, { cwd: consumer, encoding: "utf8" });
  if (r.error?.code === "ENOENT") {
    console.log(`${name}: not installed, skipped`);
    continue;
  }
  assert.equal(r.status, 0, `${name} failed:\n${r.stdout}\n${r.stderr}`);
  console.log(`${name}: ${r.stdout.trim()}`);
}

writeFileSync(
  join(consumer, "types.ts"),
  `import { openStore, type DocumentResult, type StorageAdapter } from "@holographic-memory/wasm";
import { IndexedDbStorage } from "@holographic-memory/wasm/storage/indexeddb";
import { OpfsStorage } from "@holographic-memory/wasm/storage/opfs";
import { FileStorage } from "@holographic-memory/wasm/storage/file";
const adapters: StorageAdapter[] = [new IndexedDbStorage(), new OpfsStorage(), new FileStorage("x")];
const store = await openStore({ storage: adapters[0] });
const hits: DocumentResult[] = await store.searchDocuments("q", { k: 3, bm25K1: 1.2 });
// @ts-expect-error unknown option names are rejected
await store.searchDocuments("q", { bm25Q: 1 });
void hits;
`,
);
writeFileSync(
  join(consumer, "tsconfig.json"),
  JSON.stringify({ compilerOptions: { target: "ES2022", module: "ES2022", moduleResolution: "bundler", strict: true, noEmit: true, lib: ["ES2023", "DOM"], types: ["node"], skipLibCheck: false } }),
);
sh(join(consumer, "node_modules", ".bin", "tsc"), ["-p", "tsconfig.json"], { cwd: consumer });
console.log("types: the packed .d.ts files resolve and type-check under tsc (bundler resolution, skipLibCheck off)");
