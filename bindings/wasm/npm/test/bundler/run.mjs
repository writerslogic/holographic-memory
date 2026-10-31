// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Bundles the `bundler` entry with webpack 5 (asyncWebAssembly) and runs the result in headless
// Chrome. Also bundles the `web` entry, which relies on `new URL(..., import.meta.url)` assets.
//   node test/bundler/run.mjs

import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";
import webpack from "webpack";

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, "../../dist");
const chrome = process.env.WARM_CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const entrySource = (specifier) => `
import { openStore } from ${JSON.stringify(specifier)};
window.__run = async () => {
  const store = await openStore({ dimensions: 1024 });
  await store.memorizeDocuments([
    { id: "a", text: "bundled with webpack and searched in the browser" },
    { id: "b", text: "gardening tips for spring" },
  ]);
  const hits = await store.searchDocuments("webpack browser");
  const copy = await openStore({ snapshot: await store.exportSnapshot() });
  return { top: hits[0].documentId, docs: (await copy.stats()).documents, simd: store.simd };
};
`;

async function bundle(name, specifier) {
  const work = mkdtempSync(join(tmpdir(), `warm-${name}-`));
  writeFileSync(join(work, "entry.js"), entrySource(specifier));
  writeFileSync(join(work, "index.html"), `<!doctype html><meta charset="utf-8"><script src="main.js"></script>`);
  const compiler = webpack({
    mode: "production",
    context: work,
    entry: "./entry.js",
    output: { path: work, filename: "main.js", publicPath: "" },
    experiments: { asyncWebAssembly: true },
    resolve: { alias: { "@pkg": dist } },
    performance: { hints: false },
  });
  const stats = await new Promise((res, rej) => compiler.run((e, s) => (e ? rej(e) : res(s))));
  assert.ok(!stats.hasErrors(), stats.toString({ errors: true, colors: false }));
  return work;
}

function serve(root) {
  const types = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm" };
  return new Promise((res) => {
    const server = createServer((req, resp) => {
      try {
        const path = new URL(req.url, "http://x").pathname.replace(/^\/$/, "/index.html");
        const body = readFileSync(join(root, path));
        resp.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" }).end(body);
      } catch {
        resp.writeHead(404).end();
      }
    }).listen(0, "127.0.0.1", () => res(server));
  });
}

const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ["--no-sandbox"] });
let failed = false;
try {
  for (const [name, specifier] of [
    ["bundler", join(dist, "bundler.js")],
    ["web", join(dist, "web.js")],
  ]) {
    const root = await bundle(name, specifier);
    const server = await serve(root);
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    await page.goto(`http://localhost:${server.address().port}/`);
    await page.waitForFunction("typeof window.__run === 'function'", { timeout: 30_000 });
    const result = await page.evaluate("window.__run()");
    assert.deepEqual(errors, []);
    assert.equal(result.top, "a");
    assert.equal(result.docs, 2);
    console.log(`webpack ${name} entry in Chrome: ok (simd=${result.simd})`);
    await page.close();
    server.close();
  }
} catch (error) {
  failed = true;
  console.error(error);
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);
