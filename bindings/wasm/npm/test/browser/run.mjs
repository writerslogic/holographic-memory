// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Drives real headless browsers against the built package and the demo page:
//   node test/browser/run.mjs [--browser chrome|firefox|all]
// Browser binaries: HMS_CHROME (default: the macOS Chrome app) and HMS_FIREFOX.

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";
import { startServer } from "../../../../../examples/wasm/serve.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const which = args.includes("--browser") ? args[args.indexOf("--browser") + 1] : "all";
const BROWSERS = {
  chrome: {
    browser: "chrome",
    executablePath: process.env.HMS_CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  },
  firefox: { browser: "firefox", executablePath: process.env.HMS_FIREFOX },
};

async function pageTests(browser, base, name) {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`${base}/test/page.html?run=${name}-${Date.now()}`);
  await page.waitForFunction("window.__done === true", { timeout: 120_000 });
  const results = await page.evaluate("window.__results");
  const simd = await page.evaluate("window.__simd");
  await page.close();
  for (const r of results) console.log(`  ${r.ok ? "ok  " : "FAIL"} ${r.name}${r.ok ? ` (${r.ms} ms)` : `\n${r.error}`}`);
  assert.deepEqual(errors, [], "uncaught page errors");
  assert.ok(results.length >= 7, "all page tests ran");
  assert.ok(results.every((r) => r.ok), "page tests passed");
  return { simd, tests: results.length };
}

async function demoTests(browser, server, base, backend) {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const text = (selector) => page.$eval(selector, (el) => el.textContent);
  const open = async (passphrase = "") => {
    await page.select('[data-testid="backend"]', backend);
    await page.$eval('[data-testid="passphrase"]', (el, v) => { el.value = v; }, passphrase);
    await page.click('[data-testid="open"]');
    await page.waitForFunction(() => document.querySelector('[data-testid="status"]').textContent.startsWith("Open."), { timeout: 60_000 });
  };
  const search = async (query) => {
    await page.$eval('[data-testid="query"]', (el, v) => { el.value = v; }, query);
    await page.click('[data-testid="search"]');
    await page.waitForFunction(() => /result/.test(document.querySelector('[data-testid="count"]').textContent), { timeout: 30_000 });
    return page.$$eval('[data-testid="hit"] strong', (els) => els.map((e) => e.textContent));
  };

  await page.goto(`${base}/`);
  await page.evaluate(async () => { for (const db of await indexedDB.databases()) indexedDB.deleteDatabase(db.name); });
  await open();
  await page.click('[data-testid="sample"]');
  await page.waitForFunction(() => /5 documents/.test(document.querySelector('[data-testid="status"]').textContent));
  assert.equal((await search("bm25 frequency"))[0], "BM25 ranking");
  assert.equal((await search("tomatoes frost"))[0], "Spring gardening");

  await page.$eval('[data-testid="doc-id"]', (el) => { el.value = "my-note"; });
  await page.$eval('[data-testid="doc-text"]', (el) => { el.value = "Zeppelin logistics\nAirships carried freight across the Atlantic."; });
  await page.click('[data-testid="add"]');
  await page.waitForFunction(() => /6 documents/.test(document.querySelector('[data-testid="status"]').textContent));
  assert.equal((await search("zeppelin airships"))[0], "my-note");

  // Persistence: a reload brings the documents back from the chosen backend.
  if (backend !== "memory") {
    await new Promise((resolve) => setTimeout(resolve, 600));
    await page.reload();
    await open();
    assert.match(await text('[data-testid="status"]'), /6 documents/);
    assert.equal((await search("zeppelin airships"))[0], "my-note");
  }

  // Offline: the service worker caches the app; with the server stopped, reload and use it.
  await page.evaluate(() => navigator.serviceWorker.ready);
  await page.reload();
  await page.waitForFunction(() => navigator.serviceWorker.controller !== null, { timeout: 30_000 });
  await page.reload();
  await new Promise((resolve) => server.close(resolve));
  server.closeAllConnections?.();
  await page.reload();
  await open();
  assert.equal((await search("okapi saturation"))[0], "BM25 ranking");
  await page.$eval('[data-testid="doc-id"]', (el) => { el.value = "offline-note"; });
  await page.$eval('[data-testid="doc-text"]', (el) => { el.value = "Written while the server was down\nQuokkas live on Rottnest Island."; });
  await page.click('[data-testid="add"]');
  assert.equal((await search("quokkas rottnest"))[0], "offline-note");
  await page.close();
  assert.deepEqual(errors, [], "uncaught page errors");
}

let failed = false;
for (const name of which === "all" ? ["chrome", "firefox"] : [which]) {
  const spec = BROWSERS[name];
  if (!spec?.executablePath || !existsSync(spec.executablePath)) {
    console.log(`SKIP ${name}: no executable (set HMS_${name.toUpperCase()})`);
    failed = true;
    continue;
  }
  console.log(`\n== ${name} ==`);
  const browser = await puppeteer.launch({ ...spec, headless: true, args: name === "chrome" ? ["--no-sandbox"] : [] });
  try {
    console.log(`version: ${await browser.version()}`);
    let server = await startServer(0, [["/test/", here]]);
    const base = `http://localhost:${server.address().port}`;
    const info = await pageTests(browser, base, name);
    console.log(`  page tests: ${info.tests} passed, simd=${info.simd}`);
    for (const backend of ["opfs", "indexeddb"]) {
      server = server.listening ? server : await startServer(Number(new URL(base).port), [["/test/", here]]);
      await demoTests(browser, server, base, backend);
      console.log(`  demo (${backend}, persistence + offline): passed`);
    }
  } catch (error) {
    failed = true;
    console.error(`FAIL ${name}:`, error);
  } finally {
    await browser.close();
  }
}
process.exit(failed ? 1 : 0);
