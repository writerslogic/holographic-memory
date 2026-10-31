// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// The npm and PyPI launchers against a locally served release asset. The real GitHub release download
// is not exercised here: no release exists until the coordinator publishes one.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { after, before, describe, it } from "node:test";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const BIN = process.env.WARM_MCP_BIN;
assert.ok(BIN, "set WARM_MCP_BIN to the warm-mcp binary");
const PYTHON = process.env.WARM_MCP_PYTHON ?? "python3";
const here = path.dirname(fileURLToPath(import.meta.url));
const NPM_CLI = path.join(here, "..", "npm", "bin", "cli.js");
const PY_SRC = path.join(here, "..", "pypi", "src");

const TRIPLES = {
  "darwin-arm64": "aarch64-apple-darwin",
  "darwin-x64": "x86_64-apple-darwin",
  "linux-x64": "x86_64-unknown-linux-gnu",
  "linux-arm64": "aarch64-unknown-linux-gnu",
};
const triple = TRIPLES[`${process.platform}-${process.arch}`];
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "warm-launch-"));
const sha = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");

let server;
let base;
let hits = 0;
let serveCorrupt = false;
const assetDir = path.join(scratch, "assets");
const checksums = path.join(scratch, "checksums.json");

before(async () => {
  assert.ok(triple, "unsupported test platform");
  fs.mkdirSync(assetDir);
  fs.copyFileSync(BIN, path.join(assetDir, `warm-mcp-${triple}`));
  fs.writeFileSync(checksums, JSON.stringify({ assets: { [triple]: sha(BIN) } }));
  server = http.createServer((req, res) => {
    hits++;
    const file = path.join(assetDir, path.basename(req.url ?? ""));
    if (!fs.existsSync(file)) {
      res.writeHead(404).end();
      return;
    }
    const body = fs.readFileSync(file);
    res.writeHead(200, { "content-type": "application/octet-stream" });
    res.end(serveCorrupt ? Buffer.concat([body, Buffer.from("tampered")]) : body);
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = `http://127.0.0.1:${server.address().port}`;
});
after(() => {
  server.close();
  fs.rmSync(scratch, { recursive: true, force: true });
});

function env(extra) {
  const out = {};
  for (const [k, v] of Object.entries(process.env)) if (!k.startsWith("WARM_MCP_")) out[k] = v;
  return { ...out, ...extra };
}

function launcherEnv(cache) {
  return env({ WARM_MCP_RELEASE_BASE: base, WARM_MCP_CHECKSUMS: checksums, WARM_MCP_CACHE_DIR: cache });
}

// Asynchronous on purpose: the release server above runs on this event loop.
function exec(command, args, e) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { env: e, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    const timer = setTimeout(() => child.kill("SIGKILL"), 60_000);
    child.on("error", reject);
    child.on("exit", (status) => {
      clearTimeout(timer);
      resolve({ status, stdout, stderr });
    });
  });
}

const launchers = {
  npm: (args, e) => exec("node", [NPM_CLI, ...args], e),
  pypi: (args, e) => exec(PYTHON, ["-m", "writerslogic_warm_mcp", ...args], { ...e, PYTHONPATH: PY_SRC }),
};

for (const [name, run] of Object.entries(launchers)) {
  describe(`${name} launcher`, { skip: triple ? false : "unsupported platform" }, () => {
    it("downloads, verifies, caches and runs the binary", async () => {
      const cache = path.join(scratch, `cache-${name}`);
      hits = 0;
      const first = await run(["--version"], launcherEnv(cache));
      assert.equal(first.status, 0, first.stderr);
      assert.match(first.stdout, /^warm-mcp \d+\.\d+\.\d+/);
      assert.match(first.stderr, /downloading http:\/\/127\.0\.0\.1:\d+\/warm-mcp-/);
      assert.equal(hits, 1);
      const second = await run(["--version"], launcherEnv(cache));
      assert.equal(second.status, 0, second.stderr);
      assert.equal(hits, 1, "the verified cached binary is reused without a download");
      assert.ok(!second.stderr.includes("downloading"));
    });

    it("passes arguments and the exit code through", async () => {
      const cache = path.join(scratch, `cache-${name}`);
      const bad = await run(["--no-such-flag"], launcherEnv(cache));
      assert.notEqual(bad.status, 0);
      assert.match(bad.stderr, /unexpected argument/);
    });

    it("rejects a download whose checksum does not match and leaves no binary behind", async () => {
      const cache = path.join(scratch, `cache-${name}-corrupt`);
      serveCorrupt = true;
      try {
        const res = await run(["--version"], launcherEnv(cache));
        assert.equal(res.status, 1);
        assert.match(res.stderr, /checksum mismatch/);
      } finally {
        serveCorrupt = false;
      }
      const leftovers = fs.existsSync(cache) ? fs.readdirSync(cache, { recursive: true }).filter((f) => /warm-mcp$|\.download-/.test(String(f))) : [];
      assert.deepEqual(leftovers, []);
    });

    it("replaces a cached binary that no longer matches its checksum", async () => {
      const cache = path.join(scratch, `cache-${name}-replace`);
      assert.equal((await run(["--version"], launcherEnv(cache))).status, 0);
      const cached = fs.readdirSync(cache, { recursive: true }).map(String).find((f) => f.endsWith("warm-mcp"));
      assert.ok(cached);
      fs.appendFileSync(path.join(cache, cached), "corrupt");
      hits = 0;
      const again = await run(["--version"], launcherEnv(cache));
      assert.equal(again.status, 0, again.stderr);
      assert.equal(hits, 1);
    });

    it("refuses without a checksum table, and with a table lacking this platform", async () => {
      const cache = path.join(scratch, `cache-${name}-notable`);
      const none = await run(["--version"], { ...launcherEnv(cache), WARM_MCP_CHECKSUMS: path.join(scratch, "missing.json") });
      assert.equal(none.status, 1);
      assert.match(none.stderr, /no checksum table/);
      const other = path.join(scratch, "other.json");
      fs.writeFileSync(other, JSON.stringify({ assets: { "riscv64-unknown-linux-gnu": "0".repeat(64) } }));
      const wrong = await run(["--version"], { ...launcherEnv(cache), WARM_MCP_CHECKSUMS: other });
      assert.equal(wrong.status, 1);
      assert.match(wrong.stderr, /no valid entry/);
    });

    it("refuses a plain-HTTP download from a non-loopback host", async () => {
      const cache = path.join(scratch, `cache-${name}-http`);
      const res = await run(["--version"], { ...launcherEnv(cache), WARM_MCP_RELEASE_BASE: "http://example.invalid/releases" });
      assert.equal(res.status, 1);
      assert.match(res.stderr, /non-HTTPS/);
    });

    it("WARM_MCP_BINARY runs a binary you provide without any download", async () => {
      hits = 0;
      const res = await run(["--version"], env({ WARM_MCP_BINARY: BIN, WARM_MCP_RELEASE_BASE: base, WARM_MCP_CHECKSUMS: path.join(scratch, "missing.json") }));
      assert.equal(res.status, 0, res.stderr);
      assert.equal(hits, 0);
    });

    it("serves an MCP session over stdio through the launcher", async () => {
      const cache = path.join(scratch, `cache-${name}`);
      const e = launcherEnv(cache);
      const [command, args, cenv] =
        name === "npm"
          ? ["node", [NPM_CLI, "--data-dir", path.join(scratch, `data-${name}`)], e]
          : [PYTHON, ["-m", "writerslogic_warm_mcp", "--data-dir", path.join(scratch, `data-${name}`)], { ...e, PYTHONPATH: PY_SRC }];
      const client = new Client({ name: "launcher-e2e", version: "1.0.0" });
      await client.connect(new StdioClientTransport({ command, args, env: cenv, stderr: "pipe" }));
      try {
        const res = await client.callTool({ name: "remember", arguments: { id: "via/launcher", text: "launched through the " + name + " package" } });
        assert.notEqual(res.isError, true);
        const got = await client.callTool({ name: "recall", arguments: { query: "launched package" } });
        assert.equal(got.structuredContent.results[0].memory.id, "via/launcher");
      } finally {
        await client.close();
      }
    });
  });
}
