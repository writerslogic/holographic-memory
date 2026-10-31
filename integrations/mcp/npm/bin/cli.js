#!/usr/bin/env node
// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// Launcher for the warm-mcp binary: finds, or downloads and verifies, the prebuilt binary for this
// platform, then runs it with stdio inherited so the MCP stdio transport passes straight through.
"use strict";

const { spawn } = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const https = require("node:https");
const http = require("node:http");

const pkg = require("../package.json");

const TARGETS = {
  "darwin-arm64": "aarch64-apple-darwin",
  "darwin-x64": "x86_64-apple-darwin",
  "linux-x64": "x86_64-unknown-linux-gnu",
  "linux-arm64": "aarch64-unknown-linux-gnu",
  "win32-x64": "x86_64-pc-windows-msvc",
};
const MAX_BYTES = 256 * 1024 * 1024;
const MAX_REDIRECTS = 5;

function fail(message) {
  process.stderr.write(`writerslogic-warm-mcp: ${message}\n`);
  process.exit(1);
}

function target() {
  const key = `${process.platform}-${process.arch}`;
  const triple = TARGETS[key];
  if (!triple) {
    fail(`no prebuilt warm-mcp for ${key}; build it with cargo (--features mcp) and set WARM_MCP_BINARY`);
  }
  return triple;
}

function cacheDir() {
  if (process.env.WARM_MCP_CACHE_DIR) return process.env.WARM_MCP_CACHE_DIR;
  if (process.platform === "darwin") return path.join(os.homedir(), "Library", "Caches", "writerslogic-warm-mcp");
  if (process.platform === "win32") {
    return path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"), "writerslogic-warm-mcp");
  }
  return path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache"), "writerslogic-warm-mcp");
}

function sha256File(file) {
  const hash = crypto.createHash("sha256");
  hash.update(fs.readFileSync(file));
  return hash.digest("hex");
}

function expectedChecksum(triple) {
  const file = process.env.WARM_MCP_CHECKSUMS || path.join(__dirname, "..", "checksums.json");
  let table;
  try {
    table = JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    fail(`no checksum table at ${file}; this package was not built for release. Set WARM_MCP_BINARY to a binary you trust.`);
  }
  const sum = table.assets && table.assets[triple];
  if (!/^[0-9a-f]{64}$/.test(sum || "")) fail(`checksum table has no valid entry for ${triple}`);
  return sum;
}

function download(url, destination, redirects = 0) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https:") ? https : http;
    if (!url.startsWith("https:") && !isLoopback(url)) {
      reject(new Error("refusing a non-HTTPS download from a non-loopback host"));
      return;
    }
    client
      .get(url, { headers: { "user-agent": `writerslogic-warm-mcp/${pkg.version}` } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          if (redirects >= MAX_REDIRECTS) return reject(new Error("too many redirects"));
          return resolve(download(new URL(res.headers.location, url).toString(), destination, redirects + 1));
        }
        if (res.statusCode !== 200) {
          res.resume();
          return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        }
        let size = 0;
        const out = fs.createWriteStream(destination, { mode: 0o700 });
        res.on("data", (chunk) => {
          size += chunk.length;
          if (size > MAX_BYTES) {
            res.destroy(new Error("download exceeds the size limit"));
          }
        });
        res.on("error", (e) => {
          out.destroy();
          reject(e);
        });
        res.pipe(out);
        out.on("finish", () => out.close(resolve));
        out.on("error", reject);
      })
      .on("error", reject);
  });
}

function isLoopback(url) {
  const host = new URL(url).hostname;
  return host === "localhost" || host === "127.0.0.1" || host === "[::1]" || host === "::1";
}

async function ensureBinary() {
  if (process.env.WARM_MCP_BINARY) return process.env.WARM_MCP_BINARY;
  const triple = target();
  const exe = process.platform === "win32" ? ".exe" : "";
  const dir = path.join(cacheDir(), pkg.version, triple);
  const binary = path.join(dir, `warm-mcp${exe}`);
  const expected = expectedChecksum(triple);
  if (fs.existsSync(binary) && sha256File(binary) === expected) return binary;

  const base =
    process.env.WARM_MCP_RELEASE_BASE ||
    `https://github.com/writerslogic/warm/releases/download/mcp-v${pkg.version}`;
  const url = `${base.replace(/\/$/, "")}/warm-mcp-${triple}${exe}`;
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const partial = path.join(dir, `.download-${process.pid}`);
  process.stderr.write(`writerslogic-warm-mcp: downloading ${url}\n`);
  try {
    await download(url, partial);
    const actual = sha256File(partial);
    if (actual !== expected) {
      throw new Error(`checksum mismatch: expected ${expected}, got ${actual}`);
    }
    if (process.platform !== "win32") fs.chmodSync(partial, 0o755);
    fs.renameSync(partial, binary);
  } catch (e) {
    fs.rmSync(partial, { force: true });
    fail(`download failed: ${e.message}`);
  }
  return binary;
}

ensureBinary().then((binary) => {
  const child = spawn(binary, process.argv.slice(2), { stdio: "inherit" });
  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(signal, () => child.kill(signal));
  }
  child.on("error", (e) => fail(`cannot run ${binary}: ${e.message}`));
  child.on("exit", (code, signal) => {
    process.exit(signal ? 1 : (code ?? 1));
  });
});
