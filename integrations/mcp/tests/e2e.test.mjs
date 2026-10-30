// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// End-to-end tests of hms-mcp through the official MCP TypeScript SDK client.
// Run: HMS_MCP_BIN=/path/to/hms-mcp npm test   (build with --features mcp,security for the encryption test)
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { after, describe, it } from "node:test";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const BIN = process.env.HMS_MCP_BIN;
assert.ok(BIN, "set HMS_MCP_BIN to the hms-mcp binary");
const SECURITY = process.env.HMS_MCP_SECURITY === "1";

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "hms-mcp-e2e-"));
after(() => fs.rmSync(scratch, { recursive: true, force: true }));
let counter = 0;
const freshDir = () => path.join(scratch, `store-${counter++}`);

function cleanEnv(extra = {}) {
  const env = {};
  for (const [k, v] of Object.entries(process.env)) if (!k.startsWith("HMS_MCP_")) env[k] = v;
  return { ...env, ...extra };
}

async function stdioClient(args = [], env = {}) {
  const transport = new StdioClientTransport({ command: BIN, args, env: cleanEnv(env), stderr: "pipe" });
  const client = new Client({ name: "e2e", version: "1.0.0" });
  await client.connect(transport);
  return client;
}

async function call(client, name, args = {}) {
  const res = await client.callTool({ name, arguments: args });
  return res;
}
const ok = async (client, name, args) => {
  const res = await call(client, name, args);
  assert.notEqual(res.isError, true, JSON.stringify(res.structuredContent));
  return res.structuredContent;
};
const err = async (client, name, args, code) => {
  const res = await call(client, name, args);
  assert.equal(res.isError, true, `${name} should fail`);
  assert.equal(res.structuredContent.error.code, code, JSON.stringify(res.structuredContent));
  return res.structuredContent.error;
};

describe("stdio", () => {
  const dir = freshDir();
  const args = ["--data-dir", dir];

  it("initializes and advertises capabilities and tools", async () => {
    const c = await stdioClient(args);
    try {
      const info = c.getServerVersion();
      assert.equal(info.name, "holographic-memory");
      const caps = c.getServerCapabilities();
      assert.ok(caps.tools && caps.resources && caps.prompts);
      const { tools } = await c.listTools();
      assert.deepEqual(
        tools.map((t) => t.name).sort(),
        ["export", "forget", "get", "list", "recall", "remember", "search", "stats"],
      );
      for (const t of tools) {
        assert.equal(t.inputSchema.type, "object");
        assert.equal(t.inputSchema.additionalProperties, false);
        assert.ok(t.description.length > 20);
      }
      assert.equal(tools.find((t) => t.name === "remember").annotations.readOnlyHint, false);
      assert.equal(tools.find((t) => t.name === "recall").annotations.readOnlyHint, true);
      assert.equal(tools.find((t) => t.name === "forget").annotations.destructiveHint, true);
    } finally {
      await c.close();
    }
  });

  it("remembers, recalls, searches, gets, lists, exports and forgets", async () => {
    const c = await stdioClient(args);
    try {
      const a = await ok(c, "remember", {
        text: "The deployment pipeline uses blue green releases on Kubernetes.",
        id: "ops/deploy",
        title: "Deploys",
        tags: ["ops", "k8s"],
        metadata: { source: "runbook", priority: 2 },
      });
      assert.equal(a.created, true);
      assert.equal(a.id, "ops/deploy");
      await ok(c, "remember", { text: "Alice prefers dark roast coffee and oat milk.", id: "people/alice", tags: ["people"] });
      await ok(c, "remember", { text: "Bob runs the quarterly budget review every March.", id: "people/bob", tags: ["people", "finance"] });
      const gen = await ok(c, "remember", { text: "Generated id memory about espresso machines." });
      assert.match(gen.id, /^mem-[0-9a-f]{11}-[0-9a-f]{8}$/);

      const recall = await ok(c, "recall", { query: "which release strategy does deployment use?", k: 3 });
      assert.equal(recall.results[0].memory.id, "ops/deploy");
      assert.equal(recall.results[0].memory.title, "Deploys");
      assert.deepEqual(recall.results[0].memory.tags, ["k8s", "ops"]);
      assert.equal(recall.results[0].memory.metadata.priority, 2);
      assert.match(recall.results[0].memory.text, /blue green/);
      assert.ok(recall.results[0].score > 0 && recall.results[0].score <= 1);

      const tagged = await ok(c, "search", { query: "budget coffee", tags: ["people"], k: 10 });
      assert.deepEqual(tagged.results.map((r) => r.memory.id).sort(), ["people/alice", "people/bob"]);
      const both = await ok(c, "search", { query: "budget", tags: ["people", "finance"] });
      assert.deepEqual(both.results.map((r) => r.memory.id), ["people/bob"]);
      const filtered = await ok(c, "search", { query: "deployment", filter: { source: "runbook" } });
      assert.deepEqual(filtered.results.map((r) => r.memory.id), ["ops/deploy"]);
      assert.ok("lexical_score" in filtered.results[0]);

      const got = await ok(c, "get", { id: "people/alice" });
      assert.equal(got.memory.text, "Alice prefers dark roast coffee and oat milk.");
      await err(c, "get", { id: "people/nobody" }, "not_found");

      const page1 = await ok(c, "list", { limit: 2 });
      assert.equal(page1.count, 2);
      assert.ok(page1.next_cursor);
      const page2 = await ok(c, "list", { limit: 2, cursor: page1.next_cursor });
      assert.equal(page2.count, 2);
      assert.equal(page2.next_cursor, null);
      const ids = [...page1.memories, ...page2.memories].map((m) => m.id);
      assert.deepEqual(ids, [...ids].sort());
      assert.equal(new Set(ids).size, 4);
      const people = await ok(c, "list", { prefix: "people/" });
      assert.deepEqual(people.memories.map((m) => m.id), ["people/alice", "people/bob"]);
      const fin = await ok(c, "list", { tag: "finance" });
      assert.deepEqual(fin.memories.map((m) => m.id), ["people/bob"]);

      const stats = await ok(c, "stats", {});
      assert.equal(stats.memories, 4);
      assert.equal(stats.encryption_at_rest, false);
      assert.match(stats.recall, /lexical/);

      const exp = await call(c, "export", { format: "json" });
      const arr = JSON.parse(exp.content[0].text);
      assert.equal(arr.length, 4);
      assert.equal(exp.structuredContent.count, 4);
      assert.equal(exp.structuredContent.next_cursor, null);
      const jsonl = await call(c, "export", { format: "jsonl", prefix: "people/" });
      assert.equal(jsonl.content[0].text.split("\n").length, 2);
      const md = await call(c, "export", { format: "markdown", tag: "k8s" });
      assert.match(md.content[0].text, /^## Deploys/);
      const paged = await ok(c, "export", { limit: 3 });
      assert.equal(paged.count, 3);
      assert.ok(paged.next_cursor);

      await err(c, "forget", { prefix: "people/" }, "invalid_argument");
      const del = await ok(c, "forget", { prefix: "people/", confirm: true });
      assert.equal(del.deleted, 2);
      const one = await ok(c, "forget", { id: "ops/deploy" });
      assert.equal(one.deleted, 1);
      const again = await ok(c, "forget", { ids: ["ops/deploy"] });
      assert.deepEqual(again.missing, ["ops/deploy"]);
      assert.equal((await ok(c, "stats", {})).memories, 1);
      const after = await ok(c, "recall", { query: "deployment kubernetes" });
      assert.equal(after.count, 0);
    } finally {
      await c.close();
    }
  });

  it("replaces, compare-and-sets and preserves exact text", async () => {
    const c = await stdioClient(["--data-dir", freshDir()]);
    try {
      const first = await ok(c, "remember", { id: "cas", text: "one", create_only: true });
      await err(c, "remember", { id: "cas", text: "two", create_only: true }, "already_exists");
      const second = await ok(c, "remember", { id: "cas", text: "two", if_version: first.version });
      assert.equal(second.created, false);
      assert.notEqual(second.version, first.version);
      const conflict = await err(c, "remember", { id: "cas", text: "three", if_version: first.version }, "version_conflict");
      assert.equal(conflict.details.current_version, second.version);
      await err(c, "remember", { id: "absent", text: "x", if_version: "deadbeef" }, "version_conflict");
      const g = await ok(c, "get", { id: "cas" });
      assert.equal(g.memory.text, "two");
      assert.ok(g.memory.created_at <= g.memory.updated_at);

      const exact = "\n  indented line\n\nsecond   line with  gaps\n\n";
      await ok(c, "remember", { id: "ws", text: exact });
      assert.equal((await ok(c, "get", { id: "ws" })).memory.text, exact);
      await err(c, "remember", { id: "blank", text: "  \n" }, "invalid_argument");
      await ok(c, "remember", { id: "blank", text: "  \n", allow_empty: true });
      assert.equal((await ok(c, "get", { id: "blank" })).memory.text, "  \n");
      await ok(c, "remember", { id: "empty", text: "", allow_empty: true });
      assert.equal((await ok(c, "get", { id: "empty" })).memory.text, "");
    } finally {
      await c.close();
    }
  });

  it("validates input strictly", async () => {
    const c = await stdioClient(["--data-dir", freshDir(), "--max-text-bytes", "100", "--max-tags", "2"]);
    try {
      await err(c, "remember", { text: "x".repeat(101) }, "too_large");
      await err(c, "remember", { text: "x".repeat(100) .replace(/x/g, "a b ").slice(0, 100) + "z" }, "too_large");
      await err(c, "remember", { text: "ok", unknown_field: 1 }, "invalid_argument");
      await err(c, "remember", { text: 5 }, "invalid_argument");
      await err(c, "remember", {}, "invalid_argument");
      await err(c, "remember", { text: "ok", id: "" }, "invalid_argument");
      await err(c, "remember", { text: "ok", id: "bad\nid" }, "invalid_argument");
      await err(c, "remember", { text: "ok", id: "hms:chunk:x" }, "invalid_argument");
      await err(c, "remember", { text: "ok", id: "a".repeat(257) }, "invalid_argument");
      await err(c, "remember", { text: "ok", tags: ["a", "b", "c"] }, "invalid_argument");
      await err(c, "remember", { text: "ok", tags: [""] }, "invalid_argument");
      await err(c, "remember", { text: "ok", metadata: { "hms.v": 1 } }, "invalid_argument");
      await err(c, "remember", { text: "ok", metadata: "nope" }, "invalid_argument");
      await err(c, "remember", { text: "ok", create_only: true, if_version: "x" }, "invalid_argument");
      await err(c, "recall", { query: "  " }, "invalid_argument");
      await err(c, "recall", { query: "x", k: 0 }, "invalid_argument");
      await err(c, "recall", { query: "x", k: 51 }, "invalid_argument");
      await err(c, "search", { query: "x", filter: { "hms.tag.a": true } }, "invalid_argument");
      await err(c, "search", { query: "x", fusion: "nope" }, "invalid_argument");
      await err(c, "list", { limit: 0 }, "invalid_argument");
      await err(c, "list", { limit: 201 }, "invalid_argument");
      await err(c, "export", { format: "xml" }, "invalid_argument");
      await err(c, "forget", {}, "invalid_argument");
      await err(c, "forget", { id: "a", prefix: "b", confirm: true }, "invalid_argument");
      await err(c, "stats", { namespace: "other" }, "forbidden");
      await assert.rejects(call(c, "no_such_tool", {}), /unknown tool|-32602/i);
      // 100 bytes exactly is accepted (limit, limit+1 above)
      await ok(c, "remember", { text: "y".repeat(100) });
    } finally {
      await c.close();
    }
  });

  it("serves resources and prompts", async () => {
    const dir2 = freshDir();
    const c = await stdioClient(["--data-dir", dir2]);
    try {
      await ok(c, "remember", { id: "notes/a b&c", text: "Resource body text.", title: "Note A" });
      const { resources } = await c.listResources();
      assert.ok(resources.some((r) => r.uri === "hms://stats"));
      const mem = resources.find((r) => r.name === "notes/a b&c");
      assert.equal(mem.uri, "hms://memory/notes%2Fa%20b%26c");
      assert.equal(mem.title, "Note A");
      const read = await c.readResource({ uri: mem.uri });
      assert.equal(read.contents[0].text, "Resource body text.");
      assert.equal(read.contents[0]._meta.version.length, 16);
      const stats = await c.readResource({ uri: "hms://stats" });
      assert.equal(JSON.parse(stats.contents[0].text).memories, 1);
      const { resourceTemplates } = await c.listResourceTemplates();
      assert.equal(resourceTemplates[0].uriTemplate, "hms://memory/{id}");
      await assert.rejects(c.readResource({ uri: "hms://memory/missing" }), /-32002|-32602|no such resource/);
      await assert.rejects(c.readResource({ uri: "file:///etc/passwd" }), /-32002|-32602|no such resource/);
      await assert.rejects(c.readResource({ uri: "hms://memory/%ZZ" }), /-32002|-32602|no such resource/);

      const { prompts } = await c.listPrompts();
      assert.deepEqual(prompts.map((p) => p.name).sort(), ["recall-context", "review-memories", "save-memory"]);
      const rc = await c.getPrompt({ name: "recall-context", arguments: { topic: "resource body" } });
      assert.match(rc.messages[0].content.text, /Resource body text/);
      const sm = await c.getPrompt({ name: "save-memory", arguments: { content: "x", tags: "a,b" } });
      assert.match(sm.messages[0].content.text, /remember/);
      const rv = await c.getPrompt({ name: "review-memories", arguments: {} });
      assert.match(rv.messages[0].content.text, /notes\/a b&c/);
      await assert.rejects(c.getPrompt({ name: "recall-context", arguments: {} }), /topic/);
      await assert.rejects(c.getPrompt({ name: "nope", arguments: {} }), /unknown prompt/);
    } finally {
      await c.close();
    }
  });

  it("handles concurrent calls and reports uncommitted overload", async () => {
    const c = await stdioClient(["--data-dir", freshDir()]);
    try {
      const writes = Array.from({ length: 60 }, (_, i) =>
        call(c, "remember", { id: `c/${String(i).padStart(3, "0")}`, text: `concurrent memory number ${i} about topic${i % 5}` }),
      );
      const reads = Array.from({ length: 30 }, () => call(c, "recall", { query: "concurrent topic3" }));
      const results = await Promise.allSettled([...writes, ...reads]);
      for (const result of results) {
        if (result.status === "fulfilled") {
          assert.notEqual(result.value.isError, true);
        } else {
          assert.equal(result.reason.code, -32603);
          assert.deepEqual(result.reason.data, { code: "server_busy", retryable: true });
        }
      }
      const committed = results.slice(0, writes.length).filter((r) => r.status === "fulfilled").length;
      assert.equal((await ok(c, "stats", {})).memories, committed);
      for (let i = 0; i < writes.length; i++) {
        if (results[i].status === "rejected") {
          const id = `c/${String(i).padStart(3, "0")}`;
          await err(c, "get", { id }, "not_found");
          await ok(c, "remember", { id, text: `concurrent memory number ${i} about topic${i % 5}` });
        }
      }
      assert.equal((await ok(c, "stats", {})).memories, 60);
      const same = await Promise.all(
        Array.from({ length: 20 }, (_, i) => call(c, "remember", { id: "contended", text: `writer ${i}` })),
      );
      assert.ok(same.every((r) => r.isError !== true));
      const list = await ok(c, "list", { prefix: "contended" });
      assert.equal(list.count, 1);
      // Compare-and-set under contention: exactly one winner per version.
      const base = await ok(c, "remember", { id: "race", text: "start" });
      const racers = await Promise.all(
        Array.from({ length: 10 }, (_, i) => call(c, "remember", { id: "race", text: `r${i}`, if_version: base.version })),
      );
      assert.equal(racers.filter((r) => r.isError !== true).length, 1);
    } finally {
      await c.close();
    }
  });

  it("persists across restarts", async () => {
    const dir3 = freshDir();
    let c = await stdioClient(["--data-dir", dir3]);
    const w = await ok(c, "remember", { id: "p/1", text: "Persistent fact about glaciers.", tags: ["geo"], metadata: { k: "v" } });
    await ok(c, "remember", { id: "p/2", text: "Second fact about volcanoes." });
    await ok(c, "forget", { id: "p/2" });
    await c.close();
    c = await stdioClient(["--data-dir", dir3]);
    try {
      const g = await ok(c, "get", { id: "p/1" });
      assert.equal(g.memory.version, w.version);
      assert.equal(g.memory.metadata.k, "v");
      assert.deepEqual(g.memory.tags, ["geo"]);
      assert.equal((await ok(c, "recall", { query: "glaciers" })).results[0].memory.id, "p/1");
      await err(c, "get", { id: "p/2" }, "not_found");
      assert.equal((await ok(c, "stats", {})).memories, 1);
    } finally {
      await c.close();
    }
  });

  it("refuses a second process on a held store with store_locked", async () => {
    const dir4 = freshDir();
    const first = await stdioClient(["--data-dir", dir4]);
    try {
      const child = spawn(BIN, ["--data-dir", dir4], { env: cleanEnv(), stdio: ["pipe", "pipe", "pipe"] });
      let stderr = "";
      child.stderr.on("data", (d) => (stderr += d));
      const code = await new Promise((resolve) => child.on("exit", resolve));
      assert.notEqual(code, 0);
      assert.match(stderr, /store_locked/);
      assert.match(stderr, /--transport http/);
    } finally {
      await first.close();
    }
  });

  it("namespaces: per-call argument and project strategy", async () => {
    const dir5 = freshDir();
    const c = await stdioClient(["--data-dir", dir5, "--namespace-arg"]);
    try {
      await ok(c, "remember", { id: "x", text: "alpha default" });
      await ok(c, "remember", { id: "x", text: "beta other", namespace: "other" });
      assert.equal((await ok(c, "get", { id: "x" })).memory.text, "alpha default");
      assert.equal((await ok(c, "get", { id: "x", namespace: "other" })).memory.text, "beta other");
      await err(c, "get", { id: "x", namespace: "../escape" }, "invalid_argument");
      await err(c, "get", { id: "x", namespace: "UPPER" }, "invalid_argument");
      const res = await c.readResource({ uri: "hms://namespace/other/memory/x" });
      assert.equal(res.contents[0].text, "beta other");
      assert.ok(fs.existsSync(path.join(dir5, "stores", "default")));
      assert.ok(fs.existsSync(path.join(dir5, "stores", "other")));
    } finally {
      await c.close();
    }
    const proj = path.join(scratch, "myproj");
    fs.mkdirSync(path.join(proj, ".git"), { recursive: true });
    fs.mkdirSync(path.join(proj, "src"));
    const p = await stdioClient(["--data-dir", dir5, "--namespace-strategy", "project", "--project", path.join(proj, "src")]);
    try {
      const s = await ok(p, "stats", {});
      assert.match(s.namespace, /^myproj-[0-9a-f]{8}$/);
      assert.equal(s.memories, 0);
    } finally {
      await p.close();
    }
  });

  it("exposes only the configured tools", async () => {
    const ro = await stdioClient(["--data-dir", freshDir(), "--read-only"]);
    try {
      const { tools } = await ro.listTools();
      assert.deepEqual(tools.map((t) => t.name).sort(), ["export", "get", "list", "recall", "search", "stats"]);
      await assert.rejects(call(ro, "remember", { text: "x" }), /unknown tool|-32602/i);
    } finally {
      await ro.close();
    }
    const only = await stdioClient(["--data-dir", freshDir(), "--tools", "recall,remember", "--no-resources", "--no-prompts"]);
    try {
      const { tools } = await only.listTools();
      assert.deepEqual(tools.map((t) => t.name).sort(), ["recall", "remember"]);
      const caps = only.getServerCapabilities();
      assert.equal(caps.resources, undefined);
      assert.equal(caps.prompts, undefined);
    } finally {
      await only.close();
    }
  });

  it("starts with an exact --store directory", async () => {
    const dir6 = freshDir();
    const c = await stdioClient(["--store", dir6]);
    try {
      await ok(c, "remember", { id: "s", text: "single store" });
      assert.ok(fs.existsSync(path.join(dir6, ".hms.lock")));
    } finally {
      await c.close();
    }
  });
});

describe("encryption at rest", { skip: !SECURITY }, () => {
  it("keeps plaintext off disk and requires the passphrase", async () => {
    const dir = freshDir();
    const env = { HMS_MCP_PASSPHRASE: "correct horse battery staple" };
    let c = await stdioClient(["--data-dir", dir, "--encrypt"], env);
    await ok(c, "remember", { id: "secret", text: "the launch code is zanzibar-quokka-7731" });
    assert.equal((await ok(c, "stats", {})).encryption_at_rest, true);
    await c.close();
    const walk = (d) =>
      fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]));
    for (const f of walk(dir)) {
      assert.ok(!fs.readFileSync(f).includes("zanzibar-quokka-7731"), `plaintext found in ${f}`);
    }
    c = await stdioClient(["--data-dir", dir, "--encrypt"], env);
    assert.match((await ok(c, "get", { id: "secret" })).memory.text, /zanzibar/);
    await c.close();
    const bad = spawn(BIN, ["--data-dir", dir, "--encrypt"], {
      env: cleanEnv({ HMS_MCP_PASSPHRASE: "wrong wrong wrong wrong" }),
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stderr = "";
    bad.stderr.on("data", (d) => (stderr += d));
    const code = await new Promise((r) => bad.on("exit", r));
    assert.notEqual(code, 0);
    assert.ok(!stderr.includes("wrong wrong"), "passphrase must not be logged");
    assert.ok(!stderr.includes("zanzibar"));
  });
});

describe("encryption flag without a passphrase", () => {
  it("fails fast", async () => {
    const child = spawn(BIN, ["--data-dir", freshDir(), "--encrypt"], { env: cleanEnv(), stdio: ["pipe", "pipe", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (d) => (stderr += d));
    const code = await new Promise((r) => child.on("exit", r));
    assert.notEqual(code, 0);
    assert.match(stderr, /HMS_MCP_PASSPHRASE|security/);
  });
});

async function startHttp(extraArgs = [], env = {}, existingDir = undefined) {
  const dir = existingDir ?? freshDir();
  const child = spawn(BIN, ["--transport", "http", "--bind", "127.0.0.1:0", "--data-dir", dir, ...extraArgs], {
    env: cleanEnv(env),
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stderr = "";
  child.stderr.on("data", (d) => (stderr += d));
  const url = await new Promise((resolve, reject) => {
    let buf = "";
    child.stdout.on("data", (d) => {
      buf += d;
      const m = buf.match(/listening on (http:\/\/\S+)/);
      if (m) resolve(m[1]);
    });
    child.on("exit", (code) => reject(new Error(`exited ${code}: ${stderr}`)));
  });
  return {
    url,
    dir,
    child,
    stop: () =>
      new Promise((resolve) => {
        child.on("exit", resolve);
        child.kill("SIGTERM");
      }),
  };
}

function rawRequest(url, { method = "POST", headers = {}, body } = {}) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const req = http.request({ hostname: u.hostname, port: u.port, path: u.pathname, method, headers }, (res) => {
      let data = "";
      res.on("data", (d) => (data += d));
      res.on("end", () => resolve({ status: res.statusCode, headers: res.headers, body: data }));
    });
    req.on("error", (error) => reject(new Error(
      `${method} ${u.pathname} (${Buffer.byteLength(body ?? "")} body bytes): ${error.message}`,
      { cause: error },
    )));
    if (body) req.write(body);
    req.end();
  });
}

describe("streamable http", () => {
  const initBody = JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: { protocolVersion: "2025-03-26", capabilities: {}, clientInfo: { name: "raw", version: "1" } },
  });
  const jsonHeaders = { "content-type": "application/json", accept: "application/json, text/event-stream" };

  it("serves the SDK client, persists across restarts and shuts down cleanly", async () => {
    let srv = await startHttp();
    const dir = srv.dir;
    let client = new Client({ name: "e2e-http", version: "1.0.0" });
    await client.connect(new StreamableHTTPClientTransport(new URL(srv.url)));
    try {
      assert.equal(client.getServerVersion().name, "holographic-memory");
      assert.equal((await client.listTools()).tools.length, 8);
      await ok(client, "remember", { id: "h/1", text: "http transport memory about lighthouses" });
      const many = await Promise.all(
        Array.from({ length: 30 }, (_, i) => call(client, "remember", { id: `h/c${i}`, text: `concurrent http ${i}` })),
      );
      assert.ok(many.every((r) => r.isError !== true));
      assert.equal((await ok(client, "recall", { query: "lighthouses" })).results[0].memory.id, "h/1");
      // A second concurrent client shares the same store.
      const second = new Client({ name: "e2e-http-2", version: "1.0.0" });
      await second.connect(new StreamableHTTPClientTransport(new URL(srv.url)));
      assert.equal((await ok(second, "get", { id: "h/1" })).memory.id, "h/1");
      assert.equal((await ok(second, "stats", {})).memories, 31);
      await second.close();
    } finally {
      await client.close();
    }
    const code = await srv.stop();
    assert.equal(code, 0);
    srv = await startHttp([], {}, dir);
    client = new Client({ name: "e2e-http", version: "1.0.0" });
    await client.connect(new StreamableHTTPClientTransport(new URL(srv.url)));
    try {
      assert.equal((await ok(client, "stats", {})).memories, 31);
      assert.match((await ok(client, "get", { id: "h/1" })).memory.text, /lighthouses/);
    } finally {
      await client.close();
      await srv.stop();
    }
  });

  it("enforces the bearer token, Host and Origin checks, and body size", async () => {
    const token = "t".repeat(24);
    const srv = await startHttp(["--max-request-bytes", "4096"], { HMS_MCP_TOKEN: token });
    try {
      const noAuth = await rawRequest(srv.url, { headers: jsonHeaders, body: initBody });
      assert.equal(noAuth.status, 401);
      const wrong = await rawRequest(srv.url, { headers: { ...jsonHeaders, authorization: "Bearer " + "x".repeat(24) }, body: initBody });
      assert.equal(wrong.status, 401);
      const good = await rawRequest(srv.url, { headers: { ...jsonHeaders, authorization: `Bearer ${token}` }, body: initBody });
      assert.equal(good.status, 200);
      const badHost = await rawRequest(srv.url, { headers: { ...jsonHeaders, authorization: `Bearer ${token}`, host: "evil.example" }, body: initBody });
      assert.equal(badHost.status, 403);
      const badOrigin = await rawRequest(srv.url, {
        headers: { ...jsonHeaders, authorization: `Bearer ${token}`, origin: "https://evil.example" },
        body: initBody,
      });
      assert.equal(badOrigin.status, 403);
      const big = await rawRequest(srv.url, { headers: { ...jsonHeaders, authorization: `Bearer ${token}` }, body: "x".repeat(10_000) });
      assert.equal(big.status, 413);
      const health = await rawRequest(new URL("/health", srv.url).toString(), { method: "GET" });
      assert.equal(health.status, 200);
      // The SDK client works with the token.
      const client = new Client({ name: "e2e-auth", version: "1.0.0" });
      await client.connect(
        new StreamableHTTPClientTransport(new URL(srv.url), { requestInit: { headers: { authorization: `Bearer ${token}` } } }),
      );
      assert.equal((await ok(client, "stats", {})).memories, 0);
      await client.close();
    } finally {
      await srv.stop();
    }
  });

  it("refuses to bind a non-loopback address without authorization", async () => {
    const child = spawn(BIN, ["--transport", "http", "--bind", "0.0.0.0:0", "--allow-remote", "--data-dir", freshDir()], {
      env: cleanEnv(),
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stderr = "";
    child.stderr.on("data", (d) => (stderr += d));
    const code = await new Promise((r) => child.on("exit", r));
    assert.notEqual(code, 0);
    assert.match(stderr, /needs --allow-remote and static bearer or OAuth authorization/);
  });

  it("negotiates protocol versions: initialize handshake and the stateless 2026-07-28 form", async () => {
    const srv = await startHttp();
    try {
      const parse = (body) => {
        const lines = body.split("\n").filter((l) => l.startsWith("data:") && l.slice(5).trim() !== "");
        return JSON.parse(lines.length ? lines[lines.length - 1].slice(5) : body);
      };
      for (const version of ["2025-03-26", "2025-06-18", "2025-11-25"]) {
        const res = await rawRequest(srv.url, {
          headers: jsonHeaders,
          body: JSON.stringify({
            jsonrpc: "2.0",
            id: 1,
            method: "initialize",
            params: { protocolVersion: version, capabilities: {}, clientInfo: { name: "raw", version: "1" } },
          }),
        });
        assert.equal(res.status, 200, res.body);
        const out = parse(res.body).result;
        assert.equal(out.protocolVersion, version);
        assert.equal(out.serverInfo.name, "holographic-memory");
        assert.ok(out.capabilities.tools);
      }
      const meta = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientInfo": { name: "raw", version: "1" },
        "io.modelcontextprotocol/clientCapabilities": {},
      };
      const listed = await rawRequest(srv.url, {
        headers: { ...jsonHeaders, "mcp-protocol-version": "2026-07-28", "mcp-method": "tools/list" },
        body: JSON.stringify({ jsonrpc: "2.0", id: 2, method: "tools/list", params: { _meta: meta } }),
      });
      assert.equal(listed.status, 200, listed.body);
      assert.equal(parse(listed.body).result.tools.length, 8);
      const called = await rawRequest(srv.url, {
        headers: { ...jsonHeaders, "mcp-protocol-version": "2026-07-28", "mcp-method": "tools/call", "mcp-name": "stats" },
        body: JSON.stringify({ jsonrpc: "2.0", id: 3, method: "tools/call", params: { name: "stats", arguments: {}, _meta: meta } }),
      });
      assert.equal(called.status, 200, called.body);
      assert.equal(parse(called.body).result.structuredContent.memories, 0);
    } finally {
      await srv.stop();
    }
  });

  it("accepts a configured browser origin", async () => {
    const srv = await startHttp(["--allow-origin", "http://localhost:3000"]);
    try {
      const res = await rawRequest(srv.url, { headers: { ...jsonHeaders, origin: "http://localhost:3000" }, body: initBody });
      assert.equal(res.status, 200);
    } finally {
      await srv.stop();
    }
  });
});
