// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// Drives the backend with SDK-typed commands against a real warm-mcp process (WARM_MCP_BIN).
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, describe, it } from "node:test";

import Anthropic from "@anthropic-ai/sdk";
import type { BetaMemoryTool20250818Command } from "@anthropic-ai/sdk/resources/beta";

import { WarmMemoryTool, normalizePath } from "../src/index.js";

const BIN = process.env["WARM_MCP_BIN"];
assert.ok(BIN, "set WARM_MCP_BIN to the warm-mcp binary");
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "warm-memtool-"));
after(() => fs.rmSync(scratch, { recursive: true, force: true }));
let n = 0;

function env(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) if (v !== undefined && !k.startsWith("WARM_MCP_")) out[k] = v;
  return out;
}

async function open(extra: string[] = [], dir = path.join(scratch, `s${n++}`)) {
  const tool = await WarmMemoryTool.connect({
    connection: { command: BIN, args: ["--data-dir", dir, "--namespace-arg", ...extra], env: env() },
  });
  return { tool, dir };
}

const run = (tool: WarmMemoryTool, command: BetaMemoryTool20250818Command) => {
  const runnable = tool.asTool();
  return runnable.run(command) as Promise<string>;
};
const fails = async (p: Promise<unknown>, pattern: RegExp) => {
  await assert.rejects(p, (e: Error) => {
    assert.match(e.message, pattern);
    return true;
  });
};

describe("path validation", () => {
  const bad = [
    "/memories/../etc/passwd",
    "/memories/a/../../b",
    "/memories/..",
    "/memories/%2e%2e/x",
    "/memories/%2E%2E%2Fx",
    "/memories/..%2fx",
    "/memories/a%5cb",
    "/memories/a\\b",
    "/memories/a/./b",
    "/memories//a",
    "/memories/.hidden",
    "/memoriesx",
    "/memories-other/a",
    "memories/a",
    "/etc/passwd",
    "",
    "/memories/a\u0000b",
    "/memories/a\nb",
    `/memories/${"x".repeat(129)}`,
    `/memories/${"a/".repeat(17)}f`,
  ];
  for (const p of bad) {
    it(`rejects ${JSON.stringify(p).slice(0, 40)}`, () => {
      assert.throws(() => normalizePath(p));
    });
  }
  it("accepts ordinary paths and trims a trailing slash", () => {
    assert.equal(normalizePath("/memories"), "/memories");
    assert.equal(normalizePath("/memories/"), "/memories");
    assert.equal(normalizePath("/memories/a/b.md"), "/memories/a/b.md");
    assert.equal(normalizePath("/memories/a/"), "/memories/a");
    assert.equal(normalizePath("/memories/100%.md"), "/memories/100%.md");
    assert.equal(normalizePath("/memories/naïve café.txt"), "/memories/naïve café.txt");
  });
  it("is enforced by every command before storage is touched", async () => {
    const { tool } = await open();
    try {
      const evil = "/memories/../../secrets.env";
      await fails(run(tool, { command: "view", path: evil }), /escape/);
      await fails(run(tool, { command: "create", path: evil, file_text: "x" }), /escape/);
      await fails(run(tool, { command: "str_replace", path: evil, old_str: "a", new_str: "b" }), /escape/);
      await fails(run(tool, { command: "insert", path: evil, insert_line: 0, insert_text: "x" }), /escape/);
      await fails(run(tool, { command: "delete", path: evil }), /escape/);
      await fails(run(tool, { command: "rename", old_path: "/memories/a", new_path: evil }), /escape/);
      await fails(run(tool, { command: "rename", old_path: evil, new_path: "/memories/b" }), /escape/);
      await fails(run(tool, { command: "view", path: "/memories/%2e%2e/x" }), /encoded/);
      assert.equal((await tool.store.list("/")).length, 0);
    } finally {
      await tool.close();
    }
  });
});

describe("memory tool commands", () => {
  it("views an empty store, creates, views, edits, renames and deletes", async () => {
    const { tool } = await open();
    try {
      assert.equal(
        await run(tool, { command: "view", path: "/memories" }),
        "Here're the files and directories up to 2 levels deep in /memories, excluding hidden items and node_modules:\n0B\t/memories",
      );
      assert.equal(
        await run(tool, { command: "create", path: "/memories/notes.txt", file_text: "Meeting notes:\n- timeline\n- next steps\n" }),
        "File created successfully at: /memories/notes.txt",
      );
      await fails(run(tool, { command: "create", path: "/memories/notes.txt", file_text: "again" }), /File \/memories\/notes.txt already exists/);
      assert.equal(
        await run(tool, { command: "view", path: "/memories/notes.txt" }),
        "Here's the content of /memories/notes.txt with line numbers:\n     1\tMeeting notes:\n     2\t- timeline\n     3\t- next steps",
      );
      assert.equal(
        await run(tool, { command: "view", path: "/memories/notes.txt", view_range: [2, 3] }),
        "Here's the content of /memories/notes.txt with line numbers:\n     2\t- timeline\n     3\t- next steps",
      );
      assert.equal(
        await run(tool, { command: "view", path: "/memories/notes.txt", view_range: [3, -1] }),
        "Here's the content of /memories/notes.txt with line numbers:\n     3\t- next steps",
      );
      await fails(run(tool, { command: "view", path: "/memories/notes.txt", view_range: [3, 1] }), /invalid view_range/);

      const edited = await run(tool, { command: "str_replace", path: "/memories/notes.txt", old_str: "- timeline", new_str: "- project timeline" });
      assert.match(edited, /^The memory file has been edited\./);
      assert.match(edited, /     2\t- project timeline/);
      await fails(run(tool, { command: "str_replace", path: "/memories/notes.txt", old_str: "absent", new_str: "x" }), /did not appear verbatim/);
      await fails(run(tool, { command: "str_replace", path: "/memories/missing.txt", old_str: "a", new_str: "b" }), /does not exist/);
      await run(tool, { command: "create", path: "/memories/dup.txt", file_text: "x\ny\nx\n" });
      await fails(run(tool, { command: "str_replace", path: "/memories/dup.txt", old_str: "x", new_str: "z" }), /Multiple occurrences of old_str `x` in lines: 1, 3/);
      await run(tool, { command: "str_replace", path: "/memories/dup.txt", old_str: "y\n" } as BetaMemoryTool20250818Command); // new_str omitted: the documented schema makes it optional
      assert.equal((await tool.store.get("/memories/dup.txt"))?.text, "x\nx\n");

      assert.equal(
        await run(tool, { command: "insert", path: "/memories/notes.txt", insert_line: 0, insert_text: "# Title\n" }),
        "The file /memories/notes.txt has been edited.",
      );
      await run(tool, { command: "insert", path: "/memories/notes.txt", insert_line: 4, insert_text: "- last" });
      assert.equal((await tool.store.get("/memories/notes.txt"))?.text, "# Title\nMeeting notes:\n- project timeline\n- next steps\n- last\n");
      await fails(run(tool, { command: "insert", path: "/memories/notes.txt", insert_line: 6, insert_text: "x" }), /Invalid `insert_line` parameter: 6.*\[0, 5\]/);
      await fails(run(tool, { command: "insert", path: "/memories/nope.txt", insert_line: 0, insert_text: "x" }), /does not exist/);

      assert.equal(await run(tool, { command: "rename", old_path: "/memories/notes.txt", new_path: "/memories/project/notes.md" }), "Successfully renamed /memories/notes.txt to /memories/project/notes.md");
      await fails(run(tool, { command: "rename", old_path: "/memories/dup.txt", new_path: "/memories/project/notes.md" }), /destination \/memories\/project\/notes.md already exists/);
      await fails(run(tool, { command: "rename", old_path: "/memories/ghost", new_path: "/memories/x" }), /The path \/memories\/ghost does not exist/);
      await fails(run(tool, { command: "rename", old_path: "/memories", new_path: "/memories/x" }), /Cannot rename/);
      await fails(run(tool, { command: "rename", old_path: "/memories/project", new_path: "/memories/project/inner" }), /into itself/);

      const listing = await run(tool, { command: "view", path: "/memories" });
      assert.match(listing, /\n\d+(\.\d)?[BK]\t\/memories\n/);
      assert.match(listing, /\t\/memories\/dup.txt\n/);
      assert.match(listing, /\t\/memories\/project\/\n/);
      assert.match(listing, /\t\/memories\/project\/notes.md$/);

      assert.equal(await run(tool, { command: "delete", path: "/memories/dup.txt" }), "Successfully deleted /memories/dup.txt");
      await fails(run(tool, { command: "delete", path: "/memories/dup.txt" }), /The path \/memories\/dup.txt does not exist/);
      await fails(run(tool, { command: "delete", path: "/memories" }), /Cannot delete the \/memories directory/);
      assert.equal(await run(tool, { command: "delete", path: "/memories/project" }), "Successfully deleted /memories/project");
      assert.equal((await tool.store.list("/memories/")).length, 0);
    } finally {
      await tool.close();
    }
  });

  it("handles directories: nested listing, directory rename, file/dir conflicts", async () => {
    const { tool } = await open();
    try {
      for (const p of ["a/one.txt", "a/b/two.txt", "a/b/c/three.txt", "z.txt"]) {
        await run(tool, { command: "create", path: `/memories/${p}`, file_text: `${p} body\n` });
      }
      const listing = await run(tool, { command: "view", path: "/memories" });
      const paths = listing.split("\n").slice(1).map((l) => l.split("\t")[1]);
      assert.deepEqual(paths, ["/memories", "/memories/a/", "/memories/a/b/", "/memories/a/one.txt", "/memories/z.txt"]);
      const sub = await run(tool, { command: "view", path: "/memories/a" });
      assert.deepEqual(sub.split("\n").slice(1).map((l) => l.split("\t")[1]), ["/memories/a", "/memories/a/b/", "/memories/a/b/c/", "/memories/a/b/two.txt", "/memories/a/one.txt"]);
      await fails(run(tool, { command: "view", path: "/memories/a/one.txt/x" }), /does not exist/);

      await fails(run(tool, { command: "create", path: "/memories/a", file_text: "x" }), /is a directory/);
      await fails(run(tool, { command: "create", path: "/memories/z.txt/inner", file_text: "x" }), /is a file/);
      await run(tool, { command: "rename", old_path: "/memories/a", new_path: "/memories/moved" });
      assert.deepEqual((await tool.store.list("/memories/")).map((e) => e.path), [
        "/memories/moved/b/c/three.txt",
        "/memories/moved/b/two.txt",
        "/memories/moved/one.txt",
        "/memories/z.txt",
      ]);
      assert.equal((await tool.store.get("/memories/moved/b/two.txt"))?.text, "a/b/two.txt body\n");
      await run(tool, { command: "delete", path: "/memories/moved/b" });
      assert.deepEqual((await tool.store.list("/memories/moved/")).map((e) => e.path), ["/memories/moved/one.txt"]);
      await fails(run(tool, { command: "rename", old_path: "/memories/moved", new_path: "/memories/z.txt/in" }), /is a file/);
    } finally {
      await tool.close();
    }
  });

  it("pages long files and enforces the size limit", async () => {
    const { tool } = await open(["--max-text-bytes", "4000"]);
    try {
      const lines = Array.from({ length: 250 }, (_, i) => `line ${i + 1} ${"w".repeat(4)}`);
      await tool.store.put("/memories/long.txt", `${lines.join("\n")}\n`, { createOnly: true });
      const short = new WarmMemoryTool(tool.store, { maxViewChars: 500 });
      const first = await short.view({ command: "view", path: "/memories/long.txt" } as never);
      assert.ok(first.length < 700);
      const next = first.match(/starting at line (\d+)/);
      assert.ok(next, first);
      const second = await short.view({ command: "view", path: "/memories/long.txt", view_range: [Number(next[1]), -1] } as never);
      assert.match(second, new RegExp(`${next[1]}\\tline ${next[1]} wwww`));
      assert.ok(!second.includes("\tline 1 wwww"), "the paged view resumes where the first stopped");
      await fails(run(tool, { command: "create", path: "/memories/big.txt", file_text: "x".repeat(4001) }), /limit is 4000/);
      await run(tool, { command: "create", path: "/memories/ok.txt", file_text: "x".repeat(4000) });
      await fails(run(tool, { command: "insert", path: "/memories/ok.txt", insert_line: 1, insert_text: "more" }), /limit is 4000/);
      assert.equal((await tool.store.get("/memories/ok.txt"))?.text.length, 4000);
    } finally {
      await tool.close();
    }
  });

  it("applies concurrent edits to one file without losing any", async () => {
    const { tool } = await open();
    try {
      await run(tool, { command: "create", path: "/memories/log.txt", file_text: "start\n" });
      const edits = Array.from({ length: 8 }, (_, i) =>
        run(tool, { command: "insert", path: "/memories/log.txt", insert_line: 0, insert_text: `entry ${i}` }),
      );
      await Promise.all(edits);
      const text = (await tool.store.get("/memories/log.txt"))?.text ?? "";
      for (let i = 0; i < 8; i++) assert.match(text, new RegExp(`^entry ${i}$`, "m"));
      assert.match(text, /^start$/m);
    } finally {
      await tool.close();
    }
  });

  it("persists across restarts and recalls files by meaning", async () => {
    const dir = path.join(scratch, `s${n++}`);
    let { tool } = await open([], dir);
    await run(tool, { command: "create", path: "/memories/prefs/coffee.md", file_text: "Alice prefers dark roast coffee with oat milk.\n" });
    await run(tool, { command: "create", path: "/memories/work/deploy.md", file_text: "Deployments use blue green releases on Kubernetes.\n" });
    await tool.close();
    ({ tool } = await open([], dir));
    try {
      assert.match(await run(tool, { command: "view", path: "/memories/prefs/coffee.md" }), /dark roast/);
      const hits = await tool.recall("what kind of release process do deployments use");
      assert.match(hits, /^1\. \/memories\/work\/deploy.md \(relevance 1\.00\)/);
      const missing = await tool.recall("zzzqqq nonexistent");
      assert.equal(missing, "No memory files match that query.");
      await fails(tool.recall(""), /non-empty/);
      await fails(tool.recall("x", 0), /1 to 50/);
      const recall = tool.recallTool();
      assert.equal(recall.name, "memory_recall");
      assert.match(await recall.run(recall.parse({ query: "coffee" })), /prefs\/coffee.md/);
    } finally {
      await tool.close();
    }
  });

  it("keeps namespaces separate", async () => {
    const dir = path.join(scratch, `s${n++}`);
    const mk = (namespace: string) =>
      WarmMemoryTool.connect({
        connection: { command: BIN!, args: ["--data-dir", dir, "--namespace-arg"], env: env() },
        namespace,
      });
    const alice = await mk("alice");
    try {
      await run(alice, { command: "create", path: "/memories/x.txt", file_text: "alice data\n" });
      assert.equal((await alice.store.list("/memories/")).length, 1);
    } finally {
      await alice.close();
    }
    const bob = await mk("bob");
    try {
      assert.equal((await bob.store.list("/memories/")).length, 0);
      await fails(run(bob, { command: "view", path: "/memories/x.txt" }), /does not exist/);
    } finally {
      await bob.close();
    }
  });
});

describe("official SDK tool runner", () => {
  it("serves a memory tool_use loop through client.beta.messages.toolRunner", async () => {
    const { tool } = await open();
    try {
      const seen: unknown[] = [];
      const replies = [
        {
          id: "msg_1",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-5-5",
          stop_reason: "tool_use",
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 1 },
          content: [
            { type: "tool_use", id: "toolu_1", name: "memory", input: { command: "view", path: "/memories" } },
            {
              type: "tool_use",
              id: "toolu_2",
              name: "memory",
              input: { command: "create", path: "/memories/acme.md", file_text: "Acme Corp prefers email follow-ups.\n" },
            },
          ],
        },
        {
          id: "msg_2",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-5-5",
          stop_reason: "tool_use",
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 1 },
          content: [{ type: "tool_use", id: "toolu_3", name: "memory", input: { command: "view", path: "/memories/../../etc/passwd" } }],
        },
        {
          id: "msg_3",
          type: "message",
          role: "assistant",
          model: "claude-sonnet-5-5",
          stop_reason: "end_turn",
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 1 },
          content: [{ type: "text", text: "Done." }],
        },
      ];
      let call = 0;
      const client = new Anthropic({
        apiKey: "test",
        maxRetries: 0,
        fetch: async (_url: unknown, init?: { body?: unknown }) => {
          seen.push(JSON.parse(String(init?.body)));
          return new Response(JSON.stringify(replies[call++]), { status: 200, headers: { "content-type": "application/json" } });
        },
      });
      const final = await client.beta.messages.toolRunner({
        model: "claude-sonnet-5-5",
        max_tokens: 256,
        messages: [{ role: "user", content: "Remember that Acme prefers email." }],
        tools: [tool.asTool()],
        max_iterations: 5,
      });
      assert.equal(final.stop_reason, "end_turn");
      const bodies = seen as { tools: { type: string; name: string }[]; messages: { role: string; content: unknown }[] }[];
      assert.deepEqual(bodies[0]?.tools[0], { type: "memory_20250818", name: "memory" });
      const results = bodies[1]?.messages.at(-1)?.content as { tool_use_id: string; content: string; is_error?: boolean }[];
      assert.equal(results[0]?.tool_use_id, "toolu_1");
      assert.match(results[0]?.content ?? "", /^Here're the files and directories/);
      assert.equal(results[1]?.content, "File created successfully at: /memories/acme.md");
      const blocked = bodies[2]?.messages.at(-1)?.content as { content: string; is_error?: boolean }[];
      assert.equal(blocked[0]?.is_error, true);
      assert.match(blocked[0]?.content ?? "", /escape/);
      assert.equal((await tool.store.get("/memories/acme.md"))?.text, "Acme Corp prefers email follow-ups.\n");
    } finally {
      await tool.close();
    }
  });
});
