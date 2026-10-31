// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// The six memory_20250818 commands over a `FileStore`, with the result strings the documented
// reference behaviour uses, plus semantic recall.
import type {
  BetaMemoryTool20250818CreateCommand,
  BetaMemoryTool20250818DeleteCommand,
  BetaMemoryTool20250818InsertCommand,
  BetaMemoryTool20250818RenameCommand,
  BetaMemoryTool20250818StrReplaceCommand,
  BetaMemoryTool20250818ViewCommand,
} from "@anthropic-ai/sdk/resources/beta";
import type { MemoryToolHandlers } from "@anthropic-ai/sdk/helpers/beta/memory";
import { betaMemoryTool } from "@anthropic-ai/sdk/helpers/beta/memory";

import { ancestors, isInside, normalizePath, ROOT } from "./paths.js";
import {
  ConflictError,
  type FileEntry,
  type FileStore,
  WarmToolError,
  McpFileStore,
  type McpFileStoreOptions,
} from "./store.js";

const LINE_NUMBER_WIDTH = 6;
const MAX_LINES = 999_999;
const LISTING_DEPTH = 2;

export interface WarmMemoryToolOptions {
  /** Longest `view` result in characters; longer files are cut at a line and paged with `view_range`. */
  maxViewChars?: number;
  /** Times an edit is retried when another writer changed the file in between. */
  editRetries?: number;
}

function humanSize(bytes: number): string {
  if (bytes === 0) return "0B";
  const units = ["B", "K", "M", "G"];
  const i = Math.min(Math.floor(Math.log2(bytes) / 10), units.length - 1);
  const size = bytes / 1024 ** i;
  return Number.isInteger(size) ? `${size}${units[i]}` : `${size.toFixed(1)}${units[i]}`;
}

function numbered(lines: string[], start: number): string {
  return lines.map((line, i) => `${String(i + start).padStart(LINE_NUMBER_WIDTH)}\t${line}`).join("\n");
}

/** Lines of a file the way the tool counts them: a trailing newline does not add an empty line. */
function splitLines(text: string): string[] {
  const lines = text.split("\n");
  if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
  return lines;
}

function describe(e: unknown): Error {
  if (e instanceof WarmToolError) return new Error(`${e.message} (${e.code})`);
  return e instanceof Error ? e : new Error(String(e));
}

export class WarmMemoryTool implements MemoryToolHandlers {
  private readonly maxViewChars: number;
  private readonly editRetries: number;

  constructor(
    readonly store: FileStore,
    options: WarmMemoryToolOptions = {},
  ) {
    this.maxViewChars = options.maxViewChars ?? 16_000;
    this.editRetries = options.editRetries ?? 25;
  }

  /** Connects to warm-mcp (stdio or HTTP) and wraps it. */
  static async connect(options: McpFileStoreOptions & WarmMemoryToolOptions = {}): Promise<WarmMemoryTool> {
    return new WarmMemoryTool(await McpFileStore.connect(options), options);
  }

  /** The runnable tool for `client.beta.messages.toolRunner({ tools: [...] })`. */
  asTool(): ReturnType<typeof betaMemoryTool> {
    return betaMemoryTool(this);
  }

  async close(): Promise<void> {
    await this.store.close();
  }

  private async kind(path: string): Promise<"file" | "dir" | "none"> {
    if (path === ROOT) return "dir";
    if (await this.store.get(path)) return "file";
    return (await this.store.list(`${path}/`)).length > 0 ? "dir" : "none";
  }

  private async requireFile(path: string, message: string) {
    const file = await this.store.get(path);
    if (!file) throw new Error(message);
    return file;
  }

  private checkSize(text: string): void {
    const bytes = Buffer.byteLength(text);
    if (bytes > this.store.maxFileBytes) {
      throw new Error(`Error: the file would be ${bytes} bytes; the limit is ${this.store.maxFileBytes}. Split the content into several files.`);
    }
  }

  /** Creating `path` must not put a file where a directory exists or under an existing file. */
  private async checkCreatable(path: string): Promise<void> {
    for (const ancestor of ancestors(path)) {
      if (await this.store.get(ancestor)) {
        throw new Error(`Error: ${ancestor} is a file, so ${path} cannot be created inside it`);
      }
    }
    if ((await this.store.list(`${path}/`)).length > 0) {
      throw new Error(`Error: ${path} is a directory`);
    }
  }

  async view(command: BetaMemoryTool20250818ViewCommand): Promise<string> {
    const path = normalizePath(command.path);
    try {
      const file = path === ROOT ? undefined : await this.store.get(path);
      if (file) return this.viewFile(path, file.text, command.view_range);
      const entries = await this.store.list(`${path}/`);
      if (entries.length === 0 && path !== ROOT) {
        throw new Error(`The path ${command.path} does not exist. Please provide a valid path.`);
      }
      return this.viewDirectory(path, entries);
    } catch (e) {
      throw describe(e);
    }
  }

  private viewFile(path: string, text: string, range: number[] | undefined): string {
    const lines = splitLines(text);
    if (lines.length > MAX_LINES) {
      throw new Error(`File ${path} exceeds maximum line limit of 999,999 lines.`);
    }
    let start = 1;
    let shown = lines;
    if (range && range.length === 2) {
      const first = range[0] as number;
      const last = range[1] as number;
      if (!Number.isInteger(first) || !Number.isInteger(last) || first < 1 || (last !== -1 && last < first)) {
        throw new Error(`Error: invalid view_range [${range.join(", ")}]; use [start, end] with 1 <= start <= end, or end = -1`);
      }
      start = first;
      shown = lines.slice(first - 1, last === -1 ? undefined : last);
    }
    let body = numbered(shown, start);
    const header = `Here's the content of ${path} with line numbers:\n`;
    if (header.length + body.length > this.maxViewChars) {
      const kept: string[] = [];
      let size = header.length;
      for (const [i, line] of shown.entries()) {
        const row = `${String(i + start).padStart(LINE_NUMBER_WIDTH)}\t${line}`;
        if (size + row.length + 1 > this.maxViewChars && kept.length > 0) break;
        kept.push(row.length > this.maxViewChars ? `${row.slice(0, this.maxViewChars - 200)} [line truncated]` : row);
        size += row.length + 1;
      }
      const next = start + kept.length;
      body = `${kept.join("\n")}\n[Output truncated at ${this.maxViewChars} characters. Use view_range starting at line ${next} to read the rest.]`;
    }
    return header + body;
  }

  private viewDirectory(path: string, entries: FileEntry[]): string {
    type Node = { name: string; bytes: number; dir: boolean; children: Map<string, Node> };
    const root: Node = { name: "", bytes: 0, dir: true, children: new Map() };
    const prefix = `${path}/`;
    for (const entry of entries) {
      const parts = entry.path.slice(prefix.length).split("/");
      let node = root;
      node.bytes += entry.bytes;
      parts.forEach((part, i) => {
        let child = node.children.get(part);
        if (!child) {
          child = { name: part, bytes: 0, dir: i < parts.length - 1, children: new Map() };
          node.children.set(part, child);
        }
        child.bytes += entry.bytes;
        node = child;
      });
    }
    const rows = [`${humanSize(root.bytes)}\t${path}`];
    const walk = (node: Node, base: string, depth: number) => {
      for (const child of [...node.children.values()].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))) {
        if (child.name === "node_modules") continue;
        const shown = `${base}/${child.name}`;
        rows.push(`${humanSize(child.bytes)}\t${shown}${child.dir ? "/" : ""}`);
        if (child.dir && depth < LISTING_DEPTH) walk(child, shown, depth + 1);
      }
    };
    walk(root, path, 1);
    return `Here're the files and directories up to 2 levels deep in ${path}, excluding hidden items and node_modules:\n${rows.join("\n")}`;
  }

  async create(command: BetaMemoryTool20250818CreateCommand): Promise<string> {
    const path = normalizePath(command.path);
    if (path === ROOT) throw new Error("Error: /memories is a directory");
    try {
      this.checkSize(command.file_text);
      await this.checkCreatable(path);
      await this.store.put(path, command.file_text, { createOnly: true });
    } catch (e) {
      if (e instanceof ConflictError && e.kind === "exists") throw new Error(`Error: File ${command.path} already exists`);
      throw describe(e);
    }
    return `File created successfully at: ${command.path}`;
  }

  /** Read-modify-write with compare-and-set, retried when another writer got in between. */
  private async edit(
    path: string,
    notFound: string,
    change: (text: string) => { text: string; result: string },
  ): Promise<string> {
    for (let attempt = 0; attempt < this.editRetries; attempt++) {
      const file = await this.requireFile(path, notFound);
      const { text, result } = change(file.text);
      this.checkSize(text);
      try {
        await this.store.put(path, text, { ifVersion: file.version });
        return result;
      } catch (e) {
        if (e instanceof ConflictError && e.kind === "version") {
          await new Promise((resolve) => setTimeout(resolve, Math.random() * 5 * (attempt + 1)));
          continue;
        }
        throw e;
      }
    }
    throw new Error(`Error: ${path} kept changing while it was edited; try again`);
  }

  async str_replace(command: BetaMemoryTool20250818StrReplaceCommand): Promise<string> {
    const path = normalizePath(command.path);
    const oldStr = command.old_str;
    const newStr = command.new_str ?? "";
    if (typeof oldStr !== "string" || oldStr.length === 0) throw new Error("Error: old_str must not be empty");
    try {
      return await this.edit(path, `Error: The path ${command.path} does not exist. Please provide a valid path.`, (text) => {
        const positions: number[] = [];
        for (let at = text.indexOf(oldStr); at !== -1; at = text.indexOf(oldStr, at + 1)) positions.push(at);
        if (positions.length === 0) {
          throw new Error(`No replacement was performed, old_str \`${oldStr}\` did not appear verbatim in ${command.path}.`);
        }
        if (positions.length > 1) {
          const lines = positions.map((p) => text.slice(0, p).split("\n").length);
          throw new Error(
            `No replacement was performed. Multiple occurrences of old_str \`${oldStr}\` in lines: ${lines.join(", ")}. Please ensure it is unique`,
          );
        }
        const at = positions[0] as number;
        const updated = text.slice(0, at) + newStr + text.slice(at + oldStr.length);
        const changedLine = text.slice(0, at).split("\n").length - 1;
        const all = updated.split("\n");
        const from = Math.max(0, changedLine - 2);
        const to = Math.min(all.length, changedLine + 3);
        return {
          text: updated,
          result: `The memory file has been edited. Here is the snippet showing the change (with line numbers):\n${numbered(all.slice(from, to), from + 1)}`,
        };
      });
    } catch (e) {
      throw describe(e);
    }
  }

  async insert(command: BetaMemoryTool20250818InsertCommand): Promise<string> {
    const path = normalizePath(command.path);
    try {
      return await this.edit(path, `Error: The path ${command.path} does not exist`, (text) => {
        const lines = splitLines(text);
        const at = command.insert_line;
        if (!Number.isInteger(at) || at < 0 || at > lines.length) {
          throw new Error(
            `Error: Invalid \`insert_line\` parameter: ${at}. It should be within the range of lines of the file: [0, ${lines.length}]`,
          );
        }
        lines.splice(at, 0, command.insert_text.replace(/\n+$/, ""));
        return { text: `${lines.join("\n")}\n`, result: `The file ${command.path} has been edited.` };
      });
    } catch (e) {
      throw describe(e);
    }
  }

  async delete(command: BetaMemoryTool20250818DeleteCommand): Promise<string> {
    const path = normalizePath(command.path);
    if (path === ROOT) throw new Error("Error: Cannot delete the /memories directory itself");
    try {
      const kind = await this.kind(path);
      if (kind === "none") throw new Error(`Error: The path ${command.path} does not exist`);
      if (kind === "file") await this.store.delete([path]);
      else await this.store.deletePrefix(`${path}/`);
    } catch (e) {
      throw describe(e);
    }
    return `Successfully deleted ${command.path}`;
  }

  async rename(command: BetaMemoryTool20250818RenameCommand): Promise<string> {
    const from = normalizePath(command.old_path);
    const to = normalizePath(command.new_path);
    if (from === ROOT) throw new Error("Error: Cannot rename the /memories directory itself");
    if (to === ROOT || isInside(to, from)) {
      throw new Error(`Error: Cannot move ${command.old_path} into itself or onto /memories`);
    }
    try {
      const kind = await this.kind(from);
      if (kind === "none") throw new Error(`Error: The path ${command.old_path} does not exist`);
      if ((await this.kind(to)) !== "none") throw new Error(`Error: The destination ${command.new_path} already exists`);
      for (const ancestor of ancestors(to)) {
        if (await this.store.get(ancestor)) throw new Error(`Error: ${ancestor} is a file, so ${command.new_path} cannot be created inside it`);
      }
      const sources = kind === "file" ? [from] : (await this.store.list(`${from}/`)).map((e) => e.path);
      const created: string[] = [];
      try {
        for (const source of sources) {
          const file = await this.requireFile(source, `Error: The path ${source} does not exist`);
          const target = kind === "file" ? to : `${to}${source.slice(from.length)}`;
          await this.store.put(target, file.text, { createOnly: true });
          created.push(target);
        }
      } catch (e) {
        // Copy first, delete second: a failure here leaves the original intact.
        await this.store.delete(created).catch(() => undefined);
        throw e;
      }
      await this.store.delete(sources);
    } catch (e) {
      if (e instanceof ConflictError && e.kind === "exists") {
        throw new Error(`Error: The destination ${command.new_path} already exists`);
      }
      throw describe(e);
    }
    return `Successfully renamed ${command.old_path} to ${command.new_path}`;
  }

  /** Semantic recall over memory files: the files most relevant to `query`, best first. */
  async recall(query: string, k = 5): Promise<string> {
    if (typeof query !== "string" || query.trim() === "") throw new Error("Error: query must be a non-empty string");
    if (!Number.isInteger(k) || k < 1 || k > 50) throw new Error("Error: k must be an integer from 1 to 50");
    try {
      const hits = await this.store.recall(query, k);
      if (hits.length === 0) return "No memory files match that query.";
      return hits
        .map((hit, i) => {
          const snippet = hit.text.length > 400 ? `${hit.text.slice(0, 400)}...` : hit.text;
          return `${i + 1}. ${hit.path} (relevance ${hit.score.toFixed(2)})\n${snippet}`;
        })
        .join("\n\n");
    } catch (e) {
      throw describe(e);
    }
  }

  /**
   * A companion tool that lets Claude search its memory files by meaning. Add it next to
   * `asTool()`: `tools: [memory.asTool(), memory.recallTool()]`.
   */
  recallTool() {
    return {
      name: "memory_recall",
      description:
        "Search your memory files for the ones most relevant to a query. Returns file paths with a relevance score and a text excerpt. Use `memory` view to read a file in full.",
      input_schema: {
        type: "object" as const,
        properties: {
          query: { type: "string", description: "What to look for, in natural language." },
          k: { type: "integer", minimum: 1, maximum: 50, description: "How many files to return (default 5)." },
        },
        required: ["query"],
      },
      parse: (content: unknown): { query: string; k?: number } => {
        const value = content as { query?: unknown; k?: unknown };
        if (typeof value?.query !== "string") throw new Error("query must be a string");
        if (value.k !== undefined && typeof value.k !== "number") throw new Error("k must be a number");
        return value.k === undefined ? { query: value.query } : { query: value.query, k: value.k };
      },
      run: (input: { query: string; k?: number }) => this.recall(input.query, input.k),
    };
  }
}
