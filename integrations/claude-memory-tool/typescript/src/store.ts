// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// Storage behind the memory tool: files addressed by path. `McpFileStore` keeps them in
// Holographic Memory through the hms-mcp server, so every file is also searchable.
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

export interface StoredFile {
  path: string;
  text: string;
  /** Opaque; changes on every write. */
  version: string;
}

export interface FileEntry {
  path: string;
  bytes: number;
}

export interface RecallHit {
  path: string;
  score: number;
  text: string;
}

/** Thrown by `create`/`put` preconditions. */
export class ConflictError extends Error {
  constructor(
    readonly kind: "exists" | "version",
    message: string,
  ) {
    super(message);
  }
}

export interface FileStore {
  get(path: string): Promise<StoredFile | undefined>;
  /** Create (`ifVersion` undefined, `createOnly` true) or replace (`ifVersion` set) a file. */
  put(path: string, text: string, options: { createOnly?: boolean; ifVersion?: string }): Promise<string>;
  /** Every file whose path starts with `prefix`, in path order. */
  list(prefix: string): Promise<FileEntry[]>;
  /** Deletes the given files; returns how many existed. */
  delete(paths: string[]): Promise<number>;
  /** Deletes every file under `prefix` (a directory path followed by "/"). */
  deletePrefix(prefix: string): Promise<number>;
  recall(query: string, k: number): Promise<RecallHit[]>;
  /** Largest file the store accepts, in bytes. */
  readonly maxFileBytes: number;
  close(): Promise<void>;
}

export type McpConnection =
  | {
      /** Spawn hms-mcp over stdio. Default `{ command: "hms-mcp" }`. */
      command?: string;
      args?: string[];
      env?: Record<string, string>;
    }
  | {
      /** Connect to a running `hms-mcp --transport http`. */
      url: string;
      /** Sent as `Authorization: Bearer`; defaults to the HMS_MCP_TOKEN environment variable. */
      token?: string;
    };

export interface McpFileStoreOptions {
  connection?: McpConnection;
  /** Store (namespace) on a server started with --namespace-arg, for example one per user. */
  namespace?: string;
  /** Overrides the server-reported text limit when it is smaller. */
  maxFileBytes?: number;
}

const TAG = "memory-file";
const PAGE = 200;

interface ToolFailure {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export class McpFileStore implements FileStore {
  private constructor(
    private readonly client: Client,
    private readonly namespace: string | undefined,
    readonly maxFileBytes: number,
  ) {}

  static async connect(options: McpFileStoreOptions = {}): Promise<McpFileStore> {
    const connection = options.connection ?? {};
    const client = new Client({ name: "holographic-memory-claude-tool", version: "0.1.0" });
    if ("url" in connection) {
      const token = connection.token ?? process.env["HMS_MCP_TOKEN"];
      const requestInit = token ? { headers: { authorization: `Bearer ${token}` } } : undefined;
      await client.connect(new StreamableHTTPClientTransport(new URL(connection.url), requestInit ? { requestInit } : {}));
    } else {
      await client.connect(
        new StdioClientTransport({
          command: connection.command ?? "hms-mcp",
          args: connection.args ?? [],
          ...(connection.env ? { env: { ...(process.env as Record<string, string>), ...connection.env } } : {}),
          stderr: "inherit",
        }),
      );
    }
    const store = new McpFileStore(client, options.namespace, Number.MAX_SAFE_INTEGER);
    const stats = (await store.call("stats", {})) as { limits: { max_text_bytes: number } };
    const reported = stats.limits.max_text_bytes;
    const limit = options.maxFileBytes ? Math.min(options.maxFileBytes, reported) : reported;
    return new McpFileStore(client, options.namespace, limit);
  }

  private async call(name: string, args: Record<string, unknown>): Promise<unknown> {
    const full = this.namespace ? { ...args, namespace: this.namespace } : args;
    const result = await this.client.callTool({ name, arguments: full });
    if (result.isError) {
      const failure = (result.structuredContent as { error?: ToolFailure } | undefined)?.error;
      throw new HmsToolError(failure?.code ?? "unknown", failure?.message ?? "tool call failed", failure?.details);
    }
    return result.structuredContent;
  }

  async get(path: string): Promise<StoredFile | undefined> {
    try {
      const out = (await this.call("get", { id: path })) as { memory: { text: string; version: string } };
      return { path, text: out.memory.text, version: out.memory.version };
    } catch (e) {
      if (e instanceof HmsToolError && e.code === "not_found") return undefined;
      throw e;
    }
  }

  async put(path: string, text: string, options: { createOnly?: boolean; ifVersion?: string }): Promise<string> {
    try {
      const out = (await this.call("remember", {
        id: path,
        text,
        tags: [TAG],
        allow_empty: true,
        ...(options.createOnly ? { create_only: true } : {}),
        ...(options.ifVersion ? { if_version: options.ifVersion } : {}),
      })) as { version: string };
      return out.version;
    } catch (e) {
      if (e instanceof HmsToolError && e.code === "already_exists") throw new ConflictError("exists", e.message);
      if (e instanceof HmsToolError && e.code === "version_conflict") throw new ConflictError("version", e.message);
      throw e;
    }
  }

  async list(prefix: string): Promise<FileEntry[]> {
    const entries: FileEntry[] = [];
    let cursor: string | undefined;
    for (;;) {
      const out = (await this.call("list", { prefix, limit: PAGE, tag: TAG, ...(cursor ? { cursor } : {}) })) as {
        memories: { id: string; bytes: number }[];
        next_cursor: string | null;
      };
      for (const m of out.memories) entries.push({ path: m.id, bytes: m.bytes });
      if (!out.next_cursor) return entries;
      cursor = out.next_cursor;
    }
  }

  async delete(paths: string[]): Promise<number> {
    let deleted = 0;
    for (let i = 0; i < paths.length; i += 500) {
      const out = (await this.call("forget", { ids: paths.slice(i, i + 500) })) as { deleted: number };
      deleted += out.deleted;
    }
    return deleted;
  }

  async deletePrefix(prefix: string): Promise<number> {
    const entries = await this.list(prefix);
    return this.delete(entries.map((e) => e.path));
  }

  async recall(query: string, k: number): Promise<RecallHit[]> {
    const out = (await this.call("recall", { query, k, tags: [TAG] })) as {
      results: { score: number; memory: { id: string; text: string } }[];
    };
    return out.results.map((r) => ({ path: r.memory.id, score: r.score, text: r.memory.text }));
  }

  async close(): Promise<void> {
    await this.client.close();
  }
}

export class HmsToolError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly details?: Record<string, unknown>,
  ) {
    super(message);
  }
}
