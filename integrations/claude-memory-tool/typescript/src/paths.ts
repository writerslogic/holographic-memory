// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// Path validation for the /memories virtual filesystem. Paths are storage keys, never real
// filesystem paths, but every path is still validated so nothing outside /memories is ever addressed.

export const ROOT = "/memories";
const MAX_PATH_BYTES = 240;
const MAX_SEGMENT_BYTES = 128;
const MAX_DEPTH = 16;

export class MemoryPathError extends Error {}

/** Returns the canonical form of `input`: ROOT, or ROOT/segment/..., without a trailing slash. */
export function normalizePath(input: unknown): string {
  if (typeof input !== "string" || input.length === 0) {
    throw new MemoryPathError("Path must be a non-empty string");
  }
  // Percent-encoded separators and dots are rejected outright, so a path that a later layer
  // decodes cannot turn into a traversal. Checked before and after one decode.
  const encoded = /%(2e|2f|5c|00)/i;
  if (encoded.test(input)) {
    throw new MemoryPathError(`Path ${input} contains an encoded separator or dot`);
  }
  let decoded = input;
  try {
    decoded = decodeURIComponent(input);
  } catch {
    // A malformed escape cannot decode into anything; the literal checks below still apply.
  }
  for (const candidate of [input, decoded]) {
    if (/[\u0000-\u001f\u007f\\]/.test(candidate)) {
      throw new MemoryPathError(`Path ${input} contains a control character or backslash`);
    }
    if (candidate.split("/").some((s) => s === ".." || s === ".")) {
      throw new MemoryPathError(`Path ${input} would escape /memories directory`);
    }
  }
  if (input !== ROOT && !input.startsWith(`${ROOT}/`)) {
    throw new MemoryPathError(`Path must start with /memories, got: ${input}`);
  }
  let trimmed = input;
  while (trimmed.length > ROOT.length && trimmed.endsWith("/")) trimmed = trimmed.slice(0, -1);
  if (trimmed === ROOT) return ROOT;
  const segments = trimmed.slice(ROOT.length + 1).split("/");
  if (segments.length > MAX_DEPTH) {
    throw new MemoryPathError(`Path ${input} is nested more than ${MAX_DEPTH} levels`);
  }
  for (const segment of segments) {
    if (segment.length === 0) {
      throw new MemoryPathError(`Path ${input} contains an empty segment`);
    }
    if (segment.startsWith(".")) {
      throw new MemoryPathError(`Path ${input} contains a hidden segment; names must not start with a dot`);
    }
    if (Buffer.byteLength(segment) > MAX_SEGMENT_BYTES) {
      throw new MemoryPathError(`Path ${input} has a segment longer than ${MAX_SEGMENT_BYTES} bytes`);
    }
  }
  if (Buffer.byteLength(trimmed) > MAX_PATH_BYTES) {
    throw new MemoryPathError(`Path ${input} is longer than ${MAX_PATH_BYTES} bytes`);
  }
  return trimmed;
}

/** Every proper ancestor of `path` below ROOT, shallowest first. */
export function ancestors(path: string): string[] {
  const out: string[] = [];
  const parts = path.slice(ROOT.length + 1).split("/");
  for (let i = 1; i < parts.length; i++) out.push(`${ROOT}/${parts.slice(0, i).join("/")}`);
  return out;
}

export function isInside(path: string, directory: string): boolean {
  return path === directory || path.startsWith(`${directory}/`);
}
