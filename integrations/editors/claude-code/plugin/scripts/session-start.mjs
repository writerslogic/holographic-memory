#!/usr/bin/env node
// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// SessionStart hook: tells the model that WARM is connected and how to use it. The hook
// never opens the memory store itself: the MCP server holds the store lock for the session.
import { readFileSync } from "node:fs";

let source = "startup";
try {
  const input = JSON.parse(readFileSync(0, "utf8"));
  if (typeof input.source === "string") source = input.source;
} catch {
  // No or malformed stdin: fall back to the startup text.
}

const prefix = "mcp__plugin_writerslogic-warm_writerslogic-warm__";
const lines = [
  "WARM is connected: durable, private memory stored on this machine.",
  `- Before starting work, call ${prefix}recall with the task or question to retrieve relevant earlier decisions and facts.`,
  `- Save durable facts, decisions, conventions and open questions with ${prefix}remember (one topic per memory, a stable id such as "decisions/<topic>" so later writes replace it, tags for grouping).`,
  `- Use ${prefix}list and ${prefix}get to review, ${prefix}forget to remove stale memories.`,
];
if (source === "compact") {
  lines.push("- The conversation was just compacted: recall again for anything the summary may have dropped, and save unsaved decisions now.");
} else if (source === "resume") {
  lines.push("- This session was resumed: recall for what changed since it paused.");
}

process.stdout.write(
  JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      additionalContext: lines.join("\n"),
    },
  }),
);
