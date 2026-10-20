# holographic-memory-claude-tool

A backend for Anthropic's client-side memory tool (`memory_20250818`) on Holographic Memory. Claude's `/memories` directory becomes a virtual filesystem stored in an `hms-mcp` server, so every file is also searchable by meaning.

```ts
import Anthropic from "@anthropic-ai/sdk";
import { HolographicMemoryTool } from "holographic-memory-claude-tool";

const memory = await HolographicMemoryTool.connect({ connection: { command: "hms-mcp" } }); // or { url: "http://127.0.0.1:8765/mcp" }
const client = new Anthropic();
const final = await client.beta.messages.toolRunner({
  model: "claude-sonnet-5-5",
  max_tokens: 1024,
  messages: [{ role: "user", content: "Remember that Acme Corp prefers email follow-ups." }],
  tools: [memory.asTool(), memory.recallTool()], // recallTool adds semantic search over the files
});
await memory.close();
```

`HolographicMemoryTool` implements the SDK's `MemoryToolHandlers`: `view`, `create`, `str_replace`, `insert`, `delete`, `rename` with the documented result strings. It rejects any path outside `/memories` (traversal, encoded separators, hidden or oversized segments), applies edits with compare-and-set so concurrent writers do not lose updates, and pages long files through `view_range`.

Pass `namespace: "user-123"` (server started with `--namespace-arg`) for one store per user. See `docs/integrations/mcp/memory-tool.md` in the repository for the full reference.

Licensed Apache-2.0.
