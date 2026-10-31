# Anthropic memory tool on WARM

Anthropic's client-side [memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) (`{"type": "memory_20250818", "name": "memory"}`) lets Claude read and write a `/memories` directory that your application serves. These packages serve it from a WARM store through `warm-mcp`, so Claude's memory files are also searchable by meaning.

| Language | Package | Entry points |
|---|---|---|
| Python | `writerslogic-warm-claude-tool` (`warm_memory_tool`) | `WarmMemoryTool` (async, `BetaAsyncAbstractMemoryTool`), `WarmMemoryToolSync` (`BetaAbstractMemoryTool`) |
| TypeScript | `writerslogic-warm-claude-tool` | `WarmMemoryTool` (implements the SDK's `MemoryToolHandlers`; `asTool()` wraps it with `betaMemoryTool`) |

Both connect to an `warm-mcp` server over stdio (they start it) or streamable HTTP (`url=`), and both add a companion tool, `memory_recall`, for semantic recall.

## Use

```python
import anthropic
from warm_memory_tool import WarmMemoryToolSync

client = anthropic.Anthropic()
with WarmMemoryToolSync.connect(command="warm-mcp") as memory:
    final = client.beta.messages.tool_runner(
        model="claude-sonnet-5-5", max_tokens=1024,
        messages=[{"role": "user", "content": "Remember that Acme Corp prefers email follow-ups."}],
        tools=[memory, memory.recall_tool()],
    ).until_done()
```

```ts
const memory = await WarmMemoryTool.connect({ connection: { command: "warm-mcp" } });
const final = await client.beta.messages.toolRunner({
  model: "claude-sonnet-5-5", max_tokens: 1024,
  messages: [{ role: "user", content: "Remember that Acme Corp prefers email follow-ups." }],
  tools: [memory.asTool(), memory.recallTool()],
});
await memory.close();
```

Connection options: `command`/`args`/`env` (Python) or `connection: { command, args, env }` (TypeScript) start a server; `url` (and `token`, default `WARM_MCP_TOKEN`) connect to a shared `warm-mcp --transport http`. `namespace` selects a store on a server started with `--namespace-arg`: one store per user or per agent. `max_file_bytes` can only lower the server's limit.

## Command behaviour

The command schema and the result strings follow the current documentation; where the documentation and the Anthropic SDKs' own `BetaLocalFilesystemMemoryTool` differ, the SDK's strings are used (directory entries end in `/`, `str_replace` returns a snippet).

| Command | Behaviour |
|---|---|
| `view` of a directory | `Here're the files and directories up to 2 levels deep in {path}, excluding hidden items and node_modules:` then `{size}\t{path}` lines (human sizes such as `5.5K`; a directory's size is the sum of its files). The first `view /memories` of an empty store succeeds. |
| `view` of a file | `Here's the content of {path} with line numbers:` then lines numbered from 1 in a 6-wide column. `view_range` is `[start, end]` or `[start, -1]`. A trailing newline adds no empty line. |
| long `view` | Output is cut at a line boundary after `max_view_chars` (16000) with a note giving the line to resume from with `view_range`. |
| `create` | Fails with `File {path} already exists` rather than overwriting; refuses a path that is a directory or sits under a file. |
| `str_replace` | The unique occurrence of `old_str` is replaced (`new_str` omitted deletes it). Zero or several matches fail with the documented messages, several naming the line numbers. Returns the edited snippet. |
| `insert` | `insert_text` goes after line `insert_line` (`0` is the top); an out-of-range line reports `[0, n]`. The file ends with a newline afterwards, as in the SDK reference. |
| `delete` | A file, or a directory with everything under it. `/memories` itself is refused. |
| `rename` | A file or a whole directory; the destination must not exist; `/memories` and moving a directory into itself are refused. Copies first and deletes second, so a failure leaves the original in place. |

Errors are raised as the SDK's tool error (`ToolError` in Python, a thrown `Error` in TypeScript), which the runner returns as `is_error: true`.

## Safety

- Every path in every command is validated before storage is touched: it must be `/memories` or start with `/memories/`; `..` and `.` segments, backslashes, control characters, empty and hidden (dot-prefixed) segments are rejected; percent-encoded dots, slashes and backslashes (`%2e`, `%2f`, `%5c`, any case) and NUL are rejected outright; names are limited to 128 bytes, paths to 240 bytes and 16 levels. Paths are storage keys, never filesystem paths, so there is no symlink or canonicalisation surface.
- File size is capped at the server's `max_text_bytes` (default 32 KiB) with a clear message; split larger content.
- Edits use compare-and-set on the file version and retry with jitter (25 attempts by default), so concurrent writers do not lose updates. Rename and directory delete are not transactions across files; rename undoes its copies on failure.

## Semantic recall

`memory.recall(query, k=5)` and the companion `memory_recall` tool return the files most relevant to a query with a relevance score and an excerpt. Files are stored with the tag `memory-file`, so recall covers only memory files even when the same store also holds other memories. Recall is lexical (BM25) in the default `warm-mcp` build and hybrid lexical plus semantic when the server runs with an embedding model; `docs/integrations/mcp/README.md` says which build gives which.

## Tests

```sh
# Python
WARM_MCP_BIN=/path/to/warm-mcp uv run --project integrations/claude-memory-tool/python pytest integrations/claude-memory-tool/python/tests
# TypeScript
cd integrations/claude-memory-tool/typescript && npm install && WARM_MCP_BIN=/path/to/warm-mcp npm test
```

Both suites run against a real `warm-mcp` process and drive the backend with the Anthropic SDKs' own command types. They cover every command and error string, 20 path-traversal forms, directory moves and conflicts, long-file paging and the size limit (limit and limit plus one), concurrent edits of one file, restart persistence, namespace isolation and semantic recall, and they run the SDK tool runner end to end (`client.beta.messages.tool_runner` / `toolRunner`) against a scripted model endpoint: the runner sends `{"type": "memory_20250818", "name": "memory"}`, dispatches `tool_use` blocks to the backend and returns `is_error` for a traversal attempt. No Anthropic API call is made; a live model was not used for this tool.
