# writerslogic-warm-claude-tool

A backend for Anthropic's client-side memory tool (`memory_20250818`) on WARM. Claude's `/memories` directory becomes a virtual filesystem stored in an `warm-mcp` server, so every file is also searchable by meaning.

```python
import anthropic
from warm_memory_tool import WarmMemoryToolSync

client = anthropic.Anthropic()
with WarmMemoryToolSync.connect(command="warm-mcp") as memory:   # or url="http://127.0.0.1:8765/mcp"
    runner = client.beta.messages.tool_runner(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Remember that Acme Corp prefers email follow-ups."}],
        tools=[memory, memory.recall_tool()],   # recall_tool adds semantic search over the files
    )
    print(runner.until_done().content)
```

`WarmMemoryTool` is the async variant (`BetaAsyncAbstractMemoryTool`). Both implement the six commands (`view`, `create`, `str_replace`, `insert`, `delete`, `rename`) with the documented result strings, reject any path outside `/memories` (traversal, encoded separators, hidden or oversized segments), apply edits with compare-and-set so concurrent writers do not lose updates, and page long files through `view_range`.

Pass `namespace="user-123"` (server started with `--namespace-arg`) for one store per user. See `docs/integrations/mcp/memory-tool.md` in the repository for the full reference.

Licensed Apache-2.0.
