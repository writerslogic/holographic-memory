# Framework adapter matrix

Every adapter was run against a real `hms-server` (Python) or the native Node addon (JS). Python 3.12, one isolated environment per framework (`integrations/python/run-matrix.sh`). Framework dependency trees conflict, so install exactly one framework extra per environment.

## Python (`pip install "holographic-integrations[<extra>]"`)

| Extra | Adapter docs | Tested version | Verification |
|---|---|---|---|
| `langchain` | [langchain](langchain.md) | langchain-core 1.6.12 | `langchain-tests` VectorStore (sync+async) and Retriever suites |
| `langgraph` | [langgraph](langgraph.md) | langgraph 1.2.16, langgraph-checkpoint 4.3.0 | `langgraph-checkpoint-conformance` 0.0.2, level FULL (81 passed); store checked against `InMemoryStore` |
| `llamaindex` | [llamaindex](llamaindex.md) | llama-index-core 0.14.25 | 21 filter cases against `SimpleVectorStore`; index, retriever, storage-context, memory round trips |
| `haystack` | [haystack](haystack.md) | haystack-ai 3.4.0 | `haystack.testing.document_store` mixins (81 passed, 2 upstream-only skips) |
| `dspy` | [dspy](dspy.md) | dspy 3.4.1 | real `ChainOfThought` and `ReAct` programs on `DummyLM` |
| `openai-agents` | [openai-agents](openai-agents.md) | openai-agents 0.24.0 | real `Runner` with a scripted model |
| `pydantic-ai` | [pydantic-ai](pydantic-ai.md) | pydantic-ai-slim 2.58.0 | `Agent.run` with `FunctionModel` / `TestModel` |
| `smolagents` | [smolagents](smolagents.md) | smolagents 1.26.0 | `CodeAgent` and `ToolCallingAgent` with a scripted model |
| `litellm` | [litellm](litellm.md) | litellm 1.106.0, mcp 2.3.0 | LiteLLM `mock_response` / `mock_tool_calls` loops |
| `crewai` | [crewai](crewai.md) | crewai 1.15.28 | real `Crew.kickoff()` with a scripted LLM |
| `autogen` | [autogen](autogen.md) | autogen-agentchat / core / ext 0.7.5 | real `AssistantAgent` with a replay client |
| `ag2` | [autogen](autogen.md) | ag2 1.1.2 | tool toolkit (AG2 1.x has no memory interface HMS can back) |
| `semantic-kernel` | [semantic-kernel](semantic-kernel.md) | semantic-kernel 1.45.0 | real `Kernel.invoke` of a search function |
| `mem0` | [mem0](mem0.md) | mem0ai 2.2.1 | one scenario through real mem0 and the shim; result shapes compared |
| `letta` | [letta](letta.md) | letta-client 1.12.1 | `PassagesResource` signatures; tool source executed against the real server |

Not verified anywhere: `text_queries=True` (the service embeds text itself) needs a server built with `--features local-models` and model directories; it is covered at the wire level with a mock transport only. A running Letta server was not used.

## Limits common to all adapters

They come from `hms-server` (see `docs/SERVER.md`), and each adapter's page lists how it surfaces them:

- The framework core scans the service's metadata cursor in tenant/collection scope. `list` returns at most `max_fetch` matching records and accurately reports more matches; `forget_where` scans every page. Text retrieval still uses exact id-filtered queries. Framework-specific public enumeration limits remain documented on each adapter page.
- Top-level equality filters run in the service. Other operators select document ids from paginated metadata before ranking. Vector-mode recall merges exact cosine results across bounded id groups. Fused text-mode residual queries require at most 1000 eligible records and raise `HmsError(code="filter_limit")` beyond that bound; independently normalized fused scores cannot be merged safely. `overfetch` remains accepted for configuration compatibility.
- `k` greater than `max_fetch` raises `ValueError` instead of silently truncating.
- Scores are cosine similarity in vector mode and a fused rank score in text mode.
- Metadata keys beginning `_hms_` are reserved.

## JavaScript / TypeScript (`integrations/js`, npm workspace; Node 26.8, TypeScript 5.9)

All packages run against the real native addon (`holographic-memory`), no mocks of HMS.

| Package | Adapter docs | Tested version | Verification |
|---|---|---|---|
| `@holographic-memory/core` | [js-core](js-core.md) | n/a | 19 tests |
| `@holographic-memory/vercel-ai` | [vercel-ai](vercel-ai.md) | ai 7.0.146, zod 4.1 | SDK mock models, real tool loop, `streamText` |
| `@holographic-memory/mastra` | [mastra](mastra.md) | @mastra/core 1.78.0, @mastra/memory 1.38.0 | 15 contract tests plus a real `Agent` with semantic recall (Mastra publishes no vector-store test suite) |
| `@holographic-memory/langchain` | [langchainjs](langchainjs.md) | @langchain/core 1.2.19 | 23 tests mirroring the standard VectorStore behaviors (`@langchain/standard-tests` is not published on npm) and a real `RunnableWithMessageHistory` run |
| `@holographic-memory/llamaindex` | [llamaindex-ts](llamaindex-ts.md) | llamaindex 0.12.1, @llamaindex/core 0.6.22 | 18 tests: `VectorStoreIndex`, all 13 `FilterOperator`s, KV store vs `SimpleKVStore` |
| `@holographic-memory/openai-agents` | [openai-agents-js](openai-agents-js.md) | @openai/agents 0.20.0, zod 4.6.7 | 11 tests: `HmsSession` vs `MemorySession`, real `Runner`; ordering at 99/100/101/200+ items |
