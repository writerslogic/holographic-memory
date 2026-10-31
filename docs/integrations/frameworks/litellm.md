# LiteLLM (generic OpenAI-format tools)

LiteLLM has no memory or vector-store interface. "Adapter" here is defined as: OpenAI-format function-tool schemas, a dispatcher that executes tool calls, and a bounded completion loop. The schema and dispatcher work with any OpenAI-compatible client (OpenAI SDK, Azure, Mistral, ...); only the default `completion` of the loop imports litellm.

Tested against `litellm==1.106.0` (`mcp==2.3.0`, which litellm imports when `tools=` is passed), Python 3.12.

## Install

```sh
pip install "writerslogic-warm-integrations[litellm]"
```

Needs a running `warm-server` whose `--input-dim` matches the embedder (default: 384).

## Use

```python
from writerslogic_warm_integrations.core import WarmConfig
from writerslogic_warm_integrations.litellm import MemoryTools

tools = MemoryTools(WarmConfig(url="http://127.0.0.1:8080", tenant_id="acme"), user_id="alice")

messages = [{"role": "user", "content": "Remember that I prefer dark roast."}]
response = tools.run(messages, model="gpt-5")                 # litellm.completion loop
response = await tools.arun(messages, model="gpt-5")          # litellm.acompletion loop
```

Manual control:

```python
r = litellm.completion(model=..., messages=messages, tools=MemoryTools.tools_schema())
for call in r.choices[0].message.tool_calls:
    messages.append(tools.handle_tool_call(call))             # or: await tools.ahandle_tool_call(call)
```

- `tools_schema()` returns `remember(text, tags?)`, `recall(query, k?)`, `forget(memory_id)` as `{"type": "function", "function": {...}}` (a fresh copy per call).
- `handle_tool_call` accepts LiteLLM `ChatCompletionMessageToolCall`, OpenAI SDK tool-call objects and plain dicts (arguments as a JSON string or a dict) and returns `{"role": "tool", "tool_call_id", "name", "content"}`. Bad arguments, malformed JSON or an unknown tool come back as an `Error: ...` tool message for the model to correct; service failures (`WarmError`, network) raise.
- `run`/`arun` extend `messages` in place with each assistant and tool message, pass extra keyword arguments to the completion call, accept `completion=` (any callable with the `litellm.completion` signature) and raise `RuntimeError` after `max_steps` (default 8) turns that all request tools.
- `user_id` is stored as metadata and restricts `recall` (service-side equality filter) and `forget`. `k` is clamped to `max_k` (default 20).

## Knobs

`WarmConfig`: `url`, `api_key`, `tenant_id`, `collection`, `embedder`, `text_queries`, `zero_trust_key`, `max_fetch`, `overfetch`; or pass ready `backend=` / `async_backend=` objects. Sync and async backends are created lazily.

## Filters

No filter argument; `user_id` is a top-level equality filter evaluated by the service.

## Conformance

LiteLLM's own mock completion path (`mock_tool_calls`, `mock_response`) drives real `litellm.completion`/`acompletion` calls through the loop, including `tools=` validation, against a real `warm-server`: `pytest tests/litellm`. Runaway tool loops, clamping, scoping and error messages are covered.

## Limitations

The default `HashingEmbedder` is lexical; supply an `embedder` for semantic recall. Streaming responses are not handled by `run`/`arun`; assemble the streamed tool calls and use `handle_tool_call` yourself.
