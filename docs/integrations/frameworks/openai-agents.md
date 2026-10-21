# OpenAI Agents SDK

`holographic_integrations.openai_agents` provides a persistent `Session` and a long-term memory tool set for the OpenAI Agents SDK (`openai-agents`).

Tested against `openai-agents==0.24.0` (`openai==3.35.1`), Python 3.12.

## Install

```sh
pip install "holographic-integrations[openai-agents]"
```

You need a running `hms-server` (see `docs/SERVER.md`); its `--input-dim` must equal your embedder's dimension (default embedder: 384).

## Session

```python
from agents import Agent, Runner
from holographic_integrations.core import HmsConfig
from holographic_integrations.openai_agents import HolographicSession

config = HmsConfig(url="http://127.0.0.1:8080", tenant_id="acme")
session = HolographicSession("conversation-42", config)
result = await Runner.run(Agent(name="a"), "hello", session=session)
```

`HolographicSession` subclasses `agents.memory.SessionABC` and implements `get_items(limit)`, `add_items`, `pop_item`, `clear_session`, `session_id` and `session_settings` exactly as the protocol defines them (`limit` returns the latest N items in chronological order; `SessionSettings.limit` is the default). `Runner.run`, `run_sync` and `run_streamed` all take it. `isinstance(s, agents.memory.Session)` holds.

### Storage and limits

Items are JSON text in a `core.paged.AsyncPagedLog` (one record per item, partitioned into pages of `page_size` items, because the service has no list endpoint and caps one query at `--max-top-k`, default 100).

| Limit | Value | Knob |
|---|---|---|
| Items per session | unbounded; tail located in O(log pages) queries | `page_size` (default `min(50, max_fetch)`) |
| Size of one item | `chunk_chars * max_parts` characters (default 8000 x 100) | `chunk_chars` (at most 8000), `max_parts` (at most `max_fetch`) |
| Concurrent writers | one process per session id; writers sharing one session object are serialized | |

An item longer than `chunk_chars` continues in extra records. A larger item raises `ValueError` before anything is written. Parts are written before the head record, so a reader never sees a head without its parts. Tested at 49/50/51/100/101/230 items across page boundaries, and at exactly `max_parts` and `max_parts + 1` parts.

Text queries are not used for sessions; the stored vector is only needed because the service requires one.

## Memory tools

```python
from holographic_integrations.openai_agents import memory_tools

agent = Agent(name="assistant", tools=memory_tools(config, user_id="alice"))
```

Returns `remember(text, tags?)`, `recall(query, k?)` and `forget(memory_id)` as `FunctionTool`s. `user_id` is stored as metadata; `recall` is restricted to it (equality filter, pushed to the service) and `forget` refuses ids owned by another user. `k` is clamped to `max_k` (default 20). `sync=True` builds blocking tools on `HmsBackend`; the default uses `AsyncHmsBackend`.

## Knobs

Everything is on `HmsConfig`: `url`, `api_key`, `tenant_id`, `collection` (scope inside a tenant), `embedder` (the encoder; default lexical `HashingEmbedder`), `text_queries` (let the service embed, needs its `local-models` stages), `zero_trust_key` (client-side vector masking, obfuscation not encryption), `max_fetch`, `overfetch`. The BM25 and title-weight options belong to the native engine and are not reachable through the HTTP service.

## Metadata filters

The adapter does not expose a filter argument: sessions are keyed by session id, memory by `user_id`; both are top-level equality filters evaluated by the service before ranking.

## Conformance

Run with a real `hms-server` and a scripted `Model` (no network): `Runner.run` replays history to the second run, items round-trip equal across a fresh session object, `limit`/`pop_item`/`clear_session` semantics, per-session isolation, tool loop through the SDK's own tool executor (`ToolContext`, strict schemas). `pytest tests/openai_agents`. The SDK ships no public conformance suite for third-party sessions.

## Limitations

- `OpenAIResponsesCompactionAwareSession` (`run_compaction`) is not implemented.
- No server-side search over session contents; use the memory tools for retrieval.
