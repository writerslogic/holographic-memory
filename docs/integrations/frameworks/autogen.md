# AutoGen and AG2

"AutoGen" is now three things. This page records what each is and what the adapters implement.

| Project | State (checked 2026-10) | Adapter |
|---|---|---|
| `autogen-agentchat` / `autogen-core` 0.4+ (Microsoft) | Archived, maintenance mode. Microsoft's README names **Microsoft Agent Framework** as the successor and recommends it for new projects. | `holographic_integrations.autogen.HolographicMemory`, an `autogen_core.memory.Memory` |
| `ag2` 1.x (the AG2 community fork) | Active. 1.x is a rewrite (package `ag2`); the old `autogen` import namespace is gone. Its memory interface is `ag2.knowledge.KnowledgeStore`, a path-addressed file store. | `holographic_integrations.ag2.HolographicMemoryToolkit`, remember/recall/forget tools |
| Microsoft Agent Framework | Successor to Microsoft AutoGen. | Not covered here. |

Why AG2 gets tools and not a `KnowledgeStore`: that protocol requires atomic `append` returning a byte offset, `read_range` over unbounded files, and directory listing. HMS stores at most 64 KiB per document, has no atomic append and no listing endpoint, so an implementation could not honor the contract (write-ahead logs would exceed one document within a few turns). AG2 agents get semantic memory through tools instead, which is what a vector store is for.

## Install

```sh
pip install "holographic-integrations[autogen]"   # autogen-agentchat 0.4+ Memory
pip install "holographic-integrations[ag2]"       # AG2 1.x tools
```

The two extras conflict (different frameworks); install one per environment.

## AutoGen quickstart

```python
from holographic_integrations.autogen import HolographicMemory
from holographic_integrations.core import HmsConfig

memory = HolographicMemory("assistant_notes", hms=HmsConfig(url="http://127.0.0.1:8080", tenant_id="agents", input_dim=384), k=5)
agent = AssistantAgent("helper", model_client=client, memory=[memory])
await memory.add(MemoryContent(content="Prefers dark roast", mime_type=MemoryMimeType.TEXT, metadata={"topic": "food"}))
```

`update_context` queries with the latest message and adds a system message in the same format as `ListMemory`. `query(q, k=, score_threshold=, filter=)` takes a MongoDB-style filter over the `MemoryContent.metadata` you stored. Results carry `metadata["id"]` and `metadata["score"]` (cosine). `name` is the collection, so several agents can share one tenant. `clear()` deletes the collection's records.

Content types: text, Markdown and JSON `dict` (round-trips as a dict). `bytes` and images raise `ValueError`. Supply `metadata["id"]` to make an add idempotent.

Serialization: `dump_component()` / `load_component()` work (`provider = holographic_integrations.autogen.HolographicMemory`). The config holds name, URL, tenant, dimension, `k`, threshold, overfetch and `max_fetch`. The API key (read from `HMS_API_KEY` on load) and a custom embedder are never serialized.

## AG2 quickstart

```python
from ag2 import Agent
from holographic_integrations.ag2 import HolographicMemoryToolkit

agent = Agent("assistant", config=cfg, tools=[HolographicMemoryToolkit(hms_config, k=5)])
```

Tools: `remember(text, metadata?) -> id`, `recall(query, k?, filter?) -> [{id, text, metadata, score}]`, `forget(id) -> bool` (omit with `allow_forget=False`). The toolkit's `.backend` is the `AsyncHmsBackend` for direct use.

## Knobs

Both take an `HmsConfig`: `url`, `api_key`, `tenant_id`, `collection`, `embedder` (default offline `HashingEmbedder`; set `input_dim` to match `--input-dim`), `text_queries`, `zero_trust_key`, `overfetch`, `max_fetch`.

## Filter mapping

| Filter | Evaluated |
|---|---|
| `{"k": v}` equality, several keys ANDed | by the service, before ranking |
| `$gt $gte $lt $lte $ne $in $nin $exists $contains`, `$or`, `$not`, nested keys | in the adapter on an over-fetch of `overfetch * k` candidates (bounded by `max_fetch`); results can be fewer than `k` |

Limits: `k` is at most `max_fetch` (default 100, the server's `--max-top-k`); no listing endpoint, so there is no "all memories" read; text and metadata are 64 KiB each.

## Tested

- autogen-agentchat / autogen-core / autogen-ext 0.7.5, Python 3.12: `Memory` ABC, add/query round trips, filters, JSON content, collections isolated in one tenant, `update_context` into a real `BufferedChatCompletionContext`, a real `AssistantAgent(memory=[...])` run with `ReplayChatCompletionClient` asserting the memory text reached the model call and a `MemoryQueryEvent` was emitted, component dump/load (`tests/autogen/test_autogen_memory.py`).
- ag2 1.1.2, Python 3.12: a real `Agent` with scripted `TestConfig` calling `remember` then `recall`, asserting the tool result reached the model and the record is in the service; `forget`; the toolkit survives AG2's deep copy of tools (`tests/autogen/test_ag2_toolkit.py`).
- Examples: `examples/autogen_memory.py`, `examples/ag2_memory.py` (offline, run against a live `hms-server`).
- Not run: real LLM providers; Microsoft Agent Framework.
