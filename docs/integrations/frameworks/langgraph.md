# LangGraph

`holographic_integrations.langgraph` provides a long-term-memory `BaseStore` (`HolographicStore`) and a checkpoint saver (`HolographicSaver`) over Holographic Memory (`hms-server`).

## Install

```sh
pip install "holographic-integrations[langgraph]"
hms-server --input-dim 384
```

## Quickstart

```python
from holographic_integrations.langgraph import HolographicSaver, HolographicStore

saver = HolographicSaver(url="http://127.0.0.1:8080", tenant_id="acme")
store = HolographicStore(index={"embed": embeddings, "dims": 384, "fields": ["text"]}, url="http://127.0.0.1:8080", tenant_id="acme")
app = graph.compile(checkpointer=saver, store=store)
```

Runnable example: `integrations/python/examples/langgraph_memory.py`.

## HolographicStore (`BaseStore`)

Implements `batch`/`abatch` (so `get`, `put`, `delete`, `search`, `list_namespaces` and their async twins). `index` is LangGraph's `IndexConfig` (`embed`: `Embeddings`, function or provider string; `dims`; `fields`). Without `index`, semantic search is off as for every `BaseStore`: `query` is ignored and `put(index=...)` has no effect. HMS knobs: `config=HmsConfig(...)` or keyword arguments (`url`, `api_key`, `tenant_id`, `collection` default `langgraph-store`, `zero_trust_key`, `max_fetch`, ...).

| LangGraph operation | Mapping |
|---|---|
| namespace prefix | per-level equality filters, run in the service |
| `filter` scalar equality (str up to 256 chars, int, null) | pushed to the service; every filter is then re-checked on the full value with LangGraph's own comparison rules (`$eq $ne $gt $gte $lt $lte`, nested dicts, lists) |
| `list_namespaces` prefix/suffix/wildcards/`max_depth` | one marker record per namespace (removed when its last item is deleted); suffix conditions are evaluated client-side |
| `index=False` / `index=[fields]` on `put` | not embedded (excluded from semantic search) / embedded from those fields |

Differences from `InMemoryStore`, all deliberate: several index fields are joined into one vector (one vector per record; InMemory keeps one per field); TTL is not supported (`ttl=` raises `NotImplementedError`); a search without `query` returns items sorted by (namespace, key) after fetching at most `max_fetch` matches, so `offset + limit` above `max_fetch` raises `ValueError`; namespaces are limited to 16 levels; a value must fit 64 KiB as JSON. Namespace listing returns at most `max_fetch` namespaces. Scores are cosine similarity.

## HolographicSaver (checkpointer)

Implements `put`, `put_writes`, `get_tuple`, `list`, `delete_thread`, `delete_for_runs`, `copy_thread`, `prune` and their async twins. Channel values are stored as shared blobs keyed by `(channel, version)` like `InMemorySaver`; payloads of any size are split over part records, so a 200 kB channel value works although the service caps text at 64 KiB. Checkpoints are grouped into pages of 50 by sequence number, so a thread of any length lists newest-first exactly; each page, channel blob and pending-write set is one query. Pending writes are stored one record per (checkpoint, task).

Limits: one writer per (thread, namespace) (LangGraph serializes a thread already); more than `max_fetch` tasks with writes on one checkpoint is unsupported; `delete_for_runs` leaves channel blobs in place (they may be shared; `prune` removes unreferenced blobs); `copy_thread` does not preserve each write's `task_path`; `list(None)` enumerates at most `max_fetch` (thread, namespace) heads. `max_fetch` must be at least 50.

## Tested against

| Package | Version |
|---|---|
| langgraph | 1.2.16 |
| langgraph-checkpoint | 4.3.0 |
| langgraph-checkpoint-conformance | 0.0.2 |
| langchain-core | 1.6.12 |
| Python | 3.12 |

- **Checkpoint saver:** `langgraph-checkpoint-conformance` against a live `hms-server`, all eight capabilities detected, 81 tests passed, 0 failed, level FULL. Plus real graphs: resume from a second saver instance, a 200 kB value, `get_state_history`, and a 62-checkpoint thread listed across pages with `limit` and `before`, sync and async.
- **Store:** LangGraph publishes no store conformance package (`langgraph-store-conformance` does not exist), so `InMemoryStore` is the oracle: 14 filter queries and 10 `list_namespaces` forms are compared against it on identical data, plus semantic search, index controls, batch ordering, TTL rejection and limits.
