# CrewAI

`writerslogic_warm_integrations.crewai` provides the two pluggable storage points CrewAI exposes today:

| Class | Plugs into | CrewAI interface |
|---|---|---|
| `WarmMemoryStorage` | `Memory(storage=...)` (and `Crew(memory=Memory(...))`) | `crewai.memory.storage.backend.StorageBackend` |
| `WarmKnowledgeStorage` | `Knowledge(storage=...)` | `crewai.knowledge.storage.base_knowledge_storage.BaseKnowledgeStorage` |

CrewAI replaced its per-memory-type storages (short-term, long-term, entity) with one unified `Memory` and a `StorageBackend` protocol. This adapter targets that protocol, not the removed `RAGStorage`.

## Install

```sh
pip install "writerslogic-warm-integrations[crewai]"
```

Run `warm-server --input-dim N` where `N` is the length of your CrewAI embedder's vectors. `Memory` embeds text itself and hands vectors to the backend.

## Quickstart

```python
from crewai.memory.unified_memory import Memory
from writerslogic_warm_integrations.core import WarmConfig
from writerslogic_warm_integrations.crewai import WarmMemoryStorage

storage = WarmMemoryStorage(WarmConfig(url="http://127.0.0.1:8080", api_key="...", tenant_id="my_crew", input_dim=1536))
memory = Memory(storage=storage, embedder={"provider": "openai", "config": {"model": "text-embedding-3-small"}})
crew = Crew(agents=[...], tasks=[...], memory=memory)
```

Knowledge: `Knowledge(collection_name="docs", sources=[...], storage=WarmKnowledgeStorage(config=WarmConfig(...)))`. The knowledge storage embeds with `config.embedder` (default: offline `HashingEmbedder`), so give it the same encoder you want at query time.

A runnable offline example (scripted LLM, hashing embedder) is `integrations/python/examples/crewai_memory.py`.

## Config and knobs

All WARM knobs are `WarmConfig` fields: `url`, `api_key`, `tenant_id` (hard isolation, separate store), `collection` (soft scope inside a tenant), `embedder` (only for records that arrive without an embedding), `input_dim`, `zero_trust_key` (client-side vector masking, obfuscation not encryption), `max_fetch`, `overfetch`. The CrewAI embedder, LLM and scoring weights stay on `Memory(...)` as usual.

## How CrewAI concepts map

| CrewAI | Stored as |
|---|---|
| `MemoryRecord.content` | document text |
| `scope` | `crew_scope` plus one boolean key per ancestor path, so a scope prefix is an equality filter |
| `categories` | `crew_categories` plus one boolean key per category |
| `metadata[k]` | `crew_m_<k>` |
| `importance`, `created_at`, `last_accessed`, `source`, `private` | `crew_*` keys, round-tripped exactly |
| score | `1 / (1 + sqrt(2 - 2 cos))`, the same formula as CrewAI's LanceDB backend on unit vectors, so `min_score` and the composite recall score keep their scale |

## Filter mapping

| `search` / `delete` argument | Evaluated |
|---|---|
| `scope_prefix` | by the service, before ranking (ancestor keys). Matches on path-component boundaries: `/ops` matches `/ops` and `/ops/deploy`, not `/opsx` (LanceDB's `LIKE 'prefix%'` would match `/opsx`). |
| `categories=[c]` | by the service |
| `categories=[a, b, ...]` (any of) | in the adapter, on an over-fetch of `overfetch * limit` candidates (bounded by `max_fetch`) |
| `metadata_filter` | by the service (equality on every key, as in CrewAI) |
| `older_than` | in the adapter (delete only) |

## Limits

- The service has no list or offset endpoint. `list_records`, `get_scope_info`, `list_scopes`, `list_categories` and `count` read at most `max_fetch` records (default 100, the server's `--max-top-k`) and emit a `UserWarning` when truncated. Raise `max_fetch` together with `--max-top-k` for larger scopes. `search`, `delete` and `reset` are exact at any size.
- `touch_records` is not implemented (the service does not return stored vectors, so a record cannot be rewritten to update `last_accessed`); `last_accessed` is set when the record is saved. CrewAI treats `touch_records` as optional.
- `search` returns at most `max_fetch` results.
- Embedding dimension must equal `--input-dim`; a mismatch is a `422` from the service surfaced as `WarmError`, not CrewAI's `EmbeddingDimensionMismatchError`.
- Text per record is limited to 64 KiB, metadata to 64 KiB.

## Sync and async

`save`, `search`, `delete` have `asave`, `asearch`, `adelete` counterparts on the async client; the sync methods use a separate sync client. Knowledge storage has `asave`, `asearch`, `areset` as well.

## Tested

- crewai 1.15.28, Python 3.12, against `warm-server` built from this repo.
- Conformance: `isinstance(storage, StorageBackend)` (runtime-checkable protocol), every protocol method exercised against the real server, `Memory.remember/recall(depth="shallow")/forget/info/list_records` end to end, `Knowledge.add_sources/query` end to end, and a real `Crew.kickoff()` with a scripted `BaseLLM`: the recalled memory reaches the agent prompt and the task result is saved back through the backend (`tests/crewai/test_crewai.py`).
- Deep recall (`depth="deep"`) and LLM-driven scope/category inference call the LLM and are CrewAI behavior, unchanged by the backend. They need a real LLM and were not run.
