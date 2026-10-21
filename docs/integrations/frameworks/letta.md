# Letta

`holographic_integrations.letta` puts Letta's archival memory on Holographic Memory in two ways.

1. **`ArchivalMemory` / `AsyncArchivalMemory`**: the same `create`, `list`, `delete`, `search` as letta-client's `client.agents.passages`, returning letta-client's own `Passage` and `PassageSearchResponse`. Use it directly from your application or as a drop-in for that resource.
2. **Custom tools** for agents on a Letta server: `register_tools(client)` uploads `hms_archival_insert` and `hms_archival_search` with `client.tools.upsert`; create the agent with `secrets=agent_secrets(url, api_key=..., input_dim=...)`.

```python
from holographic_integrations.core import HmsConfig
from holographic_integrations.letta import ArchivalMemory
with ArchivalMemory(HmsConfig(url="http://127.0.0.1:8080", input_dim=384)) as passages:
    passages.create(agent_id, text="Meeting: focus on performance", tags=["meetings"])
    passages.search(agent_id, query="performance", tags=["meetings"], top_k=3).results
```

Install: `pip install "holographic-integrations[letta]"` (adds `letta-client`).

## What was verified

- Tested against letta-client 1.12.1. The Letta server's built-in tool contract was read from the letta-ai/letta 0.16.8 source (`function_sets/base.py`, `core_tool_executor.py`, `agent_manager.py`): `archival_memory_insert(content, tags=None) -> None`; `archival_memory_search(query, tags=None, tag_match_mode="any", top_k=None, start_datetime=None, end_datetime=None)` returning a list of `{"id", "timestamp", "content", "tags"}`, default 5 results, empty query returns `[]`, naive datetimes read in the agent's timezone. `ArchivalMemory.archival_memory_insert/search` implement exactly that, and the uploaded tool signatures are checked against it in `tests/letta/test_letta_archival.py`.
- The method signatures are checked against `PassagesResource` in the installed letta-client; `register_tools` is exercised through a real `letta_client.Letta` with a mock transport (request shape only).
- The tool source is executed in a fresh namespace, as Letta's sandbox would, against a real `hms-server`.
- **Not verified:** a running Letta server. The open-source server is no longer published as a Python package and Letta Cloud needs an account, so the uploaded tools have not been executed by Letta's sandbox and the sandbox's `pip_requirements` installation of `holographic-integrations` is untested. The sandbox must reach your hms-server over the network.

## Knobs

`HmsConfig` (URL, API key, tenant, encoder `embedder=`, `zero_trust_key`, `max_fetch`, `text_queries`), `timezone` (naive datetimes and result timestamps, default UTC), `default_top_k` (5, Letta's default), `buckets` (listing, default 32). Each agent id is its own collection (`letta:<agent_id>`) in the tenant.

## Differences and limits

- Search is cosine over your embedder, not Letta's hybrid vector plus full-text RRF; results carry no `relevance` block. The default hashing embedder is lexical, so supply a real embedder for semantic recall.
- Tag and date filters run in the client on an over-fetched candidate list (`5 * top_k`, at most the service's `--max-top-k`, default 100), so a filtered search can return fewer than `top_k`. A bare `end_datetime` date means end of that day.
- `list` enumerates the agent's passages through hash buckets (`buckets * max_fetch`, 3,200 by default, reachable) and then orders by creation time, so cursors (`after`/`before`) are exact within that range; `search=` is a case-insensitive substring match. Keep `buckets` constant for a store.
- `delete` of a missing passage raises `KeyError`. Passage `embedding`, `metadata`, `source_id`/`file_id` are not stored.
- Only archival memory is provided. Core memory blocks and conversation search are Letta server features and are not replaced.
- Service limits apply: 64 KiB per passage, 256-byte ids, equality-only server filters.
