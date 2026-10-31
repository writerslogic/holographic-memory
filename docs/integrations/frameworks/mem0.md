# mem0 drop-in

`writerslogic_warm_integrations.mem0` provides `Memory` and `AsyncMemory` with the mem0ai 2.x method signatures and return shapes, backed by WARM. It does not import mem0.

```python
from writerslogic_warm_integrations.mem0 import Memory   # was: from mem0 import Memory
m = Memory()                                        # warm-server on 127.0.0.1:8080, 384-dim hashing embedder
m.add("I love hiking", user_id="alice", infer=False)
m.search("hiking", filters={"user_id": "alice"})    # {"results": [...]}
```

Install: `pip install "writerslogic-warm-integrations[mem0]"` (the extra only adds mem0ai for the conformance test; the shim itself needs no extra).

Tested against mem0ai 2.2.1, Python 3.12, a real `warm-server` built from this repository. `tests/mem0/test_mem0_conformance.py` runs one scenario (add, get, get_all, search, update, history, delete, delete_all) through real mem0 (local qdrant, offline) and through the shim and requires identical result shapes, key by key; `tests/mem0/test_mem0_shim.py` covers behaviour against the server.

## Surface

`add`, `search`, `get`, `get_all`, `update`, `delete`, `delete_all`, `history`, `reset`, `close`, `from_config`, context managers; `AsyncMemory` has the same as coroutines (`aclose()` closes the HTTP client; `close()` closes only the history file).

Follows mem0 2.x validation: `search`/`get_all` take `filters={"user_id"|"agent_id"|"run_id": ...}` (top-level entity arguments raise), ids are trimmed and may not contain whitespace, `top_k`/`threshold` ranges, `expiration_date` (hidden unless `show_expired=True`), actor ids from message `name`, system messages skipped on verbatim adds, identity keys in metadata ignored.

## Configuration (WARM knobs)

```python
Memory(
    warm=WarmConfig(url=..., api_key=..., tenant_id=..., collection=..., embedder=..., input_dim=...,
                  text_queries=False, zero_trust_key="", max_fetch=100),
    extractor=None, reranker=None, history_db_path=":memory:", buckets=32, strict_infer=False,
)
```

`Memory.from_config` accepts a mem0-style dict: `vector_store: {"provider": "warm", "config": {url, api_key, tenant_id, collection_name, embedding_model_dims, ...}}`, `embedder: {"provider": "hashing"|"custom", "config": {...}}`, `llm` (a chat callable), `history_db_path`, `custom_instructions`. Any other provider (qdrant, openai, ...) raises `ValueError` instead of being ignored. The encoder is yours: pass `embedder=` an object with `dim`, `embed_documents`, `embed_query`; the default hashing embedder is lexical, not semantic. With `text_queries=True` and a server built with `local-models`, the service embeds text itself (wire-level only; not exercised here).

## Differences from mem0 (read these)

- **`add(infer=True)` is not what mem0 does by default.** mem0 asks an LLM to extract facts. Without an `extractor`, this shim stores each message verbatim (mem0's `infer=False` behaviour) and emits a `UserWarning` once per instance. Pass `extractor=LLMExtractor(chat_fn)` (any chat function: `chat_fn(messages) -> str`, optionally async; its reply must be `{"memory": [{"text": ...}]}`) or any `extractor(messages, existing_texts, prompt) -> facts`. `strict_infer=True` raises instead of storing verbatim. The extraction prompt is this package's, not mem0's, and extraction is additive with hash de-duplication (no update/delete decisions). Nothing is sent anywhere except to the function you supply.
- **Scores are cosine similarity**, not mem0's blended semantic/BM25/entity score; `threshold` (default 0.1) applies to cosine. No entity boosting, no `reranker` provider objects (pass a callable `(query, results, top_k) -> results`). `explain=True` returns `score_details={"semantic": score}`.
- **Not supported:** `memory_type="procedural_memory"` (raises `NotImplementedError`), `timestamp` and `reference_date` (raise `ValueError`, as mem0 OSS does), vision messages (non-string content is skipped), `chat`, `project`.
- **History** is a SQLite file with mem0's `history` schema (`history_db_path`, default in-memory). It is local to the shim, not stored in WARM.
- **Metadata filters:** v2 syntax (`AND`/`OR`/`NOT`, `eq ne gt gte lt lte in nin contains icontains`, `"*"`). Entity ids and top-level equalities run in the service before ranking; every other term runs in the client on an over-fetched candidate list (`overfetch * top_k`, at most the service's `--max-top-k`, default 100), so such a query can return fewer than `top_k` results. `top_k` above `max_fetch` raises `ValueError` for `search`.

## Limits

- `get_all` needs enumeration, which the service does not have. Each record carries a hash bucket (`_mem0_bucket`, `buckets` of them, default 32). One query is tried first; if it hits the service's per-query cap, each bucket is queried, so up to `buckets * max_fetch` (3,200 by default) records per filter are reachable and a `RuntimeWarning` is raised if a bucket is itself full. Keep `buckets` constant for the life of a store. `top_k` is honoured (default 20). Tested at the cap minus one, at the cap, above it and at 2.5 pages with `--max-top-k 10`.
- `delete_all` and `reset` have no such ceiling: they list and delete batch by batch until nothing remains.
- The service accepts at most 64 KiB of text and metadata per record and ids of 256 bytes; mem0 ids are UUIDs. A memory with the same id replaces the old one (`update` is get, edit, replace).
- No concurrency control: two writers updating one memory race, last write wins.
- Memory ids are scoped by the WARM `collection` (default none) inside the tenant. Documents written by other clients lack the reserved keys and are invisible here.
