# LlamaIndex

`writerslogic_warm_integrations.llamaindex` puts WARM behind LlamaIndex's vector store, key-value/doc/index stores, chat stores and memory interfaces.

Tested: `llama-index-core==0.14.25`, Python 3.12, against a real `warm-server`.

```sh
pip install "writerslogic-warm-integrations[llamaindex]"
WARM_API_KEY=... warm-server --input-dim 1536 --data-dir ./hms-data   # input-dim = your embedding model's dimension
```

## Classes

| Class | LlamaIndex interface | Notes |
|---|---|---|
| `WarmVectorStore` | `BasePydanticVectorStore` | stores text; sync and async `add/delete/delete_nodes/get_nodes/clear/query` |
| `WarmMemoryBlock` | `BaseMemoryBlock[str]` | message recall for `Memory`, encoded by the WARM encoder, scoped by `session_id` |
| `WarmAsyncChatStore` | `AsyncDBChatStore` | the FIFO store for `Memory.from_defaults(chat_store=...)` |
| `WarmChatStore` | `BaseChatStore` | for `ChatMemoryBuffer` |
| `WarmKVStore` | `BaseKVStore` | any KV consumer |
| `WarmDocumentStore`, `WarmIndexStore` | `KVDocumentStore`, `KVIndexStore` | persist a full `StorageContext` in WARM |

`llama-index-core`'s own `VectorMemoryBlock` also works unchanged on `WarmVectorStore`.

## Quickstart

```python
from llama_index.core import StorageContext, VectorStoreIndex
from writerslogic_warm_integrations.llamaindex import WarmVectorStore

store = WarmVectorStore(url="http://127.0.0.1:8080", api_key="...", tenant_id="acme", input_dim=1536)
index = VectorStoreIndex.from_documents(docs, storage_context=StorageContext.from_defaults(vector_store=store))
index = VectorStoreIndex.from_vector_store(store)          # reopen later
```

Runnable end to end (index, filters, reload from storage, memory block): `integrations/python/examples/llamaindex_quickstart.py`.

## Configuration (the WARM knobs)

Every class takes an `WarmConfig` or the same fields as keywords.

| Field | Meaning |
|---|---|
| `url`, `api_key`, `tenant_id` | service, bearer key, store selection (hard isolation) |
| `collection` | scope inside a tenant; ids and queries are restricted to it. Use different collections for vector, doc and index stores sharing a tenant |
| `input_dim` | must equal the server's `--input-dim` and your embedding model |
| `embedder` | the WARM encoder, used only for records or queries that arrive without an embedding (default: lexical `HashingEmbedder`). The vector store uses the embeddings LlamaIndex computes |
| `text_queries` | the service embeds and ranks text itself (BM25 + cosine, needs `warm-server` with `local-models`); enables TEXT_SEARCH, SPARSE, HYBRID and SEMANTIC_HYBRID modes |
| `zero_trust_key` | client-side vector masking (obfuscation, not encryption) |
| `overfetch`, `max_fetch` | candidates examined for client-side filtering (`overfetch * k`, capped at `max_fetch`); set `max_fetch` to the server's `--max-top-k` |

The server's BM25 parameters, title weight and at-rest encryption are engine options, not part of the HTTP service; use the Node surface or Rust for those.

## Metadata filters

The service evaluates equality on top-level keys only. `MetadataFilters` are converted to an AST: top-level ANDed equalities on strings, ints and bools are sent to the service (applied before ranking); everything else is evaluated in the adapter over `overfetch * k` candidates (at most `max_fetch`, default 100), so **a filter with a non-equality term can return fewer than `similarity_top_k` results**. Float equality is always client-side.

Semantics follow `SimpleVectorStore` (verified differentially in `tests/llamaindex/test_vector_store.py`: every case below returns the same ids from both stores).

| Operator | Mapping |
|---|---|
| `==` | equality (pushed when str/int/bool) |
| `!=`, `nin` | client-side; a missing key matches |
| `>`, `>=`, `<`, `<=` | client-side; numbers or strings only, bool excluded |
| `in` | client-side |
| `any`, `all`, `contains` | client-side membership of list metadata (substring for string metadata) |
| `text_match`, `text_match_insensitive` | client-side substring |
| `is_empty` | missing, null, `""` or `[]` |
| `FilterCondition.AND/OR` | conjunction/disjunction, nested `MetadataFilters` supported (an extension over the reference store, which rejects nesting) |
| `FilterCondition.NOT` | none of the listed filters match (reference semantics) |

Deviation: `text_match` on list-valued metadata does membership instead of raising.

`VectorStoreQuery.node_ids` and `doc_ids` become client-side `in` filters; an empty list means no restriction (the index passes `[]` for stores that hold their own nodes).

## Query modes

`default` only, unless `text_queries=True`; then TEXT_SEARCH, SPARSE, HYBRID, SEMANTIC_HYBRID send `query_str` (and the embedding if given) to the service, which fuses BM25 and cosine; `alpha` is ignored and `score` is then a fused rank score, not a cosine. MMR, SVM and the learner modes raise `ValueError`. Text mode is tested at the wire level only (mock transport), because the model-backed server is not part of the automated environment.

## Limits (inherited from `warm-server`)

- No list/get endpoint: `get_nodes` is per-id queries; `get_nodes()` without ids and `clear()` page through at most `max_fetch` records per pass (`clear` and `delete` loop until empty).
- `similarity_top_k > max_fetch` raises `ValueError` (the server caps `top_k`). Boundary tested at cap-1, cap, cap+1.
- `get_nodes` does not return embeddings. Node text must fit one record (64 KiB, 4096 words) and metadata 64 KiB; the serialized node excluding text is stored in metadata, so keep node metadata small.
- `delete_nodes()` with neither ids nor filters raises; use `clear()` (removes only this collection).
- Async methods create one `httpx.AsyncClient` on first use, bound to that event loop; call `aclose()`.

## Storage, chat and memory classes

- KV/doc/index stores: each value is the metadata of one record (JSON, 64 KiB max). `get_all` reads `buckets` hash partitions (default 8) of at most `max_fetch` entries and raises `WarmError` rather than return a partial collection, so capacity is about `buckets * max_fetch` entries per collection (tested at the boundary). Raise `buckets` for bigger docstores. Note that with a text-storing vector store the docstore stays empty unless you add to it.
- Chat stores: messages are paged by sequence number so histories longer than `max_fetch` work; `page_size <= max_fetch`. One writer per key (appends read-modify-write a header). `delete_message(key, idx)` takes the sequence number. Boundary tested around a page edge.
- `WarmMemoryBlock`: `Memory` hands blocks the chat history plus the new input, and the block queries with the last `retrieval_context_window` messages (default 5, like `VectorMemoryBlock`). A long window dilutes the query with the assistant's replies; set it to 1 to query by the latest turn only.

## Conformance

- No published third-party conformance suite exists for LlamaIndex vector stores, so the oracle is `SimpleVectorStore`: 21 filter cases compared id-for-id, plus nested filters, `VectorStoreIndex.from_documents` / `from_vector_store` / retriever / `delete_ref_doc`, the full `StorageContext` (vector + doc + index store) reload through `load_index_from_storage`, `SimpleDocumentStore` and `SimpleChatStore` differential tests, `Memory.from_defaults(chat_store=WarmAsyncChatStore, memory_blocks=[...])`, and the stock `VectorMemoryBlock`.
- Run: `WARM_SERVER_BIN=<warm-server> pytest integrations/python/tests/llamaindex` in an environment with `writerslogic-warm-integrations[llamaindex]`. 51 tests pass against `llama-index-core 0.14.25` and the real server.
