# LangChain

`holographic_integrations.langchain` provides a `VectorStore`, a `BaseRetriever` and a `BaseChatMessageHistory` over Holographic Memory (`hms-server`).

## Install

```sh
pip install "holographic-integrations[langchain]"
hms-server --input-dim 384        # the vector length your embeddings produce
```

## Quickstart

```python
from holographic_integrations.langchain import HolographicVectorStore, HolographicRetriever

store = HolographicVectorStore(my_embeddings, url="http://127.0.0.1:8080", tenant_id="acme", collection="docs")
store.add_documents(docs, ids=[...])
store.similarity_search("question", k=4, filter={"region": "eu"})
retriever = store.as_retriever(search_kwargs={"k": 4})          # or HolographicRetriever(vectorstore=store, ...)
```

Runnable example: `integrations/python/examples/langchain_rag.py`.

## Classes

| Class | Interface | Notes |
|---|---|---|
| `HolographicVectorStore` | `VectorStore` | `add_documents/add_texts` (+ async), `delete(ids=` or `filter=)`, `get_by_ids`, `similarity_search[_with_score][_by_vector]`, `similarity_search_with_relevance_scores`, `max_marginal_relevance_search[_by_vector]`, `from_texts`, `as_retriever`. Every method has an async twin on the async backend (no thread offload). |
| `HolographicRetriever` | `BaseRetriever` | `search_type` similarity / similarity_score_threshold / mmr, `k`, `filter`, `fetch_k`, `lambda_mult`, `include_score`; `k` and `filter` overridable per `invoke`. |
| `HolographicChatMessageHistory` | `BaseChatMessageHistory` | One `session_id`; `messages`, `add_messages`, `clear` plus `aget_messages/aadd_messages/aclear`. Any session length (see below). |
| `LangChainEmbedder`, `EmbedderAsLangChain` | bridges | LangChain `Embeddings` to the core `Embedder` and back (use `HashingEmbedder` as LangChain embeddings). |

## HMS knobs

Passed as keyword arguments or as `config=HmsConfig(...)` (not both):

| Knob | Field | Default |
|---|---|---|
| Encoder | the `embedding` argument; or `HmsConfig.embedder`; or `text_queries=True` to let a model-enabled server embed text | `HashingEmbedder(input_dim or 384)` (lexical, offline) |
| Tenant / isolation | `tenant_id` (separate store), `collection` (namespace inside a store) | default tenant, no collection |
| Auth / transport | `url`, `api_key`, `timeout`, `transport` | local server |
| Vector masking | `zero_trust_key`, `mask_salt` (obfuscation, not encryption) | off |
| Search limits | `max_fetch` (match the server's `--max-top-k`), `overfetch` | 100, 5 |
| Relevance | `relevance_score_fn` | `(cosine + 1) / 2` |

BM25 weighting, title weight and at-rest encryption belong to the native engine (Node/Rust surface); the HTTP service does not expose them, so they have no knob here.

## Metadata filters

Filters are MongoDB-style dicts (`holographic_integrations.core.filters.parse`). The service evaluates **top-level equality only**, so:

| Filter | Where it runs |
|---|---|
| `{"k": v}`, several keys ANDed, scalar `v` (str, int, bool, null) | in the service, before ranking: never reduces results below `k` |
| float equality, `$ne $gt $gte $lt $lte $in $nin $exists $contains`, `$or`, `$not`, dotted nested keys `a.b` | in the client, on `overfetch * k` candidates (at most `max_fetch`): can return fewer than `k` even when more matches exist |

Booleans never equal integers. `$ne`/`$nin` match documents missing the key.

## Limits

- `k` (and MMR `fetch_k`, which is clamped) cannot exceed `max_fetch` (the server's `--max-top-k`, default 100); `k` above it raises `ValueError`.
- Ids: 1 to 256 bytes (including the `collection/` prefix), not starting with `hms:`. Text up to 64 KiB, metadata up to 64 KiB. Metadata keys starting with `_hms_` are reserved and rejected.
- Scores are cosine similarity. With `text_queries=True` they are fused rank scores, so set `relevance_score_fn` and do not use `score_threshold` with the default function.
- MMR re-embeds the candidate texts with the configured embedder, because the service does not return stored vectors; it is unavailable with `text_queries=True` and no local embedder.
- Chat history stores messages as numbered records grouped into pages of 50 (`page_size`, at most `max_fetch`), so a session of any length reads back in order with `ceil(n/page_size) + 1` queries. The next sequence number is cached per instance: use one writer per session. Each message must fit 64 KiB (content is stored twice, as searchable text and inside the serialized message).
- Without `collection`, a store sees only records written through this package without one; documents written by other clients lack the reserved bookkeeping keys and are invisible to `get_by_ids`.
- `delete()` needs `ids` or `filter`; clearing a collection is explicit (`delete_all()`).

## Tested against

| Package | Version |
|---|---|
| langchain-core | 1.6.12 |
| langchain-tests | 1.1.9 |
| Python | 3.12 |

Conformance: `langchain-tests` `VectorStoreIntegrationTests` (sync and async, 25 tests) and `RetrieversIntegrationTests` (5 tests) pass against a live `hms-server`; plus 14 adapter tests (filters, relevance, MMR, collections, chat-history page boundaries 9/10/11/20/21, text-query wire format). Text-query mode is tested at the wire level only (it needs a server built with `local-models` and model directories).
