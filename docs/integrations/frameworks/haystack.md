<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Haystack

`holographic_integrations.haystack` provides a `DocumentStore` and two retriever components for Haystack 3.

- Tested against `haystack-ai==3.4.0`, Python 3.12, `hms-server` built from this repository.
- Extra: `pip install "holographic-integrations[haystack]"`.

## Quickstart

```python
from haystack import Document, Pipeline
from holographic_integrations.haystack import HolographicDocumentStore, HolographicTextRetriever

store = HolographicDocumentStore(url="http://127.0.0.1:8080", tenant_id="docs", embedding_dim=384)
store.write_documents([Document(content="Holographic Memory runs locally.", meta={"source": "docs"})])

pipe = Pipeline()
pipe.add_component("retriever", HolographicTextRetriever(store, top_k=3))
pipe.run({"retriever": {"query": "where does it run?",
                        "filters": {"field": "meta.source", "operator": "==", "value": "docs"}}})
```

Runnable version: `integrations/python/examples/haystack_rag.py`.

## Components

| Class | Role |
|---|---|
| `HolographicDocumentStore` | Full `DocumentStore` protocol (`count_documents`, `filter_documents`, `write_documents` with every `DuplicatePolicy`, `delete_documents`, `to_dict`/`from_dict`), every method also as `*_async`. Also `delete_all_documents`, `delete_by_filter`, `update_by_filter`, `count_documents_by_filter`, `count_unique_metadata_by_filter`, `get_metadata_fields_info`, `get_metadata_field_min_max`, `get_metadata_field_unique_values`. |
| `HolographicEmbeddingRetriever` | `run(query_embedding, filters, top_k, scale_score, return_embedding)` and `run_async`. Honors `FilterPolicy` (REPLACE and MERGE). |
| `HolographicTextRetriever` | `run(query, filters, top_k, return_embedding)` and `run_async`. Embeds the query with the store's encoder, or sends text to the service under `text_queries`. |

Both retrievers are `@component`s and round-trip through `Pipeline.to_dict()/from_dict()`. Importing the package registers `holographic_integrations.haystack` with Haystack 3's deserialization allowlist (`allow_deserialization_module`), which Haystack requires for third-party components.

## Knobs

| Haystack-side argument | Meaning |
|---|---|
| `url`, `api_key` (`Secret`, default `HMS_API_KEY`), `tenant_id`, `timeout` | Connection. `tenant_id` is a separate store; use it for hard isolation. |
| `collection` | Soft scope inside a tenant: ids and queries are namespaced, so two collections may reuse ids. |
| `embedding_dim` | The service's `--input-dim` (default 384). |
| `embedder` | Encoder (`holographic_integrations.core.Embedder`) for documents without an embedding and for text queries. Default `HashingEmbedder`, which is lexical, not semantic. Only `HashingEmbedder` serializes. |
| `text_queries` | Send query text to the service, which embeds it (service must run with `--features local-models` and model stages). Scores are then the service's fused rank scores, not cosines. |
| `zero_trust_key` (`Secret`) | Client-side vector masking. Obfuscation, not encryption; the server still sees text and metadata. |
| `return_embedding` | Whether `filter_documents` returns stored embeddings (default True). Retrievers have their own flag. |
| `list_buckets`, `max_fetch`, `overfetch` | See limits below. |

BM25 weighting, title weight and at-rest encryption are engine options of the native Node and Rust surfaces; the HTTP service does not expose them, so this adapter cannot either.

## Filters

Haystack's own evaluator (`haystack.utils.filters.document_matches_filter`) decides every match, so semantics (nested AND/OR/NOT, `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`, ISO-date ordering, `None` handling, `FilterError` for malformed filters) are Haystack's. The service only narrows the candidate set:

| Filter term | Where it runs |
|---|---|
| Top-level AND of `meta.<key> == <string>` | Pushed to the service (equality on JSON). |
| `==` with numbers, booleans, `None`, or ISO-date strings | Local (Haystack treats `1 == 1.0 == True` and compares dates as instants; the service compares JSON exactly). |
| Everything else, `meta.a.b` nested keys, `id`, `content`, OR, NOT, ranges, `in` | Local, after enumeration or an over-fetch. |

Malformed filters raise `FilterError` before any request, even on an empty store.

## Limits

The service has no list or get endpoint and caps `top_k` at `--max-top-k` (default 100), so:

- **Whole-store operations** (`count_documents`, `filter_documents`, `delete_by_filter`, `update_by_filter`, the metadata statistics) enumerate through queries. They first try one query; if it returns `max_fetch` records they fan out over `list_buckets` hash buckets (default 16). That covers roughly `list_buckets * max_fetch` documents (about 1,600 by default, fewer when the hash is uneven) and raises `HmsError` beyond it. `list_buckets` is part of the stored data layout: set it when creating a tenant and do not change it. Boundary behavior is tested at 9, 10 and 11 documents with `max_fetch=10`.
- **Retrieval** with a residual filter fetches `top_k * overfetch` candidates (capped at `max_fetch`) and filters locally, so it can return fewer than `top_k` documents. Pure string-equality filters are exact.
- `top_k` above `max_fetch` raises `ValueError`.
- `write_documents` with `FAIL` or `SKIP` (and the default `NONE`, which is `FAIL`) checks existing ids one query per document; use `OVERWRITE` for bulk loads.
- A document's text must fit one engine chunk (64 KiB); metadata, including the stored embedding and blob, must fit 64 KiB. Split long text with Haystack's `DocumentSplitter` before writing.
- Metadata keys `hs_bucket`, `hs_payload` and anything starting `_hms_` are reserved. Embeddings, blobs and sparse embeddings are stored inside the metadata payload so documents round-trip exactly.
- `DuplicatePolicy.NONE` behaves as `FAIL`, as in `InMemoryDocumentStore`.
- Listing order is unspecified (results are sorted by id).
- `delete_all_documents` has no `recreate_index` parameter, so those two upstream tests skip.
- Not provided: sparse retrieval, hybrid retrieval beyond what the service does under `text_queries`.

## Conformance

`integrations/python/tests/haystack/test_haystack_conformance.py` runs Haystack's own `haystack.testing.document_store` suites against a live `hms-server` (`--input-dim 768`, one tenant per test): `DocumentStoreBaseExtendedTests` (count, write, delete, all filter tests including malformed filters, delete-all, delete-by-filter, update-by-filter) plus `CountDocumentsByFilterTest`, `CountUniqueMetadataByFilterTest`, `GetMetadataFieldsInfoTest`, `GetMetadataFieldMinMaxTest`, `GetMetadataFieldUniqueValuesTest`. Result with haystack-ai 3.4.0: 81 passed, 2 skipped (the `recreate_index` variants, see above). Only `assert_documents_are_equal` is overridden (compare as a set sorted by id) and `test_write_documents`, which Haystack requires every store to define.

`test_haystack_store.py` adds real-server tests for pipelines, serialization, collections, masking, async, the enumeration boundary and the `text_queries` wire format (the last against a mock transport: on-device models were not run for this adapter).

Run:

```sh
HMS_SERVER_BIN=target/release/hms-server uv run --extra haystack pytest tests/haystack
```
