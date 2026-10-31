# Semantic Kernel (Python)

`writerslogic_warm_integrations.semantic_kernel` implements Semantic Kernel's current vector store abstraction: `WarmStore` (a `VectorStore`) and `WarmCollection` (a `VectorStoreCollection` and `VectorSearch`). The legacy `MemoryStoreBase` connector style, including the removed adapter in `writerslogic_warm_sdk.semantic_kernel`, is superseded by this. This separate integration has not yet passed the current dependency security gate; inspect the [security notes](../../SECURITY.md#optional-python-integration-dependencies) before deployment.

## Install

```sh
pip install "writerslogic-warm-integrations[semantic-kernel]"
```

## Quickstart

```python
@vectorstoremodel
@dataclass
class Note:
    id: Annotated[str, VectorStoreField("key")]
    text: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    team: Annotated[str, VectorStoreField("data", is_indexed=True)]
    embedding: Annotated[list[float] | str | None, VectorStoreField("vector", dimensions=384)] = None

store = WarmStore(WarmConfig(url="http://127.0.0.1:8080", tenant_id="sk", input_dim=384), embedding_generator=my_sk_embedder)
async with store.get_collection(Note, collection_name="notes") as notes:
    await notes.upsert(Note(id="1", text="...", team="platform", embedding="..."))
    hits = await notes.search(values="query", top=3, filter="lambda x: x.team == 'platform'")
    fn = notes.create_search_function(function_name="search_notes", top=3)   # kernel.add_function(...)
```

See `integrations/python/examples/semantic_kernel_store.py`. Semantic Kernel's vector store API is async only, so there is no sync path.

## Record mapping

| Model | Stored |
|---|---|
| key field | document id (strings only; at most 256 bytes) |
| text data field | document text: `text_field=` if given, else first `is_full_text_indexed` field, else first data field |
| other data fields | metadata under their storage name (JSON types only; `_warm_` prefix is reserved) |
| vector field (at most one) | used as the vector when it holds a list of floats. A string or `None` is the source text: the field's or store's SK embedding generator embeds it before upsert; with no generator the backend embeds the text with `WarmConfig.embedder` (or the service does, with `text_queries=True`) |

Vectors are not returned (`include_vectors` is ignored): the service does not return stored vectors. `collection_name` scopes records inside the tenant; `ensure_collection_exists` is a no-op (collections are implicit) and `ensure_collection_deleted` deletes the collection's records. `WarmStore.list_collection_names()` only knows collections opened through that store instance.

## Search, scores, filters

- `search(values=...)` embeds with the field/collection embedding generator when present, else with `WarmConfig.embedder`. `search(vector=...)` uses your vector. `top + skip` must be at most `max_fetch` (default 100, the server's `--max-top-k`).
- Scores follow the vector field's distance function: `COSINE_SIMILARITY` returns the cosine (higher is better); `COSINE_DISTANCE` and `DEFAULT` return `1 - cosine` (lower is better). Other distance functions are rejected when the collection is created, because the service ranks by cosine.
- `hybrid_search` works only with `WarmConfig(text_queries=True)` against a server built with `local-models` (BM25 + cosine fusion; scores are then fused rank scores, not cosines). Otherwise it raises `VectorStoreOperationNotSupportedException`.
- Filters are lambdas or strings, a list meaning AND:

| Expression | Evaluated |
|---|---|
| `x.f == c` combined with `and` at the top level | by the service, before ranking |
| `!= < <= > >= in not in`, `x.f.contains(c)`, `or`, `not` | in the adapter on an over-fetch of `overfetch * (top + skip)` candidates (bounded by `max_fetch`); the result can be shorter than `top` |
| text field, vector field, unknown field, chained comparison, other calls | `VectorStoreOperationException` |

- `get(top=, skip=, order_by=)` without keys reads at most `max_fetch` records and orders them in the adapter.
- `include_total_count` reports the number of candidates fetched, not the collection size.
- `upsert` with an existing key replaces the record. `delete` of an absent key is a no-op.

## Tested

semantic-kernel 1.45.0, Python 3.12, real `warm-server`. `tests/semantic_kernel/test_semantic_kernel.py` runs the idioms from Semantic Kernel's own vector store tests (single and batch upsert/get/delete, `get` with ordering and paging, search with values and vectors, lambda and string filters, `skip`), a custom `EmbeddingGeneratorBase` service, distance-function orientation, model validation errors, collection isolation inside one tenant, a filter whose only match lies beyond the service's page size, and a real `Kernel` invoking `create_search_function` through `kernel.invoke`. Semantic Kernel does not ship a reusable connector conformance suite in the installed package, so this mirrors its in-repo integration tests rather than importing them.

Not run: a real embedding service (the offline `HashingEmbedder` stands in), and `hybrid_search` against on-device models.
