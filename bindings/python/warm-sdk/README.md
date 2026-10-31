<a href="https://writerslogic.com/warm/"><img src="https://raw.githubusercontent.com/writerslogic/warm/main/assets/crystal-halo/crystal-detailed-static.png" width="96" alt="WARM logo" align="left"></a>

# WARM SDK

Python HTTP client and framework adapters for a WARM service.

[Website](https://writerslogic.com/warm/) · [Documentation](https://writerslogic.com/warm/docs/) · [Source](https://github.com/writerslogic/warm) · [License](https://github.com/writerslogic/warm/blob/main/LICENSE)

<br clear="left">

The client talks to `warm-server`, which ships in the WARM repository. The server
stores vectors you supply and exposes document and query endpoints. See the
[server guide](https://writerslogic.com/warm/docs/latest/SERVER/) to run it.

## Installation

```bash
uv add warm-sdk
```

Framework adapters are optional extras. Install only the integration you need:

```bash
uv add "warm-sdk[langchain]"
```

The `all`, `embedchain`, `phidata`, and legacy `semantic-kernel` extras are retired.
Install individual maintained integrations instead of the former aggregate.
Embedchain forces obsolete dependencies; its incomplete adapter has been removed.
The legacy `writerslogic_warm_sdk.semantic_kernel` module has also been removed.
Phidata's upstream is archived, and its WARM adapter could not instantiate because
six abstract methods were unimplemented. Its module is removed rather than
advertising a working vector store. Agno is the upstream successor; WARM does not
yet provide or claim a compatible Agno adapter.
The separate [current Semantic Kernel adapter](https://github.com/writerslogic/warm/blob/main/docs/integrations/frameworks/semantic-kernel.md)
uses a different API and is not a drop-in replacement; its dependency environment
still needs a security review before deployment.

Other extras bring their own dependency trees; audit the environment you deploy.
See the [dependency security notes](https://github.com/writerslogic/warm/blob/main/docs/SECURITY.md#optional-python-integration-dependencies).

The earlier standalone `langchain-holographic` distribution is retired. For LangChain, install
`warm-sdk[langchain]` and import `WarmVectorStore` from
`writerslogic_warm_sdk.langchain`. This adapter connects to `warm-server`; update the old
`langchain_holographic` import and its local-store configuration when migrating.

The LangChain adapter requires `langchain-core>=0.3.86`. It preserves supplied
IDs through `add_texts`, `add_documents`, `from_texts`, and search results. Supply
stable IDs for writes you may need to reconcile or retry. The adapter forwards `filter` from similarity retrievers to WARM. Metadata and ID counts must
match the input texts before embedding begins; the embedder must return one
vector per text before any upload. Call `store.close()` when finished. Similarity
retrieval is supported; scored retrieval, MMR, and the adapter's deletion APIs are
not implemented. The security notes track unresolved dependency advisories in other extras;
these contract tests do not establish their safety.

The Haystack adapter is partial: it supports writes with `OVERWRITE` and tenant-scoped
counts and deletion. `NONE` (and an omitted policy) also means overwrite for WARM.
`FAIL` and `SKIP` raise `NotImplementedError` before writing because atomic duplicate
handling is not exposed by the server. Document enumeration/filtering raises
`NotImplementedError`; it cannot be implemented faithfully from IDs/metadata-only
listing or similarity search. This adapter does not yet implement the complete
Haystack DocumentStore protocol or support FilterRetriever. Call `store.close()`
when finished. Do not treat unsupported operations as empty successful results.

DSPy users must configure memory-only caching before the first cache or LM use:
`dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)`.
See the [tested dependency profile and its limits](https://github.com/writerslogic/warm/blob/main/docs/integrations/frameworks/dspy.md#cache-safety):
newer LiteLLM versions can force an older vulnerable Filelock. The SDK lock
alone does not constrain a consumer installation.

The CrewAI tool requires `crewai>=1.15.33` (currently Python 3.11–3.13 for this SDK)
and closes its per-call HTTP client on
success or failure. Its Chroma dependency remains subject to the security notes;
the tested tool path does not certify CrewAI's built-in storage or other features.
The SmolAgents tool requires `smolagents>=1.26.0`. Its optional `top_k` input accepts
omission or `None` as the default of three results. Use it as a context manager,
or call `tool.close()` when finished, to release its persistent HTTP connection:

```python
from writerslogic_warm_sdk.smolagents import WarmSearchTool

with WarmSearchTool(embedder=my_embedder, url="http://localhost:8080", api_key=my_api_key) as tool:
    context = tool("notes about retention", top_k=3)
```

The `pydantic-ai` extra installs `pydantic-ai-slim>=2.64.0`, which provides the
framework without every provider SDK. Add the provider extra you actually use,
for example `pydantic-ai-slim[openai]`; local tests need no paid provider.

The `fast` extra adds `orjson` for request encoding, which cuts the per-query cost of sending an embedding:

```bash
uv add "warm-sdk[fast]"
```

Requires Python 3.11 or newer. Python 3.10 reached end of life on October 1, 2026.

## Quick Start

```python
from writerslogic_warm_sdk import WarmClient

client = WarmClient(url="http://localhost:8080", api_key="...")

client.add_documents([
    {"id": "doc-1", "text": "Paris is the capital of France.", "vector": [0.12, -0.03, 0.88], "metadata": {"lang": "en"}},
])

response = client.query(query_vector=[0.10, -0.01, 0.90], top_k=3, filter={"lang": "en"})
print(response["matches"])
client.delete_document("doc-1")
client.close()
```

`AsyncWarmClient` has the same methods as coroutines; close it with `await client.aclose()`. Embeddings come from your own model: the SDK sends vectors, it does not compute them.

`client.stats()` returns authenticated, tenant-scoped server statistics, including
`documents`. The async equivalent is `await client.stats()`; authorization and
transport failures propagate rather than appearing as zero counts.

`client.read_document_chunk(doc_id, index=0)` returns one stored passage and its
`document` metadata, `chunkCount`, byte span, and opaque store `revision`. Use
`await client.read_document_chunk(...)` with the async client. Missing documents
or indices raise `WarmError` (404); `text is None` means text was not retained.
Each response contains at most 8 MiB of chunk text plus document metadata. Compare
revision strings across multi-chunk reads and restart the read sequence if they
differ. Chunk spans and overlapping passages cannot recover discarded source
whitespace or unretained data; this is not a lossless original-document export.
This method requires a server exposing the chunk-read endpoint.

After bulk ingestion, `client.maintain_document_index()` explicitly builds the server's
exact dense document index and returns `{"documentIndexTrained": true}` when the
store is eligible. Smaller or unsupported stores return `false` and use exact scanning.
Use `await client.maintain_document_index()` with the async client. Rebuild after updates
or restart; the index is a runtime cache. Searches preserve the existing cosine results,
and metadata filters continue to apply before ranking. This method requires an
`warm-server` version exposing `POST /api/v1/index/maintain`.

## Timeouts and transport failures

Both clients accept a timeout in seconds, `None`, or an `httpx.Timeout` with separate
connect/read/write settings. For example, `timeout=httpx.Timeout(30.0, connect=5.0)`
allows five seconds to connect and thirty seconds for subsequent I/O.

The synchronous client's optimized transport handles plain HTTP without environment
proxies. It applies each phase's socket timeout and preserves HTTPX
phase-specific timeout/network errors. A read/write failure can occur after the server has
received a mutation; reconcile its outcome before retrying. The transport does not
resend failed requests. HTTPS URLs and configured proxies normally use HTTPX's own
transport; passing an HTTPS request directly to `KeepAliveTransport` is rejected
before any bytes are sent. A body-producer exception is raised before starting the
HTTP exchange, so subsequent requests can still use the client. Connections are owned
by their worker thread; the client's close registry does not retain sockets after a
thread exits.

## Source Evidence

`complete_evidence` packs complete source observations in your retrieval order. Each observation contains a turn handle, session handle, observation time, role, and verbatim text. Supply opaque caller IDs as lowercase SHA-256 hex handles, and keep their association with your original records.

```python
from hashlib import sha256
from writerslogic_warm_sdk import complete_evidence, decode_source_packet

def handle(identity):
    return sha256(identity.encode("utf-8")).hexdigest()

session = handle("session-7")
candidates = [
    [(handle("turn-1"), session, "2026-10-13T10:00:00Z", "user",
      "I have two aquarium tanks.")],
    [(handle("turn-2"), session, "2026-10-13T10:00:00Z", "assistant",
      "You mentioned two aquarium tanks.")],
]

selection = complete_evidence(candidates, selected=[0], budget_bytes=2048)
observations = decode_source_packet(selection.packet)
assert observations == [candidates[index] for index in selection.selected]
```

The indices in `selected` are mandatory; packing raises `ValueError` if they exceed the budget. Additional complete groups fit in candidate order, with shared text and session metadata stored once. Decoding preserves every occurrence, role, observation time, and source text. The budget counts UTF-8 packet bytes; measure reader tokens separately. Packets contain source observations and no inferred answer or event date. These helpers run locally and make no service calls.

## Integrations

| Framework | Extra | Import |
|-----------|-------|--------|
| LangChain | `langchain` | `from writerslogic_warm_sdk.langchain import WarmVectorStore` |
| LlamaIndex | `llamaindex` | `from writerslogic_warm_sdk.llamaindex import WarmVectorStore` |
| Haystack | `haystack` | `from writerslogic_warm_sdk.haystack import WarmDocumentStore` |
| DSPy | `dspy` | `from writerslogic_warm_sdk.dspy import WarmRM` |
| CrewAI | `crewai` | `from writerslogic_warm_sdk.crewai import WarmSearchTool` |
| Pydantic AI | `pydantic-ai` | `from writerslogic_warm_sdk.pydantic_ai import get_warm_tool` |
| SmolAgents | `smolagents` | `from writerslogic_warm_sdk.smolagents import WarmSearchTool` |


The LlamaIndex adapter supports dense similarity queries with flat equality
metadata filters combined with AND. Filters run on the server before top-k
selection. Plain node text and metadata are stored separately, without prepending
metadata to the returned text. Unsupported search modes, operators, nested filters, ID restrictions
and projection/hybrid options raise `NotImplementedError`; empty ID lists mean
no restriction, matching LlamaIndex retrievers. Repeated filter keys
raise `ValueError` instead of overwriting a constraint. The adapter owns one
HTTP client; call `store.close()` when finished and construct a new store to
change connection settings. These contracts are tested with llama-index-core
0.10.0 and 0.14.25; this does not resolve the optional dependency advisories above.

Use `delete_nodes([node_id, ...])` to delete known nodes. Missing nodes are already
deleted; other errors propagate. Multiple deletions are sequential, not atomic,
so a failure can follow earlier committed deletions. Reference-document deletion
(`delete(ref_doc_id)`) and filtered deletion are unsupported and raise before any
write. A reference-document ID must never be treated as a node ID: they can name
different records. Full LlamaIndex relationship/node-type reconstruction remains
outside this adapter's current TextNode result contract.


## Vector Masking

Passing `zero_trust_key` masks every embedding client-side with a keyed signed permutation (`VectorMask`, scrypt-derived). The server can still rank vectors because inner products and cosine similarity are preserved exactly. This is obfuscation, **not encryption**: the server learns all pairwise similarities, and known plain/masked pairs reveal the key for the coordinates they cover. Pass a per-collection `mask_salt` (at least 16 bytes) shared by all clients of that collection. Masks produced by 0.1.0/0.1.1 are not compatible and were not secure.

## License

Apache-2.0. See the [LICENSE](https://github.com/writerslogic/warm/blob/main/LICENSE).
