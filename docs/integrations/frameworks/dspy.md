<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSPy

`writerslogic_warm_integrations.dspy` provides a retriever for DSPy programs.

- Tested against `dspy==3.4.1`, Python 3.12, `warm-server` built from this repository.
- Tested dependency profile:

  ```sh
  uv add "writerslogic-warm-integrations[dspy]" "dspy==3.4.1" "litellm==1.103.10" "filelock>=4.2.6"
  ```

  LiteLLM 1.106.6 requires Filelock below 4; the current patched Filelock requires
  the compatible older LiteLLM selected above. Do not assume an unconstrained
  latest installation matches this profile.

## Current DSPy retriever shape

DSPy no longer has a retriever base class to subclass. A retriever is a callable returning `dspy.Prediction(passages=[...])` (see `dspy.retrievers.Embeddings`), called inside a `dspy.Module`. `WarmRetriever` follows that shape and is itself a `dspy.Module`, so callbacks, `acall`, `Module.deepcopy()` and optimizers work. The older `dspy.settings.configure(rm=...)` plus `dspy.Retrieve` path is covered by `WarmRM`.

## Quickstart

```python
import dspy
from writerslogic_warm_integrations.dspy import WarmRetriever

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)

retriever = WarmRetriever(k=3, url="http://127.0.0.1:8080", tenant_id="kb", input_dim=384)
retriever.add(["WARM runs locally."], metadatas=[{"source": "docs"}])

class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = retriever
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        return self.answer(context=self.retrieve(question).passages, question=question)
```

Runnable (offline, `DummyLM`): `integrations/python/examples/dspy_rag.py`.

## Cache safety

Configure memory-only caching before any DSPy program, LM, or first access to
`dspy.cache`, as in the quickstart. DSPy 3.4.1 lazily constructs its default disk
cache; calling `configure_cache(enable_disk_cache=False, enable_memory_cache=True)`
first prevents that construction. This setting is process-wide and must not be
changed back by another component. It trades persistent response caching for
in-process caching; WARM's stored memories are unaffected.

DiskCache 5.6.3 remains an installed DSPy dependency with an unsafe-pickle advisory
([GHSA-w8v5-vhqr-4h9v](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v)). This
configuration avoids its use in the verified retrieval/cache workflow; it does
not patch DiskCache, remove the advisory, or secure arbitrary application code
that uses it directly. Do not load untrusted serialized DSPy programs or caches.

## API

| Member | Behavior |
|---|---|
| `retriever(query, k=None, filter=None)` | `dspy.Prediction` with `passages`, `scores`, `ids`, `metadatas`. `aforward`/`acall` for async. |
| `add(passages, metadatas=None, ids=None)` / `aadd` | Index passages; generated ids when omitted; an existing id is replaced. |
| `delete(ids)` / `adelete` | Remove by id; returns how many existed. |
| `batch_search(queries, ...)` | One prediction per query. |
| `as_tool(name, desc)` | `dspy.Tool` returning passage strings, for `dspy.ReAct`. |
| `WarmRM(retriever)` | Legacy `rm` for `dspy.settings.configure`; returns `dotdict` rows with `long_text`, `score`, `id`, `metadata`. |

## Knobs

Constructor keywords are `WarmConfig` fields: `url`, `api_key`, `tenant_id`, `collection`, `timeout`, `zero_trust_key` (vector masking: obfuscation, not encryption), `text_queries`, `input_dim`, `overfetch`, `max_fetch`. `embedder` accepts a core `Embedder`, a `dspy.Embedder`, or any callable `list[str] -> array` (the dimension is probed). Default is the lexical `HashingEmbedder`. `score_threshold` drops weak hits; scores are cosine similarity with a client embedder and the service's fused rank score under `text_queries`, so pick the threshold for the mode you run. BM25 weighting, title weight and at-rest encryption are not exposed by the HTTP service.

## Filters

DSPy has no filter concept; `filter=` is an WARM extension. A mapping of keys to values is equality pushed to the service. MongoDB-style operators (`$gt`, `$gte`, `$lt`, `$lte`, `$ne`, `$in`, `$nin`, `$exists`, `$or`, `$and`, `$not`) run locally after fetching `k * overfetch` candidates (capped at `max_fetch`), so such filters can return fewer than `k` passages.

## Limits

- `k` above `max_fetch` (the service's `--max-top-k`, default 100) raises `ValueError`.
- A passage must fit one engine chunk (64 KiB) and its metadata 64 KiB.
- The service cannot list documents, so there is no "all passages" call; keep the ids you need to delete.
- `WarmRM` deduplicates across a multi-query call and applies `k` per query.

## Verification

`integrations/python/tests/dspy/test_dspy_retriever.py` against a live `warm-server`, one tenant per test, 10 passed with dspy 3.4.1: a real `dspy.ChainOfThought` RAG program with `DummyLM` (asserts the retrieved passage reached the prompt), a real `dspy.ReAct` run using `as_tool()`, `Module.deepcopy()`, legacy `dspy.Retrieve` with `WarmRM`, a `dspy.Embedder`-shaped callable, filters, thresholds, id replacement/deletion and the async paths. DSPy has no retriever conformance suite, so these are tests of ours against DSPy's documented calling conventions. Run:

```sh
WARM_SERVER_BIN=target/release/warm-server uv run --extra dspy pytest tests/dspy
```

`test_cache_isolation.py` additionally runs the offline example in a fresh process
that rejects every DiskCache import, then exercises a cache miss and an in-memory
write/read. This catches lazy default-cache creation even though `DummyLM` alone
does not use that cache. The default configuration fails the guard; the documented
memory-only setup passes. Eleven tests pass with DSPy 3.4.1, LiteLLM 1.103.10,
Filelock 4.2.6, and Python 3.12. This is a scoped mitigation, not a clean dependency
scan or a guarantee about other DSPy features.
