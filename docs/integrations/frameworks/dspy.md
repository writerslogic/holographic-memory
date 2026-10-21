<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSPy

`holographic_integrations.dspy` provides a retriever for DSPy programs.

- Tested against `dspy==3.4.1`, Python 3.12, `hms-server` built from this repository.
- Extra: `pip install "holographic-integrations[dspy]"`.

## Current DSPy retriever shape

DSPy no longer has a retriever base class to subclass. A retriever is a callable returning `dspy.Prediction(passages=[...])` (see `dspy.retrievers.Embeddings`), called inside a `dspy.Module`. `HolographicRetriever` follows that shape and is itself a `dspy.Module`, so callbacks, `acall`, `Module.deepcopy()` and optimizers work. The older `dspy.settings.configure(rm=...)` plus `dspy.Retrieve` path is covered by `HolographicRM`.

## Quickstart

```python
import dspy
from holographic_integrations.dspy import HolographicRetriever

retriever = HolographicRetriever(k=3, url="http://127.0.0.1:8080", tenant_id="kb", input_dim=384)
retriever.add(["Holographic Memory runs locally."], metadatas=[{"source": "docs"}])

class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = retriever
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        return self.answer(context=self.retrieve(question).passages, question=question)
```

Runnable (offline, `DummyLM`): `integrations/python/examples/dspy_rag.py`.

## API

| Member | Behavior |
|---|---|
| `retriever(query, k=None, filter=None)` | `dspy.Prediction` with `passages`, `scores`, `ids`, `metadatas`. `aforward`/`acall` for async. |
| `add(passages, metadatas=None, ids=None)` / `aadd` | Index passages; generated ids when omitted; an existing id is replaced. |
| `delete(ids)` / `adelete` | Remove by id; returns how many existed. |
| `batch_search(queries, ...)` | One prediction per query. |
| `as_tool(name, desc)` | `dspy.Tool` returning passage strings, for `dspy.ReAct`. |
| `HolographicRM(retriever)` | Legacy `rm` for `dspy.settings.configure`; returns `dotdict` rows with `long_text`, `score`, `id`, `metadata`. |

## Knobs

Constructor keywords are `HmsConfig` fields: `url`, `api_key`, `tenant_id`, `collection`, `timeout`, `zero_trust_key` (vector masking: obfuscation, not encryption), `text_queries`, `input_dim`, `overfetch`, `max_fetch`. `embedder` accepts a core `Embedder`, a `dspy.Embedder`, or any callable `list[str] -> array` (the dimension is probed). Default is the lexical `HashingEmbedder`. `score_threshold` drops weak hits; scores are cosine similarity with a client embedder and the service's fused rank score under `text_queries`, so pick the threshold for the mode you run. BM25 weighting, title weight and at-rest encryption are not exposed by the HTTP service.

## Filters

DSPy has no filter concept; `filter=` is an HMS extension. A mapping of keys to values is equality pushed to the service. MongoDB-style operators (`$gt`, `$gte`, `$lt`, `$lte`, `$ne`, `$in`, `$nin`, `$exists`, `$or`, `$and`, `$not`) run locally after fetching `k * overfetch` candidates (capped at `max_fetch`), so such filters can return fewer than `k` passages.

## Limits

- `k` above `max_fetch` (the service's `--max-top-k`, default 100) raises `ValueError`.
- A passage must fit one engine chunk (64 KiB) and its metadata 64 KiB.
- The service cannot list documents, so there is no "all passages" call; keep the ids you need to delete.
- `HolographicRM` deduplicates across a multi-query call and applies `k` per query.

## Verification

`integrations/python/tests/dspy/test_dspy_retriever.py` against a live `hms-server`, one tenant per test, 10 passed with dspy 3.4.1: a real `dspy.ChainOfThought` RAG program with `DummyLM` (asserts the retrieved passage reached the prompt), a real `dspy.ReAct` run using `as_tool()`, `Module.deepcopy()`, legacy `dspy.Retrieve` with `HolographicRM`, a `dspy.Embedder`-shaped callable, filters, thresholds, id replacement/deletion and the async paths. DSPy has no retriever conformance suite, so these are tests of ours against DSPy's documented calling conventions. Run:

```sh
HMS_SERVER_BIN=target/release/hms-server uv run --extra dspy pytest tests/dspy
```
