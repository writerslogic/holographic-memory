# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Haystack indexing and query pipelines on WARM.

Start a service first:  WARM_API_KEY=dev warm-server --input-dim 384
Then:                   WARM_URL=http://127.0.0.1:8080 WARM_API_KEY=dev python haystack_rag.py

The default encoder is the dependency-free lexical HashingEmbedder. For semantic retrieval pass
`embedder=` (any object with embed_documents/embed_query and `dim`) or run the service with
on-device model stages and set `text_queries=True`.
"""

import os

from haystack import Document, Pipeline
from haystack.utils import Secret

from writerslogic_warm_integrations.haystack import WarmDocumentStore, WarmTextRetriever

store = WarmDocumentStore(
    url=os.environ.get("WARM_URL", "http://127.0.0.1:8080"),
    api_key=Secret.from_env_var("WARM_API_KEY", strict=False),
    tenant_id=os.environ.get("WARM_TENANT", "haystack-example"),
    embedding_dim=int(os.environ.get("WARM_INPUT_DIM", "384")),
)
store.delete_all_documents()
store.write_documents(
    [
        Document(content="Haystack pipelines connect components into retrieval workflows.", meta={"source": "docs", "year": 2023}),
        Document(content="WARM stores vectors and text locally on your machine.", meta={"source": "docs", "year": 2025}),
        Document(content="Pasta needs ripe tomato and basil.", meta={"source": "recipes", "year": 2024}),
    ]
)

pipeline = Pipeline()
pipeline.add_component("retriever", WarmTextRetriever(store, top_k=2))
result = pipeline.run(
    {
        "retriever": {
            "query": "where is my memory stored?",
            "filters": {"field": "meta.source", "operator": "==", "value": "docs"},
        }
    }
)
for doc in result["retriever"]["documents"]:
    print(f"{doc.score:.3f}  {doc.content}  {doc.meta}")
assert result["retriever"]["documents"][0].meta["source"] == "docs"
print("documents in store:", store.count_documents())
