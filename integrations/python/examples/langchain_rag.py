# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""LangChain: vector store, retriever with a metadata filter, and persistent chat history.

Needs a running warm-server started with `--input-dim 384` (the default):
    WARM_API_KEY=... warm-server --input-dim 384
Set WARM_URL / WARM_API_KEY if not local. Embeddings here are the offline lexical
`HashingEmbedder`; pass any LangChain `Embeddings` (OpenAI, HuggingFace, Ollama) instead.
"""

import os
import uuid

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from writerslogic_warm_integrations.core import HashingEmbedder
from writerslogic_warm_integrations.langchain import (
    EmbedderAsLangChain,
    WarmChatMessageHistory,
    WarmRetriever,
    WarmVectorStore,
)

URL = os.environ.get("WARM_URL", "http://127.0.0.1:8080")
KEY = os.environ.get("WARM_API_KEY")
TENANT = "example-" + uuid.uuid4().hex[:8]  # a private store for this run

store = WarmVectorStore(
    EmbedderAsLangChain(HashingEmbedder(384)), url=URL, api_key=KEY, tenant_id=TENANT, collection="docs"
)
store.add_documents(
    [
        Document(page_content="Quarterly revenue grew twelve percent in Europe", metadata={"region": "eu", "year": 2025}),
        Document(page_content="Revenue in North America was flat this quarter", metadata={"region": "na", "year": 2025}),
        Document(page_content="Our Europe office moved to a larger building", metadata={"region": "eu", "year": 2024}),
    ],
    ids=["a", "b", "c"],
)

retriever = WarmRetriever(vectorstore=store, k=2, filter={"region": "eu"}, include_score=True)
for doc in retriever.invoke("how did revenue change"):
    print(f"{doc.metadata['warm_score']:.3f}  {doc.id}  {doc.page_content}")

# operators run client-side on over-fetched candidates (documented limit)
recent = store.similarity_search("revenue", k=3, filter={"year": {"$gte": 2025}})
print([d.id for d in recent])

history = WarmChatMessageHistory("session-1", url=URL, api_key=KEY, tenant_id=TENANT)
history.add_messages([HumanMessage("What grew in Europe?"), AIMessage("Revenue, by twelve percent.")])
print([f"{m.type}: {m.content}" for m in history.messages])

store.delete_all()
history.clear()
