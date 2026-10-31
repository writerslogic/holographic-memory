# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""LlamaIndex on WARM: index, filtered retrieval, reload from storage, memory block.

Start a server first (dimension must match the embedder; MockEmbedding is used so no model or
API key is needed):

    WARM_API_KEY= warm-server --input-dim 8 --data-dir ./hms-data
    WARM_URL=http://127.0.0.1:8080 python examples/llamaindex_quickstart.py
"""

import asyncio
import os
import uuid

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

from writerslogic_warm_integrations.llamaindex import (
    WarmDocumentStore,
    WarmIndexStore,
    WarmMemoryBlock,
    WarmVectorStore,
)

URL = os.environ.get("WARM_URL", "http://127.0.0.1:8080")
DIM = 8
tenant = f"example-{uuid.uuid4().hex[:8]}"  # isolated store per run
warm = dict(url=URL, tenant_id=tenant, input_dim=DIM)


def main() -> None:
    embed = MockEmbedding(embed_dim=DIM)
    ctx = StorageContext.from_defaults(
        vector_store=WarmVectorStore(**warm, collection="vectors"),
        docstore=WarmDocumentStore(**warm, collection="docs"),
        index_store=WarmIndexStore(**warm, collection="indexes"),
    )
    docs = [Document(text=f"note {i}", doc_id=f"doc{i}", metadata={"team": "red" if i % 2 else "blue", "n": i}) for i in range(6)]
    index = VectorStoreIndex.from_documents(docs, storage_context=ctx, embed_model=embed)

    filters = MetadataFilters(filters=[
        MetadataFilter(key="team", value="red"),
        MetadataFilter(key="n", value=2, operator=FilterOperator.GT),
    ])
    hits = index.as_retriever(similarity_top_k=3, filters=filters).retrieve("note")
    print("filtered:", sorted(h.node.metadata["n"] for h in hits))

    reopened = load_index_from_storage(
        StorageContext.from_defaults(
            vector_store=WarmVectorStore(**warm, collection="vectors"),
            docstore=WarmDocumentStore(**warm, collection="docs"),
            index_store=WarmIndexStore(**warm, collection="indexes"),
        ),
        index_id=index.index_id,
        embed_model=embed,
    )
    print("reloaded retrieves:", len(reopened.as_retriever(similarity_top_k=6).retrieve("note")))


async def memory_demo() -> None:
    block = WarmMemoryBlock(**{**warm, "input_dim": DIM}, retrieval_context_window=1)
    await block.aput([ChatMessage(role="user", content="my cat is called Miso")], session_id="chat-1")
    print("recalled:", await block.aget([ChatMessage(role="user", content="cat called Miso")], session_id="chat-1"))
    await block.aclose()


if __name__ == "__main__":
    main()
    asyncio.run(memory_demo())
