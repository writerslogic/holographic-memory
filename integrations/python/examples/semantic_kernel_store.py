# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Semantic Kernel vector store backed by Holographic Memory.

    HMS_API_KEY=demo hms-server --input-dim 64 --dim 1024
    HMS_URL=http://127.0.0.1:8080 HMS_API_KEY=demo python semantic_kernel_store.py
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

from holographic_integrations.core import HashingEmbedder, HmsConfig
from holographic_integrations.semantic_kernel import HolographicStore


@vectorstoremodel
@dataclass
class Note:
    id: Annotated[str, VectorStoreField("key")]
    text: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)]
    team: Annotated[str, VectorStoreField("data", is_indexed=True)]
    embedding: Annotated[list[float] | str | None, VectorStoreField("vector", dimensions=64)] = None


async def main() -> None:
    config = HmsConfig(
        url=os.environ.get("HMS_URL", "http://127.0.0.1:8080"),
        api_key=os.environ.get("HMS_API_KEY"),
        tenant_id="sk_demo",
        input_dim=64,
        embedder=HashingEmbedder(64),  # swap for a real model and match --input-dim
    )
    async with HolographicStore(config).get_collection(Note, collection_name="notes") as notes:
        await notes.ensure_collection_deleted()
        await notes.upsert([
            Note(id="1", text="The release train leaves every Thursday", team="platform"),
            Note(id="2", text="Invoices are paid on the first of the month", team="finance"),
            Note(id="3", text="Rollbacks need two approvers", team="platform"),
        ])
        hits = await notes.search(values="when does the release train leave", top=2, filter="lambda x: x.team == 'platform'")
        async for hit in hits.results:
            print(f"{hit.score:.3f}  {hit.record.id}  {hit.record.text}")

        kernel = Kernel()
        search = notes.create_search_function(function_name="search_notes", description="Search notes", top=1)
        kernel.add_function(plugin_name="notes", function=search)
        result = await kernel.invoke(plugin_name="notes", function_name="search_notes", query="invoice payment date")
        print(result.value[0])


asyncio.run(main())
