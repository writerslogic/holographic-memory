# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Text and hybrid query modes need an hms-server built with local-models and model directories.

Those are not available in CI, so this checks the wire format against a mock transport only:
it proves what the adapter sends, not what a model-backed server returns.
"""

import json

import httpx
import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

from holographic_integrations.llamaindex import HolographicVectorStore


def test_hybrid_mode_sends_text_query():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"matches": []})

    store = HolographicVectorStore(input_dim=8, text_queries=True, transport=httpx.MockTransport(handler))
    store.query(VectorStoreQuery(query_str="find me", mode=VectorStoreQueryMode.HYBRID, similarity_top_k=4))
    assert seen == [{"top_k": 4, "query": "find me", "filter": {"_hms_collection": ""}}]


def test_text_mode_rejected_without_text_queries():
    store = HolographicVectorStore(input_dim=8, transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(ValueError, match="text_queries"):
        store.query(VectorStoreQuery(query_str="x", mode=VectorStoreQueryMode.TEXT_SEARCH))
