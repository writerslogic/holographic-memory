# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""langchain-tests' own standard suites against a live hms-server."""

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import RetrieversIntegrationTests, VectorStoreIntegrationTests
from langchain_tests.integration_tests.vectorstores import EMBEDDING_SIZE

from holographic_integrations.langchain import HolographicRetriever, HolographicVectorStore


def _store(server, **kw) -> HolographicVectorStore:
    return HolographicVectorStore(
        VectorStoreIntegrationTests.get_embeddings(), url=server.url, tenant_id=server.tenant(), **kw
    )


class TestHolographicVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture
    def vectorstore(self, server_factory) -> VectorStore:  # type: ignore[override]
        store = _store(server_factory(EMBEDDING_SIZE))
        yield store
        store.close()


class TestHolographicRetriever(RetrieversIntegrationTests):
    """Runs the retriever suite against a store holding five documents."""

    _store_ref = None

    @pytest.fixture(autouse=True)
    def _populate(self, server_factory):
        server = server_factory(EMBEDDING_SIZE)
        store = _store(server)
        store.add_documents([Document(page_content=f"document number {i}", metadata={"i": i}) for i in range(5)])
        type(self)._store_ref = store
        yield
        store.close()

    @property
    def retriever_constructor(self):
        return HolographicRetriever

    @property
    def retriever_constructor_params(self):
        return {"vectorstore": type(self)._store_ref}

    @property
    def retriever_query_example(self) -> str:
        return "document number 2"
