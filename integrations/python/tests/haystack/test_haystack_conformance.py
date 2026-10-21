# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Haystack's own document store conformance suites, run against a real hms-server."""

import pytest

pytest.importorskip("haystack")

from haystack.dataclasses import Document  # noqa: E402
from haystack.document_stores.types import DuplicatePolicy  # noqa: E402
from haystack.testing.document_store import (  # noqa: E402
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    DocumentStoreBaseExtendedTests,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
)

from holographic_integrations.haystack import HolographicDocumentStore  # noqa: E402

DIM = 768  # haystack's fixture embeddings


class TestHolographicDocumentStore(
    DocumentStoreBaseExtendedTests,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    @pytest.fixture
    def document_store(self, server_factory):
        srv = server_factory(DIM, max_tenants=100000)
        store = HolographicDocumentStore(url=srv.url, tenant_id=srv.tenant(), embedding_dim=DIM)
        yield store
        store.close()

    @staticmethod
    def assert_documents_are_equal(received, expected):
        # The service has no stable listing order; the contract is set equality.
        assert sorted(received, key=lambda d: d.id) == sorted(expected, key=lambda d: d.id)

    def test_write_documents(self, document_store):
        # DuplicatePolicy.NONE behaves as FAIL, like the in-memory store.
        doc = Document(content="test doc")
        assert document_store.write_documents([doc]) == 1
        assert document_store.count_documents() == 1
        with pytest.raises(Exception, match="already exists"):
            document_store.write_documents([doc])
        document_store.write_documents([doc], policy=DuplicatePolicy.SKIP)
