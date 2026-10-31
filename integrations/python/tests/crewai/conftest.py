# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def no_chroma_storage(monkeypatch):
    from chromadb.api.shared_system_client import SharedSystemClient
    from chromadb.config import System

    assert not SharedSystemClient._identifier_to_system
    attempts = []

    def reject(self, *args, **kwargs):
        attempts.append("System")
        raise AssertionError("WARM storage must not construct a Chroma system")

    monkeypatch.setattr(System, "__init__", reject)
    yield
    assert attempts == [], attempts
