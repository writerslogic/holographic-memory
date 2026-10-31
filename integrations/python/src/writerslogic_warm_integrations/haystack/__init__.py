# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

from haystack.core.serialization import allow_deserialization_module

from .document_store import WarmDocumentStore
from .retrievers import WarmEmbeddingRetriever, WarmTextRetriever

__all__ = ["WarmDocumentStore", "WarmEmbeddingRetriever", "WarmTextRetriever"]

# Haystack 3 refuses to deserialize components from modules it does not trust. Importing this
# package is the user's opt-in for these classes only.
allow_deserialization_module("writerslogic_warm_integrations.haystack")

HolographicDocumentStore = WarmDocumentStore
HolographicEmbeddingRetriever = WarmEmbeddingRetriever
HolographicTextRetriever = WarmTextRetriever
__all__ += ['HolographicDocumentStore', 'HolographicEmbeddingRetriever', 'HolographicTextRetriever']
