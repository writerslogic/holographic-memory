# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Bridge from LangChain `Embeddings` to the core `Embedder` protocol."""

from __future__ import annotations

from typing import List, Optional, Sequence

from langchain_core.embeddings import Embeddings


class EmbedderAsLangChain(Embeddings):
    """Expose a core `Embedder` (for example `HashingEmbedder`) as LangChain `Embeddings`."""

    def __init__(self, embedder):
        self.embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed_query(text)


class LangChainEmbedder:
    """Wrap a LangChain `Embeddings`. `dim` is probed once unless given (a probe costs one embed call)."""

    def __init__(self, embeddings: Embeddings, dim: Optional[int] = None):
        self.embeddings = embeddings
        self.dim = dim if dim is not None else len(embeddings.embed_query("dimension probe"))

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    async def aembed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return await self.embeddings.aembed_documents(list(texts))

    async def aembed_query(self, text: str) -> List[float]:
        return await self.embeddings.aembed_query(text)
