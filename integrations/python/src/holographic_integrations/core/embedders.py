# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Embedders: the `encoder` knob.

The service stores vectors you supply (or, with on-device model stages, embeds text itself).
Adapters take whatever embedding object their framework already uses and wrap it in the
`Embedder` protocol; frameworks without one (chat history, mem0, sessions) take `embedder=`
directly. `HashingEmbedder` is a dependency-free default for offline use and tests. It is
lexical (hashed word unigrams and bigrams), not semantic.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Awaitable, Callable, List, Optional, Protocol, Sequence, Union, runtime_checkable

_TOKEN = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """`dim` is the vector length the service was started with (`--input-dim`)."""

    dim: int

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


async def aembed_documents(embedder: Embedder, texts: Sequence[str]) -> List[List[float]]:
    native = getattr(embedder, "aembed_documents", None)
    if native is not None:
        return await native(texts)
    return await asyncio.to_thread(embedder.embed_documents, texts)


async def aembed_query(embedder: Embedder, text: str) -> List[float]:
    native = getattr(embedder, "aembed_query", None)
    if native is not None:
        return await native(text)
    return await asyncio.to_thread(embedder.embed_query, text)


class HashingEmbedder:
    """Signed feature hashing of lowercase word unigrams and bigrams, L2-normalized."""

    def __init__(self, dim: int = 384, bigrams: bool = True):
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim
        self.bigrams = bigrams

    def _features(self, text: str) -> List[str]:
        words = _TOKEN.findall(text.lower())
        feats = list(words)
        if self.bigrams:
            feats.extend(f"{a} {b}" for a, b in zip(words, words[1:]))
        return feats

    def embed_query(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0  # the service rejects all-zero vectors; empty text maps to a fixed axis
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


class FunctionEmbedder:
    """Wrap plain callables (for example a sentence-transformers `encode`) as an `Embedder`.

    `documents` maps a list of texts to a list of vectors; `query` defaults to `documents`
    applied to one text. Either may be a coroutine function for the async path.
    """

    def __init__(
        self,
        documents: Callable[[Sequence[str]], Union[Sequence[Sequence[float]], Awaitable[Sequence[Sequence[float]]]]],
        dim: int,
        query: Optional[Callable[[str], Union[Sequence[float], Awaitable[Sequence[float]]]]] = None,
    ):
        self.dim = dim
        self._documents = documents
        self._query = query

    @staticmethod
    def _run(value):
        if asyncio.iscoroutine(value):
            return asyncio.run(value)
        return value

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [list(map(float, row)) for row in self._run(self._documents(texts))]

    def embed_query(self, text: str) -> List[float]:
        if self._query is not None:
            return list(map(float, self._run(self._query(text))))
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        rows = self._documents(texts)
        if asyncio.iscoroutine(rows):
            rows = await rows
        return [list(map(float, row)) for row in rows]

    async def aembed_query(self, text: str) -> List[float]:
        if self._query is None:
            return (await self.aembed_documents([text]))[0]
        row = self._query(text)
        if asyncio.iscoroutine(row):
            row = await row
        return list(map(float, row))
