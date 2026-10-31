# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Source registry. Modules import their client library lazily, so a missing optional
dependency only matters for the source that needs it."""

from __future__ import annotations

import importlib
from typing import Dict, List, Tuple, Type

from ..model import ImporterError
from .base import Source

# name -> (module, class)
_REGISTRY: Dict[str, Tuple[str, str]] = {
    "jsonl": ("files", "JsonlSource"),
    "csv": ("files", "CsvSource"),
    "parquet": ("files", "ParquetSource"),
    "sqlite": ("files", "SqliteSource"),
    "duckdb": ("files", "DuckDbSource"),
    "numpy": ("files", "NumpySource"),
    "markdown": ("markdown", "MarkdownSource"),
    "chroma": ("chroma", "ChromaSource"),
    "sqlite-vec": ("sqlite_vec", "SqliteVecSource"),
    "lancedb": ("lancedb", "LanceDbSource"),
    "qdrant": ("qdrant", "QdrantSource"),
    "faiss": ("faiss", "FaissSource"),
    "hnswlib": ("hnswlib", "HnswlibSource"),
    "annoy": ("annoy", "AnnoySource"),
    "pinecone": ("pinecone", "PineconeSource"),
    "weaviate": ("weaviate", "WeaviateSource"),
    "milvus": ("milvus", "MilvusSource"),
    "pgvector": ("pgvector", "PgVectorSource"),
    "redis": ("redis", "RedisSource"),
    "elasticsearch": ("elastic", "ElasticsearchSource"),
    "opensearch": ("elastic", "OpenSearchSource"),
    "mongodb": ("mongodb", "MongoDbSource"),
    "vespa": ("vespa", "VespaSource"),
    "openai-vector-store": ("openai_vs", "OpenAiVectorStoreSource"),
    "langchain": ("langchain", "LangChainSource"),
    "llamaindex": ("llamaindex", "LlamaIndexSource"),
    "mem0": ("mem0", "Mem0Source"),
    "mem0-platform": ("mem0", "Mem0PlatformSource"),
    "zep": ("zep", "ZepSource"),
    "letta": ("letta", "LettaSource"),
}


def source_names() -> List[str]:
    return sorted(_REGISTRY)


def source_class(name: str) -> Type[Source]:
    try:
        module, cls = _REGISTRY[name]
    except KeyError:
        raise ImporterError(f"unknown source '{name}'; known: {', '.join(source_names())}") from None
    return getattr(importlib.import_module(f"{__name__}.{module}"), cls)


__all__ = ["Source", "source_class", "source_names"]
