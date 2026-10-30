# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import datetime as _dt
from .client import AsyncHolographicClient, HolographicClient
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

__all__ = ['FactStatus', 'HistoryMode', 'TimeLike', 'MemoryError', 'to_millis', 'to_datetime', 'Remembered', 'ChainLink', 'Explanation', 'Recalled', 'FactRecord', 'ForgetReport', 'ConsolidationReport', 'MemoryClient', 'AsyncMemoryClient']

FactStatus = Literal['current', 'superseded', 'expired', 'not-yet-valid']
HistoryMode = Literal['current', 'all', 'historical']
TimeLike = int | _dt.datetime

class MemoryError(Exception):
    code: str
    status: int
    def __init__(self, message: str, code: str = 'invalid_request', status: int = 422) -> None: ...

def to_millis(value: TimeLike | None) -> int | None: ...
def to_datetime(millis: int) -> _dt.datetime: ...

@dataclass(frozen=True)
class Remembered:
    id: str
    created: bool
    supersedes: list[str]
    superseded_by: list[str]
    ingest_time: int

@dataclass(frozen=True)
class ChainLink:
    id: str
    relation: Literal['predecessor', 'successor']
    effective_from: int
    ingest_time: int

@dataclass(frozen=True)
class Explanation:
    retrieved: bool
    rank: int | None
    lexical_score: float | None
    semantic_score: float | None
    retrieval_score: float | None
    lexical_weight: float
    semantic_weight: float
    fusion: str
    matched_terms: list[str]
    recency_factor: float
    historical_factor: float
    final_score: float | None
    status: FactStatus
    as_of: int
    known_at: int | None
    valid_from: int | None
    valid_to: int | None
    within_validity: bool
    chain: list[ChainLink]

@dataclass(frozen=True)
class Recalled:
    id: str
    text: str | None
    score: float
    status: FactStatus
    ingest_time: int | None
    event_time: int | None
    valid_from: int | None
    valid_to: int | None
    subject: str | None
    entities: list[str]
    supersedes: list[str]
    superseded_by: list[str]
    metadata: dict[str, Any]
    source_uri: str | None
    explanation: Explanation | None = ...

@dataclass(frozen=True)
class FactRecord:
    id: str
    text: str | None
    status: FactStatus
    ingest_time: int | None
    event_time: int | None
    valid_from: int | None
    valid_to: int | None
    effective_from: int
    subject: str | None
    entities: list[str]
    supersedes: list[str]
    superseded_by: list[str]
    metadata: dict[str, Any]

@dataclass(frozen=True)
class ForgetReport:
    forgotten: list[str]
    dry_run: bool
    purged: bool

@dataclass(frozen=True)
class ConsolidationReport:
    created: list[Remembered]
    forgotten: list[str]
    dry_run: bool

class MemoryClient:
    def __init__(self, client: HolographicClient) -> None: ...
    def capabilities(self, contract_version: int = 1) -> dict[str, Any]: ...
    @classmethod
    def connect(cls, url: str = 'http://localhost:8080', **options: Any) -> MemoryClient: ...
    def close(self) -> None: ...
    def __enter__(self) -> MemoryClient: ...
    def __exit__(self, *exc: Any) -> None: ...
    def remember(self, text: str, *, id: str | None = None, event_time: TimeLike | None = None, valid_from: TimeLike | None = None, valid_to: TimeLike | None = None, subject: str | None = None, supersedes: Sequence[str] | None = None, entities: Sequence[str] | None = None, metadata: Mapping[str, Any] | None = None, embedding: Sequence[float] | None = None, source_uri: str | None = None) -> Remembered: ...
    def remember_many(self, facts: Iterable[Mapping[str, Any]]) -> list[Remembered]: ...
    def recall(self, query: str, *, k: int | None = None, as_of: TimeLike | None = None, known_at: TimeLike | None = None, history: HistoryMode | None = None, subject: str | None = None, entities: Sequence[str] | None = None, filter: Mapping[str, Any] | None = None, embedding: Sequence[float] | None = None, lexical_weight: float | None = None, semantic_weight: float | None = None, min_semantic_score: float | None = None, fusion: Literal['blend', 'rrf'] | None = None, candidate_limit: int | None = None, explain: bool = False) -> list[Recalled]: ...
    def explain(self, query: str, fact_id: str, *, as_of: TimeLike | None = None, known_at: TimeLike | None = None, embedding: Sequence[float] | None = None, lexical_weight: float | None = None, semantic_weight: float | None = None, fusion: Literal['blend', 'rrf'] | None = None, candidate_limit: int | None = None) -> Explanation: ...
    def forget(self, *, ids: Iterable[str] | None = None, subject: str | None = None, entity: str | None = None, ingested_before: TimeLike | None = None, event_before: TimeLike | None = None, matching: Mapping[str, Any] | None = None, include_chain: bool = False, dry_run: bool = False, purge: bool = False) -> ForgetReport: ...
    def get(self, fact_id: str) -> FactRecord | None: ...
    def history(self, fact_id: str) -> list[FactRecord]: ...
    def apply_forgetting(self, *, include_chain: bool = False, dry_run: bool = True, purge: bool = False) -> ForgetReport: ...
    def consolidate(self, *, dry_run: bool = True, allow_destructive: bool = False) -> ConsolidationReport: ...

class AsyncMemoryClient:
    def __init__(self, client: AsyncHolographicClient) -> None: ...
    async def capabilities(self, contract_version: int = 1) -> dict[str, Any]: ...
    @classmethod
    def connect(cls, url: str = 'http://localhost:8080', **options: Any) -> AsyncMemoryClient: ...
    async def aclose(self) -> None: ...
    async def __aenter__(self) -> AsyncMemoryClient: ...
    async def __aexit__(self, *exc: Any) -> None: ...
    async def remember(self, text: str, *, id: str | None = None, event_time: TimeLike | None = None, valid_from: TimeLike | None = None, valid_to: TimeLike | None = None, subject: str | None = None, supersedes: Sequence[str] | None = None, entities: Sequence[str] | None = None, metadata: Mapping[str, Any] | None = None, embedding: Sequence[float] | None = None, source_uri: str | None = None) -> Remembered: ...
    async def remember_many(self, facts: Iterable[Mapping[str, Any]]) -> list[Remembered]: ...
    async def recall(self, query: str, *, k: int | None = None, as_of: TimeLike | None = None, known_at: TimeLike | None = None, history: HistoryMode | None = None, subject: str | None = None, entities: Sequence[str] | None = None, filter: Mapping[str, Any] | None = None, embedding: Sequence[float] | None = None, lexical_weight: float | None = None, semantic_weight: float | None = None, min_semantic_score: float | None = None, fusion: Literal['blend', 'rrf'] | None = None, candidate_limit: int | None = None, explain: bool = False) -> list[Recalled]: ...
    async def explain(self, query: str, fact_id: str, *, as_of: TimeLike | None = None, known_at: TimeLike | None = None, embedding: Sequence[float] | None = None, lexical_weight: float | None = None, semantic_weight: float | None = None, fusion: Literal['blend', 'rrf'] | None = None, candidate_limit: int | None = None) -> Explanation: ...
    async def forget(self, *, ids: Iterable[str] | None = None, subject: str | None = None, entity: str | None = None, ingested_before: TimeLike | None = None, event_before: TimeLike | None = None, matching: Mapping[str, Any] | None = None, include_chain: bool = False, dry_run: bool = False, purge: bool = False) -> ForgetReport: ...
    async def get(self, fact_id: str) -> FactRecord | None: ...
    async def history(self, fact_id: str) -> list[FactRecord]: ...
    async def apply_forgetting(self, *, include_chain: bool = False, dry_run: bool = True, purge: bool = False) -> ForgetReport: ...
    async def consolidate(self, *, dry_run: bool = True, allow_destructive: bool = False) -> ConsolidationReport: ...
