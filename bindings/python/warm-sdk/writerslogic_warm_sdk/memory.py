# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Agent memory over a WARM service: remember, recall, forget, explain.

Facts carry optional validity intervals and an event time that is distinct from the time the
store ingested them. A newer fact can supersede an older one; the old one is kept as
queryable history. Every policy default is conservative: nothing is superseded, rescored,
consolidated or forgotten unless you ask.

Times are Unix milliseconds (``int``) or timezone-aware ``datetime`` values; results carry
milliseconds, and :func:`to_datetime` converts them.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Union

import httpx

from .client import AsyncWarmClient, WarmClient

__all__ = [
    "AsyncMemoryClient",
    "ChainLink",
    "ConsolidationReport",
    "Explanation",
    "FactRecord",
    "FactStatus",
    "ForgetReport",
    "HistoryMode",
    "MemoryClient",
    "MemoryError",
    "Recalled",
    "Remembered",
    "TimeLike",
    "to_datetime",
    "to_millis",
]

FactStatus = Literal["current", "superseded", "expired", "not-yet-valid"]
HistoryMode = Literal["current", "all", "historical"]
TimeLike = Union[int, _dt.datetime]
_API = "/api/v1/memory"


class MemoryError(Exception):  # noqa: A001 - the service's name for this error class
    """The service rejected a memory request. ``code`` and ``status`` come from its error body."""

    def __init__(self, message: str, code: str = "invalid_request", status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def to_millis(value: Optional[TimeLike]) -> Optional[int]:
    """Unix milliseconds for an ``int`` (returned unchanged) or a ``datetime`` (naive means UTC)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("a time must be an int of milliseconds or a datetime")
    if isinstance(value, int):
        return value
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        delta = value - _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
        return (delta.days * 86_400 + delta.seconds) * 1000 + delta.microseconds // 1000
    raise TypeError("a time must be an int of milliseconds or a datetime")


def to_datetime(millis: int) -> _dt.datetime:
    """The UTC ``datetime`` of a Unix-millisecond time."""
    return _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(milliseconds=millis)


@dataclass(frozen=True)
class Remembered:
    id: str
    created: bool
    """False when the identical fact was already stored (an idempotent retry)."""
    supersedes: List[str]
    superseded_by: List[str]
    ingest_time: int

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> "Remembered":
        return cls(d["id"], d["created"], list(d["supersedes"]), list(d["supersededBy"]), d["ingestTime"])


@dataclass(frozen=True)
class ChainLink:
    id: str
    relation: Literal["predecessor", "successor"]
    effective_from: int
    ingest_time: int

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> "ChainLink":
        return cls(d["id"], d["relation"], d["effectiveFrom"], d["ingestTime"])


@dataclass(frozen=True)
class Explanation:
    """Why a fact matched and how it was adjusted."""

    retrieved: bool
    rank: Optional[int]
    lexical_score: Optional[float]
    semantic_score: Optional[float]
    retrieval_score: Optional[float]
    lexical_weight: float
    semantic_weight: float
    fusion: str
    matched_terms: List[str]
    recency_factor: float
    historical_factor: float
    final_score: Optional[float]
    status: FactStatus
    as_of: int
    known_at: Optional[int]
    valid_from: Optional[int]
    valid_to: Optional[int]
    within_validity: bool
    chain: List[ChainLink]

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> "Explanation":
        return cls(
            retrieved=d["retrieved"],
            rank=d.get("rank"),
            lexical_score=d.get("lexicalScore"),
            semantic_score=d.get("semanticScore"),
            retrieval_score=d.get("retrievalScore"),
            lexical_weight=d["lexicalWeight"],
            semantic_weight=d["semanticWeight"],
            fusion=d["fusion"],
            matched_terms=list(d["matchedTerms"]),
            recency_factor=d["recencyFactor"],
            historical_factor=d["historicalFactor"],
            final_score=d.get("finalScore"),
            status=d["status"],
            as_of=d["asOf"],
            known_at=d.get("knownAt"),
            valid_from=d.get("validFrom"),
            valid_to=d.get("validTo"),
            within_validity=d["withinValidity"],
            chain=[ChainLink._from(link) for link in d["chain"]],
        )


@dataclass(frozen=True)
class Recalled:
    id: str
    text: Optional[str]
    score: float
    status: FactStatus
    ingest_time: Optional[int]
    event_time: Optional[int]
    valid_from: Optional[int]
    valid_to: Optional[int]
    subject: Optional[str]
    entities: List[str]
    supersedes: List[str]
    superseded_by: List[str]
    metadata: Dict[str, Any]
    source_uri: Optional[str]
    explanation: Optional[Explanation] = None

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> "Recalled":
        explanation = d.get("explanation")
        return cls(
            id=d["id"],
            text=d.get("text"),
            score=d["score"],
            status=d["status"],
            ingest_time=d.get("ingestTime"),
            event_time=d.get("eventTime"),
            valid_from=d.get("validFrom"),
            valid_to=d.get("validTo"),
            subject=d.get("subject"),
            entities=list(d["entities"]),
            supersedes=list(d["supersedes"]),
            superseded_by=list(d["supersededBy"]),
            metadata=dict(d["metadata"]),
            source_uri=d.get("sourceUri"),
            explanation=Explanation._from(explanation) if explanation else None,
        )


@dataclass(frozen=True)
class FactRecord:
    """A stored fact with its history links, as ``get`` and ``history`` return it."""

    id: str
    text: Optional[str]
    status: FactStatus
    ingest_time: Optional[int]
    event_time: Optional[int]
    valid_from: Optional[int]
    valid_to: Optional[int]
    effective_from: int
    subject: Optional[str]
    entities: List[str]
    supersedes: List[str]
    superseded_by: List[str]
    metadata: Dict[str, Any]

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> "FactRecord":
        return cls(
            id=d["id"],
            text=d.get("text"),
            status=d["status"],
            ingest_time=d.get("ingestTime"),
            event_time=d.get("eventTime"),
            valid_from=d.get("validFrom"),
            valid_to=d.get("validTo"),
            effective_from=d["effectiveFrom"],
            subject=d.get("subject"),
            entities=list(d["entities"]),
            supersedes=list(d["supersedes"]),
            superseded_by=list(d["supersededBy"]),
            metadata=dict(d["metadata"]),
        )


@dataclass(frozen=True)
class ForgetReport:
    forgotten: List[str]
    dry_run: bool
    purged: bool

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> "ForgetReport":
        return cls(list(d["forgotten"]), d["dryRun"], d["purged"])


@dataclass(frozen=True)
class ConsolidationReport:
    created: List[Remembered]
    forgotten: List[str]
    dry_run: bool

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> "ConsolidationReport":
        return cls([Remembered._from(r) for r in d["created"]], list(d["forgotten"]), d["dryRun"])


def _drop_none(values: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _fact_payload(
    text: str,
    *,
    id: Optional[str],
    event_time: Optional[TimeLike],
    valid_from: Optional[TimeLike],
    valid_to: Optional[TimeLike],
    subject: Optional[str],
    supersedes: Optional[Sequence[str]],
    entities: Optional[Sequence[str]],
    metadata: Optional[Mapping[str, Any]],
    embedding: Optional[Sequence[float]],
    source_uri: Optional[str],
) -> Dict[str, Any]:
    return _drop_none(
        {
            "id": id,
            "text": text,
            "eventTime": to_millis(event_time),
            "validFrom": to_millis(valid_from),
            "validTo": to_millis(valid_to),
            "subject": subject,
            "supersedes": list(supersedes) if supersedes is not None else None,
            "entities": list(entities) if entities is not None else None,
            "metadata": dict(metadata) if metadata is not None else None,
            "embedding": list(embedding) if embedding is not None else None,
            "sourceUri": source_uri,
        }
    )


def _recall_payload(
    query: str,
    *,
    k: Optional[int],
    as_of: Optional[TimeLike],
    known_at: Optional[TimeLike],
    history: Optional[HistoryMode],
    subject: Optional[str],
    entities: Optional[Sequence[str]],
    filter: Optional[Mapping[str, Any]],
    embedding: Optional[Sequence[float]],
    lexical_weight: Optional[float],
    semantic_weight: Optional[float],
    min_semantic_score: Optional[float],
    fusion: Optional[Literal["blend", "rrf"]],
    candidate_limit: Optional[int],
    explain: bool,
) -> Dict[str, Any]:
    return _drop_none(
        {
            "query": query,
            "k": k,
            "asOf": to_millis(as_of),
            "knownAt": to_millis(known_at),
            "history": history,
            "subject": subject,
            "entities": list(entities) if entities is not None else None,
            "filter": dict(filter) if filter is not None else None,
            "embedding": list(embedding) if embedding is not None else None,
            "lexicalWeight": lexical_weight,
            "semanticWeight": semantic_weight,
            "minSemanticScore": min_semantic_score,
            "fusion": fusion,
            "candidateLimit": candidate_limit,
            "explain": explain or None,
        }
    )


def _scope_payload(
    ids: Optional[Iterable[str]],
    subject: Optional[str],
    entity: Optional[str],
    ingested_before: Optional[TimeLike],
    event_before: Optional[TimeLike],
    matching: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    chosen = [
        scope
        for scope in (
            {"kind": "ids", "ids": list(ids)} if ids is not None else None,
            {"kind": "subject", "subject": subject} if subject is not None else None,
            {"kind": "entity", "entity": entity} if entity is not None else None,
            {"kind": "ingestedBefore", "time": to_millis(ingested_before)}
            if ingested_before is not None
            else None,
            {"kind": "eventBefore", "time": to_millis(event_before)} if event_before is not None else None,
            {"kind": "matching", "filter": dict(matching)} if matching is not None else None,
        )
        if scope is not None
    ]
    if len(chosen) != 1:
        raise ValueError(
            "forget takes exactly one scope: ids, subject, entity, ingested_before, event_before or matching"
        )
    return chosen[0]


def _raise_for(response: httpx.Response) -> Any:
    if response.is_success:
        return response.json()
    try:
        error = response.json()["error"]
        raise MemoryError(error["message"], error.get("code", "error"), response.status_code)
    except (ValueError, KeyError, TypeError):
        response.raise_for_status()
        raise


class MemoryClient:
    """Memory operations over a :class:`WarmClient`.

    >>> memory = MemoryClient(WarmClient("http://localhost:8080"))
    >>> memory.remember("Alice lives in Paris", subject="alice/city", valid_from=2020_01_01)
    """

    def __init__(self, client: WarmClient) -> None:
        self._client = client

    @classmethod
    def connect(cls, url: str = "http://localhost:8080", **options: Any) -> "MemoryClient":
        """A client over a new :class:`WarmClient`; ``options`` are its arguments."""
        return cls(WarmClient(url, **options))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MemoryClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _post(self, op: str, payload: Mapping[str, Any]) -> Any:
        response = self._client.client.post(f"{self._client.url}{_API}/{op}", json=payload)
        return _raise_for(response)

    def capabilities(self, contract_version: int = 1) -> Dict[str, Any]:
        """Negotiate the service's temporal Memory contract; unsupported versions fail."""
        return self._post("capabilities", {"contractVersion": contract_version})["capabilities"]

    def remember(
        self,
        text: str,
        *,
        id: Optional[str] = None,
        event_time: Optional[TimeLike] = None,
        valid_from: Optional[TimeLike] = None,
        valid_to: Optional[TimeLike] = None,
        subject: Optional[str] = None,
        supersedes: Optional[Sequence[str]] = None,
        entities: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        source_uri: Optional[str] = None,
    ) -> Remembered:
        """Store one fact. Retrying the same call is safe: it reports ``created=False``."""
        payload = _fact_payload(
            text,
            id=id,
            event_time=event_time,
            valid_from=valid_from,
            valid_to=valid_to,
            subject=subject,
            supersedes=supersedes,
            entities=entities,
            metadata=metadata,
            embedding=self._client.encrypt_vector(list(embedding)) if embedding is not None else None,
            source_uri=source_uri,
        )
        return Remembered._from(self._post("remember", payload)["remembered"][0])

    def remember_many(self, facts: Iterable[Mapping[str, Any]]) -> List[Remembered]:
        """Store facts atomically. Each mapping takes ``remember``'s keyword names plus ``text``."""
        payload = {"facts": [self._wire_fact(fact) for fact in facts]}
        return [Remembered._from(r) for r in self._post("remember", payload)["remembered"]]

    def _wire_fact(self, fact: Mapping[str, Any]) -> Dict[str, Any]:
        embedding = fact.get("embedding")
        return _fact_payload(
            fact["text"],
            id=fact.get("id"),
            event_time=fact.get("event_time"),
            valid_from=fact.get("valid_from"),
            valid_to=fact.get("valid_to"),
            subject=fact.get("subject"),
            supersedes=fact.get("supersedes"),
            entities=fact.get("entities"),
            metadata=fact.get("metadata"),
            embedding=self._client.encrypt_vector(list(embedding)) if embedding is not None else None,
            source_uri=fact.get("source_uri"),
        )

    def recall(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        as_of: Optional[TimeLike] = None,
        known_at: Optional[TimeLike] = None,
        history: Optional[HistoryMode] = None,
        subject: Optional[str] = None,
        entities: Optional[Sequence[str]] = None,
        filter: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        lexical_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        min_semantic_score: Optional[float] = None,
        fusion: Optional[Literal["blend", "rrf"]] = None,
        candidate_limit: Optional[int] = None,
        explain: bool = False,
    ) -> List[Recalled]:
        """Facts relevant to ``query``, best first. Only facts current now unless ``history`` says otherwise."""
        payload = _recall_payload(
            query,
            k=k,
            as_of=as_of,
            known_at=known_at,
            history=history,
            subject=subject,
            entities=entities,
            filter=filter,
            embedding=self._client.encrypt_vector(list(embedding)) if embedding is not None else None,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            min_semantic_score=min_semantic_score,
            fusion=fusion,
            candidate_limit=candidate_limit,
            explain=explain,
        )
        return [Recalled._from(r) for r in self._post("recall", payload)["results"]]

    def explain(
        self,
        query: str,
        fact_id: str,
        *,
        as_of: Optional[TimeLike] = None,
        known_at: Optional[TimeLike] = None,
        embedding: Optional[Sequence[float]] = None,
        lexical_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        fusion: Optional[Literal["blend", "rrf"]] = None,
        candidate_limit: Optional[int] = None,
    ) -> Explanation:
        """Why ``fact_id`` does or does not come back for ``query``."""
        payload = _recall_payload(
            query,
            k=None,
            as_of=as_of,
            known_at=known_at,
            history=None,
            subject=None,
            entities=None,
            filter=None,
            embedding=self._client.encrypt_vector(list(embedding)) if embedding is not None else None,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            min_semantic_score=None,
            fusion=fusion,
            candidate_limit=candidate_limit,
            explain=False,
        )
        payload["id"] = fact_id
        return Explanation._from(self._post("explain", payload)["explanation"])

    def forget(
        self,
        *,
        ids: Optional[Iterable[str]] = None,
        subject: Optional[str] = None,
        entity: Optional[str] = None,
        ingested_before: Optional[TimeLike] = None,
        event_before: Optional[TimeLike] = None,
        matching: Optional[Mapping[str, Any]] = None,
        include_chain: bool = False,
        dry_run: bool = False,
        purge: bool = False,
    ) -> ForgetReport:
        """Remove facts by exactly one scope. Repeating a call removes nothing more.

        ``include_chain`` also removes every version linked by supersession; ``purge`` compacts
        the store afterwards so deleted text leaves the journal; ``dry_run`` only reports.
        """
        payload: Dict[str, Any] = {
            "scope": _scope_payload(ids, subject, entity, ingested_before, event_before, matching),
            "includeChain": include_chain,
            "dryRun": dry_run,
            "purge": purge,
        }
        return ForgetReport._from(self._post("forget", payload))

    def get(self, fact_id: str) -> Optional[FactRecord]:
        record = self._post("get", {"id": fact_id})["fact"]
        return FactRecord._from(record) if record else None

    def history(self, fact_id: str) -> List[FactRecord]:
        """Every version of the fact's supersession chain, in effect order."""
        return [FactRecord._from(r) for r in self._post("history", {"id": fact_id})["history"]]

    def apply_forgetting(self, *, include_chain: bool = False, dry_run: bool = True, purge: bool = False) -> ForgetReport:
        """Run the service's configured forgetting policy. Defaults to a dry run; the default policy selects nothing."""
        payload = {"includeChain": include_chain, "dryRun": dry_run, "purge": purge}
        return ForgetReport._from(self._post("applyForgetting", payload))

    def consolidate(self, *, dry_run: bool = True, allow_destructive: bool = False) -> ConsolidationReport:
        """Run the service's configured consolidation policy. Defaults to a dry run; the default policy plans nothing."""
        payload = {"dryRun": dry_run, "allowDestructive": allow_destructive}
        return ConsolidationReport._from(self._post("consolidate", payload))


class AsyncMemoryClient:
    """Async counterpart of :class:`MemoryClient` over an :class:`AsyncWarmClient`."""

    def __init__(self, client: AsyncWarmClient) -> None:
        self._client = client

    @classmethod
    def connect(cls, url: str = "http://localhost:8080", **options: Any) -> "AsyncMemoryClient":
        return cls(AsyncWarmClient(url, **options))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncMemoryClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _post(self, op: str, payload: Mapping[str, Any]) -> Any:
        response = await self._client.client.post(f"{self._client.url}{_API}/{op}", json=payload)
        return _raise_for(response)

    async def capabilities(self, contract_version: int = 1) -> Dict[str, Any]:
        """Negotiate the service's temporal Memory contract; unsupported versions fail."""
        return (await self._post("capabilities", {"contractVersion": contract_version}))["capabilities"]

    def _mask(self, embedding: Optional[Sequence[float]]) -> Optional[List[float]]:
        return self._client.encrypt_vector(list(embedding)) if embedding is not None else None

    async def remember(
        self,
        text: str,
        *,
        id: Optional[str] = None,
        event_time: Optional[TimeLike] = None,
        valid_from: Optional[TimeLike] = None,
        valid_to: Optional[TimeLike] = None,
        subject: Optional[str] = None,
        supersedes: Optional[Sequence[str]] = None,
        entities: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        source_uri: Optional[str] = None,
    ) -> Remembered:
        payload = _fact_payload(
            text,
            id=id,
            event_time=event_time,
            valid_from=valid_from,
            valid_to=valid_to,
            subject=subject,
            supersedes=supersedes,
            entities=entities,
            metadata=metadata,
            embedding=self._mask(embedding),
            source_uri=source_uri,
        )
        return Remembered._from((await self._post("remember", payload))["remembered"][0])

    async def remember_many(self, facts: Iterable[Mapping[str, Any]]) -> List[Remembered]:
        wire = [
            _fact_payload(
                fact["text"],
                id=fact.get("id"),
                event_time=fact.get("event_time"),
                valid_from=fact.get("valid_from"),
                valid_to=fact.get("valid_to"),
                subject=fact.get("subject"),
                supersedes=fact.get("supersedes"),
                entities=fact.get("entities"),
                metadata=fact.get("metadata"),
                embedding=self._mask(fact.get("embedding")),
                source_uri=fact.get("source_uri"),
            )
            for fact in facts
        ]
        body = await self._post("remember", {"facts": wire})
        return [Remembered._from(r) for r in body["remembered"]]

    async def recall(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        as_of: Optional[TimeLike] = None,
        known_at: Optional[TimeLike] = None,
        history: Optional[HistoryMode] = None,
        subject: Optional[str] = None,
        entities: Optional[Sequence[str]] = None,
        filter: Optional[Mapping[str, Any]] = None,
        embedding: Optional[Sequence[float]] = None,
        lexical_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        min_semantic_score: Optional[float] = None,
        fusion: Optional[Literal["blend", "rrf"]] = None,
        candidate_limit: Optional[int] = None,
        explain: bool = False,
    ) -> List[Recalled]:
        payload = _recall_payload(
            query,
            k=k,
            as_of=as_of,
            known_at=known_at,
            history=history,
            subject=subject,
            entities=entities,
            filter=filter,
            embedding=self._mask(embedding),
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            min_semantic_score=min_semantic_score,
            fusion=fusion,
            candidate_limit=candidate_limit,
            explain=explain,
        )
        return [Recalled._from(r) for r in (await self._post("recall", payload))["results"]]

    async def explain(
        self,
        query: str,
        fact_id: str,
        *,
        as_of: Optional[TimeLike] = None,
        known_at: Optional[TimeLike] = None,
        embedding: Optional[Sequence[float]] = None,
        lexical_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        fusion: Optional[Literal["blend", "rrf"]] = None,
        candidate_limit: Optional[int] = None,
    ) -> Explanation:
        payload = _recall_payload(
            query,
            k=None,
            as_of=as_of,
            known_at=known_at,
            history=None,
            subject=None,
            entities=None,
            filter=None,
            embedding=self._mask(embedding),
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            min_semantic_score=None,
            fusion=fusion,
            candidate_limit=candidate_limit,
            explain=False,
        )
        payload["id"] = fact_id
        return Explanation._from((await self._post("explain", payload))["explanation"])

    async def forget(
        self,
        *,
        ids: Optional[Iterable[str]] = None,
        subject: Optional[str] = None,
        entity: Optional[str] = None,
        ingested_before: Optional[TimeLike] = None,
        event_before: Optional[TimeLike] = None,
        matching: Optional[Mapping[str, Any]] = None,
        include_chain: bool = False,
        dry_run: bool = False,
        purge: bool = False,
    ) -> ForgetReport:
        payload: Dict[str, Any] = {
            "scope": _scope_payload(ids, subject, entity, ingested_before, event_before, matching),
            "includeChain": include_chain,
            "dryRun": dry_run,
            "purge": purge,
        }
        return ForgetReport._from(await self._post("forget", payload))

    async def get(self, fact_id: str) -> Optional[FactRecord]:
        record = (await self._post("get", {"id": fact_id}))["fact"]
        return FactRecord._from(record) if record else None

    async def history(self, fact_id: str) -> List[FactRecord]:
        body = await self._post("history", {"id": fact_id})
        return [FactRecord._from(r) for r in body["history"]]

    async def apply_forgetting(
        self, *, include_chain: bool = False, dry_run: bool = True, purge: bool = False
    ) -> ForgetReport:
        payload = {"includeChain": include_chain, "dryRun": dry_run, "purge": purge}
        return ForgetReport._from(await self._post("applyForgetting", payload))

    async def consolidate(self, *, dry_run: bool = True, allow_destructive: bool = False) -> ConsolidationReport:
        payload = {"dryRun": dry_run, "allowDestructive": allow_destructive}
        return ConsolidationReport._from(await self._post("consolidate", payload))
