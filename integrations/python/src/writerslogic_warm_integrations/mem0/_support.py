# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Pure helpers shared by the sync and async mem0-compatible memories.

Shapes, validation messages and filter syntax follow mem0ai 2.2.1 (`mem0/memory/main.py`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import uuid
import warnings
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from writerslogic_warm_integrations.core import And, Cmp, Eq, Filter, WarmConfig, Not, Or, Record
from writerslogic_warm_integrations.core.filters import all_of
from writerslogic_warm_integrations.core.embedders import Embedder, HashingEmbedder

logger = logging.getLogger(__name__)

BUCKET_KEY = "_mem0_bucket"
ENTITY_PARAMS = ("user_id", "agent_id", "run_id")
IDENTITY_KEYS = frozenset(ENTITY_PARAMS) | {"actor_id"}
PROMOTED = ("user_id", "agent_id", "run_id", "actor_id", "role", "attributed_to", "expiration_date")
_CORE_KEYS = frozenset({"data", "hash", "created_at", "updated_at", "id", "text_lemmatized", BUCKET_KEY, *PROMOTED})
DEFAULT_BUCKETS = 32


# --------------------------------------------------------------------------- validation


def trim_entity_id(value: Optional[Any], name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    trimmed = value.strip()
    if trimmed == "":
        raise ValueError(f"Invalid {name}: cannot be empty or whitespace-only. Provide a valid identifier.")
    if any(c.isspace() for c in trimmed):
        raise ValueError(f"Invalid {name}: cannot contain whitespace. Provide a valid identifier without spaces.")
    return trimmed


def validate_search_params(threshold: Optional[float] = None, top_k: Optional[int] = None) -> None:
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a valid number")
        if threshold < 0 or threshold > 1:
            raise ValueError(f"Invalid threshold: {threshold}. Must be between 0 and 1 (inclusive).")
    if top_k is not None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be a valid integer")
        if top_k < 0:
            raise ValueError(f"Invalid top_k: {top_k}. Must be a non-negative integer.")


def trim_query(query: Any) -> str:
    if not isinstance(query, str):
        raise ValueError("Invalid query: must be a non-empty string.")
    trimmed = query.strip()
    if not trimmed:
        raise ValueError("Invalid query: cannot be empty or whitespace-only.")
    return trimmed


def reject_entity_kwargs(kwargs: Mapping[str, Any], method: str) -> None:
    bad = set(ENTITY_PARAMS) & set(kwargs)
    if bad:
        raise ValueError(
            f"Top-level entity parameters {bad} are not supported in {method}(). Use filters={{'user_id': '...'}} instead."
        )


def normalize_expiration(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError("expiration_date must be a valid date in YYYY-MM-DD format.") from exc
    raise ValueError("expiration_date must be a date string in YYYY-MM-DD format.")


def is_expired(metadata: Mapping[str, Any]) -> bool:
    exp = metadata.get("expiration_date")
    if not exp:
        return False
    try:
        return date.fromisoformat(str(exp)) < datetime.now(timezone.utc).date()
    except ValueError:
        return False


def effective_scope_filters(filters: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    eff = dict(filters) if filters else {}
    for key in ENTITY_PARAMS:
        if key in eff:
            eff[key] = trim_entity_id(eff[key], key)
    if not any(key in eff for key in ENTITY_PARAMS):
        raise ValueError(
            "filters must contain at least one of: user_id, agent_id, run_id. Example: filters={'user_id': 'u1'}"
        )
    return eff


# --------------------------------------------------------------------------- filters


@dataclass(frozen=True)
class _Icontains(Filter):
    key: str
    value: str

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        found = metadata.get(self.key)
        return isinstance(found, str) and self.value.lower() in found.lower()


@dataclass(frozen=True)
class NotExpired(Filter):
    def matches(self, metadata: Mapping[str, Any]) -> bool:
        return not is_expired(metadata)


def _condition(key: str, cond: Any) -> Filter:
    if not isinstance(cond, dict):
        if cond == "*":
            return Cmp(key, "exists", True)
        return Eq(key, cond)
    terms: List[Filter] = []
    for op, value in cond.items():
        if op == "eq":
            terms.append(Eq(key, value))
        elif op in ("ne", "gt", "gte", "lt", "lte", "in", "nin", "contains"):
            terms.append(Cmp(key, op, value))
        elif op == "icontains":
            terms.append(_Icontains(key, str(value)))
        else:
            raise ValueError(f"Unsupported metadata filter operator: {op}")
    return all_of(terms)


def _sub(spec: Mapping[str, Any]) -> Filter:
    return translate_filters(spec) or And(())


def translate_filters(filters: Mapping[str, Any]) -> Optional[Filter]:
    """mem0 v2 filter syntax (AND/OR/NOT lists, eq/ne/gt/gte/lt/lte/in/nin/contains/icontains, "*") to a core filter."""
    terms: List[Filter] = []
    for key, value in filters.items():
        if key == "AND":
            if not isinstance(value, list):
                raise ValueError("AND operator requires a list of conditions")
            terms.extend(_sub(s) for s in value)
        elif key == "OR":
            if not isinstance(value, list) or not value:
                raise ValueError("OR operator requires a non-empty list of conditions")
            terms.append(Or(tuple(_sub(s) for s in value)))
        elif key == "NOT":
            if not isinstance(value, list) or not value:
                raise ValueError("NOT operator requires a non-empty list of conditions")
            terms.extend(Not(_sub(s)) for s in value)
        else:
            terms.append(_condition(key, value))
    if not terms:
        return None
    return all_of(terms)


# --------------------------------------------------------------------------- items


def _md5(text: str) -> str:
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()


def bucket_of(memory_id: str, buckets: int) -> int:
    return zlib.crc32(memory_id.encode("utf-8")) % buckets


def to_item(record_id: str, text: str, meta: Mapping[str, Any], score: Optional[float] = None, *, with_score: bool = True) -> Dict[str, Any]:
    """Shape of mem0's `MemoryItem.model_dump()` plus promoted keys and extra metadata."""
    item: Dict[str, Any] = {
        "id": record_id,
        "memory": text,
        "hash": meta.get("hash"),
        "metadata": None,
        "score": score,
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
    }
    if not with_score:
        del item["score"]
    for key in PROMOTED:
        if key in meta:
            item[key] = meta[key]
    extra = {k: v for k, v in meta.items() if k not in _CORE_KEYS}
    if extra:
        item["metadata"] = extra
    return item


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_identity(metadata: Mapping[str, Any], existing: Mapping[str, Any], context: str) -> Dict[str, Any]:
    clean = {}
    for key, value in metadata.items():
        if key in IDENTITY_KEYS:
            if value != existing.get(key):
                logger.warning("%s: ignoring metadata['%s'] - identity fields cannot be set through metadata", context, key)
        else:
            clean[key] = value
    return clean


def make_record(text: str, base_meta: Mapping[str, Any], buckets: int, extra: Optional[Mapping[str, Any]] = None) -> Record:
    mid = str(uuid.uuid4())
    meta = dict(base_meta)
    if extra:
        meta.update(extra)
    meta["hash"] = _md5(text)
    meta.setdefault("created_at", now_iso())
    meta["updated_at"] = meta["created_at"]
    meta[BUCKET_KEY] = bucket_of(mid, buckets)
    return Record(mid, text, meta)


def entity_metadata(user_id: Optional[str], agent_id: Optional[str], run_id: Optional[str], metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    user_id, agent_id, run_id = (trim_entity_or_none(v, n) for v, n in ((user_id, "user_id"), (agent_id, "agent_id"), (run_id, "run_id")))
    base = strip_identity(metadata, {}, "add()") if metadata else {}
    for key, value in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
        if value:
            base[key] = value
    if not (user_id or agent_id or run_id):
        raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be provided.")
    return base


def trim_entity_or_none(value: Optional[Any], name: str) -> Optional[str]:
    return trim_entity_id(value, name)


def normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, dict):
        return [messages]
    if isinstance(messages, list):
        return messages
    raise ValueError("messages must be str, dict, or list[dict]")


def parse_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for m in messages:
        if isinstance(m.get("content"), str) and m.get("role"):
            lines.append(f"{m['role']}: {m['content']}")
    return "\n".join(lines)


def verbatim_records(messages: Sequence[Any], base_meta: Mapping[str, Any], buckets: int) -> List[Tuple[Record, Dict[str, Any]]]:
    out = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") is None or not isinstance(m.get("content"), str):
            logger.warning("Skipping invalid message format: %s", m)
            continue
        if m["role"] == "system":
            continue
        extra: Dict[str, Any] = {"role": m["role"]}
        name = m.get("name")
        if name:
            extra["actor_id"] = name
        rec = make_record(m["content"], base_meta, buckets, extra)
        out.append((rec, {"id": rec.id, "memory": rec.text, "event": "ADD", "actor_id": name if name else None, "role": m["role"]}))
    return out


# --------------------------------------------------------------------------- extraction

Fact = Union[str, Mapping[str, Any]]
# extractor(messages, existing_memory_texts, prompt) -> facts; may be a coroutine function for AsyncMemory.
Extractor = Callable[[List[Dict[str, Any]], List[str], Optional[str]], Any]

_EXTRACTION_PROMPT = """You maintain a long-term memory for an assistant. Read the new messages and list the durable facts, preferences, plans and details worth remembering about the speaker(s). Each fact is one short self-contained sentence. Skip chit-chat, questions without facts, and anything already present in the existing memories. If nothing is worth remembering return an empty list.

Reply with JSON only: {"memory": [{"text": "<fact>"}]}"""

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_facts(raw: Any) -> List[Fact]:
    """Parse an LLM reply of the form {"memory": [{"text": ...}]} (fenced or embedded JSON tolerated)."""
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if not raw or not str(raw).strip():
        return []
    text = _FENCE.sub("", str(raw))
    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            logger.error("extraction reply is not JSON")
            return []
        try:
            data = json.loads(text[start : end + 1], strict=False)
        except json.JSONDecodeError:
            logger.error("extraction reply is not JSON")
            return []
    facts = data.get("memory", []) if isinstance(data, dict) else []
    return facts if isinstance(facts, list) else []


class LLMExtractor:
    """mem0-style fact extraction through any chat function you supply.

    `llm(messages)` receives OpenAI-format chat messages and returns the reply text; it may be a
    coroutine function (required for `AsyncMemory` when it is async). Nothing is sent anywhere
    except to the function you pass.
    """

    def __init__(self, llm: Callable[[List[Dict[str, str]]], Any], instructions: Optional[str] = None):
        self.llm = llm
        self.instructions = instructions

    def _prompt(self, messages: List[Dict[str, Any]], existing: List[str], prompt: Optional[str]) -> List[Dict[str, str]]:
        system = _EXTRACTION_PROMPT
        extra = prompt or self.instructions
        if extra:
            system += f"\n\nAdditional instructions: {extra}"
        user = "Existing memories:\n" + ("\n".join(f"- {t}" for t in existing) or "(none)") + "\n\nNew messages:\n" + parse_messages(messages)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def __call__(self, messages: List[Dict[str, Any]], existing: List[str], prompt: Optional[str] = None) -> Any:
        reply = self.llm(self._prompt(messages, existing, prompt))
        if hasattr(reply, "__await__"):
            async def finish() -> List[Fact]:
                return parse_facts(await reply)
            return finish()
        return parse_facts(reply)


def facts_to_records(
    facts: Sequence[Fact], existing_hashes: set, base_meta: Mapping[str, Any], buckets: int
) -> List[Tuple[Record, Dict[str, Any]]]:
    out: List[Tuple[Record, Dict[str, Any]]] = []
    seen = set(existing_hashes)
    for fact in facts:
        text = fact.get("text") if isinstance(fact, Mapping) else fact
        if not isinstance(text, str) or not text.strip():
            continue
        h = _md5(text)
        if h in seen:
            continue
        seen.add(h)
        extra = {"attributed_to": fact["attributed_to"]} if isinstance(fact, Mapping) and fact.get("attributed_to") else None
        rec = make_record(text, base_meta, buckets, extra)
        out.append((rec, {"id": rec.id, "memory": text, "event": "ADD"}))
    return out


# --------------------------------------------------------------------------- history


class History:
    """mem0's `history` table (same columns) in SQLite. `":memory:"` keeps it for the process lifetime."""

    def __init__(self, path: str = ":memory:"):
        self._lock = threading.Lock()
        self._con = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._con:
            self._con.execute(
                """CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY, memory_id TEXT, old_memory TEXT, new_memory TEXT, event TEXT,
                    created_at DATETIME, updated_at DATETIME, is_deleted INTEGER, actor_id TEXT, role TEXT)"""
            )

    def add(
        self, memory_id: str, old: Optional[str], new: Optional[str], event: str, *,
        created_at: Optional[str] = None, updated_at: Optional[str] = None, is_deleted: int = 0,
        actor_id: Optional[str] = None, role: Optional[str] = None,
    ) -> None:
        with self._lock, self._con:
            self._con.execute(
                "INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, actor_id, role)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), memory_id, old, new, event, created_at or now_iso(), updated_at, is_deleted, actor_id, role),
            )

    def get(self, memory_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, actor_id, role"
                " FROM history WHERE memory_id = ? ORDER BY created_at ASC, DATETIME(updated_at) ASC",
                (memory_id,),
            ).fetchall()
        keys = ("id", "memory_id", "old_memory", "new_memory", "event", "created_at", "updated_at", "is_deleted", "actor_id", "role")
        out = [dict(zip(keys, r)) for r in rows]
        for row in out:
            row["is_deleted"] = bool(row["is_deleted"])
        return out

    def reset(self) -> None:
        with self._lock, self._con:
            self._con.execute("DELETE FROM history")

    def close(self) -> None:
        with self._lock:
            self._con.close()


# --------------------------------------------------------------------------- configuration

_VERBATIM_WARNING = (
    "writerslogic_warm_integrations.mem0: add(infer=True) has no extractor configured, so messages are stored "
    "VERBATIM (infer=False behaviour); mem0 would have asked an LLM to extract facts. Pass extractor=LLMExtractor(your_llm) "
    "(or Memory.from_config({'llm': your_llm})) to extract facts, or pass infer=False to silence this."
)


class Settings:
    """Options shared by `Memory` and `AsyncMemory`, resolved from constructor arguments or a mem0-style dict."""

    def __init__(
        self,
        config: Optional[Union[Mapping[str, Any], "Settings"]] = None,
        *,
        warm: Optional[WarmConfig] = None,
        extractor: Optional[Extractor] = None,
        reranker: Optional[Callable[[str, List[Dict[str, Any]], int], List[Dict[str, Any]]]] = None,
        history_db_path: Optional[str] = None,
        buckets: Optional[int] = None,
        strict_infer: bool = False,
        custom_instructions: Optional[str] = None,
    ):
        cfg = dict(config) if isinstance(config, Mapping) else {}
        warm_kwargs: Dict[str, Any] = {}
        embedder: Optional[Embedder] = None

        vs = cfg.get("vector_store")
        if vs:
            provider = vs.get("provider", "warm")
            if provider != "warm":
                raise ValueError(f"vector_store.provider {provider!r} is not supported here; use 'warm' (or omit vector_store)")
            c = dict(vs.get("config") or {})
            names = {"collection_name": "collection", "embedding_model_dims": "input_dim"}
            allowed = {"url", "api_key", "tenant_id", "timeout", "zero_trust_key", "mask_salt", "text_queries",
                       "max_fetch", "overfetch", "batch_size", "collection_name", "embedding_model_dims"}
            unknown = set(c) - allowed
            if unknown:
                raise ValueError(f"unsupported vector_store.config keys for warm: {sorted(unknown)}")
            warm_kwargs = {names.get(k, k): v for k, v in c.items()}
        emb = cfg.get("embedder")
        if emb:
            provider = emb.get("provider", "hashing")
            ec = dict(emb.get("config") or {})
            if provider == "hashing":
                embedder = HashingEmbedder(int(ec.get("embedding_dims", warm_kwargs.get("input_dim") or 384)))
            elif provider == "custom":
                embedder = ec["embedder"]
            else:
                raise ValueError(f"embedder.provider {provider!r} is not supported here; use 'hashing' or 'custom' with config={{'embedder': <Embedder>}}")
        llm = cfg.get("llm")
        if llm is not None and extractor is None:
            fn = llm if callable(llm) else (llm.get("config") or {}).get("fn") if isinstance(llm, Mapping) and llm.get("provider") == "callable" else None
            if fn is None:
                raise ValueError("llm must be a chat callable or {'provider': 'callable', 'config': {'fn': <callable>}}; hosted-LLM providers are not instantiated by this shim")
            extractor = LLMExtractor(fn, cfg.get("custom_instructions"))
        if cfg.get("reranker") is not None and reranker is None:
            if not callable(cfg["reranker"]):
                raise ValueError("reranker must be a callable (query, results, top_k) -> results")
            reranker = cfg["reranker"]

        if warm is None:
            if embedder is not None:
                warm_kwargs["embedder"] = embedder
            warm = WarmConfig(**warm_kwargs)
        self.warm = warm
        self.extractor = extractor
        self.reranker = reranker
        self.history_db_path = history_db_path or cfg.get("history_db_path") or ":memory:"
        self.buckets = buckets or DEFAULT_BUCKETS
        if self.buckets < 1:
            raise ValueError("buckets must be >= 1")
        self.strict_infer = strict_infer
        self.custom_instructions = custom_instructions or cfg.get("custom_instructions")
        self._warned = False

    def check_infer(self, infer: bool) -> bool:
        """Whether an extractor should run for this add(); warns once (or raises) when infer was requested without one."""
        if not infer:
            return False
        if self.extractor is not None:
            return True
        if self.strict_infer:
            raise ValueError("add(infer=True) needs an extractor; none is configured (strict_infer=True)")
        if not self._warned:
            self._warned = True
            warnings.warn(_VERBATIM_WARNING, UserWarning, stacklevel=4)
        return False


def reject_unsupported_add(timestamp: Any, memory_type: Optional[str], agent_id: Optional[str]) -> None:
    if timestamp is not None:
        raise ValueError("The 'timestamp' parameter is not supported in the open-source memory SDK.")
    if memory_type is not None and memory_type != "procedural_memory":
        raise ValueError("Invalid 'memory_type'. Please pass procedural_memory to create procedural memories.")
    if memory_type == "procedural_memory" and agent_id is not None:
        raise NotImplementedError("procedural memories need mem0's LLM summarizer and are not implemented; use a regular add()")
