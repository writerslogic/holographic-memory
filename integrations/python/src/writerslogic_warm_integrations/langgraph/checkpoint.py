# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""LangGraph checkpoint saver over WARM.

Validated against `langgraph-checkpoint-conformance` (all eight capabilities).

Layout, per (thread, checkpoint namespace):

- one `head` record: the latest checkpoint id and the next sequence number;
- one record per checkpoint (metadata fields, parent id, sequence, page) whose payload is the
  serialized checkpoint without channel values;
- one record per channel blob `(channel, version)`, shared by every checkpoint that uses that
  version, as in `InMemorySaver`;
- one record per `(checkpoint, task)` holding that task's pending writes.

Payloads are base64 in the record text and are split over 40,000-character part records, so
a value is bounded by the service's request limits, not its 64 KiB text limit. The service has
no list or range query, so checkpoints are grouped into pages of 50 by sequence number and read
newest page first. A listing is therefore exact for any thread length, but costs one query
per page and per channel blob; a checkpoint with more than 100 tasks that wrote, or a thread
written by two concurrent writers per namespace, is outside what this layout guarantees.

Every record carries a constant vector: nothing here is semantically searchable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
from typing import Any, AsyncIterator, Dict, Generator, Iterator, List, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from ..core import AsyncWarmBackend, WarmBackend, WarmConfig, Record
from ..core.filters import And, Eq

_PAGE = 50
_PART = 40_000
_WRITE_GROUP = 100  # records per request: keeps a request of parts under the service's 8 MiB text budget

Call = Tuple[str, tuple]
Program = Generator[Call, Any, Any]


def _h(*parts: Any) -> str:
    return "lg" + hashlib.sha256(json.dumps(parts).encode()).hexdigest()


def _pack(typed: Tuple[str, bytes]) -> str:
    return f"{typed[0]}:{base64.b64encode(typed[1]).decode()}"


def _unpack(text: str) -> Tuple[str, bytes]:
    kind, _, data = text.partition(":")
    return kind, base64.b64decode(data)


def _split(text: str) -> List[str]:
    return [text[i : i + _PART] for i in range(0, len(text), _PART)] or [""]


def _eq(**kw: Any) -> And:
    return And(tuple(Eq(k, v) for k, v in kw.items()))


class WarmSaver(BaseCheckpointSaver[str]):
    """Checkpoint saver for LangGraph graphs.

    Pass `config` (an `WarmConfig`) or keyword options (`url`, `api_key`, `tenant_id`, ...). The
    default collection is `langgraph-checkpoints`. `serde` is LangGraph's serializer protocol.
    """

    def __init__(self, *, config: Optional[WarmConfig] = None, serde: Optional[SerializerProtocol] = None, **warm_kwargs: Any) -> None:
        super().__init__(serde=serde)
        if config is not None and warm_kwargs:
            raise ValueError("pass either config or keyword options, not both")
        if config is None:
            warm_kwargs.setdefault("collection", "langgraph-checkpoints")
            config = WarmConfig(**warm_kwargs)
        if config.max_fetch < _PAGE:
            raise ValueError(f"max_fetch must be at least {_PAGE}")
        self.config = config
        self._const = [1.0] + [0.0] * (config.input_dim - 1)
        self._sync: Optional[WarmBackend] = None
        self._async: Optional[AsyncWarmBackend] = None

    @property
    def backend(self) -> WarmBackend:
        if self._sync is None:
            self._sync = WarmBackend(self.config)
        return self._sync

    @property
    def abackend(self) -> AsyncWarmBackend:
        if self._async is None:
            self._async = AsyncWarmBackend(self.config)
        return self._async

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        if self._async is not None:
            await self._async.aclose()
            self._async = None

    # ---- drivers: the programs below are written once and run on either backend ---------

    def _run(self, program: Program) -> Any:
        backend = self.backend
        try:
            call = next(program)
            while True:
                try:
                    result = getattr(backend, call[0])(*call[1])
                except Exception as exc:  # delivered to the program so its own handlers apply
                    call = program.throw(exc)
                else:
                    call = program.send(result)
        except StopIteration as done:
            return done.value

    async def _arun(self, program: Program) -> Any:
        backend = self.abackend
        try:
            call = next(program)
            while True:
                try:
                    result = await getattr(backend, call[0])(*call[1])
                except Exception as exc:
                    call = program.throw(exc)
                else:
                    call = program.send(result)
        except StopIteration as done:
            return done.value

    # ---- record helpers (programs) ------------------------------------------------------

    def _records(self, rid: str, text: str, meta: Dict[str, Any]) -> List[Record]:
        chunks = _split(text)
        head = Record(rid, chunks[0], {**meta, "cp_nparts": len(chunks)}, self._const)
        parts = [
            Record(f"{rid}#{i}", chunk, {"kind": "part", "cp_thread": meta["cp_thread"], "cp_idx": i}, self._const)
            for i, chunk in enumerate(chunks)
            if i > 0
        ]
        return [head, *parts]

    def _write(self, records: List[Record]) -> Program:
        for i in range(0, len(records), _WRITE_GROUP):
            yield ("remember", (records[i : i + _WRITE_GROUP],))

    def _payloads(self, recs: List[Optional[Record]]) -> Program:
        """Join each record's text with its part records."""
        want: List[str] = []
        for r in recs:
            if r is not None:
                want.extend(f"{r.id}#{i}" for i in range(1, r.metadata["cp_nparts"]))
        parts = {}
        if want:
            for rec in (yield ("get", (want,))):
                if rec is not None:
                    parts[rec.id] = rec.text
        out: List[Optional[str]] = []
        for r in recs:
            if r is None:
                out.append(None)
                continue
            rid = self._rid(r)
            pieces = [r.text, *(parts.get(f"{rid}#{i}") for i in range(1, r.metadata["cp_nparts"]))]
            if any(p is None for p in pieces):
                raise RuntimeError(f"payload of {rid} is missing parts")
            out.append("".join(pieces))
        return out

    @staticmethod
    def _rid(record: Record) -> str:
        return record.id

    def _delete_ids(self, rec_ids: List[str], nparts: Dict[str, int]) -> Program:
        ids = list(rec_ids)
        for rid in rec_ids:
            ids.extend(f"{rid}#{i}" for i in range(1, nparts.get(rid, 1)))
        removed = yield ("forget", (ids,))
        return removed

    def _head_record(self, thread: str, ns: str, latest: str, next_seq: int) -> Record:
        meta = {"kind": "head", "cp_thread": thread, "cp_ns": ns, "cp_latest": latest, "cp_next": next_seq, "cp_nparts": 1}
        return Record(_h("head", thread, ns), "head", meta, self._const)

    # ---- reading ------------------------------------------------------------------------

    def _head(self, thread: str, ns: str) -> Program:
        rec = (yield ("get", ([_h("head", thread, ns)],)))[0]
        return rec

    def _pending(self, thread: str, ns: str, cp_id: str) -> Program:
        page = yield ("list", (_eq(kind="writes", cp_thread=thread, cp_ns=ns, cp_id=cp_id),))
        texts = yield from self._payloads(page.records)
        rows: List[Tuple[Tuple[str, str, int], Tuple[str, str, Any]]] = []
        for text in texts:
            doc = json.loads(text)
            for w in doc["writes"]:
                rows.append((writes_sort_key(doc["path"], doc["task"], w["idx"]), (doc["task"], w["channel"], self.serde.loads_typed((w["type"], base64.b64decode(w["data"]))))))
        rows.sort(key=lambda r: r[0])
        return [row for _, row in rows]

    def _tuple(self, record: Record) -> Program:
        thread, ns, cp_id = record.metadata["cp_thread"], record.metadata["cp_ns"], record.metadata["cp_id"]
        (text,) = yield from self._payloads([record])
        doc = self.serde.loads_typed(_unpack(text))
        checkpoint: Checkpoint = doc["checkpoint"]
        values: Dict[str, Any] = {}
        versions = checkpoint["channel_versions"]
        blob_ids = [_h("blob", thread, ns, ch, ver) for ch, ver in versions.items()]
        if blob_ids:
            blobs = yield ("get", (blob_ids,))
            texts = yield from self._payloads(blobs)
            for ch, text_ in zip(versions, texts):
                if text_ is None:
                    continue
                typed = _unpack(text_)
                if typed[0] != "empty":
                    values[ch] = self.serde.loads_typed(typed)
        writes = yield from self._pending(thread, ns, cp_id)
        parent = record.metadata.get("cp_parent") or None
        return CheckpointTuple(
            config={"configurable": {"thread_id": thread, "checkpoint_ns": ns, "checkpoint_id": cp_id}},
            checkpoint={**checkpoint, "channel_values": values},
            metadata=doc["metadata"],
            parent_config={"configurable": {"thread_id": thread, "checkpoint_ns": ns, "checkpoint_id": parent}} if parent else None,
            pending_writes=writes,
        )

    def _get_tuple(self, config: RunnableConfig) -> Program:
        c = config["configurable"]
        thread, ns = c["thread_id"], c.get("checkpoint_ns", "")
        cp_id = get_checkpoint_id(config)
        if not cp_id:
            head = yield from self._head(thread, ns)
            if head is None:
                return None
            cp_id = head.metadata["cp_latest"]
        rec = (yield ("get", ([_h("cp", thread, ns, cp_id)],)))[0]
        if rec is None:
            return None
        return (yield from self._tuple(rec))

    def _namespaces(self, thread: str, ns: Optional[str]) -> Program:
        """(thread, ns, max sequence) heads for a thread (or every thread), optionally one namespace."""
        flt: Dict[str, Any] = {"kind": "head"}
        if thread is not None:
            flt["cp_thread"] = thread
        if ns is not None:
            flt["cp_ns"] = ns
        page = yield ("list", (_eq(**flt),))
        return [(r.metadata["cp_thread"], r.metadata["cp_ns"], r.metadata["cp_next"] - 1) for r in page.records]

    def _list(self, config: Optional[RunnableConfig], filter: Optional[Dict[str, Any]], before: Optional[RunnableConfig], limit: Optional[int]) -> Program:
        thread = config["configurable"]["thread_id"] if config else None
        ns = config["configurable"].get("checkpoint_ns") if config else None
        only_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None
        pushed = {f"m_{k}": v for k, v in (filter or {}).items() if isinstance(v, (str, bool, int)) or v is None}
        results: List[CheckpointTuple] = []
        for t, n, top in sorted((yield from self._namespaces(thread, ns))):
            for page_no in range(top // _PAGE, -1, -1):
                page = yield ("list", (_eq(kind="cp", cp_thread=t, cp_ns=n, cp_page=page_no, **pushed),))
                if len(page.records) >= self.config.max_fetch:
                    raise RuntimeError(f"checkpoint page {page_no} of thread {t!r} holds more than max_fetch records")
                for rec in sorted(page.records, key=lambda r: r.metadata["cp_id"], reverse=True):
                    cid = rec.metadata["cp_id"]
                    if only_id and cid != only_id:
                        continue
                    if before_id and cid >= before_id:
                        continue
                    if limit is not None and len(results) >= limit:
                        return results
                    tup = yield from self._tuple(rec)
                    if filter and not all(v == tup.metadata.get(k) for k, v in filter.items()):
                        continue
                    results.append(tup)
        return results

    # ---- writing ------------------------------------------------------------------------

    def _put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> Program:
        c = config["configurable"]
        thread, ns = c["thread_id"], c["checkpoint_ns"]
        cp = checkpoint.copy()
        values: Dict[str, Any] = cp.pop("channel_values")  # type: ignore[misc]
        cp_id = checkpoint["id"]
        head = yield from self._head(thread, ns)
        existing = (yield ("get", ([_h("cp", thread, ns, cp_id)],)))[0]
        next_seq = head.metadata["cp_next"] if head else 0
        seq = existing.metadata["cp_seq"] if existing else next_seq
        records: List[Record] = []
        for ch, ver in new_versions.items():
            typed = self.serde.dumps_typed(values[ch]) if ch in values else ("empty", b"")
            rid = _h("blob", thread, ns, ch, ver)
            records += self._records(rid, _pack(typed), {"kind": "blob", "cp_thread": thread, "cp_ns": ns})
        full_meta = get_checkpoint_metadata(config, metadata)
        payload = _pack(self.serde.dumps_typed({"checkpoint": cp, "metadata": full_meta}))
        rid = _h("cp", thread, ns, cp_id)
        meta: Dict[str, Any] = {
            "kind": "cp", "cp_thread": thread, "cp_ns": ns, "cp_id": cp_id, "cp_parent": c.get("checkpoint_id") or "",
            "cp_seq": seq, "cp_page": seq // _PAGE,
        }
        for k, v in full_meta.items():
            if isinstance(v, (str, bool, int)) or v is None:
                meta[f"m_{k}"] = v
        records += self._records(rid, payload, meta)
        yield from self._write(records)
        if not existing:
            latest = max(head.metadata["cp_latest"], cp_id) if head else cp_id
            yield from self._write([self._head_record(thread, ns, latest, seq + 1)])
        return {"configurable": {"thread_id": thread, "checkpoint_ns": ns, "checkpoint_id": cp_id}}

    def _put_writes(self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str) -> Program:
        c = config["configurable"]
        thread, ns, cp_id = c["thread_id"], c.get("checkpoint_ns", ""), c["checkpoint_id"]
        rid = _h("writes", thread, ns, cp_id, task_id)
        rec = (yield ("get", ([rid],)))[0]
        old_parts = rec.metadata["cp_nparts"] if rec else 1
        entries: Dict[int, Dict[str, Any]] = {}
        if rec:
            (text,) = yield from self._payloads([rec])
            entries = {w["idx"]: w for w in json.loads(text)["writes"]}
        for i, (channel, value) in enumerate(writes):
            idx = WRITES_IDX_MAP.get(channel, i)
            if idx >= 0 and idx in entries:
                continue
            typed = self.serde.dumps_typed(value)
            entries[idx] = {"idx": idx, "channel": channel, "type": typed[0], "data": base64.b64encode(typed[1]).decode()}
        text = json.dumps({"task": task_id, "path": task_path, "writes": list(entries.values())})
        records = self._records(rid, text, {"kind": "writes", "cp_thread": thread, "cp_ns": ns, "cp_id": cp_id, "cp_task": task_id})
        yield from self._write(records)
        stale = [f"{rid}#{i}" for i in range(len(records), old_parts)]
        if stale:
            yield ("forget", (stale,))

    def _delete_checkpoint_records(self, recs: List[Record]) -> Program:
        """Delete checkpoints with their parts and writes (blobs are shared and left in place)."""
        for rec in recs:
            m = rec.metadata
            yield from self._delete_ids([self._rid(rec)], {self._rid(rec): m["cp_nparts"]})
            page = yield ("list", (_eq(kind="writes", cp_thread=m["cp_thread"], cp_ns=m["cp_ns"], cp_id=m["cp_id"]),))
            yield from self._delete_ids([self._rid(w) for w in page.records], {self._rid(w): w.metadata["cp_nparts"] for w in page.records})

    def _repoint_head(self, thread: str, ns: str) -> Program:
        """Recompute a namespace's latest checkpoint after deletions; drop the head when none is left."""
        head = yield from self._head(thread, ns)
        if head is None:
            return
        for page_no in range((head.metadata["cp_next"] - 1) // _PAGE, -1, -1):
            page = yield ("list", (_eq(kind="cp", cp_thread=thread, cp_ns=ns, cp_page=page_no),))
            if page.records:
                latest = max(r.metadata["cp_id"] for r in page.records)
                yield from self._write([self._head_record(thread, ns, latest, head.metadata["cp_next"])])
                return
        yield ("forget", ([head.id],))

    def _delete_thread(self, thread: str) -> Program:
        while True:
            page = yield ("list", (_eq(cp_thread=thread),))
            if not page.records:
                return
            yield ("forget", ([r.id for r in page.records],))

    def _delete_for_runs(self, run_ids: Sequence[str]) -> Program:
        touched = set()
        for run in run_ids:
            while True:
                page = yield ("list", (_eq(kind="cp", m_run_id=run),))
                if not page.records:
                    break
                yield from self._delete_checkpoint_records(page.records)
                touched.update((r.metadata["cp_thread"], r.metadata["cp_ns"]) for r in page.records)
        for thread, ns in touched:
            yield from self._repoint_head(thread, ns)

    def _copy_thread(self, source: str, target: str) -> Program:
        for t, ns, top in (yield from self._namespaces(source, None)):
            rows: List[CheckpointTuple] = []
            for page_no in range(0, top // _PAGE + 1):
                page = yield ("list", (_eq(kind="cp", cp_thread=t, cp_ns=ns, cp_page=page_no),))
                for rec in sorted(page.records, key=lambda r: r.metadata["cp_seq"]):
                    rows.append((yield from self._tuple(rec)))
            for tup in rows:
                cfg = {"configurable": {"thread_id": target, "checkpoint_ns": ns}}
                if tup.parent_config:
                    cfg["configurable"]["checkpoint_id"] = tup.parent_config["configurable"]["checkpoint_id"]
                cp = dict(tup.checkpoint)
                yield from self._put(cfg, cp, tup.metadata, dict(cp["channel_versions"]))  # type: ignore[arg-type]
                stored = {"configurable": {"thread_id": target, "checkpoint_ns": ns, "checkpoint_id": cp["id"]}}
                by_task: Dict[str, List[Tuple[str, Any]]] = {}
                for task, channel, value in tup.pending_writes or []:
                    by_task.setdefault(task, []).append((channel, value))
                for task, ws in by_task.items():
                    yield from self._put_writes(stored, ws, task, "")

    def _prune(self, thread_ids: Sequence[str], strategy: str) -> Program:
        if strategy not in ("keep_latest", "delete"):
            raise ValueError(f"unknown prune strategy {strategy!r}")
        for thread in thread_ids:
            if strategy == "delete":
                yield from self._delete_thread(thread)
                continue
            for t, ns, top in (yield from self._namespaces(thread, None)):
                head = yield from self._head(t, ns)
                latest = head.metadata["cp_latest"]
                keep_rec = (yield ("get", ([_h("cp", t, ns, latest)],)))[0]
                (text,) = yield from self._payloads([keep_rec])
                keep_versions = self.serde.loads_typed(_unpack(text))["checkpoint"]["channel_versions"]
                keep_blobs = {_h("blob", t, ns, ch, ver) for ch, ver in keep_versions.items()}
                for page_no in range(0, top // _PAGE + 1):
                    page = yield ("list", (_eq(kind="cp", cp_thread=t, cp_ns=ns, cp_page=page_no),))
                    doomed = [r for r in page.records if r.metadata["cp_id"] != latest]
                    yield from self._delete_checkpoint_records(doomed)
                while True:
                    page = yield ("list", (_eq(kind="blob", cp_thread=t, cp_ns=ns),))
                    doomed_blobs = [r for r in page.records if self._rid(r) not in keep_blobs]
                    if not doomed_blobs:
                        break
                    yield from self._delete_ids([self._rid(r) for r in doomed_blobs], {self._rid(r): r.metadata["cp_nparts"] for r in doomed_blobs})
                yield from self._repoint_head(t, ns)

    # ---- BaseCheckpointSaver: sync --------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self._run(self._get_tuple(config))

    def list(
        self, config: Optional[RunnableConfig], *, filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None, limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        yield from self._run(self._list(config, filter, before, limit))

    def put(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        return self._run(self._put(config, checkpoint, metadata, new_versions))

    def put_writes(self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        self._run(self._put_writes(config, writes, task_id, task_path))

    def delete_thread(self, thread_id: str) -> None:
        self._run(self._delete_thread(thread_id))

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        self._run(self._delete_for_runs(run_ids))

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self._run(self._copy_thread(source_thread_id, target_thread_id))

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        self._run(self._prune(thread_ids, strategy))

    # ---- async ----------------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await self._arun(self._get_tuple(config))

    async def alist(
        self, config: Optional[RunnableConfig], *, filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None, limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for tup in await self._arun(self._list(config, filter, before, limit)):
            yield tup

    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig:
        return await self._arun(self._put(config, checkpoint, metadata, new_versions))

    async def aput_writes(self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        await self._arun(self._put_writes(config, writes, task_id, task_path))

    async def adelete_thread(self, thread_id: str) -> None:
        await self._arun(self._delete_thread(thread_id))

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        await self._arun(self._delete_for_runs(run_ids))

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await self._arun(self._copy_thread(source_thread_id, target_thread_id))

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        await self._arun(self._prune(thread_ids, strategy))

    def get_next_version(self, current: Optional[str], channel: None) -> str:
        current_v = 0 if current is None else (current if isinstance(current, int) else int(current.split(".")[0]))
        return f"{current_v + 1:032}.{random.random():016}"
