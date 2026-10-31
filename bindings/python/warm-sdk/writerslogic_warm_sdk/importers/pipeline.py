# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Streaming, resumable, idempotent import pipeline."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .checkpoint import Checkpoint
from .common import redact_uri, scrub, stable_hash, to_jsonable, vector_list
from .mapping import Mapper, Projector, Target, load_embedder, valid_vector
from .model import WarmDoc, ImporterError, Record, Reject
from .options import ImportOptions
from .report import Report, Stats
from .sink import BatchRejected, WarmSink
from .sources.base import Source
from .verify import Reservoir, run_verification

SERVER_MAX_BATCH_DOCS = 1000  # server default for --max-batch
SERVER_MAX_EMBEDDING_VALUES = 1_048_576  # per batch request
_VECTOR_JSON_BYTES = 24  # generous bytes per serialized float
# Refusals that concern the request, not one document: bisecting would only repeat them.
_FATAL_CODES = {"invalid_tenant", "not_found", "method_not_allowed", "tenant_limit", "unauthorized"}


def _doc_bytes(doc: WarmDoc) -> int:
    n = len(doc.id) + len(doc.text or "") + len(json.dumps(doc.metadata, separators=(",", ":"))) + 32
    if doc.vector is not None:
        n += len(doc.vector) * _VECTOR_JSON_BYTES
    return n


def _chain(prev: str, ident: str) -> str:
    return hashlib.sha256((prev + "\x00" + ident).encode("utf-8", "surrogatepass")).hexdigest()


def _record_fingerprint(record: Record) -> str:
    return stable_hash(
        "warm-import-prefix-v2", record.namespace, str(record.id), record.text,
        to_jsonable(vector_list(record.vector)), to_jsonable(record.metadata),
        to_jsonable(record.created_at), to_jsonable(record.updated_at), record.error,
    )


def default_checkpoint_path(opts: ImportOptions, fingerprint: str) -> str:
    base = opts.state_dir or os.environ.get("WARM_IMPORT_STATE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "warm-import"
    )
    return os.path.join(base, f"{fingerprint[:24]}.json")


class _Rejects:
    """Append-only JSONL of rejected records (ids and reasons only, never content)."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self._fh = None

    def add(self, ident: str, reason: str, detail: str) -> None:
        if self.path is None:
            return
        if self._fh is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps({"id": ident, "reason": reason, "detail": scrub(detail)}) + "\n")

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def run_import(
    source: Source,
    sink: Optional[WarmSink],
    opts: ImportOptions,
    *,
    tenant: Optional[str] = None,
    target: Optional[Target] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Report:
    """Import every record of `source` into the WARM server behind `sink`.

    With `opts.dry_run` (or no sink) nothing is written: records are read, mapped and
    counted, and the fidelity check runs locally. Returns the `Report`. On a fatal
    error raises `ImporterError` carrying the partial `report`; a checkpoint then
    allows resuming."""
    opts.validate()
    dry = opts.dry_run or sink is None
    say = progress or (lambda msg: None)
    stats = Stats()
    descriptor = source.describe()

    target = target or Target(input_dim=opts.target_dim)
    if target.input_dim is None:
        target.input_dim = opts.target_dim
    if opts.embedder and target.embedder is None:
        target.embedder = load_embedder(opts.embedder)
    report = Report(source.name, descriptor, _options_dict(opts), stats)
    report.dry_run = dry
    try:
        return _run(source, sink, opts, tenant, target, say, stats, descriptor, report, dry)
    except BaseException as exc:
        report.error = scrub(f"{type(exc).__name__}: {exc}")
        if opts.report_path:
            report.write(opts.report_path)
        if isinstance(exc, ImporterError):
            exc.report = report  # type: ignore[attr-defined]
        raise


def _run(source, sink, opts, tenant, target, say, stats, descriptor, report, dry) -> Report:
    if sink is not None:
        info = sink.probe()
        if info.get("input_dim") is not None:
            if opts.target_dim and opts.target_dim != info["input_dim"]:
                raise ImporterError(f"--target-dim {opts.target_dim} disagrees with the server ({info['input_dim']})")
            target.input_dim = info["input_dim"]
        if target.server_embeds is None:
            target.server_embeds = info.get("server_embeds")
    report.server = {
        "url": redact_uri(sink.url) if sink else None,
        "input_dim": target.input_dim,
        "embeds_text": target.server_embeds,
        "client_side_embedder": opts.embedder,
    }
    report.source_count = source.count()

    fingerprint = stable_hash(
        source.name, descriptor, redact_uri(sink.url) if sink else None, tenant, opts.mapping_fingerprint()
    )
    ckpt: Optional[Checkpoint] = None
    rejects = _Rejects(None)
    if not dry and not opts.no_checkpoint:
        path = opts.checkpoint_path or default_checkpoint_path(opts, fingerprint)
        ckpt = Checkpoint.open(path, fingerprint, opts.restart)
        rejects = _Rejects(path + ".rejects.jsonl")
        report.server["checkpoint"] = path
    if ckpt is not None and ckpt.position:
        stats.restore(ckpt.stats)
        report.resumed_from = ckpt.position

    mapper = Mapper(source.name, opts, target, stats, tenant, stable_hash(source.name, descriptor))
    if sink is not None:
        sink.split_policy = opts.split_policy
    reservoir = Reservoir(opts.verify_sample, opts.seed)
    projector_cache: Dict[int, Projector] = {}
    seen_tenants: Dict[Optional[str], None] = {}

    def new_space(vec: List[float]) -> Optional[List[float]]:
        """The vector this import would send for `vec` (dry-run recall)."""
        if opts.vectors in ("carry", "auto"):
            return vec if target.input_dim in (None, len(vec)) else None
        if opts.vectors == "project" and target.input_dim:
            proj = projector_cache.get(len(vec))
            if proj is None:
                proj = projector_cache[len(vec)] = Projector(len(vec), target.input_dim, opts.projection_seed)
            return proj.apply(vec)
        return None

    start_pos = ckpt.position if ckpt else 0
    cursor = ckpt.cursor if ckpt else None
    prefix_hash = ckpt.prefix_hash if ckpt else ""

    def open_stream() -> Iterator[Record]:
        """Records after the committed prefix. A non-seekable source is re-read from the start and
        the prefix is checked against the checkpoint's hash; if the order or content changed, the
        import restarts from record 0 (safe: writes replace by id)."""
        nonlocal start_pos, cursor, prefix_hash, position, last_cursor
        it = source.records(None)
        if start_pos:
            running, skipped = "", 0
            for rec in it:
                running = _chain(running, _record_fingerprint(rec))
                skipped += 1
                if skipped >= start_pos:
                    break
            if skipped == start_pos and running == prefix_hash:
                yield from it
                return
            say("source order or content changed since the checkpoint; restarting from the first record")
            report.resumed_from = None
            report.server["resume_prefix_mismatch"] = True
            stats.counts.clear(); stats.rejects.clear(); stats.skips.clear()
            stats.namespaces.clear(); stats.coercions.clear(); stats.source_dims.clear()
            start_pos, cursor, prefix_hash = 0, None, ""
            position, last_cursor = 0, None
            it = source.records(None)
        yield from it

    position = start_pos
    last_cursor = cursor
    batch: List[WarmDoc] = []
    batch_bytes = 0
    pending_hashes: List[str] = []
    pending_cursor: Any = cursor
    pending_samples: List[Tuple[Record, List[WarmDoc]]] = []
    batches_done = 0

    def effective_batch() -> int:
        dim = target.input_dim or 0
        cap = SERVER_MAX_EMBEDDING_VALUES // dim if dim else SERVER_MAX_BATCH_DOCS
        return max(1, min(opts.batch_size, SERVER_MAX_BATCH_DOCS, cap))

    def encode_client_side(docs: List[WarmDoc]) -> None:
        todo = [d for d in docs if d.vector is None and d.text]
        if not todo or target.embedder is None:
            return
        vectors = target.embedder([d.text for d in todo])
        if len(vectors) != len(todo):
            raise ImporterError("the embedder returned a different number of vectors than texts")
        for d, v in zip(todo, vectors):
            vec = vector_list(v)
            if not valid_vector(vec):
                raise ImporterError("the embedder returned an unusable vector")
            if target.input_dim and len(vec) != target.input_dim:
                raise ImporterError(f"the embedder returned {len(vec)} dimensions; the server takes {target.input_dim}")
            d.vector = vec
        stats.count("vectors_encoded_client_side", len(todo))

    def commit(docs: List[WarmDoc]) -> None:
        """Send `docs`; on a per-document refusal bisect to isolate the offending documents."""
        try:
            sink.send(docs)
            stats.count("docs_acked", len(docs))
            stats.count("batches_sent")
            for d in docs:
                seen_tenants[d.tenant] = None
        except BatchRejected as exc:
            if exc.code in _FATAL_CODES or exc.status in (404, 405):
                raise ImporterError(scrub(f"the WARM server refused the request: {exc}")) from None
            if len(docs) == 1:
                d = docs[0]
                key = d.source_key[1] if d.source_key else d.id
                stats.reject("server_rejected", str(key))
                rejects.add(str(key), "server_rejected", f"{exc.code}: {exc.message}")
                stats.count("server_refusals")
                if stats.counts["server_refusals"] > opts.max_rejects:
                    raise ImporterError(
                        f"the server refused more than {opts.max_rejects} documents (last: {exc.code}: {exc.message}); "
                        "this looks systematic. Fix the cause or raise --max-rejects"
                    ) from None
                return
            mid = len(docs) // 2
            commit(docs[:mid])
            commit(docs[mid:])
        finally:
            report.server["stale_parts_deleted_this_run"] = sink.stale_parts_deleted

    def flush() -> None:
        nonlocal batch, batch_bytes, position, last_cursor, batches_done, prefix_hash
        if not pending_hashes:
            return
        if batch:
            if dry:
                stats.count("docs_planned", len(batch))
            else:
                encode_client_side(batch)
                commit(batch)
        for rec, docs in pending_samples:
            reservoir.offer(rec, docs)
        for ident in pending_hashes:
            prefix_hash = _chain(prefix_hash, ident)
        position += len(pending_hashes)
        last_cursor = pending_cursor
        batches_done += 1
        batch, batch_bytes = [], 0
        pending_hashes.clear()
        pending_samples.clear()
        if ckpt is not None and batches_done % max(1, opts.checkpoint_every) == 0:
            rejects.flush()
            ckpt.position, ckpt.cursor, ckpt.stats, ckpt.prefix_hash = position, last_cursor, stats.snapshot(), prefix_hash
            ckpt.save()
            say(f"committed {position} records")

    try:
        for rec in open_stream():
            if opts.limit is not None and position - start_pos + len(pending_hashes) >= opts.limit:
                break
            stats.count("records_read")
            pending_hashes.append(_record_fingerprint(rec))
            pending_cursor = rec.cursor
            try:
                docs = mapper.map(rec)
            except Reject as rej:
                if rej.reason == "empty" and opts.skip_empty:
                    stats.skip("empty", str(rec.id))
                else:
                    stats.reject(rej.reason, str(rec.id))
                    rejects.add(str(rec.id), rej.reason, rej.detail)
                docs = []
            if docs:
                pending_samples.append((rec, docs))
                for d in docs:
                    batch.append(d)
                    batch_bytes += _doc_bytes(d)
            if len(batch) >= effective_batch() or batch_bytes >= opts.max_batch_bytes:
                flush()
        flush()
        if ckpt is not None:
            rejects.flush()
            ckpt.position, ckpt.cursor, ckpt.stats = position, last_cursor, stats.snapshot()
            ckpt.prefix_hash, ckpt.complete = prefix_hash, True
            ckpt.save()
    finally:
        rejects.close()
    report.complete = True

    if not dry and sink is not None:
        report.server["index_maintenance"] = (
            sink.maintain_index(list(seen_tenants) or [None]) if opts.extra.get("maintain_index", True) else "skipped"
        )

    if opts.vectors in ("reencode", "none") and dry:
        report.verification = {"performed": False, "reason": "a dry run cannot predict a server-side re-encoding"}
    else:
        report.verification = run_verification(
            source=source,
            sink=None if dry else sink,
            opts=opts,
            target=target,
            reservoir=reservoir,
            new_space=new_space,
            source_name=source.name,
            default_tenant=tenant,
        )
    if opts.report_path:
        report.write(opts.report_path)
    return report


def _options_dict(opts: ImportOptions) -> Dict[str, Any]:
    d = dataclasses.asdict(opts)
    d.pop("extra", None)
    return d


def exit_code(report: Report, opts: ImportOptions) -> int:
    if report.error:
        return 1
    if report.stats.rejected_total and not opts.allow_rejects:
        return 3
    return 0


__all__ = ["run_import", "exit_code", "default_checkpoint_path"]
