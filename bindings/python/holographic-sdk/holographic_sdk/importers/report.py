# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run statistics and the fidelity report."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from .common import scrub

_SAMPLE_IDS = 20
_NAMESPACE_CAP = 1000


class Stats:
    """Counters for one run. Nothing is dropped without an entry here."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.rejects: Dict[str, int] = {}
        self.reject_samples: Dict[str, List[str]] = {}
        self.skips: Dict[str, int] = {}
        self.skip_samples: Dict[str, List[str]] = {}
        self.namespaces: Counter = Counter()
        self.coercions: Counter = Counter()
        self.source_dims: Counter = Counter()
        self.started = time.monotonic()

    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] += n

    def reject(self, reason: str, ident: str) -> None:
        self.rejects[reason] = self.rejects.get(reason, 0) + 1
        sample = self.reject_samples.setdefault(reason, [])
        if len(sample) < _SAMPLE_IDS:
            sample.append(ident)

    def skip(self, reason: str, ident: str) -> None:
        self.skips[reason] = self.skips.get(reason, 0) + 1
        sample = self.skip_samples.setdefault(reason, [])
        if len(sample) < _SAMPLE_IDS:
            sample.append(ident)

    def namespace(self, name: str, n: int = 1) -> None:
        if name in self.namespaces or len(self.namespaces) < _NAMESPACE_CAP:
            self.namespaces[name] += n
        else:
            self.counts["namespaces_over_cap_docs"] += n

    @property
    def rejected_total(self) -> int:
        return sum(self.rejects.values())

    @property
    def skipped_total(self) -> int:
        return sum(self.skips.values())

    def snapshot(self) -> Dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "rejects": dict(self.rejects),
            "skips": dict(self.skips),
            "namespaces": dict(self.namespaces),
            "coercions": dict(self.coercions),
            "source_dims": {str(k): v for k, v in self.source_dims.items()},
        }

    def restore(self, snap: Dict[str, Any]) -> None:
        self.counts.update(snap.get("counts", {}))
        for key, n in snap.get("rejects", {}).items():
            self.rejects[key] = self.rejects.get(key, 0) + n
        for key, n in snap.get("skips", {}).items():
            self.skips[key] = self.skips.get(key, 0) + n
        self.namespaces.update(snap.get("namespaces", {}))
        self.coercions.update(snap.get("coercions", {}))
        for key, n in snap.get("source_dims", {}).items():
            self.source_dims[int(key)] += n


class Report:
    """Fidelity report: what was read, written, changed, skipped, rejected, verified."""

    def __init__(self, source: str, descriptor: Dict[str, Any], options: Dict[str, Any], stats: Stats):
        self.source = source
        self.descriptor = descriptor
        self.options = options
        self.stats = stats
        self.source_count: Optional[int] = None
        self.server: Dict[str, Any] = {}
        self.verification: Dict[str, Any] = {"performed": False}
        self.source_consistency: Dict[str, Any] = {
            "mode": "observed_stream",
            "snapshot_isolation": "source_managed",
            "resume_validation": "content_verified_prefix_v2",
            "deleted_source_records": "retained",
        }
        self.dry_run = False
        self.resumed_from: Optional[int] = None
        self.complete = False
        self.error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        elapsed = time.monotonic() - self.stats.started
        sent = self.stats.counts.get("docs_acked", 0)
        return {
            "source": self.source,
            "descriptor": self.descriptor,
            "options": self.options,
            "dry_run": self.dry_run,
            "complete": self.complete,
            "error": self.error,
            "resumed_from_record": self.resumed_from,
            "server": self.server,
            "source_count_reported": self.source_count,
            "source_consistency": self.source_consistency,
            "counts": dict(self.stats.counts),
            "skipped": {
                "total": self.stats.skipped_total,
                "by_reason": dict(self.stats.skips),
                "sample_ids": self.stats.skip_samples,
            },
            "rejected": {
                "total": self.stats.rejected_total,
                "by_reason": dict(self.stats.rejects),
                "sample_ids": self.stats.reject_samples,
            },
            "namespaces": dict(self.stats.namespaces),
            "source_vector_dims": {str(k): v for k, v in self.stats.source_dims.items()},
            "metadata_coercions": dict(self.stats.coercions),
            "verification": self.verification,
            "elapsed_seconds": round(elapsed, 3),
            "docs_per_second": round(sent / elapsed, 2) if elapsed > 0 and sent else 0.0,
        }

    def write(self, path: str) -> None:
        data = json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".report-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(scrub(data))
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def summary(self) -> str:
        d = self.to_dict()
        c = d["counts"]
        lines = [
            f"source: {self.source}  {'(dry run)' if self.dry_run else ''}".rstrip(),
            f"records read: {c.get('records_read', 0)}"
            + (f" of {self.source_count} reported by the source" if self.source_count is not None else ""),
            f"documents {'planned' if self.dry_run else 'acknowledged'}: "
            f"{c.get('docs_planned', 0) if self.dry_run else c.get('docs_acked', 0)}",
            f"skipped: {d['skipped']['total']} {d['skipped']['by_reason']}",
            f"rejected: {d['rejected']['total']} {d['rejected']['by_reason']}",
        ]
        notable = {
            k: v
            for k, v in c.items()
            if k.startswith(("ids_rewritten", "texts_split", "vectors_", "metadata_", "timestamps_"))
        }
        if notable:
            lines.append(f"changes: {notable}")
        v = d["verification"]
        if v.get("performed"):
            lines.append(
                "verification: "
                f"recall@{v.get('k')}={v.get('recall_at_k_mean')} "
                f"self-hit={v.get('self_hit_rate')} round-trip mismatches={v.get('roundtrip_mismatches')} "
                f"({v.get('queries')} queries, basis: {v.get('basis')})"
            )
        elif v.get("reason"):
            lines.append(f"verification: not performed ({v['reason']})")
        return scrub("\n".join(lines))
