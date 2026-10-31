# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Disk-backed destination ownership inventory for bounded-memory import preflight."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, List, Tuple

from .mapping import MAX_RECORD_PARTS, mapped_part_id
from .model import WarmDoc, ImporterError

_OWNER_KEYS = (
    "warm_source", "hms_namespace", "warm_source_id", "warm_part", "warm_parts",
    "warm_import_identity", "warm_import_record", "warm_import_source", "warm_import_parts",
    "warm_import_base_id", "warm_id_mapping",
)


def _legacy_key(metadata: Dict[str, Any]) -> str:
    if not all(key in metadata for key in ("warm_source", "hms_namespace", "warm_source_id")):
        return ""
    return json.dumps([metadata[key] for key in ("warm_source", "hms_namespace", "warm_source_id")],
                      ensure_ascii=True, separators=(",", ":"))


class ImportInventory:
    def __init__(self) -> None:
        self.db = sqlite3.connect("")
        self.db.execute("CREATE TABLE documents (tenant TEXT, id TEXT, owner TEXT, record TEXT, legacy TEXT, metadata TEXT, PRIMARY KEY (tenant, id))")
        self.db.execute("CREATE INDEX records ON documents (tenant, record)")
        self.db.execute("CREATE INDEX legacy ON documents (tenant, legacy)")
        self.loaded: set[str] = set()

    def close(self) -> None:
        self.db.close()

    def record(self, tenant: str, ident: str, metadata: Dict[str, Any]) -> None:
        if not isinstance(metadata, dict):
            metadata = {}
        kept = {key: metadata[key] for key in _OWNER_KEYS if key in metadata}
        self.db.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                        (tenant, ident, kept.get("warm_import_identity", ""),
                         kept.get("warm_import_record", ""), _legacy_key(kept), json.dumps(kept)))

    def load(self, tenant: str, documents: Iterable[Dict[str, Any]]) -> None:
        for document in documents:
            self.record(tenant, document["id"], document.get("metadata") or {})
        self.db.commit()
        self.loaded.add(tenant)

    def prepare(self, tenant: str, docs: List[WarmDoc], split_policy: str) -> List[Tuple[str, str]]:
        seen: Dict[str, str] = {}
        groups: Dict[str, List[WarmDoc]] = {}
        for doc in docs:
            owner = doc.metadata.get("warm_import_identity")
            record = doc.metadata.get("warm_import_record")
            if not isinstance(owner, str) or not isinstance(record, str):
                raise ImporterError("import documents require mapper-generated ownership identities")
            if doc.id in seen and seen[doc.id] != owner:
                raise ImporterError("mapped document identity collision; use --id-mapping v2 in a separate tenant")
            seen[doc.id] = owner
            existing = self.db.execute("SELECT owner, metadata FROM documents WHERE tenant = ? AND id = ?",
                                       (tenant, doc.id)).fetchone()
            if existing is not None and existing[0] != owner:
                old = json.loads(existing[1])
                legacy_match = (not existing[0] and doc.adopt_legacy and _legacy_key(old)
                                and _legacy_key(old) == _legacy_key(doc.metadata)
                                and old.get("warm_part") == doc.metadata.get("warm_part")
                                and ("warm_part" in old) == ("warm_part" in doc.metadata)
                                and old.get("warm_parts", 1) == doc.metadata["warm_import_parts"])
                if not legacy_match:
                    raise ImporterError("destination identity conflict; use a separate tenant or explicitly adopt matching legacy imports")
            groups.setdefault(record, []).append(doc)

        obsolete: Dict[str, str] = {}
        for record, group in groups.items():
            meta = group[0].metadata
            parts = meta["warm_import_parts"]
            if not isinstance(parts, int) or isinstance(parts, bool) or not 1 <= parts <= MAX_RECORD_PARTS:
                raise ImporterError("invalid imported part count")
            expected = {mapped_part_id(meta["warm_import_base_id"], record, meta["warm_id_mapping"], parts, i)
                        for i in range(parts)}
            rows = self.db.execute("SELECT id, owner, metadata FROM documents WHERE tenant = ? AND (record = ? OR (owner = '' AND legacy = ?)) LIMIT ?",
                                   (tenant, record, _legacy_key(meta), MAX_RECORD_PARTS * 2 + 1)).fetchall()
            if len(rows) > MAX_RECORD_PARTS * 2:
                raise ImporterError("too many prior layouts for this source record; inspect the destination before replacing it")
            changed = any(ident not in expected or json.loads(encoded).get("warm_import_parts", json.loads(encoded).get("warm_parts", 1)) != parts
                          for ident, _, encoded in rows)
            if not changed:
                continue
            if split_policy != "replace":
                raise ImporterError("source layout changed; --split-policy replace explicitly replaces obsolete owned parts")
            if any(not owner for _, owner, _ in rows):
                raise ImporterError("adopt legacy imports in their original layout before replacing their layout")
            if {doc.id for doc in group} != expected:
                raise ImporterError("a replacement layout must fit one atomic server batch; reduce the source record size")
            obsolete.update((ident, owner) for ident, owner, _ in rows if ident not in expected)
        return list(obsolete.items())

    def remove(self, tenant: str, ident: str) -> None:
        self.db.execute("DELETE FROM documents WHERE tenant = ? AND id = ?", (tenant, ident))
