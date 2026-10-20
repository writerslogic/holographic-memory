# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Import options. Every policy has a default and a knob."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

VECTOR_MODES = ("auto", "carry", "reencode", "project", "none")
NAMESPACE_MODES = ("metadata", "tenant")
ID_PREFIX_MODES = ("auto", "always", "never")
OVERSIZE_MODES = ("reject", "truncate")


@dataclass
class ImportOptions:
    """How records become HMS documents and how the run behaves.

    vectors:
      auto      carry a source vector when it has the server's dimension; otherwise let the
                server (or `embedder`) encode the text. A dimension mismatch never silently
                projects: it fails the run before anything is written, naming the options.
      carry     send source vectors verbatim; records without a usable vector are rejected.
      reencode  ignore source vectors and encode every text (server on-device model, or the
                client-side `embedder` callable).
      project   deterministic seeded Gaussian projection of source vectors to the server
                dimension (zero padding when the source is smaller). Approximate: see the
                recall figure in the fidelity report.
      none      send text and metadata only (needs a server that embeds text).
    """

    vectors: str = "auto"
    embedder: Optional[str] = None  # "package.module:callable" taking list[str] -> list[list[float]]
    projection_seed: int = 0x484D53
    target_dim: Optional[int] = None  # server input dimension; probed when a server is reachable

    namespace_mode: str = "metadata"  # metadata: hms_namespace key; tenant: one HMS tenant per namespace
    default_namespace: str = ""  # namespace assigned to records the source gives none
    id_prefix_namespace: str = "auto"  # prefix ids with "<namespace>::" (auto: when namespace is non-empty)
    tenant_prefix: str = ""  # prepended to namespace-derived tenant ids

    batch_size: int = 256
    max_batch_bytes: int = 7 * 1024 * 1024  # request body estimate; under the server's 8 MiB id/text and metadata budgets
    max_rejects: int = 1000  # abort when more individual documents than this are refused by the server
    state_dir: Optional[str] = None  # default checkpoint directory ($HMS_IMPORT_STATE_DIR, else ~/.cache/hms-import)
    no_checkpoint: bool = False
    checkpoint_path: Optional[str] = None
    checkpoint_every: int = 1  # batches between checkpoint writes
    restart: bool = False  # ignore an incomplete checkpoint and start over
    retries: int = 5
    backoff_seconds: float = 0.5
    limit: Optional[int] = None  # stop after this many source records (testing)

    max_text_words: int = 4000  # server limit is 4096 words and 64 KiB per document
    max_text_bytes: int = 60 * 1024
    oversize_metadata: str = "reject"  # reject | truncate (drops largest keys, recorded in hms_dropped_keys)
    max_metadata_bytes: int = 60 * 1024
    skip_empty: bool = True  # skip records with neither text nor a usable vector (counted, listed)

    dry_run: bool = False
    verify_sample: int = 50  # queries for the old-vs-new recall check (0 disables)
    verify_k: int = 10
    seed: int = 1
    report_path: Optional[str] = None
    allow_rejects: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.vectors not in VECTOR_MODES:
            raise ValueError(f"vectors must be one of {VECTOR_MODES}")
        if self.namespace_mode not in NAMESPACE_MODES:
            raise ValueError(f"namespace_mode must be one of {NAMESPACE_MODES}")
        if self.id_prefix_namespace not in ID_PREFIX_MODES:
            raise ValueError(f"id_prefix_namespace must be one of {ID_PREFIX_MODES}")
        if self.oversize_metadata not in OVERSIZE_MODES:
            raise ValueError(f"oversize_metadata must be one of {OVERSIZE_MODES}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.verify_sample < 0 or self.verify_k < 1:
            raise ValueError("verify_sample must be >= 0 and verify_k >= 1")
        if self.max_text_words < 1 or self.max_text_bytes < 256:
            raise ValueError("text limits are too small")

    def mapping_fingerprint(self) -> List[Any]:
        """Options that change which documents a record becomes; part of the checkpoint key."""
        return [
            self.vectors,
            self.embedder,
            self.projection_seed,
            self.target_dim,
            self.namespace_mode,
            self.default_namespace,
            self.id_prefix_namespace,
            self.tenant_prefix,
            self.max_text_words,
            self.max_text_bytes,
            self.oversize_metadata,
            self.max_metadata_bytes,
        ]
