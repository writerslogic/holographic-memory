<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ancillary files

## audit.bin

Opt-in (`audit_enabled`). Append-only, fixed 105-byte entries, no header:

```
0   8   timestamp_ms  u64
8   1   op            1 memorize, 2 delete, 3 compact
9   32  id_hash       SHA-256 of the id (security feature) or FxHash in the first 8 bytes
41  64  signature     Ed25519 over bytes 0..41, or zeros when signing is off
```

Each append is fsynced. A reader ignores a trailing partial entry and skips entries whose op is
not 1..=3. On open the writer cuts a trailing partial entry so later entries stay aligned.
Note the id hash differs by build feature: it is not comparable across builds with and without
`security`.

## Keys

`encryption.salt`: 16 bytes. `hms_signing.key`: the 32-byte Ed25519 seed, mode 0600.
`hms_provenance.key`: same layout. All raw, no header.

## Provenance (feature `provenance`)

`provenance_log.jsonl` is one JSON object per line chaining `prev_hash` (hex SHA-256 of the
previous line's bytes); `provenance_log.head` is a COSE-signed head; `provenance_log.statuslist`
is a raw revocation bit string (bit `i` of byte `i/8`); `kel.json` is a pretty JSON array of
key events. These are written with plain, non-atomic writes. **(unverified byte-level; outside
the stability promise.)**

## connection_graph.log (feature `experimental`)

Event log of the experimental connection graph. Unstable; no promise.

## Standalone files (not part of a store)

`HMSGRF04` graph caches (`qgraph::Graph::save`) and `QuantizedIvf::to_bytes` blobs (magic, little-endian
header, CRC32 trailer) are benchmark caches. They are validated against their inputs when read
and are never read from a store directory.

`HMSVGR01` vertex index files (`qgraph::VGraph::save`, opened by `VGraph::open`): the magic, 19
little-endian `u64` header words, an `(offset, bytes)` table of 15 sections, then each section on a
4096-byte boundary so the file can be memory-mapped and used in place. Written to a temporary file,
synced and renamed into place. Open recomputes every derived size from the header and checks every
section length and stored id before use; a mismatch is an error. Little-endian targets only; outside
the stability promise (a format change bumps the magic and needs a re-save).

Persisted updatable vector directories use the same vertex encoding together with versioned
row, checkpoint and update-log files; see [VECTOR-UPDATES.md](VECTOR-UPDATES.md). Those files
are durable data rather than rebuildable store caches.

## Memory-API stamps

A fact is a document; its stamp is in the document `version` string as `mem1:<compact JSON>` or,
when over 1024 bytes, in metadata key `_mem`. Specified in
[../memory-api/PERSISTED-FORMAT.md](../memory-api/PERSISTED-FORMAT.md); `tests/golden/v4-memory` holds both forms.
