<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WARM persisted format

This is the specification of everything an WARM store writes to disk. It is derived from the
code at `main` and checked against real stores: `tests/golden/` holds committed stores for
every format that has existed, and `tests/format_*.rs` reads them. A statement here that no
test or hexdump supports is marked **(unverified)**. The tests and `warm-format` named in these
pages arrive in the follow-up pull requests (golden stores and tests, then the verifier).

"Must", "may" and "never" are normative for writers; readers follow the recovery rules in
[JOURNAL.md](JOURNAL.md). All integers are little-endian unless stated. CRC32 is the IEEE
polynomial (the `crc32fast` / zlib CRC).

| Document | Covers |
|---|---|
| [LAYOUT.md](LAYOUT.md) | directory tree, every file, `store.json`, `format.json`, tenants, lock, open sequence |
| [JOURNAL.md](JOURNAL.md) | arena segments, frame header, LZ4, end marker, torn tail vs corruption, compaction generations |
| [MUTATIONS.md](MUTATIONS.md) | frame payload: `HMS-TXN-2` and the JSON mutation records, limits, replay |
| [ENCRYPTION.md](ENCRYPTION.md) | frame envelope, key derivation, text-segment AEAD, what is not encrypted |
| [CHECKPOINT.md](CHECKPOINT.md) | `documents.ckpt`: the document store in place of the frames before it (schemas 4–5) |
| [CACHES.md](CACHES.md) | `documents.text`, `indices.json`, NSG/IVF files, `shard_meta.json` |
| [VECTOR-UPDATES.md](VECTOR-UPDATES.md) | persisted vector checkpoints, full-precision rows, update batches and consolidation |
| [ANCILLARY.md](ANCILLARY.md) | `audit.bin`, keys, provenance files, standalone qgraph/IVF files, memory-API stamps |
| [CRASH-CONSISTENCY.md](CRASH-CONSISTENCY.md) | what survives a crash, per file, and the test that shows it |
| [ENCODERS.md](ENCODERS.md) | text encoder names, determinism contract, embedding spaces |
| [VERIFY.md](VERIFY.md) | `warm-format verify` / `upgrade`, finding codes, exit codes |

Compatibility promises are in [../STABILITY.md](../STABILITY.md).

## Version numbers

Each artefact has its own version; a store is the combination.

| Artefact | Where | Current | Readable |
|---|---|---|---|
| Arena manifest | `vectors_data.bin/format.json` `version` | 4 | 2, 3, 4 (no manifest = 1, read as 2) |
| Store schema | `store.json` `version` | 5 | 2, 3, 4, 5 |
| Document checkpoint | `documents.ckpt` magic `WARMCKP\0\x01` (`WARMCKPE\x01` sealed) | 1 | 1, schemas 4–5 |
| Journal payload | magic `HMS-TXN-3\0` | 3 | 2, 3 |
| Text segment | `documents.text` header byte 4 | 2 | cache: any value is discarded and rebuilt |
| Vector updates | `WARMVUP01`, `WARMVRW01`, `WARMVLG01` | 1 | 1 |
| Memory stamp | `mem1:` / `_mem.v` | 1 | 1 |

The same numbers are constants in `src/core/format/mod.rs`; `tests/format_golden.rs` fails if one
moves without a golden store for the new value.

The current development writer creates schema 5 stores. Schema 5 adds scoped meaning
maintenance mutations without changing the binary journal encoding. Existing schemas
2–4 keep their writer format; enabling the new mutations requires an explicit
`warm-format upgrade` into a new destination.

## History

| Format | First written by | Notes |
|---|---|---|
| Arena 1 | pre-manifest releases | no `format.json`; `seg_0.bin` directly in `vectors_data.bin/`; read as 2 and the manifest is written on open |
| Arena 2 | release v0.6.1 (commit 28c4f94) | 1 GiB segments preallocated, zero-header terminator, no checksum sidecars |
| Arena 2 + end markers + sidecars | commit bf9c322 | same manifest version; frames followed by a 16-byte end marker; `*.crc32` sidecars |
| Arena 3 | development only | 64 MiB fixed segments; never on `main` as a written default; the reader still accepts it |
| Arena 4 | commit fd5adf3 | 1 GiB declared segment, file grows in 64 MiB steps; `CURRENT` generations from compaction |
| Text segment 1 | fd5adf3 | headerless plaintext cache |
| Text segment 2 | #111 | `WARMT` header; encrypted records |
| Stores without `store.json` | before the schema | refused; rebuild with `warm-admin reencode` |

Golden stores: `v061-*` (real v0.6.1 build), `v1-premanifest`, `rejected-pre-schema` (derived
from v0.6.1), `v2-marker`, `v3-fixed64` (synthesized with the current writer and a seeded
manifest), `v4-*`, and `v5-governance`.
