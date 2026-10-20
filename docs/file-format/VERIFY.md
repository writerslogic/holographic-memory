<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# hms-format

```
hms-format verify <store-or-data-dir> [--passphrase-env VAR] [--deep] [--json]
hms-format plan <store>            hms-format upgrade <src> <dst> [--dry-run]
hms-format versions                hms-format dump <store> [--frame N]
```

`verify` never modifies the store. `--deep` opens a private copy with the engine and compares
the loaded documents, vector count and relations with an independent replay of the journal.
Without a passphrase an encrypted store is checked for framing and CRC32 only
(`payloadsAuthenticated: false`).

## Exit codes

| Code | Verdict | Meaning |
|---|---|---|
| 0 | ok | all checks passed (info findings allowed) |
| 10 | recoverable | opens after the engine repairs it: torn tail, bad cache, partial audit entry |
| 20 | corrupt | committed data or structure is damaged |
| 30 | unsupported | newer format, unknown encoder, or a store this build never opened (no `store.json`) |
| 64 / 74 | | usage / unreadable path |

## Finding codes

Info: `no-checksum-sidecar`, `legacy-zero-terminator`, `pre-manifest-store`, `orphan-generation`,
`text-segment-cache`, `index-cache-fresh`, `index-cache-stale`, `payloads-not-authenticated`,
`unrecognized-file`, `salt-unused`, `tenant-name`.
Warning: `torn-tail`, `segment-size`, `segment-size-unaligned` (final segment), `index-cache-invalid`,
`audit-torn-tail`, `audit-unknown-op`, `shard-meta-invalid`.
Error: `frame-corrupt`, `manifest-checksum`, `manifest-invalid`, `schema-invalid`, `segment-gap`,
`segment-size`, `payload-structure`, `payload-unrecognized`, `auth-failed`, `salt-invalid`,
`pointer-invalid`, `missing-file`, `key-invalid`, `deep-open`, `not-a-store`.
Unsupported: `storage-format-newer`, `storage-format-unknown`, `schema-newer`, `schema-version`,
`transaction-newer`, `transaction-older`, `unknown-magic`, `unknown-encoder`, `unknown-metric`,
`pre-schema-store`.

## Upgrade

`upgrade` writes a new store; the source is locked and never modified. Registered steps:

| Step | Does |
|---|---|
| `arena-1-to-4`, `arena-2-to-4`, `arena-3-to-4` | re-append every committed frame payload (opaque, encrypted frames unchanged) into a new format-4 arena; `seq` is renumbered |
| `text-segment-1-to-2` | drops `documents.text` (rebuilt on open) |
| `index-caches` | drops `indices.json` and NSG/IVF files (bound to the old journal length) |

The result is verified, its frame list (length and CRC32) must equal the source's, and it is
published with one rename. A store with no `store.json`, or a damaged or newer store, is refused
(use `hms-admin reencode`, restore a snapshot, or the newer release). A store already at format 4
needs nothing. Adding a format version means adding a registry entry in `src/core/format/upgrade.rs`
and a golden store.
