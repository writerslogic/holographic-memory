<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Store layout

A store is one directory. The engine creates it on first open.

```
<store>/
  .hms.lock                  exclusive flock; content "pid=<n>\n"; not part of the data
  store.json  store.json.crc32
  vectors_data.bin/          the arena (a directory despite the name)
    format.json  format.json.crc32
    seg_0.bin seg_1.bin ...  journal segments
    CURRENT                  only after a compaction: names the live generation directory
    gen-<pid>-<nanos>/       the live generation after a compaction: seg_*.bin, format.json,
                             and in schemas 4–5 documents.ckpt (CHECKPOINT.md)
  documents.text             cache (CACHES.md)
  indices.json               cache (CACHES.md); older stores also have indices.json.crc32, ignored
  nsg_index.bin ivf_index.bin  or nsg_index_<i>.bin ivf_index_<i>.bin    caches
  shard_meta.json            shard count after auto-sharding
  encryption.salt            16 bytes, encrypted stores only
  audit.bin  hms_signing.key  hms_provenance.key  provenance_log.jsonl* kel.json  (ANCILLARY.md)
```

Server data directory: `<data-dir>/tenants/<hex(utf8 tenant name)>/` is one store per tenant.
`warm-format verify <data-dir>` checks each store.
Server ownership files live separately at `<data-dir>/tenant-locks/<hex>.owner`. Their lifetime
advisory locks survive a live-store rename; they are control metadata, not store data. The
`tenants/` namespace contains only tenant directories.
Other server control metadata in `<data-dir>` is not read as store data.

A snapshot is a copy of the whole directory taken while no engine holds the lock;
`warm-admin migrate` makes a verified copy (lock file left out). There is no other snapshot
artefact.

## store.json

Pretty-printed JSON, written once (atomic: temp file, fsync, rename, directory fsync; the
sidecar first). Keys, in this order:

| Key | Type | Meaning |
|---|---|---|
| `version` | integer | 5 for new stores; 2, 3 and 4 remain supported |
| `dimensions` | integer | hypervector dimension, 1..=1000000, fixed for the store |
| `text_encoder` | string | see [ENCODERS.md](ENCODERS.md) |
| `vector_encoder` | string | `signed-projection-v2` |
| `metric` | string | `jaccard` |
| `encrypted` | bool | frames are AES-256-GCM sealed |
| `meaning_enabled` | bool | meaning memory configured |
| `embedding_space` | object or null | `{model, revision, dimensions, normalization, metric}` |

On open the engine builds the same object from its configuration and requires equality. A
different value is refused ("incompatible store schema"); a `version` above 5 is refused as
newer than supported; a store with `vectors_data.bin` but no `store.json` is refused as legacy.
The sidecar `store.json.crc32` is the CRC32 of the file bytes as 8 lowercase hex digits and a
newline; absent on stores written before sidecars, and then unverified.

## format.json

```json
{ "magic": "HMS_ARENA", "version": 4, "segment_size": 1073741824 }
```

`segment_size` must be 64 MiB (67108864) when `version` is 3 and 1 GiB (1073741824) for 2 and
4. Magic other than `HMS_ARENA` or `version` above 4 is refused. Same sidecar rule as
`store.json`. With a `CURRENT` pointer the engine reads `vectors_data.bin/format.json`, falling
back to the generation's copy.

Journal segments must be numbered contiguously from `seg_0.bin`. Opening rejects gaps and
active segments larger than `segment_size` without resizing or recovering those files.

## CURRENT

ASCII name, no newline, matching `gen-[A-Za-z0-9-]+`. The directory must exist and not be a
symlink. Without `CURRENT` the live segments are in `vectors_data.bin/` itself.

## Open sequence and what open writes

1. create the directory; take the lock; validate or create `store.json`
2. open the arena: find the live generation and check segment numbering, create or validate
   `format.json`, check the active segment's size, recover the active segment (JOURNAL.md),
   write the end marker at the recovered end, extend the active file to its allocation size
3. zero and unlink a `documents.text.canonical` left by an interrupted purge; then, with a
   checkpoint in the live generation and no frames after it, map it and leave
   `documents.text` alone until the store is built from it; with frames after it, truncate and
   recreate `documents.text`, build the store from the checkpoint and replay those frames
   (CHECKPOINT.md); otherwise truncate and recreate `documents.text` and replay the journal;
   load index caches if fresh

So opening modifies: the lock file, `documents.text`, the end of the active segment, and the
manifest/schema when missing. `open_changes_only_what_the_spec_lists` in `tests/format_layout.rs` pins this list; a refused open modifies
nothing but the lock file.
