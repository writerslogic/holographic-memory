<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Caches

Every file here can be deleted at any time; the engine rebuilds it from the journal. None is
part of the compatibility promise beyond "a cache that cannot be read is discarded".

## documents.text (text segment)

```
0  4  magic  "HMST"
4  1  version 2
5  1  flags   bit 0: records are encrypted (ENCRYPTION.md)
6  2  zero
8  ...  records, back to back, no framing
```

A record is a chunk's UTF-8 text (plaintext stores) or its sealed form. Locations
`(offset, length)` live only in memory, so the file cannot be parsed without replaying the
journal. The engine **truncates and recreates it on every open** and re-appends chunk text
during replay. A file without `HMST` (version 1: headerless plaintext) is overwritten the same
way. Deleted text therefore remains in the file only until the next open.

## indices.json

```json
{"checkpoint":["vectors_data.bin",4736],"shardCount":1,
 "shards":[{"ivf":false,"ivfCrc":null,"nsg":false,"nsgCrc":null}]}
```

`checkpoint` is the live generation directory name (`vectors_data.bin` or `gen-...`) and the
journal length (`segment_index * segment_size + end offset`). It is rewritten by every flush
(atomic write). On open the caches are used only if the checkpoint equals the arena's, the shard
count matches, and the file's CRC32 equals `nsgCrc`/`ivfCrc`; otherwise they are ignored or
discarded (a log line, not an error). A stale checkpoint is normal after any later append.
Invalid JSON is ignored.

## nsg_index.bin, ivf_index.bin

`nsg_index_<i>.bin` / `ivf_index_<i>.bin` per shard when `shardCount > 1`. `bincode` 2
(`standard` config, serde) of the index structure, sealed if the store is encrypted. The layout
is internal to the engine version and **not specified**: a file that fails its CRC32 or does not
decode is discarded and the index retrained. Written atomically, files before `indices.json`.

## shard_meta.json

`{"shard_count": N}`, written when a store auto-shards. Read on open to rebuild the shard
topology; a missing or malformed file means one shard.
