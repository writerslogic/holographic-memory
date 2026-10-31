# Source: faiss

`warm-import faiss --index FILE [--id-map F] [--texts F] [--metadata F] [--namespace NS]`

Reads a FAISS index written with `faiss.write_index`. FAISS stores vectors and integer ids only, so text, string ids and metadata come from sidecar files you supply. Verified against real `faiss-cpu` 1.15 (Flat, HNSWFlat, IDMap2, IVFFlat); no mocks.

## What is read

| Index type | Vectors | Ids |
|---|---|---|
| Flat, HNSWFlat and other types with `reconstruct_n` | in storage order, batches of 1024 | row number |
| `IndexIDMap` / `IndexIDMap2` | the wrapped index, same order | the map's own int64 ids |
| `IndexIVF*` with stored vectors | list by list via `reconstruct_from_offset` | the ids added with `add_with_ids` (sequential otherwise) |

Quantized indexes (PQ, SQ, OPQ pre-transforms) reconstruct lossy vectors; the import carries those approximations, not the original floats. Indexes that cannot reconstruct (for example IVF-PQ without stored codes access) fail with a message naming the index type; nothing is partially imported.

## Sidecars

All sidecars are row-aligned with the index storage order and streamed in lockstep (bounded memory). A sidecar with fewer or more rows than the index is a fatal error, never a silent truncation.

- `--id-map`: `.npy`, `.txt`, `.jsonl` or a `.json` list (one external id per row), or a `.json` object `{"<faiss id>": "<external id>"}` that renames native ids.
- `--texts`: `.jsonl` (a JSON string per line, or `{"text": ...}`) or `.txt` (one per line, no newlines inside a text).
- `--metadata`: `.jsonl`, one object per row. `--created-at-key` / `--updated-at-key` name metadata keys copied into the WARM timestamps (the originals stay in metadata).
- `--namespace`: one namespace for every row.

IVF indexes are enumerated list by list, which has no row order, so row-aligned `--id-map` lists, `--texts` and `--metadata` are refused for IVF; use an object id map or rebuild as Flat/IDMap. An `IndexIDMap` wrapped around an IVF index is not supported.

## Resume

Seekable: the cursor is the storage ordinal. `--mmap` memory-maps flat-style files.

## Credentials

None.
