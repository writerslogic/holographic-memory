<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Persisted vector updates, version 1

## Structure and visibility

`UpdatableVGraph` owns a directory containing an immutable `HMSVGR01` vertex
index, an immutable full-precision row file, a checkpoint manifest, and an
append-only update log. Existing standalone vertex files remain readable and
unchanged. Creating an updatable index requires the original vectors: quantized
vertex codes cannot reconstruct them without accumulating error on rebuilds.
External ids are `u32`, excluding `u32::MAX`; dense graph slots are private.

A mutable delta uses the construction implementation's greedy search and Vamana
robust prune, including reciprocal links. Slots are never reused before
consolidation. Base deletions use a bitmap; delta deletions retain routing slots
but remove their external-id lookup. Replace invalidates the previous slot and
inserts a new one in one operation. One id may occur only once between commits.
Updates become visible when their in-memory operation completes; durability
requires `commit`. Dropping without committing discards the pending batch.

With no updates, search calls the existing vertex search unchanged, with reused
per-thread scratch. With updates it widens the base beam and candidate count by
the inverse live fraction, preserving routing through deleted slots,
filters deleted answers, doubles the beam and retries when too few live candidates
remain, searches the delta with the same compensation, and merges candidates using original
full-precision distances with external-id tie breaking. Original rows stay mapped
and are touched only for candidate reranking and consolidation. `k` is clamped
to the live count before allocation. Empty indexes are represented without a
vertex file because the standalone vertex format requires a nonempty graph.

[MERIT](https://arxiv.org/html/2607.29173v1), sections 3–4, motivates bounded
reverse-neighbor discovery, local MST repair, and versioned edge invalidation.
This design retains deleted slots as routing vertices instead of breaking those
paths. It therefore needs neither reverse edges nor versioned base edges. Local
base repair would add an overlay lookup to the serving loop and duplicate topology
ownership; it is deferred pending measurements of sustained high deletion rates.
This is an engineering tradeoff, not a claim of MERIT-equivalent update costs.
Consolidation removes routing tombstones. The default advisory policy recommends
it at 10% allocated delta slots or 20% base tombstones, with a minimum 1024 delta
slots; thresholds are configurable. Commit never hides a rebuild in its latency.

## Ownership and synchronization

A directory lock excludes other writable opens, including other processes.
An atomic writer permit serializes insert/delete/replace/commit/consolidate;
concurrent writers receive `WriterBusy`. A reader/writer lock protects only
in-memory state. No lock guard is held during file I/O. Searches borrow one
consistent state for their CPU work; consolidation snapshots the state, releases
the guard, builds and writes files, and publishes the new mapped base with a
short write guard. Readers continue on the old state while rebuilding.
An ambiguous append, sync, or checkpoint publication failure requires reopening
before further writes. Searches can continue to observe the in-memory state.

## Publication and recovery

Each committed batch is one length-prefixed CRC32 frame. A frame is validated in
full, including unique ids and operation preconditions, before any operation is
applied. Replay uses the same insertion order and pruning as live updates.
Incomplete final headers or payloads are truncated to the previous frame boundary
on open; a complete frame with a bad checksum is a typed corruption error, even
at EOF. A batch gets one durable file sync using `durable_file` after append.
The directory entry for the log is durable before updates can be accepted.

Consolidation first commits any pending batch, then writes fresh generation
files named by the committed sequence. Files are synced before publication.
The manifest is atomically replaced and its parent directory synced. Its
checkpoint sequence makes old frames at or below that sequence redundant.
Only afterwards is the log atomically rotated to an empty log whose origin is
the checkpoint. Thus interruption before publication replays against the old
base; interruption after publication opens the new base and skips the old prefix.
A torn rotation exposes either complete log. Old generation files are removed
only after publication and rotation; mappings held by readers remain valid.

## Binary contract

All integers and floats are little-endian. Every header contains the byte-order
marker `0x0102030405060708`; big-endian hosts refuse these files. CRC32 is IEEE.
Unknown versions are refused as newer than supported, never interpreted as v1.
Limits are validated before allocation: dimensions 1..=2^20, ids below 2^32-1,
frame payload at most 64 MiB and at most 65536 operations. Non-finite vectors and
squared norms that overflow f32 are rejected. Checked arithmetic bounds lengths.

* `CURRENT`: `HMSVUP01`, order (u64), identity (16 bytes), sixteen u64 words,
  CRC32 of all preceding bytes. Words: dimension, metric (0 angular, 1 Euclidean),
  checkpoint sequence, degree, build ef, alpha f32 bits, seed, quantized-search
  flag, vertex bits, residual bits, aligned flag, reordered flag, id bytes,
  screen flag, block bits, reserved zero.
* `rows-<sequence>.bin`: `HMSVRW01`, order, dimension, count, metric, sequence,
  row offset (all u64), header CRC32 (u32), reserved zero (u32): 64 bytes.
  Sorted external ids (`count` u32) follow. Rows start at the next 64-byte boundary;
  each is `dimension` f32 values followed by its CRC32. The header and id ordering
  are validated at open; row checksums and finite values are validated on access.
* `updates.log`: `HMSVLG01`, order, identity (16 bytes), dimension, metric,
  origin sequence (u64), header CRC32: 60 bytes. The identity binds it to CURRENT.
* Batch frame: payload length u32, its bitwise complement u32, sequence u64,
  operation count u32, CRC32 u32, payload. The checksum covers the first 20 header
  bytes and payload. Sequences start at origin + 1 and increase exactly by one.
  Records: tag u8 (1 insert, 2 delete, 3 replace), external id u32; insert/replace
  additionally carry exactly `dimension` f32 values. No trailing bytes allowed.

The vertex format stays version 1. Golden vector directories and byte drift tests
cover CURRENT, row files and update logs independently of text-store formats.
Python and Node bindings are follow-up work.

## Verification and measurements

Acceptance includes randomized update sequences against exact kNN and a rebuilt
base, bytewise torn-tail recovery, checksum rejection, reopen equivalence,
consolidation interruption points, 50k-vector recall comparison, and concurrent
readers. Measurements and exact reproduction commands are recorded here as they
are completed; SIFT-1M timing runs only on Linux under the shared timing lock.
