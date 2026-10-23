<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Crash consistency

Assumes a filesystem where rename is atomic and the sync calls below are honoured.
Each row names the test that exercises it; the tests interrupt processes, they do not cut power.

## When a returned commit is durable

A journal commit writes the frame and the end marker after it into the active segment's
mapping, then, before the call returns:

| Platform | Sync performed | Survives |
|---|---|---|
| Linux | `msync(MS_SYNC)` of the frame range. The kernel runs it as `fdatasync` of that range (`vfs_fsync_range`): the pages are written and, on filesystems with write barriers (the default for ext4, XFS, Btrfs), the drive cache is flushed. The segment's length was synced when it was created or grown. | process crash, kernel crash, power loss |
| macOS | `msync(MS_SYNC)` of the frame range, then `fcntl(F_FULLFSYNC)` on the segment file. `msync` and `fsync` alone leave the data in the drive's write cache. A volume that refuses `F_FULLFSYNC` gets `fsync` instead, which does not flush the drive cache there. | process crash, kernel crash, power loss (on volumes that support `F_FULLFSYNC`) |
| Windows | `FlushViewOfFile`, then `FlushFileBuffers` on the segment file. | process crash, OS crash, power loss, if the drive honours the flush |

`HmsCore::flush` performs the same sync over the whole active segment. Manifests, `CURRENT`,
`encryption.salt` and new segments are synced with `File::sync_all` (`F_FULLFSYNC` on macOS,
without the `fsync` fallback, so a volume that refuses `F_FULLFSYNC` normally fails store
creation before any commit).

## Per file

| Operation | Guarantee after a crash at any byte | Test |
|---|---|---|
| Journal append | The transaction is entirely present or entirely absent. A torn frame is cut on open; everything before it is intact. A commit that returned is durable as the table above describes (frame and end marker are synced together). | `tests/format_crash.rs::torn_append_at_every_byte` |
| Corruption vs tear | A bad frame followed by committed data fails open; it is never served as a shorter store. | `storage::tests::open_fails_on_corruption_inside_committed_data` |
| `store.json`, `format.json` | Written once, atomically. The sidecar lands first, so a crash between them reads as a mismatch, never a silent accept. | `storage::tests::manifest_checksum_mismatch_fails_open` |
| `indices.json` and caches | Cache files first, `indices.json` last, all atomic. No sidecar, so any crash leaves an openable store; stale or missing caches are rebuilt. | `tests/format_crash.rs::stale_indices_sidecar_is_ignored` |
| Compaction | A crash before `CURRENT` is replaced leaves the old generation; after, the new one. Both complete. Leftover `gen-*` is garbage. | `storage::tests::interrupted_compaction_recovers_a_complete_generation`, `layout_tests::legacy_compaction_tail_survives_each_plain_and_encrypted_publication_interruption` |
| Checkpoint compaction | The checkpoint is written atomically and synced inside the staging generation before `CURRENT` is replaced, so a crash at any publication step opens either the old generation (frames, or its own checkpoint plus frames) or the new checkpoint plus the frames replayed into it, with every acknowledged write. | `compaction::tests::checkpoint_compaction_survives_each_publication_interruption` |
| `encryption.salt` | Atomic and fsynced before the first frame. | `tests/format_crash.rs::salt_is_written_atomically` |
| Vector update batch | One CRC32 frame per durable batch; an incomplete final frame is truncated, a complete bad frame fails open. | `qgraph::update::tests::torn_last_batch_at_every_byte_and_checksum_corruption` |
| Vector consolidation | A synced checkpoint publishes a complete generation; its sequence suppresses old log replay before rotation. | `qgraph::update::tests::consolidation_interruption_recovers_committed_state` |
| `audit.bin` | Entry append is fsynced; a torn entry is cut on the next open. | `tests/format_crash.rs::audit_torn_entry_is_repaired` |
| `documents.text` | Not durable; rebuilt on every open and rewritten by a purge through `documents.text.canonical` (a leftover is scrubbed at open). | `golden` tests |
| Provenance files, `kel.json`, `shard_meta.json` | Plain or atomic single-file writes, no cross-file atomicity. **Not covered.** | |

Not guaranteed: durability on filesystems that reorder or lose fsync; two processes on one
store (the lock prevents it, and a network filesystem may not honour the lock); an encrypted
store whose salt file is lost.
