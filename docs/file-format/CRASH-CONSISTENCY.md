<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Crash consistency

Assumes a filesystem where `fsync`/`msync` and rename are atomic and ordered as POSIX describes.
Each row names the test that exercises it.

| Operation | Guarantee after a crash at any byte | Test |
|---|---|---|
| Journal append | The transaction is entirely present or entirely absent. A torn frame is cut on open; everything before it is intact. A commit that returned is durable (frame and end marker are flushed together). | `tests/format_crash.rs::torn_append_at_every_byte` |
| Corruption vs tear | A bad frame followed by committed data fails open; it is never served as a shorter store. | `storage::tests::open_fails_on_corruption_inside_committed_data` |
| `store.json`, `format.json` | Written once, atomically. The sidecar lands first, so a crash between them reads as a mismatch, never a silent accept. | `storage::tests::manifest_checksum_mismatch_fails_open` |
| `indices.json` and caches | Cache files first, `indices.json` last, all atomic. No sidecar, so any crash leaves an openable store; stale or missing caches are rebuilt. | `tests/format_crash.rs::stale_indices_sidecar_is_ignored` |
| Compaction | A crash before `CURRENT` is replaced leaves the old generation; after, the new one. Both complete. Leftover `gen-*` is garbage. | `storage::tests::interrupted_compaction_recovers_a_complete_generation`, `layout_tests::legacy_compaction_tail_survives_each_plain_and_encrypted_publication_interruption` |
| `encryption.salt` | Atomic and fsynced before the first frame. | `tests/format_crash.rs::salt_is_written_atomically` |
| `audit.bin` | Entry append is fsynced; a torn entry is cut on the next open. | `tests/format_crash.rs::audit_torn_entry_is_repaired` |
| `documents.text` | Not durable; rebuilt on every open. | `golden` tests |
| Provenance files, `kel.json`, `shard_meta.json` | Plain or atomic single-file writes, no cross-file atomicity. **Not covered.** | |

Not guaranteed: durability on filesystems that reorder or lose fsync; two processes on one
store (the lock prevents it, and a network filesystem may not honour the lock); an encrypted
store whose salt file is lost.
