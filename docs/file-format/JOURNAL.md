<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The journal (arena segments)

The journal is the store. Everything else is derived from it or is a cache. It is an append-only
sequence of frames across `seg_0.bin`, `seg_1.bin`, ... (numbered contiguously from 0).

## Frame

```
offset  size  field
0       4     crc32     CRC32 of the DECOMPRESSED payload
4       4     raw_len   decompressed length, 1..=52428800 (50 MiB)
8       4     comp_len  stored length, 1..=raw_len
12      4     seq       writer counter (advisory)
16      comp_len        payload: LZ4 block if comp_len < raw_len, else the raw bytes
```

- LZ4 is the raw block format (no size prefix, no frame); `raw_len` is the exact output size.
  A block never expands more than 255 times, so `raw_len <= 255 * comp_len` for compressed frames.
- The writer compresses and stores compressed only if that is strictly smaller.
- `seq` is `0` for the first frame of a new arena and `max(seq in the active segment) + 1` after
  a reopen. It is not unique across segments and not a format version. Readers ignore it.
- Frames are not aligned and not padded. The next frame starts at `16 + comp_len`.

## Segment end

After each frame the writer puts a 16-byte end marker in the same flush: `crc=0, raw_len=0,
comp_len=0, seq=0x454E444C` (`"LDNE"` little-endian). A header with `raw_len == comp_len == 0`
is the end of the log if it is the marker or all 16 bytes are zero (the form before markers).
Anything else with both lengths zero is invalid. Fewer than 16 bytes left also ends a segment.

A frame is never split across segments. When it does not fit in `segment_size` the writer moves
to the next file and leaves the rest of the old one as is. A reader at the end of a non-final
segment (marker, zeros or fewer than 16 bytes) continues at offset 0 of the next.

## Segment file sizes

Format 2: exactly 1 GiB. Format 3: exactly 64 MiB. Format 4: the final segment is a multiple
of 64 MiB, at least 64 MiB, at most 1 GiB, and grows in 64 MiB steps (every earlier segment is
1 GiB; on Windows the engine allocates the full size). Bytes beyond the end marker are zero. A
shorter file (a trimmed copy) is extended with zeros on open; a longer file is refused.
Global journal offsets are `segment_index * segment_size + offset` (the `checkpoint`
in [CACHES.md](CACHES.md)).

## Recovery on open (final segment only)

Walk frames from offset 0. At a frame that fails any check (lengths, bounds, LZ4, CRC32):

- if no valid frame or end marker exists anywhere after it (scanning to the last non-zero byte),
  it is a **torn tail**: an append the crash interrupted. The log ends at that frame's offset,
  an end marker is written there, and appending continues;
- otherwise it is **corruption inside committed data**: open fails and the damage is left in
  place. The writer flushes frame and marker together, so a valid frame after a bad one proves
  the bad one was committed.

Non-final segments are not re-walked at open; a bad frame there fails when replay reads it.
Nothing is skipped or repaired silently.

## Compaction

Compaction writes the live state into a new directory `gen-<pid>-<nanos>/` (same geometry as the
live arena: a format-2 store compacts to format 2), replays frames appended meanwhile, then
atomically replaces `CURRENT`. Old generation directories are removed afterwards; a leftover
`gen-*` that `CURRENT` does not name is garbage from an interrupted run and may be deleted.
Frames written after a compaction go to the new generation. Compaction does not change the
format version.

Compaction frames contain document text inline (it is resolved from the text segment), so a
compacted journal is self-contained.
