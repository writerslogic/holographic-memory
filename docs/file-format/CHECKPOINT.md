<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Document checkpoint (`documents.ckpt`)

Schema 4 stores (`store.json` `version` 4, the default for stores created by 0.7.0) may hold a
document checkpoint in a journal generation: `vectors_data.bin/gen-*/documents.ckpt`. It is the
store's data, not a cache. The generation is the checkpoint followed by the frames in its
segments, and the store's state is the checkpoint's documents with those frames applied in
order. The frames that came before the checkpoint are not kept: the term data is stored once.

## When one is written

A compaction (`HmsCore::compact`, `compact_scrubbing`, purge and forget) of a store that takes
checkpoints writes one in place of the document frames it would otherwise write. A store takes
one when checkpoints are enabled (`HmsConfig::document_checkpoint.enabled`, on by default), every
mutation applied since the open was a document write, encoding or delete, the vector shards are
empty, and every chunk is lexical-only (no vector, embedding or pending encoding) and holds no
word positions (`IngestOptions::positions`). Any other
store compacts to frames, as before, and its next generation has no checkpoint.

`flush` and closing the engine run such a compaction when the live generation holds at least
`document_checkpoint.min_journal_bytes` (256 KiB by default) of frames. Ingest closes into a single
checkpoint; later writes append frames after it until the next one folds them in.

## Opening

1. The live generation is the one `CURRENT` names (LAYOUT.md). No `documents.ckpt` there: the
   frames are replayed as in any store.
2. A checkpoint in a store whose schema is below 4 fails the open.
3. The checkpoint is checked whole before use: magic, generation binding (CRC32 of the
   generation directory's name), vector dimensions, CRC32 of every byte after the CRC word (or
   the AEAD tag, ENCRYPTION.md), and every section, offset, dictionary entry and slot owner. A
   checkpoint that fails any check fails the open: it is never skipped, since the frames it
   replaced are gone.
4. The file is mapped and unfiltered lexical queries without a query embedding rank from the
   mapping with the live store's scoring code, so the results are the live store's.
5. Document writes in every encode mode (eager, deferred, lexical), deferred encodings and
   document deletes go to a store over the checkpoint that holds the documents written since;
   a checkpoint document they replace or delete is marked dead. Queries rank the checkpoint's
   live slots and that store's together, with the document count, average length and document
   frequencies counted over live chunks only (dead slots are skipped while a posting list
   decodes). The checkpoint holds no vector state, so chunk vectors, embeddings and pending
   encodings belong to the documents written since: chunk-vector queries and deferred encoding
   read that store and the vector shards without building the live store. Frames after the
   checkpoint are replayed the same way at open, which rebuilds the dead set: the frames are
   its persistent record.
6. The first access of any other kind, and any other write, builds the live store from the
   checkpoint (no frame is read), removes the dead documents and installs the ones written
   since.

Building the live store re-encodes each posting list in the live encoding, restores every document and slot (free slots
included) as recorded, reads each chunk's terms back from the postings, and moves chunk text to
`documents.text`. The result answers every query exactly as the replay of the frames the
checkpoint replaced would: `tests/document_checkpoint.rs` compares hits, scores (as bits) and
records with a store that never checkpointed, through writes and reopens.

## Layout

All integers little endian; varints are LEB128.

```
0    8  magic "HMSCKP\0\x01"
8    8  u64: CRC32 of bytes 16.. (low 32 bits)
16   8  u64: CRC32 of the generation directory's name (low 32 bits)
24   8  u64: vector dimensions of the store
32   40 u64: reserved, zero
72   32 u64: slots, chunks, words, documents
104  16 u64: terms per posting form (unstemmed, stemmed)
120  160 (offset, length) of each section, in the order below; then reserved words to byte 328
328  sections, back to back, ending at the end of the file
```

Sections:

1. per posting form, a dictionary of the sorted terms in blocks of 32 front-coded entries
   (prefix length, suffix bytes, posting count, posting byte length) and a block table of `u32`
   pairs (the block's dictionary offset, the posting offset of its first term);
2. the posting lists of every term, form 0 then form 1 in dictionary order: per posting one
   varint `slot delta << 1 | (frequency != 1)`, followed by the frequency as a varint only when
   it is not 1 (the first delta is the slot itself);
3. per slot, `u32` word count, `u32` document ordinal (`u32::MAX` for a free slot) and `u32`
   chunk index;
4. document ids in id order: `documents + 1` `u32` offsets, then the UTF-8 bytes;
5. per document, `documents + 1` `u32` offsets, then a record: version, source URI (flag and
   bytes), metadata JSON, chunk count, and per chunk a flags byte (bit 0: the chunk id follows,
   otherwise it is the default id for its index; bit 1: its terms are stemmed), the byte span as
   two varints, and its text (`0` none, `2` inline bytes).

## Durability and crash safety

The checkpoint is written with the compaction's staging generation: temporary file, sync,
rename, directory sync, before the generation's segments are synced and before `CURRENT` is
replaced (CRASH-CONSISTENCY.md). A crash before the switch leaves the previous generation; after
it, the new generation, checkpoint included. A published checkpoint is never written again; the
next one goes to the next generation.

## Purge and forget

A delete is a frame after the checkpoint; the document's terms stay in the checkpoint until the
next compaction, as they stayed in the journal before checkpoints. A purge or forget runs a
scrubbing compaction before it returns: the store's slots and text segment are first laid out
as a store that never held the document lays them out, the new checkpoint is built from that
store, and the previous generation, checkpoint included, is overwritten with zeros, synced
and removed (`tests/document_checkpoint.rs::purge_leaves_no_trace_in_any_checkpoint`).
