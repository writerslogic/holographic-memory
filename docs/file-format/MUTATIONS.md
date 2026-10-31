<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frame payloads: transactions

The decompressed payload of a plaintext frame (or the decrypted payload of an encrypted one)
is a transaction. Schema 2 writes this JSON form:

```
"HMS-TXN-2\0"  followed by  a JSON array of 1..=4096 mutations (UTF-8, no trailing bytes)
```

Schemas 3–6 write the `HMS-TXN-3` binary form described below; readers accept both forms.
A transaction is applied atomically: one frame, all or none. An unknown `HMS-TXN-<digits>\0` magic
is a different journal version: newer ones are refused with "newer than supported", older ones
(none were released) with the re-encode message. Any other prefix is refused.

Each mutation is a JSON object with exactly one key naming the variant (serde external tagging).

| Variant | Body | Replay effect |
|---|---|---|
| `AuditPending` | `{timestampMs, op, idHash, signature, effectId}` | schema 6: append a bounded, immutable audit intent to the delivery FIFO |
| `AuditAcknowledged` | `{id: [u8;16]}` | schema 6: remove the matching FIFO head after durable delivery |
| `Vector` | `{id, vector}` | set the vector `id`; drops meaning children derived from `id` |
| `Atom` | `{id, vector}` | meaning memory only |
| `Composite` | `{id, vector}` | set a composite vector |
| `StoredTriple` | `{subject_id, relation_id, object_id, composite_id, deleted}` | meaning memory |
| `Document` | `{id, source_uri, version, metadata, chunks}` | replace the document `id` with its chunks |
| `EncodeDocument` | `{id, version, fingerprint, vectors}` | install `vectors` (one per chunk) on the `pending` chunks of document `id` if it is still the revision they were computed for: same `version`, chunk count and content `fingerprint` (u64 over version, byte spans and chunk text); otherwise nothing |
| `Derived` | `{id, children}` | record meaning children of `id` |
| `DeleteDocument` | `{id}` | remove the document and its chunk vectors |
| `Delete` | `{id}` | remove the vector `id` and what derives from it |
| `DeleteAtom` | `{id}` | schema 5, meaning memory: remove only the atom; preserve raw vectors, composites and triples with the same id |
| `DeleteComposite` | `{id}` | schema 5, meaning memory: remove only the composite; preserve raw vectors and atoms with the same id; governance emits it only for an unowned composite |
| `RefreshMeaningIndices` | `{}` | schema 5, meaning memory: rebuild atom/composite postings and IDF weights from their live vectors |
| `Triplet` | `{id, subject, relation, object}` | meaning memory; needs `meaning_enabled` |
| `Relation` | `{sourceId, relationType, targetId, properties, validFrom, validTo}` | replace the edge |
| `RemoveRelation` | `{source, relation, target}` | remove the edge |
| `RelationType` | `{name, transitive, symmetric}` | declare a type |
| `Rule` | `{name, input_relations, output_relation}` | needs `meaning_enabled` |

Field naming is per struct (Relation, RelationType and AuditPending are camelCase, the rest snake_case);
treat the table as exact. Floats use shortest round-trip form (`0.0`).

`vector` is `{"dim": <store dimensions>, "indices": [<u32>...]}`: sorted strictly ascending,
each below `dim`.

A `Document` chunk:
`{id, text, start_byte, end_byte, vector, embedding, terms, word_count, stemmed}` where
`id` is `hms:chunk:<byte length of document id>:<document id>:<n>`, `text` is the chunk (or
absent when the document was stored without text), `[start_byte, end_byte)` index the original
document text with `end - start <= 65536` and the document `<= 8 MiB`, `text.len() == end -
start`, `embedding` is null or an array of the embedding-space length, `terms` is an object
term to count (stemmed lowercase tokens when `stemmed` is true), `word_count` the chunk's word
count. Documents have 1..=4096 chunks. A document `version` of the form `mem1:<json>` is a
memory-API stamp ([ANCILLARY.md](ANCILLARY.md)).

A chunk memorized with deferred or lexical-only encoding (`IngestOptions.encode`) also carries
`vector_state`: `"pending"` (its encoding will be installed by an `EncodeDocument`) or
`"lexical"` (never encoded). Such a chunk has an empty `vector` and is not in the vector
index. The field is absent when the vector is the chunk's encoding, so chunks written before
the flag existed read as encoded and a store without deferred or lexical documents serializes
exactly as before.

IDs are 1..=65535 bytes with no NUL. `hms:chunk:` and `hms:meaning:` prefixes are reserved.

## Replay

Frames are applied in file order, segment by segment. Each mutation is validated again on
replay with the limits above; a failure fails the open. Later records win: a second `Document`
with the same id replaces the first, a `Delete` removes the earlier `Vector`. The journal
keeps deleted data until compaction.

Scoped maintenance mutations require `meaning_enabled` and store schema 5. They use the
existing JSON mutation tag inside `HMS-TXN-3`; schemas 2–4 reject them on validation and
replay. A governance transaction applies atom changes, recomposes affected owned triples,
and refreshes derived indices atomically. An explicit `warm-format upgrade` changes the
schema in a new copy; normal opens never opt old stores into these mutations.

## Binary frames (`HMS-TXN-3`, schemas 3–5)

A store records its journal format in `store.json` `version`: 2 writes the JSON array above,
3 (new stores since 0.7.0) writes binary payloads; readers accept both in any store. Varints
are unsigned LEB128 (at most 10 bytes); `str` and `bytes` are a varint length then the bytes.

```
payload   = "HMS-TXN-3\0" varint(count) dictionary mutation{count}
dictionary= varint(n) { varint(shared) str(suffix) }{n}   terms strictly ascending (bytes),
            each = previous[..shared] + suffix, UTF-8
mutation  = 0x00 bytes(JSON of one mutation, as an element of the HMS-TXN-2 array)
          | 0x01 document
          | 0x02 str(id)                                   DeleteDocument
document  = str(id) (0x00 | 0x01 str(sourceUri)) str(version) bytes(metadata JSON)
            varint(chunks) chunk{chunks}
chunk     = flags [str(id)] [str(text)] varint(start_byte) varint(end_byte) varint(word_count)
            varint(dim) varint(k) varint(index gap){k} [varint(n) f32le{n}]
            varint(t) varint(term id gap){t} varint(frequency){t}
flags     = bit0 explicit chunk id (else hms:chunk:<len(id)>:<id>:<index>), bit1 text,
            bit2 stemmed, bit3 embedding, bits4-5 vector state (0 encoded, 1 pending,
            2 lexical); other bits zero
```

Gaps: the first value is absolute, each next one the difference to the previous (vector
indices and term ids strictly ascending). Term ids index the frame's dictionary, so a chunk's
terms come out in byte order. A binary payload decodes to exactly the mutations its
`HMS-TXN-2` form would hold; `warm-format` and `verify` read it through that JSON form.

## Durable audit delivery (schema 6)

An audited vector memorize/delete transaction carries its `AuditPending` in the same frame.
Compaction carries pending intents in FIFO order and appends its own intent to the staged
generation before publication. Audit-only mutations do not invalidate document checkpoints.
`op` is `"Memorize"`, `"Delete"`, or `"Compact"`; `idHash` is 32 bytes, `effectId` is 16 random
bytes, and `signature` is a pair of fixed 32-byte arrays. The pending FIFO has at most 1024
entries. Capacity, duplicate pending identities, and acknowledgement order are checked before
appending a transaction and again during replay. A full queue rejects an audited operation
before its business mutation commits.

One ordered delivery lease serializes delivery and acknowledgement, without holding the core
writer lease across sidecar I/O. The drainer syncs the audit record before committing its
acknowledgement; any error stops the drain. Only the last stream record can have been delivered
without acknowledgement, so matching that identity and payload safely deduplicates restart
recovery. A matching record is synced again before acknowledgement. `flush_audit()` drains the
queue through the tail captured when it acquires the delivery lease; new arrivals cannot make
that call run indefinitely. Open resumes pending delivery, including intents committed before
audit logging was disabled. Delivery errors fail opening rather than discard pending records.

This covers the existing audited vector memorize, vector delete, and compaction operations.
It does not add audit coverage for document/meaning APIs, recover provenance effects, or make
HTTP idempotency durable across restarts.
