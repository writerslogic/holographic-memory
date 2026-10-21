<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Frame payloads: transactions

The decompressed payload of a plaintext frame (or the decrypted payload of an encrypted one) is

```
"HMS-TXN-2\0"  followed by  a JSON array of 1..=4096 mutations (UTF-8, no trailing bytes)
```

A transaction is applied atomically: one frame, all or none. Another `HMS-TXN-<digits>\0` magic
is a different journal version: newer ones are refused with "newer than supported", older ones
(none were released) with the re-encode message. Any other prefix is refused.

Each mutation is a JSON object with exactly one key naming the variant (serde external tagging).

| Variant | Body | Replay effect |
|---|---|---|
| `Vector` | `{id, vector}` | set the vector `id`; drops meaning children derived from `id` |
| `Atom` | `{id, vector}` | meaning memory only |
| `Composite` | `{id, vector}` | set a composite vector |
| `StoredTriple` | `{subject_id, relation_id, object_id, composite_id, deleted}` | meaning memory |
| `Document` | `{id, source_uri, version, metadata, chunks}` | replace the document `id` with its chunks |
| `EncodeDocument` | `{id, version, fingerprint, vectors}` | install `vectors` (one per chunk) on the `pending` chunks of document `id` if it is still the revision they were computed for: same `version`, chunk count and content `fingerprint` (u64 over version, byte spans and chunk text); otherwise nothing |
| `Derived` | `{id, children}` | record meaning children of `id` |
| `DeleteDocument` | `{id}` | remove the document and its chunk vectors |
| `Delete` | `{id}` | remove the vector `id` and what derives from it |
| `Triplet` | `{id, subject, relation, object}` | meaning memory; needs `meaning_enabled` |
| `Relation` | `{sourceId, relationType, targetId, properties, validFrom, validTo}` | replace the edge |
| `RemoveRelation` | `{source, relation, target}` | remove the edge |
| `RelationType` | `{name, transitive, symmetric}` | declare a type |
| `Rule` | `{name, input_relations, output_relation}` | needs `meaning_enabled` |

Field naming is per struct (Relation and RelationType are camelCase, the rest snake_case);
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
