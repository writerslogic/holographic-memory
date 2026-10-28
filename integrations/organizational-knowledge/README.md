# Organizational knowledge adapter (JavaScript)

Imports the flat curated documents produced by the ToIP EGWG prototype
([`mcp_data_generation`, step 4](https://github.com/trustoverip/egwg-reporting-tf-prototype-mcp))
into an HMS store through the existing document APIs (`DocumentMemory`,
`memorizeDocument`, `searchDocuments`, `deleteDocument`). It adds no retrieval,
chunking, embedding, or persistence code of its own. It is an adapter example, not
an agreed ToIP contract, a connector, or an MCP backend. The nested model proposed
in upstream issue #8 is rejected, not converted.

Zero dependencies. Needs Node 20+ and the native binding of this checkout
(`holographic-memory` resolves by package self-reference from inside the checkout;
elsewhere, install it). It uses the same document ID, metadata, chunking, and dimensions
as `examples/organizational_knowledge.rs`, so one store can be written by either tool
(see Interoperability).

## Commands

Run from the HMS checkout root. Use a dedicated store directory; it is created if missing.

```sh
node integrations/organizational-knowledge/example/cli.cjs import ./toip-js-store toip integrations/organizational-knowledge/example/fixtures
node integrations/organizational-knowledge/example/cli.cjs search ./toip-js-store toip quartzmarker --k 5
node integrations/organizational-knowledge/example/cli.cjs forget ./toip-js-store toip Zoom zoom-example-project-chat-GMT20260101-100000
node integrations/organizational-knowledge/example/cli.cjs forget ./toip-js-store toip Zoom zoom-example-project-chat-GMT20260101-100000 --apply
```

`import` takes files and directories (each `.json` directly inside, not recursive; one
document per file, at most 16 MiB). The fixtures are synthetic. `forget` without
`--apply` prints the document ID and does not open the store.

```js
const { HolographicMemorySystem } = require('holographic-memory')
const { OrganizationalKnowledge, readCuratedFiles } = require('./integrations/organizational-knowledge/index.cjs')

async function main() {
  const knowledge = new OrganizationalKnowledge(new HolographicMemorySystem(4096, './toip-js-store'), { namespace: 'toip' })
  await knowledge.importDocuments(await readCuratedFiles(['/path/to/data/curated']))
  const hits = await knowledge.search('governance reporting', { k: 5, filter: { content_type: 'Transcript' } })
  await knowledge.forget([{ sourceSystem: 'Zoom', id: 'zoom-hiero-ai-summary-GMT20260407-135959' }], { apply: true })
  return hits
}
main().then(console.log, error => { console.error(error); process.exitCode = 1 })
```

Types are in `index.d.cts`. An optional `DocumentMemory` `embedder` and `rerank` can be passed
to the constructor; the default is local lexical retrieval with no model.

## Mapping

| Curated field | HMS |
|---|---|
| namespace (option) + `source_system` + `id` | Document ID `JSON.stringify([namespace, source_system, id])`, at most 1024 bytes |
| `identifier` (SAID) | Document `version`, verbatim, 1 to 1024 bytes |
| `content` | Text, unmodified; 500-word chunks, 50-word overlap; chunk text stored |
| `source_url` | `sourceUri` when non-empty |
| every other field | Metadata, unmodified (`organizational_unit`, `project`, `source_identifier`, `identifier_type`, `collected_at`, ...) |
| namespace | Metadata `hms_namespace` (reserved) |

The stable identity uses the curated `id`, not `source_identifier`: one Zoom recording
yields a Transcript, Chat, ClosedCaptioning, and AI Summary that share `source_identifier`.
The identity does not depend on the SAID, so a revised document replaces the earlier one
instead of accumulating. Hit offsets are UTF-8 byte offsets into the exact curated `content`.

## Behavior

- **Validation before writes.** Every document is checked, and duplicate identities in one
  import are rejected (naming both sources), before the first write. Rejected: missing or
  blank `id`, `source_system`, `identifier`; non-string `content`; whitespace-only content;
  `organizational_unit` that is not an array of strings; non-string optional fields;
  control characters in identity parts; lone surrogates anywhere (the engine would silently
  rewrite them); a `__proto__` key anywhere (the binding returns it as a prototype, not data);
  `source_url` that is not an absolute http(s) URL without credentials, whitespace, or control
  characters; `hms_namespace` in the input; metadata over 64 KiB; the nested model; and the
  engine's chunking limits (4096 chunks, 64 KiB per chunk, and with an `embedder` 1,048,576
  embedding values per document). The CLI validates adapter rules before opening the store,
  so malformed input does not create one. The engine's chunking limits can only be checked on
  an open store: they are checked before the first write, but a document that violates them
  leaves an empty store behind.
- **Bounded input.** At most 10,000 documents per import; files are at most 16 MiB each and
  512 MiB in total. Split larger syncs; identities are independent, so this is safe.
- **Overlapping calls.** Writes (`importDocuments`, `forget`) on one instance run one after
  another in call order, so the last call for an identity wins.
- **Replacement.** Importing the same namespace, `source_system`, and `id` replaces the
  document in one atomic commit. Order is the caller's: the last import wins.
- **Not one transaction.** Each document commits atomically, but an import is not atomic
  across documents. A write failure throws `partial_failure` listing `committed` IDs;
  re-running the import is idempotent.
- **Forget.** Explicit `{ sourceSystem, id }` targets inside the namespace; no wildcard.
  Without `apply: true` it only reports IDs. Repeating is safe (`absent` lists missing ones).
- **Reimport restores.** Importing a forgotten document again makes it searchable again.
  Forgetting is not durable against a later sync from an upstream that still holds it.

## Limitations

- Namespace filtering is a query filter, not authorization. Anyone who can open the store or
  call the engine API can read every namespace. Use separate stores or encryption for isolation.
- Forget removes the document from HMS retrieval. It does not claim physical erasure of
  journals, page cache, backups, snapshots, or model inferences, and it does not remove
  anything upstream. Upstream deletion is not detected or propagated.
- The node binding has no document listing, and an empty query returns nothing, so there is
  no namespace-wide or source-system-wide forget, and no prune of documents upstream dropped.
- `identifier` is stored, not verified: SAIDs are not checked and are not authentication.
- Speaker identities and recording timestamps are not inferred; offsets refer to curated text.
- A `filter` is an exact match on a top-level metadata value: `organizational_unit` must
  equal the whole array.
- One writer per store directory.
- JSON numbers are IEEE doubles: integers above 2^53 in extra metadata fields are not
  preserved exactly (the Rust example keeps them). The curated schema has no numeric fields.
- No retrieval-quality, speed, or benchmark claim is made.

## Interoperability

The Rust example reads and deletes what this adapter wrote, and the reverse, one writer at a
time. Stricter here: this adapter also rejects non-string optional fields, non-http(s)
`source_url`, `hms_namespace` in the input, control characters in identity parts, and lone
surrogates, which the Rust example accepts.

```sh
node integrations/organizational-knowledge/example/cli.cjs import ./toip-js-store toip integrations/organizational-knowledge/example/fixtures
cargo run --locked --example organizational_knowledge -- search ./toip-js-store toip quartzmarker
cargo run --locked --example organizational_knowledge -- forget ./toip-js-store toip Zoom zoom-example-project-chat-GMT20260101-100000 --apply
node integrations/organizational-knowledge/example/cli.cjs search ./toip-js-store toip "reporting profile"
```

This sequence was run once against a debug build and the synthetic fixtures; it is not part of
the automated tests.

## Tests

```sh
node --test integrations/organizational-knowledge/test/*.test.cjs
```

`document.test.cjs` needs no engine. `knowledge.test.cjs` uses the real native binding,
temporary stores, and child processes for reopen.
