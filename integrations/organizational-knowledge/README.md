# Organizational knowledge adapter (JavaScript)

Imports the flat curated documents produced by the ToIP EGWG prototype
([`mcp_data_generation`, step 4](https://github.com/trustoverip/egwg-reporting-tf-prototype-mcp))
into an WARM store through the existing document APIs (`DocumentMemory`,
`memorizeDocument`, `searchDocuments`, `deleteDocument`). It adds no retrieval,
chunking, embedding, or persistence code of its own. It is an adapter example, not
an agreed ToIP contract, a connector, or an MCP backend. The nested model proposed
in upstream issue #8 is rejected, not converted.

Zero dependencies. Needs Node 20+ and the native binding of this checkout
(`writerslogic-warm` resolves by package self-reference from inside the checkout;
elsewhere, install it). It uses the same document ID, metadata, chunking, and dimensions
as `examples/organizational_knowledge.rs`, so one store can be written by either tool
(see Interoperability).

## Commands

Run from the WARM checkout root. Use a dedicated store directory; it is created if missing.

```sh
node integrations/organizational-knowledge/example/cli.cjs import ./toip-js-store toip integrations/organizational-knowledge/example/fixtures
node integrations/organizational-knowledge/example/cli.cjs search ./toip-js-store toip quartzmarker --k 5
node integrations/organizational-knowledge/example/cli.cjs forget ./toip-js-store toip Zoom zoom-example-project-chat-GMT20260101-100000
node integrations/organizational-knowledge/example/cli.cjs forget ./toip-js-store toip Zoom zoom-example-project-chat-GMT20260101-100000 --apply
node integrations/organizational-knowledge/example/cli.cjs forget-all ./toip-js-store toip Zoom              # preview a source system
node integrations/organizational-knowledge/example/cli.cjs forget-all ./toip-js-store toip --all --apply
node integrations/organizational-knowledge/example/cli.cjs prune ./toip-js-store toip Zoom ./data/curated --apply
```

`import` takes files and directories (each `.json` directly inside, not recursive; one
document per file, at most 16 MiB). The fixtures are synthetic. `forget` without
`--apply` prints the document ID and does not open the store. `forget-all` and `prune` are
previews unless `--apply` is given, take `--limit N` (default 10,000 per run), and, like `search`
and `forget --apply`, do not create a store: a path without a `store.json` is treated as missing. `prune SOURCE_SYSTEM PATH...` keeps the SOURCE_SYSTEM documents found in PATH and
removes the rest of that source system's documents in the namespace; it refuses when PATH holds none.

```js
const { WarmMemorySystem } = require('writerslogic-warm')
const { OrganizationalKnowledge, readCuratedFiles } = require('./integrations/organizational-knowledge/index.cjs')

async function main() {
  const knowledge = new OrganizationalKnowledge(new WarmMemorySystem(4096, './toip-js-store'), { namespace: 'toip' })
  await knowledge.importDocuments(await readCuratedFiles(['/path/to/data/curated']))
  const hits = await knowledge.search('governance reporting', { k: 5, filter: { content_type: 'Transcript' } })
  await knowledge.forget([{ sourceSystem: 'Zoom', id: 'zoom-hiero-ai-summary-GMT20260407-135959' }], { apply: true })
  return hits
}
main().then(console.log, error => { console.error(error); process.exitCode = 1 })
```

Types are in `index.d.cts`. An optional `DocumentMemory` `embedder` and `rerank` can be passed
to the constructor; the default is local lexical retrieval with no model. The CLI does not
take an embedder. With a local ONNX model (needs `@huggingface/transformers`, an optional peer
dependency this checkout does not install, and a model directory holding `onnx/model.onnx` plus
the tokenizer files), the store must be created with the embedder's embedding space:

```js
const { createLocalEmbedder } = require('writerslogic-warm/semantic')

const embedder = await createLocalEmbedder({ modelPath: '/path/to/model-dir', modelId: 'BAAI/bge-small-en-v1.5', revision: 'local' })
const { model, revision, dimensions } = embedder.space
const memory = new WarmMemorySystem(4096, './toip-js-store', { embeddingModel: model, embeddingRevision: revision, embeddingDimensions: dimensions })
const knowledge = new OrganizationalKnowledge(memory, { namespace: 'toip', embedder })
const hits = await knowledge.search('governance reporting', { lexicalWeight: 0, semanticWeight: 1 })
```

## Mapping

| Curated field | WARM |
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
  embedding values per document). The CLI validates everything, including the engine's chunking
  limits on a throwaway store in the system temp directory, before it opens STORE, so rejected
  input does not create it. A library caller's `importDocuments` checks the same limits on its
  own open store before the first write.
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
- **Forget by scope.** `forgetScope({ sourceSystem | all: true, keep? }, { apply, limit })` removes
  a source system's documents, or the namespace's, except an optional nonempty `keep` list of
  `{ sourceSystem, id }` (a prune: pass the identities the upstream still holds). It lists the store
  through `listDocuments`, matches only documents whose identity and `hms_namespace` both belong to
  this namespace, previews unless `apply: true`, and handles at most `limit` documents per call;
  `truncated: true` means more match, so apply and repeat (a preview shows the same first `limit`
  each time). Options other than `apply` and `limit` are rejected: `keep` belongs in the scope. It throws `unsupported_binding` when the loaded
  binding has no `listDocuments`. A prune is only as complete as the `keep` list: documents missing
  from it are removed, so a partial export of the upstream removes the rest.
- **Reimport restores.** Importing a forgotten document again makes it searchable again.
  Forgetting is not durable against a later sync from an upstream that still holds it.

## Limitations

- Namespace filtering is a query filter, not authorization. Anyone who can open the store or
  call the engine API can read every namespace. Use separate stores or encryption for isolation.
- Forget removes the document from WARM retrieval. It does not claim physical erasure of
  journals, page cache, backups, snapshots, or model inferences, and it does not remove
  anything upstream. Upstream deletion is not detected or propagated.
- Namespace-wide forget and prune need the binding's `listDocuments`, added alongside this adapter
  in `src/bindings/node`; a binding built before that has none, and `forgetScope` then fails with
  `unsupported_binding`. Rebuild with `npm run build`. Upstream deletions are still not detected:
  pruning needs a complete list of what the upstream holds.
- `identifier` is stored, not verified: SAIDs are not checked and are not authentication.
- Speaker identities and recording timestamps are not inferred; offsets refer to curated text.
- A `filter` is an exact match on a top-level metadata value: `organizational_unit` must
  equal the whole array.
- One writer per store directory.
- JSON numbers are IEEE doubles: integers above 2^53 in extra metadata fields are not
  preserved exactly. The curated schema has no numeric fields.
- No retrieval-quality, speed, or benchmark claim is made.

## Interoperability

The Rust example reads and deletes what this adapter wrote, and the reverse, one writer at a
time. Differences checked against the Rust example binary: this adapter rejects non-string
optional fields (`title: 12`, `title: null`) and a `javascript:` `source_url`, which the Rust
example imports; it reads files up to 16 MiB where the Rust example reads 8 MiB; and it stores
integers above 2^53 in extra metadata fields as rounded doubles where the Rust example keeps
them exactly. Both reject control characters in `id`, a reserved `hms_namespace` key, and
lone surrogates.

```sh
node integrations/organizational-knowledge/example/cli.cjs import ./toip-js-store toip integrations/organizational-knowledge/example/fixtures
cargo run --locked --example organizational_knowledge -- search ./toip-js-store toip quartzmarker
cargo run --locked --example organizational_knowledge -- forget ./toip-js-store toip Zoom zoom-example-project-chat-GMT20260101-100000 --apply
node integrations/organizational-knowledge/example/cli.cjs search ./toip-js-store toip "reporting profile"
```

This sequence was run against a debug build and the synthetic fixtures. The automated
tests do not call cargo; `tests/organizational_knowledge_cli.test.cjs` covers the cross-tool path.

## Tests

```sh
node --test integrations/organizational-knowledge/test/*.test.cjs
```

`document.test.cjs` needs no engine. `knowledge.test.cjs` uses the real native binding,
temporary stores, and child processes for reopen; the `forgetScope` and `forget-all` tests are
skipped when the binding has no `listDocuments`. The binding's own test is
`tests/node/list-documents.test.cjs`. Last run on Node 20.20.2 and 26.8.1, macOS arm64
only, against the checkout's binding and a binding built from the 5ccca1e4 source.

`embedder.test.cjs` is skipped unless a local ONNX model directory is given:

```sh
WARM_TEST_ONNX_MODEL_DIR=/path/to/model-dir node --test integrations/organizational-knowledge/test/embedder.test.cjs
```

It checks only that a real embedder's vectors are stored and searched through the adapter
(three synthetic documents, one model) and that forget removes the document from semantic
results. It ran once against `BAAI/bge-small-en-v1.5` (384 dimensions), exported to ONNX locally
from cached weights with optimum, so it is not an official ONNX release. It says nothing about
retrieval quality.
