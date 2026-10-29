# Organizational knowledge across applications

HMS can index normalized documents from multiple source systems without owning
their collection or normalization. The runnable adapter below targets the flat
curated JSON documents currently produced by
[the ToIP EGWG prototype](https://github.com/trustoverip/egwg-reporting-tf-prototype-mcp).
It is an integration example, not an agreed ToIP contract or an installed MCP
backend. The proposed nested model in issue #8 is not supported by this example.

## Import and retrieve

Run from the HMS checkout. Use a dedicated store directory; the example uses the
default unencrypted configuration. It performs local lexical retrieval without
model downloads, API keys, or paid services.

```sh
cargo run --locked --example organizational_knowledge -- import ./toip-example toip /absolute/path/to/curated/document.json
cargo run --locked --example organizational_knowledge -- search ./toip-example toip "governance reporting"
```

Import one file per invocation. Required fields are `id`, `source_system`,
`identifier`, `content`, and a string array `organizational_unit`. Other fields
are preserved in metadata. Input files are bounded to 8 MiB; engine limits still
apply to chunk count, document text, and metadata. This example is not a bulk
synchronizer: a multi-file import is not one transaction.
The adapter rejects upstream `hms_namespace` fields because that name is reserved
for query scoping. Identity components must be nonempty and contain no control
characters. Schema and chunking validation finish before the import opens a store.

| Upstream field | HMS representation |
|---|---|
| Namespace + source system + stable `id` | Collision-safe JSON tuple as document ID |
| `identifier` | Document version, preserved verbatim |
| `content` | Original text, chunked at 500 words with 50-word overlap |
| `source_url` | Source URI, when present |
| All fields except `content` | Metadata, including original IDs and provenance fields |

The adapter does not verify SAIDs or treat their presence as authentication.
Stable source identity is required separately from the content identifier, so a
new version replaces the same document instead of accumulating stale versions.
Different applications can reuse an ID without overwriting one another. Namespace
filtering scopes this example's queries; it is not an authorization boundary.

Retrieved passages include UTF-8 byte offsets into the exact imported `content`,
source URI, version, and metadata. These offsets refer to curated text, not the
original recording or pre-normalization artifact. Speaker attribution and
recording timestamps require upstream span mappings; the adapter does not infer
them. It does not treat copies or summaries as independent corroboration.

## Replace and forget

Reimport a document with the same namespace, source system, and stable ID to
replace it. The supplied version is retained; upstream is responsible for
delivering versions in the right order. This is current-state document indexing,
not the memory API's supersession history.

```sh
cargo run --locked --example organizational_knowledge -- forget ./toip-example toip Zoom meeting-1
cargo run --locked --example organizational_knowledge -- forget ./toip-example toip Zoom meeting-1 --apply
```

The first command previews the selected ID without opening the store. `--apply`
removes its chunks from HMS retrieval; repeating it is safe. Remove or suppress
the upstream document as well: importing it again intentionally restores it.
This operation does not erase upstream artifacts, backups, model inferences, or
filesystem snapshots, and is not a physical-erasure guarantee. See the memory
API's [forgetting and purge contract](../memory-api/README.md#forgetting) for the
separate fact-store operations and their limits. Use one writer per store.

## JavaScript integration

The [JavaScript adapter](../../integrations/organizational-knowledge/README.md)
wraps the released `DocumentMemory` API. With the Node native bindings built as
described in [Contributing](../../CONTRIBUTING.md#set-up), run from this checkout:

```sh
node integrations/organizational-knowledge/example/cli.cjs import ./toip-example toip integrations/organizational-knowledge/example/fixtures
node integrations/organizational-knowledge/example/cli.cjs search ./toip-example toip "quartzmarker"
node integrations/organizational-knowledge/example/cli.cjs forget ./toip-example toip Zoom zoom-example-project-chat-GMT20260101-100000
node integrations/organizational-knowledge/example/cli.cjs forget-all ./toip-example toip Zoom
node integrations/organizational-knowledge/example/cli.cjs prune ./toip-example toip Zoom /absolute/path/to/data/curated
```

`forget-all` removes one source system's documents, or the whole namespace with
`--all`; `prune` keeps the source system's documents found in the given files and
removes the rest. Both preview unless `--apply` is given, and a run handles at
most `--limit` documents (10,000 by default). They list the store through
`listDocuments`, which a binding built before it was added lacks; rebuild with
`npm run build`. A prune is only as complete as the files it is given. The Rust
example forgets one document at a time.

The CLI is lexical by default. From CommonJS in the checkout, import the adapter
with `require('./integrations/organizational-knowledge/index.cjs')`; it is not
part of the released npm package's exports. Programmatic use can supply a local embedder and
reranker through `DocumentMemory`; the embedder must match the store's embedding
space. The Rust and JavaScript examples use the same identity tuple, dimensions,
and default store configuration, so they can open one store sequentially.
JavaScript validates the entire input set before writing, reports committed IDs
on a partial failure, and supports file or directory input. The entire import is
not one transaction. The JavaScript library checks the engine's chunk limits on
the open store before its first write; its CLI checks them on a throwaway store
first, so rejected input does not create yours. Its file reader accepts up to 16 MiB of
JSON per file and 512 MiB in total; the Rust file reader accepts 8 MiB per file.
Both enforce the engine's content and chunk limits. JavaScript also rejects
`__proto__` metadata keys and represents numbers as IEEE doubles; large integer
metadata may differ from the Rust reader.

Metadata filters use exact top-level equality. In particular, filtering on
`organizational_unit` must match the complete array. This differs from the
prototype's working-group substring filter and needs an explicit translation in
a prototype-side adapter. Namespace scoping remains a query convention; the host
must enforce authorization independently.

## Optional backend boundary

The current examples consume flat curated documents and use HMS chunking. They
are not wired into the prototype's MCP server. An optional backend should
preserve the workstream boundaries in
[issue #8](https://github.com/trustoverip/egwg-reporting-tf-prototype-mcp/issues/8):

| Prototype contract | Integration responsibility |
|---|---|
| Raw Artifact and Normalized Document | Keep collection, normalization, SAID rules, and authoritative provenance upstream |
| Embedded Chunk | Preserve the agreed chunk identity, text, metadata, and parent document reference when adding an indexing adapter |
| Vector Index Entry | Treat HMS as a rebuildable derived index; retain the existing backend as an option |
| Search Result | Join authoritative metadata at retrieval and preserve the existing MCP tools and response contract |

HMS's internal document ID is the namespaced stable source identity; `version`
and `metadata.identifier` retain the upstream content identifier. Internal HMS
chunk IDs are not the proposed SAID-based chunk IDs. Mapping those identities,
implementing the nested schema, matching working-group filter semantics, and
wiring the retrieval-time provenance join remain prototype integration work.
These examples neither verify SAIDs nor implement the proposed Embedded Chunk
or Search Result schemas.

The prototype already replaces changed documents transactionally, supports
`--prune`, and filters working groups before vector ranking. Its lexical scorer
reranks vector-shortlisted candidates. HMS provides an alternative lexical or
hybrid document search path; whether that improves results needs a comparison.
See the prototype's
[indexer](https://github.com/trustoverip/egwg-reporting-tf-prototype-mcp/blob/main/toip-mcp-server/src/ingest.ts)
and [retriever](https://github.com/trustoverip/egwg-reporting-tf-prototype-mcp/blob/main/toip-mcp-server/src/search.ts).

For retention or source retraction, coordinate upstream suppression with removal
from every derived index and any separately created facts or summaries. Document
deletion does not cascade into independent memory facts. The memory API can
retain supersession history and explicitly forget or purge facts; adopting it is
a separate choice from indexing normalized documents.

## Reproducible local checks

```sh
cargo test --locked --example organizational_knowledge
node --test integrations/organizational-knowledge/test/*.test.cjs
node --test tests/organizational_knowledge_cli.test.cjs
./node_modules/.bin/tsc --noEmit --strict --target es2022 --module nodenext --moduleResolution nodenext --types node tests/organizational_knowledge_types.test.cts
```

The tests exercise overlapping IDs from different applications and namespaces,
original Unicode source spans, provenance metadata, replacement, deletion,
idempotence, restart persistence, malformed inputs, and input-size boundaries.
The CLI check builds the Rust example offline and exercises both languages
against the same persistent store, including rejection before store creation,
deletion preview, replacement, reimport after deletion, and namespace isolation.
These checks use synthetic text and local storage. They establish adapter correctness,
not retrieval quality, benchmark standing, or a speed advantage.

Local verification on 2026-10-28 used macOS arm64, Rust 1.96, and Node 26.8.1:

| Check | Result |
|---|---|
| Rust example tests, default and all features | 4 passed in each configuration |
| Example Clippy, default and all features | Passed with `-D warnings` |
| Rust 1.89, no default features | Example check passed |
| JavaScript adapter tests | 31 passed |
| Rust and JavaScript CLI interoperability | 2 passed |
| Strict TypeScript file-reader/import composition | Passed |
| Strict MkDocs build | Passed; Rust API reference generation excluded |

These are focused integration checks; the full repository test matrix was not
rerun for this example. No paid services or model downloads were used.

Before proposing production adoption, compare identical corpus snapshots,
queries, chunk boundaries, embeddings, filters, and relevance judgments against the prototype's
existing backend. Measure source coverage and ranking quality separately from
latency. Associative/holographic retrieval needs its own evaluation; this adapter
does not establish a benefit for splatter or superposition.
