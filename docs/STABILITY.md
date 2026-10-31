<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Stability and compatibility

WARM follows [semantic versioning](https://semver.org). The crate is **0.x today**: until 1.0 a
minor release may break any surface, but the on-disk promise below is already kept (every
released store, v0.6.1, opens in the current code; see `tests/golden/`). The 1.x promises start
at 1.0.0.

## Surfaces

| Surface | Versioned as | Stable in 1.x | Not covered |
|---|---|---|---|
| Rust crate (`writerslogic-warm`) | crate version | items reachable without the `experimental` feature, outside `core::format::scan` internals | `experimental` feature, `doc(hidden)`, `pub(crate)` |
| Python (`warm-py` wheels, `warm-sdk`) | PyPI version | documented classes and functions | underscore names; the `phase` substrate is experimental |
| Node (`writerslogic-warm` on npm) | npm version | the exports in `index.d.ts`, `memory.d.ts`, `semantic.d.ts` | undocumented properties |
| HTTP (`warm-server`) | URL prefix `/api/v1` | documented endpoints, request and response fields, error codes | new optional fields may appear; new endpoints |
| [C ABI](runtimes/c/README.md) | `writerslogic-warm-c` package version and `hms1_*` ABI major | documented v1 symbols, ownership rules and fixed value-struct layouts; size-tagged option structs grow append-only | Rust layouts, internal symbols and unversioned convenience macros |
| [Go](runtimes/go/README.md), [Java](runtimes/java/README.md) and [.NET](runtimes/dotnet/README.md) | binding package version and underlying C ABI major | documented wrappers over the versioned C contract | private FFI helpers; package availability is recorded in each runtime guide |
| [Swift](runtimes/swift/README.md) and [Kotlin](runtimes/kotlin/README.md) | UniFFI crate and binding package version | experimental until a stable binding contract is declared | generated UniFFI interfaces, metadata and packaging layouts |
| [WebAssembly](runtimes/wasm/README.md) | WASM crate and npm package version | experimental until a stable binding contract is declared | generated JavaScript glue, scalar/SIMD artifact layout and experimental APIs |
| Persisted format | per artefact, see below | yes | caches (below) |

WritersLogic maintainers review changes to these contracts. C callers check
`hms1_abi_version()` before using the library; a new ABI major uses a new symbol
prefix so old and new majors can coexist. The language-specific runtime guides
record supported platforms, build requirements and verification limits.

Additive changes (new optional field, new endpoint, new function) are minor. Removing or
changing behaviour is major. Security fixes may break an unsafe behaviour in a patch release
and say so in the changelog.

## The file-format promise

1. **Backward:** a store written by any 1.x opens in every later 1.x, read-write, without a
   migration step. The same holds for released 0.x stores (v0.6.1 is in `tests/golden/`).
2. **Never silently converted:** the engine keeps writing a store in the format it has. A
   format change is performed by `warm-format upgrade`, which writes a new store and leaves the
   old one untouched.
3. **Forward:** an older reader meeting a version it does not know (arena manifest, `store.json`
   schema, journal magic) refuses to open and says "newer than supported"; it does not
   guess. A newer release only produces such a store when a feature that needs it is used, and
   documents which. Exception, 0.7.0: new stores are created with `store.json` schema 3, whose
   journal frames are binary (`HMS-TXN-3`, [file-format/MUTATIONS.md](file-format/MUTATIONS.md));
   0.6.x refuses them. Schema 2 stores keep writing `HMS-TXN-2` JSON frames, and 0.7 reads both.
   Stores created by 0.7.0 are schema 4: schema 3 plus document checkpoints, which replace the
   journal frames before them ([file-format/CHECKPOINT.md](file-format/CHECKPOINT.md)); a
   release that predates checkpoints refuses schema 4 rather than opening it without its
   checkpointed documents. Schema 2 and 3 stores open, replay their frames and never get a
   checkpoint. The current development writer creates new stores with schema 5, which additionally permits namespace-scoped
   durable meaning maintenance. Schemas 2, 3 and 4 remain readable and writable in their
   original format; governance that needs scoped mutations returns an upgrade error. Run
   `warm-format upgrade <source> <destination>` explicitly to enable schema 5 in a new copy.
   Schema-only upgrades retain the existing arena and checkpoints unchanged; old readers
   refuse schema 5 rather than misreading scoped deletions.
4. **Breaking the format** happens only in a major release, with the old format still readable
   or an `warm-format upgrade` step, and a golden store for each side.
5. **Caches** (`documents.text`, `indices.json`, NSG/IVF files, `shard_meta.json` aside) may
   change at any release; the engine discards and rebuilds what it cannot read.
6. **Encoders are permanent:** an encoder name always produces the same vectors. Changing output
   means a new name; stores keep the encoder they were created with.

How this is enforced: `tests/format_golden.rs` opens every committed golden store and checks
contents and searches; it also regenerates the canonical corpus and fails if the journal bytes
differ from the newest golden. A format change therefore cannot land without a version bump
(`src/core/format/mod.rs`) and a new golden store, and `scripts/format-gate.sh` rejects any
edit or deletion of an existing golden file.

Standalone persisted vector update directories have their own version 1 contract in
[file-format/VECTOR-UPDATES.md](file-format/VECTOR-UPDATES.md). Their committed golden
checkpoints and logs are read and regenerated by `vector_format_golden_and_drift_gate`;
they are covered by the same append-only golden-file gate. The standalone vertex encoding
inside a checkpoint remains `WARMVGR01`.

## Deprecation

A deprecated Rust, Python, Node or HTTP item stays for at least one minor release and six
months, whichever is longer, and is marked in docs and (where the language allows) by a
deprecation attribute or warning. A deprecated file-format feature is read for the whole major
series.

## Explicitly unstable

- the `experimental` feature and everything it gates: research modules, connection graph and
  phase graph, the capped-union encoder, `connection_graph.log`;
- `conversation-memory`, `private-search`, `local-models` (model stage APIs and model directory layout), `provenance*`
  file layouts (`provenance_log.jsonl`, `kel.json`) until they get their own specification;
- the standalone qgraph/IVF cache files, benchmark binaries and their output formats,
  `warm-eval`, `warm-admin` output JSON field order;
- the NSG/IVF bincode layout and anything else the format documents call a cache.
