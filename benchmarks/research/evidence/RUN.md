# Local evidence research prototype

This standalone, unpublished crate exercises source-checked assertions, bounded
event aggregation, source repair and proof-cut planning through WARM storage. The
fixtures contain generated local data. Supplied entity IDs, predicates, event
alternatives, amounts and times are an **oracle-extraction control**: the program
does not extract these annotations or call a model, API or cloud service. This
is not a production memory pipeline, a released API or evidence of benchmark gains.

From this directory, use a private target directory and the checked-in lockfile:

```sh
cd benchmarks/research/evidence
export CARGO_TARGET_DIR="$PWD/target/cli-local"
cargo test --offline --locked
cargo run --offline --locked -- demo
cargo run --offline --locked -- run fixtures/payments.json "$PWD/target/payment-store"
cargo run --offline --locked -- run fixtures/identity.json "$PWD/target/identity-store"
cargo run --offline --locked -- proof fixtures/proof.json
```

The checked-in lockfile pins the public dependencies. Run without `--offline`
once to populate the Cargo cache; subsequent runs can use the offline commands
above. The optional entity-resolution bridge uses the public `narrative-lens`
repository at commit `97d47eca34474409e963064dc8319559a99044f8`; a sibling
checkout is not required. The default feature set uses no models and does not
call that resolver. To check the bridge, use:

```sh
cargo test --locked --features narrative-lens
```

`demo` creates actual temporary `EvidenceStore` instances and prints one JSON
value. Its payment reports exercise a missing amount repaired from another
source, repeated mentions of the same receipt, actual versus planned payments,
attendance versus payment scope, and exact UTF-8 spans containing `Café λ`.
The supplied amounts imply 700 USD cents across two distinct actual payments.
The identity fixture permits one or two visits; a one-world budget must abstain
instead of treating a partial search as an exact answer.

The demo also plans a cut over two alternative proofs of a contact fact while
protecting an independent preference. Only this isolated fixture maps primitive
proof IDs directly to exact stored source IDs, with every source version pinned
in `fixtures/forgetting-demo.json`. It applies that cut to its separate temporary
store and requires reopen checks for removal and protection, rejection of
same-version replay, and acceptance of explicit fresh-version relearning. The
demo fails if those checks fail. There is no general automatic cut application command.

Accepted source deletion first records revoked source/version pairs and pending
deletions in a separate bounded control record. Opening the store recovers an
interrupted operation before exposing sources; a retried operation is idempotent.
The control record retains identifiers and version pins, not source text, quotes
or claims. Its limits are 512 revoked versions, 256 pending deletions and
64 KiB serialized metadata; exceeding them fails before a new deletion begins.
This is exact-version replay rejection, not a policy against retaining future
information about the same topic. Changing a version explicitly permits new
learning. It does not promise physical erasure of identifiers or backups.

`run FILE.json STORE` accepts an object with exactly `namespace`, `sources`,
`claims`, `terms`, `query` and `limits`; `fixtures/payments.json` is a complete
example. JSON files must be regular files of at most 8 MiB. Structural, span,
scope, metadata, query, retrieval-limit and execution validation occurs before
opening the destination store. The command then writes the supplied bundle and
prints its retrieval report. An existing store can contain additional annotated
sources in the same namespace. Namespace selection is not an authorization boundary.
Use a dedicated store directory: this research format uses 256 dimensions,
lexical whole-document encoding, stored text and positions, with at most 64 KiB
of source text or per-source metadata. It stores at most 1,024 sources and 4,096
claims per namespace; combined source text and combined metadata are each capped
at 8 MiB. The first evidence span owns a claim; updating a source replaces its
owned annotations, and surviving cross-source references must stay valid.

Retrieval begins with local lexical search, then enumerates exact supplied
subject/predicate/modality matches for repair. This full-scan catalogue is an
oracle annotation index, not evidence of general semantic retrieval. Catalogue
scans and `catalogue_source_bytes` are separate from the bounded answer packet's
source count, `source_bytes` and serialized `packet_bytes`; the packet limit is
not a bound on total store reads. `catalogue_source_bytes` measures one namespace
snapshot's text, not cumulative bytes read across repeated snapshots or a disk-I/O
measurement. Whole sources and their source/claim
dependencies must fit before admission. Incomplete source coverage or exhausted
identity/time worlds produces an unresolved answer with no numeric result.
`packet_bytes` measures compact UTF-8 JSON containing exactly `sources` and
`claims`; it excludes the query, answer and diagnostic trace. It is not a model
token budget, and this prototype has no model reader.

Every answer has `observed_only: true`: resolved numbers describe the admitted
assertions, not an exhaustive history of real events. A bounded range reflects
complete evaluation of the declared alternatives. Missing values, incompatible
amounts, contradictory scoped assertions and missing comparison operands cannot
be silently converted to zero or an exact answer. Caller annotations and valid
quotation spans do not establish natural-language entailment.

`proof FILE.json` accepts exactly `{ "problem": ProofProblem, "limits":
SearchLimits }` and prints a logical source-cut plan with derivability witnesses.
Only `solved` certifies the minimum-cost cut under the declared finite graph;
`budget_exhausted` does not certify optimality. The proof graph is supplied by the
caller, not inferred from stored text. This command never mutates a store.
Physical deletion requires a separately verified mapping to current stored
sources and versions; backups and unrestricted future inference are outside
this prototype's guarantees.
