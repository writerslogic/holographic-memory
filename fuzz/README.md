<img src="https://raw.githubusercontent.com/writerslogic/warm/main/assets/crystal-halo/crystal-detailed-static.png" width="88" alt="WritersLogic Adaptive Retention & Memory" align="left">

<h1>WARM fuzzing</h1>
<p><strong>Deterministic, allocation-bounded fuzz targets for every parser of untrusted input in the WritersLogic Adaptive Retention & Memory.</strong></p>

<br clear="left">

Targets must stay deterministic and must reject or clamp a length before allocating from it.
Install `cargo-fuzz` and a nightly toolchain, then from the repository root:

```sh
cargo +nightly fuzz run <target> fuzz/seeds/<target> -- -max_total_time=60
```

Add `--features <feature>` for the gated targets (see the table). Seeds live in
`fuzz/seeds/<target>/`; the corpus cargo-fuzz grows is `fuzz/corpus/` (ignored by git).
Dictionaries are in `fuzz/dictionaries/` (`-dict=fuzz/dictionaries/server.dict`).

| Target | Untrusted input it covers | Feature |
| --- | --- | --- |
| `wire_inputs` | dense float vectors into `EntangledHVec` | none |
| `ivf_decode` | persisted quantized IVF index bytes (`QuantizedIvf::from_bytes`), re-encode identity | `experimental` |
| `document_chunking` | `DocumentInput` JSON into `chunk_document`; chunks must be valid char-boundary slices | none |
| `document_engine` | `DocumentInput` + `SearchOptions` JSON through memorize, search and delete in a live store | none |
| `store_open` | a valid store with one file overwritten, truncated, appended to or replaced: manifests and CRC sidecars, arena frames, document log, indices; `verify_store` then open then search | none |
| `jumbf_decode` | JUMBF (ISO 19566-5) box hierarchies | `provenance` |
| `cose_envelopes` | COSE_Sign1, the signed WARM manifest, the SCITT signed statement | `provenance` |
| `cawg_identity` | CAWG identity assertion CBOR with its embedded COSE and did:jwk | `provenance` |
| `did_parse` | did:key, did:jwk, did:web and DID documents | `provenance` |
| `provenance_json` | W3C VC, Sigstore bundle JSON and the KERI key event log file | `provenance` |
| `private_query` | server-side query words and client-side answer words of the private search protocol | `private-search` |
| `operand_question` | free-text question parsing | `experimental` |
| `connection_graph_log` | the append-only bincode event log replayed on open | `experimental` |
| `llm_output` | text produced by an on-device model (JSON extraction, facts, query plan, dates) | `local-models` |

## Reproduce a crash from CI

Download the `fuzz-artifacts` artifact of the failed run, then:

```sh
cargo +nightly fuzz run <target> [--features <feature>] path/to/crash-<hash>
cargo +nightly fuzz fmt <target> path/to/crash-<hash>     # pretty-print the input
```

A crash becomes a regression test next to the code it broke, and the input is added to
`fuzz/seeds/<target>/`.

## Regenerate the binary seeds

```sh
cargo run --manifest-path fuzz/Cargo.toml --example gen_seeds \
  --features provenance,experimental -- fuzz/seeds
```
