# Production readiness

## API stability

- `HmsCore`, the Node.js `HolographicMemorySystem`, and quantized-phase Python APIs are supported surfaces.
- Modules behind the `experimental` Cargo feature may change between minor releases.
- Research binaries under `src/bin` are reproducibility tools, not service interfaces.

## Persistence compatibility

Persistent arenas contain a `format.json` manifest. HMS currently writes format version 1,
automatically adopts pre-manifest v1 stores, and refuses to open stores written by a newer
format. Back up a store before changing major HMS versions or running compaction.

Call `flush()` before a controlled shutdown. Use `storageHealth()` to monitor format,
segment count, allocated capacity, and used bytes.

## Index lifecycle

`indexStatus()` reports trained indices and threshold-based recommendations.
`maintainIndices()` trains recommended indices and is safe to call repeatedly. For latency-
sensitive applications, run maintenance outside the request path. `explainQuery()` reports the
selected route and dynamic NSG/IVF parameters.

`hardwareCapabilities()` reports AVX2, AVX-512, and NEON availability. HMS currently uses its
adaptive merge/galloping kernel for sparse sorted indices; capability reporting prevents future
dense kernels from silently assuming unsupported instructions.

## Capacity planning

Measure with your own embeddings and query distribution. Synthetic self-recall is a regression
signal, not evidence of semantic relevance. Run:

```sh
cargo run --release --bin hms-eval -- \
  --vectors 10000 --queries 500 --dimensions 16384 --assert-min-recall 0.90
```

Record the JSON output with the HMS version, CPU, operating system, and dataset revision.

## Choosing HMS

Use HMS when local execution, deterministic vector-symbolic composition, compact sparse vectors,
or associative/relational reasoning matters. Prefer a conventional vector database when you need
distributed replication, managed backups, mature filtering, multi-tenant authorization, or a
large ecosystem of pretrained embedding integrations.

## Security and failure testing

- Keep signing keys owner-readable and source encryption passphrases from a secret manager.
  Set `encryptionPassphraseEnv` to the name of an injected environment variable so the secret is
  removed from HMS configuration after key derivation.
- Exercise backup restoration and interrupted compaction in staging.
- Run the fuzz targets described in `fuzz/README.md` before storage or wire-format releases.
- Treat CII registration and continuous long-duration fuzzing as deployment-policy decisions.
