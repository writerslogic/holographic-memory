# Security Model

## Threat Model

### Assets Protected

1. **Stored vectors**: Hyperdimensional representations of user content stored in arena segments and index files.
2. **Operation history**: Which IDs were memorized, deleted, or compacted, and when.
3. **Signing keys**: Ed25519 private keys used for audit log non-repudiation.
4. **Encryption keys**: AES-256-GCM derived keys for data-at-rest protection.

### Threat Categories

| Threat | Mitigation | Feature Flag |
|--------|-----------|-------------|
| Unauthorized read of stored data | AES-256-GCM encryption at rest | `security` |
| Tampering with arena log | CRC32 for accidental corruption; AES-GCM authentication when encrypted | default / `security` |
| Repudiation of operations | Signed append-only audit trail | `security` |
| Key extraction from disk | Ed25519 private key stored as raw 32 bytes; protect via OS file permissions | `security` |
| Inference of original content from vectors | No inherent protection; see [PRIVACY.md](PRIVACY.md) | — |
| Side-channel on signing | ed25519-dalek uses constant-time operations | `security` |

### Trust Boundaries

- **System boundary**: Vector APIs retain representations; document APIs retain source passages by default, plus terms, metadata, source identifiers, and optional dense embeddings. Lossy encoding does not prevent inference.
- **Storage boundary**: Arena payloads and index files can be encrypted. Schema metadata, filesystem structure, audit/provenance sidecars, and signing keys have separate policies; encryption does not conceal every file or access pattern.
- **Document text cache**: With store encryption enabled, version 2 `documents.text`
  records use AES-256-GCM with their segment offsets as authenticated data. Equal passages
  receive independent nonces; altered or relocated records fail authentication. Legacy
  plaintext caches are rebuilt from the arena at open. Schema metadata, filesystem structure,
  audit/provenance sidecars and signing keys retain the separate policies described above.
- **N-API boundary**: Node.js callers interact via typed N-API bindings. Input validation occurs at the Rust boundary (ID length checks, dimension bounds, passphrase presence).

## Cryptographic Choices

### Ed25519 Signing

- **Library**: `ed25519-dalek` v3 (pure Rust, constant-time)
- **Key generation**: `rand::rng()` (OS CSPRNG via `getrandom`)
- **Key storage**: Raw 32-byte secret key at `{storage_path}/hms_signing.key`
- **What is signed**: Legacy entries cover `[timestamp_ms: u64][op: u8][id_hash: 32]` (41 bytes). Schema-6 effects prepend `WARM-AUDIT-2\0` and their 16-byte delivery identity (70 bytes total); use `AuditEntry::signable_bytes()`.
- **Verification**: Public key derived from stored secret key on load

### AES-256-GCM Encryption

- **Library**: `aes-gcm` v0.11 (AES-NI hardware acceleration where available)
- **Key derivation**: Argon2id (default parameters) from user passphrase + 16-byte random salt
- **Salt storage**: `{storage_path}/encryption.salt` (generated once, persisted)
- **Nonce**: 12-byte random per encryption operation (prepended to ciphertext)
- **Scope**: Encrypts serialized log entries before LZ4 compression in arena, and index files (NSG, IVF) on disk

### ID Hashing

- **With `security` feature**: SHA-256 (32-byte output)
- **Without**: FxHash (8-byte, zero-padded to 32)
- **Purpose**: Audit log stores hashes of vector IDs, never raw IDs

## Limitations

1. **No automatic audit-key rotation**: Historical audit signatures remain valid under their original public keys. Preserve those independently trusted keys when selecting a new signing key; the audit records do not carry a signer identifier or a rotation trust chain. Re-signing old events would change their original evidence and is not required to verify them.
2. **Library access control**: The embedded library has no user/role model; its caller
   controls access. The separate HTTP server supports [scoped keys and tenant isolation](server/auth.md).
3. **Passphrase in config**: The encryption passphrase is passed via `SecurityConfig`. The calling application is responsible for secure passphrase management (environment variables, secret managers).
4. **CRC32 is not cryptographic**: Without the `security` feature, arena integrity relies on CRC32, which detects accidental corruption but not adversarial tampering.
5. **No forward secrecy or externally anchored history**: A compromised signing key can forge entries. Per-entry signatures do not detect deletion, truncation, or reordering of otherwise valid records without an independently retained history.
6. **Salt reuse across sessions**: The Argon2 salt is generated once per storage path. This is acceptable for single-user local storage but not for multi-tenant deployments.

### Optional model dependency review

`local-models` requires Candle 0.11.1 or later; the lockfile resolves ZIP 9.2.0.
This replaces the ZIP 8.6 dependency affected by
[GHSA-m665-v7pm-7grj](https://github.com/zip-rs/zip2/security/advisories/GHSA-m665-v7pm-7grj),
[GHSA-q7v9-x93c-xqv3](https://github.com/advisories/GHSA-q7v9-x93c-xqv3),
[GHSA-8hxf-qr8p-62x4](https://github.com/advisories/GHSA-8hxf-qr8p-62x4),
[GHSA-hjqx-j4rw-g7xv](https://github.com/advisories/GHSA-hjqx-j4rw-g7xv), and
[GHSA-9hj8-73fh-9963](https://github.com/advisories/GHSA-9hj8-73fh-9963).
The upstream fix is [Candle PR 4078](https://github.com/huggingface/candle/pull/4078),
included in the [0.11.1 release](https://crates.io/crates/candle-core/0.11.1).
The minimum version applies to CPU, Metal and CUDA dependency declarations.

WARM loads these models through `VarBuilder::from_mmaped_safetensors` in
`src/core/models/mod.rs`; it does not invoke Candle's PyTorch ZIP reader or archive
extraction. The prior reachability disposition did not declare ZIP 8.6 generally
safe. Reassess the boundary before adding an archive loader, and retain model and
binding regression checks when updating Candle.

### JavaScript development dependencies

The integration test workspace's LlamaIndex fixture pulls `deepmerge` 4.3.1
through `html-to-text`. [GHSA-gcqx-w48m-w2h4](https://github.com/advisories/GHSA-gcqx-w48m-w2h4)
has no patched deepmerge release as of October 31, 2026. It is a development
dependency, not bundled into the WARM adapters; no production safety claim is
made for consumers' independently installed LlamaIndex dependencies. Keep the
alert open pending the upstream dependency change. The workspace's `tsup` 9
upgrade removes the vulnerable esbuild development-server dependency.

### Optional Python integration dependencies

The SDK's `all`, `embedchain`, `phidata`, and legacy `semantic-kernel` extras
and their three legacy adapter modules are retired. Embedchain forces an obsolete dependency
stack; the old Semantic Kernel implementation also cannot satisfy its framework
contract. Phidata's [upstream is archived](https://github.com/agno-agi/phidata),
and its WARM vector-store adapter also fails construction with six unimplemented
abstract methods. Its retirement reflects support/correctness failures, not a
newly discovered vulnerability. This retirement does not remove the separate
`writerslogic-warm-integrations` package's current Semantic Kernel adapter, whose
resolved dependencies still require independent review.

The SDK now requires Python 3.11 or newer. [Python 3.10 reached end of life on
October 1, 2026](https://devguide.python.org/versions/); retaining it also prevented
patched Pillow and Filelock releases. `pydantic-ai` now installs the official slim
distribution, without all provider SDKs. The refreshed lock uses Pillow 13.0.0,
Filelock 4.2.6, and Pydantic AI Slim 2.64.0. The Filelock constraint is a repository
resolver guard, not a dependency override or a guarantee for independently
resolved consumer environments. It selects LiteLLM 1.103.10 because 1.106.6
requires Filelock below 4; the documented DSPy profile pins that compatible pair
and passes its local retrieval/cache tests. An unconstrained latest installation
does not reproduce this profile.

The full SDK lock scan on October 31, 2026 still matches four distinct Chroma
advisories and one DiskCache advisory (OSV also returns duplicate PYSEC aliases):

- Chroma 1.5.9 remains in `import-chroma`, the importer test group, and CrewAI's
  transitive dependencies. [Code injection](https://github.com/advisories/GHSA-36p7-vc44-83pf),
  [pre-authentication code injection](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c),
  [tenant authorization](https://github.com/advisories/GHSA-xph7-9rjv-w5fr), and
  [cross-tenant access](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr)
  advisories include that release. WARM's importer reads local collections or an
  external Chroma service; it does not start a Chroma HTTP server. This is not a
  demonstrated mitigation of all four advisories. Do not deploy these extras as
  a security-qualified integration until their actual paths are verified or the
  dependency is patched/replaced.
- DSPy requires DiskCache 5.6.3, whose default pickle deserialization permits code
  execution if an attacker can write its cache directory
  ([GHSA-w8v5-vhqr-4h9v](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v)).
  Shared or untrusted cache directories are unsafe. The [verified DSPy profile](integrations/frameworks/dspy.md#cache-safety)
  disables disk caching before first use and tests a fresh process that rejects
  DiskCache imports. This mitigates that cache path, not arbitrary use of the
  dependency or untrusted serialized DSPy programs.

These are dependency matches, not five reproduced WARM exploits. The base HTTP
client depends only on `httpx` and `sniffio`; install only needed extras and audit
the resolved deployment environment. A consumer's PyPI installation is not bound
by the repository lock. Neither this SDK scan nor retirement establishes the
security of the separate Python/JavaScript integration packages. No advisories
are suppressed. The [remediation record](research/technical-audit-20261029/REMEDIATION.md)
preserves the full before/after scan, resolver trials, and remaining gates.

The earlier note associated mem0ai 0.1.116 with GHSA-7587-9684-52r6. The current
[advisory range](https://github.com/advisories/GHSA-7587-9684-52r6) starts at 0.1.117,
so that specific attribution was incorrect. Other legacy mem0 advisories were
separate findings; retirement removes the old dependency fork rather than
asserting it was safe.

### Documentation generator dependency

TypeDoc 0.28.20 depends on minimatch 10, affected by
[GHSA-g3hv-75wg-m249](https://github.com/advisories/GHSA-g3hv-75wg-m249).
The advisory concerns attacker-controlled patterns at `optimizationLevel >= 2`;
the inspected TypeDoc calls use default optimization. This is build tooling,
not the WARM service. Minimatch 11.0.1 is patched but removes `matchOne`, which
TypeDoc's glob traversal calls. A forced major override was rejected as
incompatible. Keep the alert open and upgrade when TypeDoc supports the patched
API; verify generated reference pages and glob traversal, not just installation.

## Audit Trail

For schemas 2–5, `{storage_path}/audit.bin` is an append-only binary file with fixed-size 105-byte entries:

```
[timestamp_ms: u64][op: u8][id_hash: 32 bytes][signature: 64 bytes]
```

- **Operations**: Memorize (1), Delete (2), Compact (3)
- **Timestamps**: Milliseconds since Unix epoch
- **Queryable**: `audit_since(timestamp_ms)` returns all entries at or after a given time
- **Tamper evidence**: When signing is enabled, each entry carries an Ed25519 signature over its timestamp, operation, and ID hash

Audit reads reject complete records with unknown operation codes, including records
outside the requested timestamp range, and report the corrupt record's byte offset.
This is structural validation, not automatic signature verification. A trailing
incomplete record is excluded from reads and durably truncated on open or before
the next append under the audit writer lock. Complete records are retained, including
records whose write completed but whose sync reported an error. Recovery preserves
alignment; it does not reconstruct missing audit events or establish exactly-once
delivery. A failed repair prevents the new audit append, but the associated core
mutation may already be committed as described in the
[committed-effect contract](production-readiness.md#durability-and-concurrency).

Schema 6 preserves legacy history and writes new entries to `audit.effects.bin` with a
version header, effect identity, and per-record checksum. Enabled vector memorize/delete
and compaction operations journal their audit intents atomically with the business change.
Open and `flush_audit()` recover queued delivery without reapplying that change. Delivery
is ordered and deduplicated across the write/acknowledgement crash interval, with at most
1024 pending effects. A full backlog rejects additional audited changes before commit.
Original signatures survive recovery unchanged; retain the original verification key.
Neither CRC32 nor individual signatures prove that no whole record was removed. Historical
history gaps, provenance delivery, and document/meaning audit coverage remain separate work.
See the [persisted layout](file-format/ANCILLARY.md#auditeffectsbin-schema-6) and
[delivery invariants](file-format/MUTATIONS.md#durable-audit-delivery-schema-6).

## IDF Clipping Poisoning Defense (Meaning Memory)

### Threat

When meaning memory is enabled, AtomMemory and CompositeMemory use IDF-weighted posting-list overlap scanning to rank candidates during structural queries and Hopfield attractor cleanup. An attacker who can insert vectors with carefully chosen active dimensions could exploit extreme IDF weights to dominate similarity scores, causing the system to return attacker-controlled atoms as query results.

For example, an adversary could insert atoms that activate rarely-used dimensions, inflating those dimensions' IDF weights. Subsequent queries touching those dimensions would disproportionately favor the poisoned atoms.

### Mitigation: Proportional IDF Clipping

The `IdfWeights` module applies proportional clipping to bound all IDF weights at `clip_factor * median(weights)`. This is computed as follows:

1. Collect all positive IDF weights.
2. Sort them and find the median.
3. Cap any weight exceeding `clip_factor * median` to that threshold.

The default `idf_clip_factor` is **3.0**, meaning no dimension's IDF weight can exceed 3x the median weight. This limits the maximum influence any single dimension can have on overlap scores, regardless of how many poisoned vectors are inserted.

### Configuration

The clip factor is configurable via `MeaningConfig`:

```rust
config.meaning.idf_clip_factor = 3.0; // default: cap at 3x median
```

Lower values provide stronger poisoning resistance at the cost of reduced discrimination between rare and common dimensions. Evaluate the effect on your own retrieval and adversarial fixtures; no clipping value provides a general poisoning-resistance guarantee.

### Limitations

- IDF clipping is a statistical defense, not a cryptographic one. It bounds the impact of poisoned dimensions but does not prevent insertion of adversarial vectors.
- Clipping is applied globally during `recompute()` and is not per-query. A large batch of poisoned inserts between recomputation cycles could temporarily skew results.
- The defense assumes a well-populated memory. With fewer than ~10 atoms, the median is unstable and clipping may be ineffective.

## Enabling Security Features

### Rust

```rust
// In Cargo.toml
[dependencies]
writerslogic-warm = { version = "0.6", features = ["security"] }
```

```rust
let mut config = WarmConfig::default();
config.security.signing_enabled = true;
config.security.encryption_enabled = true;
config.security.encryption_passphrase = Some("your-passphrase".to_string());
config.security.audit_enabled = true;

let warm = WarmCore::new(16384, Some("./storage".to_string()), Some(config))?;
```

### Node.js (N-API)

```javascript
const warm = new WarmMemorySystem(16384, './storage', {
  signingEnabled: true,
  encryptionEnabled: true,
  encryptionPassphraseEnv: 'WARM_PASSPHRASE',
  auditEnabled: true,
});

// Query audit trail
const entries = await warm.auditSince(Date.now() - 86400000); // last 24h
```

`auditSince` requires a nonnegative safe integer timestamp. Each result includes
hexadecimal `signature` and `signableBytes`, plus `effectId` for schema-6 deliveries.
The `signed` flag means a signature is present; it does not verify authenticity.
Use Node's `crypto.verify(null, Buffer.from(entry.signableBytes, 'hex'), trustedPublicKey,
Buffer.from(entry.signature, 'hex'))` with an independently trusted Ed25519 public key.
Verification authenticates the signed bytes; decode those bytes or compare the displayed
fields against them when processing an untrusted response. Legacy messages are 41 bytes;
effect-aware messages are 70 bytes and include the delivery ID. Rust callers can name
`writerslogic_warm::AuditEntry` and `AuditOp` and use `AuditEntry::signable_bytes()`.
Neither interface establishes completeness or ordering of the entire history.

Custom builds fail if requested encryption/signing support is unavailable. The standard npm build includes `security`; use `securityStatus()` to confirm active configuration. Audit/provenance sidecar failures can occur after the main arena mutation commits. See [production-readiness.md](production-readiness.md) for transaction and failure boundaries.
