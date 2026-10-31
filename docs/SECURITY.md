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
- **What is signed**: Each audit entry's signable prefix: `[timestamp_ms: u64][op: u8][id_hash: 32]` = 41 bytes
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

1. **No key rotation**: Changing the signing key invalidates verification of prior audit entries. Key rotation requires a re-signing migration (not yet implemented).
2. **Library access control**: The embedded library has no user/role model; its caller
   controls access. The separate HTTP server supports [scoped keys and tenant isolation](server/auth.md).
3. **Passphrase in config**: The encryption passphrase is passed via `SecurityConfig`. The calling application is responsible for secure passphrase management (environment variables, secret managers).
4. **CRC32 is not cryptographic**: Without the `security` feature, arena integrity relies on CRC32, which detects accidental corruption but not adversarial tampering.
5. **No forward secrecy or externally anchored history**: A compromised signing key can forge entries. Per-entry signatures do not detect deletion, truncation, or reordering of otherwise valid records without an independently retained history.
6. **Salt reuse across sessions**: The Argon2 salt is generated once per storage path. This is acceptable for single-user local storage but not for multi-tenant deployments.

### Optional model dependency review

`local-models` pulls Candle 0.11 and its transitive `zip` 8.6 dependency. That
version is affected by [GHSA-m665-v7pm-7grj](https://github.com/zip-rs/zip2/security/advisories/GHSA-m665-v7pm-7grj).
WARM loads these models through `VarBuilder::from_mmaped_safetensors` in
`src/core/models/mod.rs`; it does not call Candle's PyTorch ZIP reader, ZIP
streaming, or filesystem extraction. The advisory's inspection/extraction
chain is therefore not reachable through this integration. This disposition
does not declare ZIP 8.6 safe for arbitrary use. Reassess before adding a ZIP
model loader or archive-validation/extraction path, and upgrade the transitive
dependency when Candle supports a patched version.

The October 31, 2026 review also covers ZIP advisories
[GHSA-q7v9-x93c-xqv3](https://github.com/advisories/GHSA-q7v9-x93c-xqv3),
[GHSA-8hxf-qr8p-62x4](https://github.com/advisories/GHSA-8hxf-qr8p-62x4),
[GHSA-hjqx-j4rw-g7xv](https://github.com/advisories/GHSA-hjqx-j4rw-g7xv), and
[GHSA-9hj8-73fh-9963](https://github.com/advisories/GHSA-9hj8-73fh-9963).
These affect ZIP extraction, AES authentication and legacy decompression, none
of which the safetensors loader invokes. Candle's fix is merged upstream in
[PR 4078](https://github.com/huggingface/candle/pull/4078), but its latest
published release remains 0.11.0. Keep the dependency alerts open until a
patched Candle release can be adopted and the local-model tests pass.

### JavaScript development dependencies

The integration test workspace's LlamaIndex fixture pulls `deepmerge` 4.3.1
through `html-to-text`. [GHSA-gcqx-w48m-w2h4](https://github.com/advisories/GHSA-gcqx-w48m-w2h4)
has no patched deepmerge release as of October 31, 2026. It is a development
dependency, not bundled into the WARM adapters; no production safety claim is
made for consumers' independently installed LlamaIndex dependencies. Keep the
alert open pending the upstream dependency change. The workspace's `tsup` 9
upgrade removes the vulnerable esbuild development-server dependency.

### Legacy Embedchain optional dependency

The `embedchain` and `all` Python extras still resolve Embedchain 0.1.128,
which requires `mem0ai<0.2` and `chromadb<0.6`. Their existing mem0ai 0.1.116
lock is retained; it is affected by
[GHSA-7587-9684-52r6](https://github.com/advisories/GHSA-7587-9684-52r6) and
[GHSA-jfv9-68m5-gjjr](https://github.com/advisories/GHSA-jfv9-68m5-gjjr).
Do not use that extra to expose a mem0 server or to process untrusted
Databricks queries. The independent `import-mem0` extra and importer test
group now resolve mem0ai 2.2.1. Removing the legacy constraint requires an
upstream Embedchain update or retirement of that optional adapter; this
cleanup does not claim the legacy dependency stack is vulnerability-free.

## Audit Trail

The audit log at `{storage_path}/audit.bin` is an append-only binary file with fixed-size 105-byte entries:

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

Custom builds fail if requested encryption/signing support is unavailable. The standard npm build includes `security`; use `securityStatus()` to confirm active configuration. Audit/provenance sidecar failures can occur after the main arena mutation commits. See [production-readiness.md](production-readiness.md) for transaction and failure boundaries.
