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
| Tampering with arena log | CRC32 integrity (default) + Ed25519 signatures (security) | default / `security` |
| Repudiation of operations | Signed append-only audit trail | `security` |
| Key extraction from disk | Ed25519 private key stored as raw 32 bytes; protect via OS file permissions | `security` |
| Inference of original content from vectors | Hypervectors are lossy; see [PRIVACY.md](PRIVACY.md) | default |
| Side-channel on signing | ed25519-dalek uses constant-time operations | `security` |

### Trust Boundaries

- **System boundary**: All data entering via `memorize()`, `memorize_text()`, or `memorize_vector()` is encoded into lossy hypervectors before storage. Original content is not retained.
- **Storage boundary**: Arena segments, index files, and audit logs reside on local disk. Encryption at rest protects against offline access.
- **N-API boundary**: Node.js callers interact via typed N-API bindings. Input validation occurs at the Rust boundary (ID length checks, dimension bounds, passphrase presence).

## Cryptographic Choices

### Ed25519 Signing

- **Library**: `ed25519-dalek` v2 (pure Rust, constant-time)
- **Key generation**: `rand::thread_rng()` (OS CSPRNG via `getrandom`)
- **Key storage**: Raw 32-byte secret key at `{storage_path}/hms_signing.key`
- **What is signed**: Each audit entry's signable prefix: `[timestamp_ms: u64][op: u8][id_hash: 32]` = 41 bytes
- **Verification**: Public key derived from stored secret key on load

### AES-256-GCM Encryption

- **Library**: `aes-gcm` v0.10 (AES-NI hardware acceleration where available)
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
2. **No access control**: HMS has no user/role model. Access control must be enforced by the calling application.
3. **Passphrase in config**: The encryption passphrase is passed via `SecurityConfig`. The calling application is responsible for secure passphrase management (environment variables, secret managers).
4. **CRC32 is not cryptographic**: Without the `security` feature, arena integrity relies on CRC32, which detects accidental corruption but not adversarial tampering.
5. **No forward secrecy**: A compromised signing key allows forging future entries (but not altering past entries already written to disk).
6. **Salt reuse across sessions**: The Argon2 salt is generated once per storage path. This is acceptable for single-user local storage but not for multi-tenant deployments.

## Audit Trail

The audit log at `{storage_path}/audit.bin` is an append-only binary file with fixed-size 105-byte entries:

```
[timestamp_ms: u64][op: u8][id_hash: 32 bytes][signature: 64 bytes]
```

- **Operations**: Memorize (1), Delete (2), Compact (3)
- **Timestamps**: Milliseconds since Unix epoch
- **Queryable**: `audit_since(timestamp_ms)` returns all entries after a given time
- **Tamper evidence**: When signing is enabled, each entry carries an Ed25519 signature over its timestamp, operation, and ID hash

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

Lower values provide stronger poisoning resistance at the cost of reduced discrimination between rare and common dimensions. A clip factor of 2.0 is recommended for adversarial environments. Values below 1.5 may degrade retrieval quality for legitimate queries.

### Limitations

- IDF clipping is a statistical defense, not a cryptographic one. It bounds the impact of poisoned dimensions but does not prevent insertion of adversarial vectors.
- Clipping is applied globally during `recompute()` and is not per-query. A large batch of poisoned inserts between recomputation cycles could temporarily skew results.
- The defense assumes a well-populated memory. With fewer than ~10 atoms, the median is unstable and clipping may be ineffective.

## Enabling Security Features

### Rust

```rust
// In Cargo.toml
[dependencies]
holographic-memory = { version = "0.5", features = ["security"] }
```

```rust
let mut config = HmsConfig::default();
config.security.signing_enabled = true;
config.security.encryption_enabled = true;
config.security.encryption_passphrase = Some("your-passphrase".to_string());
config.security.audit_enabled = true;

let hms = HmsCore::new(16384, Some("./storage".to_string()), Some(config))?;
```

### Node.js (N-API)

```javascript
const hms = new HolographicMemorySystem(16384, './storage', {
  signingEnabled: true,
  encryptionEnabled: true,
  encryptionPassphrase: process.env.HMS_PASSPHRASE,
  auditEnabled: true,
});

// Query audit trail
const entries = await hms.auditSince(Date.now() - 86400000); // last 24h
```
