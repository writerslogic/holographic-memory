# Privacy Analysis

## Inherent Privacy Properties

HMS stores data as **sparse hyperdimensional vectors** (Binary Spatter Code with rho = 1/256). This encoding provides inherent privacy properties independent of any cryptographic features.

### Lossy Encoding

Text is encoded via deterministic character trigram hashing into a D-dimensional space where only D/256 bits are active. At D=16384:

- **Active indices**: ~64 out of 16,384 dimensions
- **Information capacity**: ~64 * log2(16384) = ~896 bits per vector
- **Input capacity**: Unbounded (arbitrary-length text maps to fixed-size vector)

The encoding is a many-to-one mapping. Multiple distinct inputs produce identical vectors (hash collisions across trigrams). **Original text cannot be reconstructed from the stored vector.**

### Quantifying Information Loss

For a document of length L characters:
- Number of trigrams: L - 2
- Each trigram maps to D/256 indices via seeded hash
- Bundling applies majority rule, discarding minority indices
- Final vector retains only the statistically dominant patterns

**Reconstruction bound**: Given a stored vector with 64 active indices, an adversary must search a space of C(16384, 64) ~ 10^174 possible index combinations. Even with knowledge of the trigram encoding scheme, the many-to-one mapping prevents unique inversion.

### Similarity Preservation vs. Privacy

HMS preserves semantic similarity (similar texts produce similar vectors) while destroying exact content. This is the fundamental privacy/utility tradeoff:

| Property | Preserved | Destroyed |
|----------|-----------|-----------|
| Semantic similarity between documents | Yes | - |
| Document length | No | Yes |
| Exact word content | No | Yes |
| Word order (beyond trigram window) | No | Yes |
| Character-level content | No | Yes |
| Punctuation and formatting | No | Yes |

## Differential Privacy

### Mechanism

When `PrivacyConfig.dp_enabled = true`, the `bundle()` operation satisfies epsilon-differential privacy via the Laplace mechanism.

### Formal Guarantee

Let B(D) denote the bundle of a dataset D. For any two datasets D and D' differing in at most one element, and for any set of outputs S:

```
P[B_dp(D) in S] <= exp(epsilon) * P[B_dp(D') in S]
```

### Implementation

1. Per-index frequency counts are computed across input vectors
2. Independent Laplace noise ~ Lap(0, 1/epsilon) is added to each count
3. Noisy counts are thresholded at n/2 (majority rule)
4. Top D/256 indices by noisy count are selected

### Sensitivity Analysis

The sensitivity of the frequency count function is 1: adding or removing one vector from the bundle changes any single index's count by at most 1. Therefore, adding Lap(0, 1/epsilon) noise to each count achieves epsilon-DP.

### Epsilon Guidance

| Epsilon | Privacy Level | Use Case |
|---------|--------------|----------|
| 0.1 | Strong | Medical records, PII |
| 1.0 | Moderate | General documents (default) |
| 10.0 | Weak | Non-sensitive content |

Lower epsilon = more noise = more privacy = lower accuracy. The utility impact depends on the number of vectors being bundled: larger bundles are more robust to noise.

### Composition

Each `bundle_dp()` call consumes epsilon from the privacy budget. For k sequential bundle operations on overlapping data, the total privacy cost is k * epsilon (basic composition) or sqrt(2k * ln(1/delta)) * epsilon + k * epsilon * (exp(epsilon) - 1) (advanced composition).

The calling application is responsible for tracking cumulative privacy budget.

## Privacy Under Encryption

When `SecurityConfig.encryption_enabled = true`:

1. **Data at rest**: All arena segments and index files are AES-256-GCM encrypted. An adversary with disk access sees only ciphertext.
2. **Audit log**: Stores SHA-256 hashes of vector IDs, not raw IDs. The audit trail reveals operation timestamps and types but not content.
3. **Key derivation**: Argon2id with random salt prevents rainbow table attacks on the passphrase.

### Combined Protection Layers

| Layer | Protects Against |
|-------|-----------------|
| Lossy encoding (default) | Content reconstruction from vectors |
| Differential privacy (optional) | Membership inference on bundled vectors |
| Encryption at rest (optional) | Unauthorized disk access |
| Audit trail hashing | ID enumeration from audit log |
| Ed25519 signing (optional) | Tampering with audit history |

## Limitations

1. **Similarity oracle**: An adversary with query access can probe the system to determine if two texts are similar. This is inherent to any similarity search system.
2. **Trigram fingerprinting**: Short, unique strings (proper nouns, IDs) may be identifiable by their trigram pattern if the adversary knows the encoding scheme and the candidate set is small.
3. **No query privacy**: Queries are processed in plaintext in memory. HMS does not implement private information retrieval (PIR).
4. **Temporal patterns**: The audit log reveals when operations occurred, even if content is hidden.
5. **DP is per-bundle**: Differential privacy applies to the `bundle()` operation only, not to individual `memorize()` calls. Individual vectors are deterministic given the input text.
