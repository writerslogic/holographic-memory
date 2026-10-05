# Privacy and security boundaries

## Local processing and retained data

The core engine makes no external embedding API calls. Optional ONNX model adapters load
explicit local directories with `local_files_only: true`; obtaining those models is a separate
application setup step. Queries and model inference run in plaintext in process memory.

Deterministic text hashes and sparse projections are lossy representations, **not encryption
or anonymization**. An attacker can encode likely inputs and compare them, infer membership,
or use lexical and similarity signals. No resistance to such attacks is implied by the
number of possible bit combinations. The built-in text encoder captures lexical overlap;
semantic retrieval requires suitable supplied embeddings.

Document ingestion retains raw passages by default. Setting `storeText: false` removes those
passages but still stores lexical term counts, metadata, source URIs, IDs, and any embeddings.
Do not use it as a privacy guarantee. Deletion removes logical records; filesystem snapshots,
backups, obsolete segments, and storage devices may retain previous bytes. Compaction is not
secure erasure.

## Encryption and signing

With `security` compiled and encryption enabled, arena payloads and persisted ANN caches use
AES-256-GCM. Argon2id derives the encryption key from a passphrase and per-store salt. Prefer
`encryptionPassphraseEnv` to embedding a secret in application configuration. Schema metadata,
file names, sizes, topology, salts, and access patterns are not encrypted. Signing keys and
optional audit/provenance sidecars have their own storage policies.

Configuration fails when an unavailable security capability is requested. Schema validation
prevents switching an existing encrypted store to plaintext, and authenticated decryption
errors stop opening the store. `securityStatus()` distinguishes compiled support from active
encryption, signing, audit, and DP settings. Checksums detect accidental corruption; CRC32 is
not authentication. An audit log records operation timing and ID hashes; low-entropy IDs may
still be guessed. Signing does not provide confidentiality.

## Client-side vector masking

`core::mask::VectorMask` (feature `security`) permutes vector coordinates with a secret
permutation derived from a passphrase by Argon2id. A client can mask vectors before sending
them to a store it does not fully trust. It is **obfuscation, not encryption**, and not
homomorphic encryption:

- Every pairwise similarity is preserved exactly, so the store learns the full similarity
  graph, vector equality, vector weight, and per-coordinate activation frequency.
- Masking is deterministic. Each known (plain, masked) pair reveals the image of that
  vector's active coordinates as a set; enough pairs recover the permutation.
- An attacker who can encode likely inputs can still test membership once any part of the
  permutation is known, and frequency analysis of common coordinates needs no known pairs.
- The engine itself applies no mask. A process holding both the passphrase and the vectors
  gains nothing from masking; use storage encryption for confidentiality at rest.

Set algebra (bind, bundle, similarity) commutes with the mask. Cyclic permutation used for
ordered binding does not, so compose sequences before masking.

## Credential-gated agent access

`core::provenance::access` (feature `provenance`) admits agents by W3C Verifiable Credentials
carrying an `eddsa-jcs-2022` proof from an issuer `did:key` the host has explicitly trusted,
with an optional expiry and explicit revocation. `HmsCore::query_as` refuses a query unless
the named agent holds a current read grant. The registry is in-memory and per-process. It
decides authorization only: the host must authenticate that the caller controls the DID it
presents, and in-process callers can still use the ungated `query`. PoSME receipts and RATS
attestation evidence are not verified anywhere in HMS.

## Differentially private bundling

DP applies only to the configured **bundle operation**, not ingestion, retrieval, document
metadata, source passages, or the whole application. The intended neighboring datasets differ
by adding or removing one input vector in a fixed, public dimension D. Each input is clipped
to its first m = max(1, floor(D / 256)) active coordinates. The mechanism:

1. Counts clipped contributions across all D coordinates, including coordinates absent from input.
2. Adds independent Laplace noise of scale m / epsilon to every count.
3. Releases the fixed-size top m coordinates of the noisy histogram.

A single input changes the histogram by at most m in L1 norm. The Laplace mechanism therefore
has the stated epsilon bound in the ideal real-arithmetic model under add/remove adjacency;
top-m selection is post-processing. Replacing one input can change L1 distance by 2m and has
a 2-epsilon bound under basic composition. The implementation uses floating-point sampling;
it has not been independently audited as a finite-precision DP mechanism. See the
[Laplace mechanism and post-processing analysis](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf).

`HmsCore::bundle` uses the configured public dimensions even for empty input. The low-level
`EntangledHVec::bundle_dp` infers dimensions from the first vector (or zero for empty
input); do not use that helper across datasets whose inferred dimensions can differ. Epsilon
must be finite and positive. Lower values add more noise, but no epsilon value is universally
appropriate for a category of sensitive data.

Repeated releases consume additional budget. The caller must account for every release and
choose an adjacency model and privacy budget suitable for its use case. HMS does not provide
a persistent budget accountant, query privacy, private information retrieval, or a claim that
non-DP outputs become private when bundling is enabled.
