# Privacy limits

HMS processes everything locally: the core engine makes no network calls, and the model
adapters only read local directories. That removes the cloud as a threat. It does not make the
stored representation private. This page summarises the limits; [Privacy and security
boundaries](../PRIVACY.md) is the complete statement and wins on any difference.

## The stored representation is not anonymous

- **Hypervector sketches and term counts are not encryption.** The text encoder is
  deterministic and public. Anyone with a store can encode candidate texts and compare, infer
  membership, or use lexical and similarity signals. The number of possible bit patterns is not a
  defence.
- **`storeText: false` is not a privacy guarantee.** It drops the passages but keeps term
  counts, metadata, source URIs, IDs and any embeddings you supplied.
- **Embeddings leak.** A dense embedding is a function of the text; inversion attacks exist.
  Treat supplied embeddings as sensitive as the text.
- **Queries run in plaintext** in process memory, and model inference sees plaintext.

## What encryption covers

With the `security` feature and encryption enabled, arena payloads and persisted index caches
are encrypted with AES-256-GCM under an Argon2id-derived key, and an existing encrypted store
cannot be silently reopened as plaintext. A wrong passphrase fails authenticated decryption and
the store does not open (the [quickstart](../quickstart/index.md) demonstrates this).

What it does not cover:

- **Passage text in `documents.text`.** In the current build the engine's file-backed passage
  cache is written in plaintext even when encryption is on. The quickstart asserts exactly this,
  so the page and the code cannot drift apart: when the engine encrypts that file, the
  quickstart check fails and this section is rewritten.
- Schema metadata, file names, sizes, topology, salts and access patterns.
- Memory of the running process, swap and crash dumps.
- Deletion: removed records can persist in obsolete segments, snapshots and backups. Compaction
  is not secure erasure.
- Integrity is not authentication for checksummed files: CRC32 detects accidents only.

## Masking vectors for a remote store

`VectorMask` (and the Python SDK's `zero_trust_key`) permutes coordinates with a secret derived
from a passphrase so a store can rank vectors without seeing the originals. It is obfuscation,
not encryption: every pairwise similarity is preserved, so the store learns the full
similarity graph, and known plaintext-and-masked pairs reveal the permutation progressively.

## Differential privacy

Differential privacy applies only to the configured **bundle** operation, not to ingestion,
retrieval, metadata or passages. The mechanism has not been audited as a finite-precision
implementation, and HMS keeps no persistent budget accountant.

## Not implemented

Search over encrypted data, query privacy against the server, PoSME receipts, RATS attestation
verification and `did:web` resolution for access credentials. The experimental
[private search](../PRIVATE-SEARCH.md) scoring hides the query vector from the scoring server
only, under the limits stated there.
