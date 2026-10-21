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

With the `security` feature and encryption enabled, arena payloads, persisted index caches and
the file-backed passage text segment (`documents.text`) are encrypted with AES-256-GCM under an Argon2id-derived key, and an existing encrypted store
cannot be silently reopened as plaintext. A wrong passphrase fails authenticated decryption and
the store does not open (the [quickstart](../quickstart/index.md) demonstrates this).

What it does not cover:

- Schema metadata, file names, sizes, topology, salts and access patterns.
- Memory of the running process, swap and crash dumps.
- Deletion: removed records can persist in obsolete segments, snapshots and backups. Compaction
  is not secure erasure.
- Integrity is not authentication for checksummed files: CRC32 detects accidents only.

## Text segment record layout

`documents.text` is a file-backed cache of chunk text. It is rebuilt from the journal when the
store opens, so it is never the only copy of a passage.

- Header: magic `HMST`, version `2`, and a flags byte. Bit 0 set means the records are encrypted.
- Unencrypted store: each record is the raw UTF-8 text.
- Encrypted store: each record is `[nonce: 12 bytes][ciphertext + tag: 16 bytes after the ciphertext]`
  (AES-256-GCM). The additional authenticated data is the record's offset in the segment as a
  little-endian `u64`, so a record cannot be moved to another offset undetected.

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
