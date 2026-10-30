<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security lifecycle and recovery evidence

This phase targets immutable identity publication, nonce generation, safe scrub ownership and concrete numeric bounds. It does not certify cryptography, finite-precision DP, hostile-host isolation or hardware durability. The authored local runtime checks have passed in final validation; [the final evidence record](final-validation/RESULTS.md) and campaign ledger distinguish those results from external assurance limits and the platform gate.

## Failure-state model

Key/salt initialization is `absent → owned temporary bytes → file sync → no-clobber publication → directory sync → reopen fixed identity`. Before publication, failure leaves no destination; interruption can leave a private temporary file. After publication, a failed directory sync returns an error and a retry reopens the same destination. Another initializer cannot replace the winner. A preexisting malformed, symlinked, nonregular or permissively readable Unix signing key fails closed. Parent directories are trusted service-account storage; this is not an openat-based sandbox against hostile directory replacement. Temporary files may remain after process termination and are not automatically interpreted as identities or deleted by filename guesses.

The normal core still owns its store lock; the new publication rule also protects standalone managers and different stores sharing a custom signing path. Secret reads allocate fixed arrays and one sentinel byte. Both successful and failed derivations zeroize their local key buffers. This does not establish absence of every compiler/allocator/stack copy. Key rotation still requires the caller's durable configuration to select the new key after its signed rotation record; creation alone is not rotation.

`scrub_remove_file` opens the final component without following symlinks/reparse points, rejects nonregular files, and refuses multiply linked Unix files before overwriting. It retains the documented copy-on-write/SSD/snapshot limits. Scrubbing compaction also drops stale dense bases before optional rebuilding, so a deleted owner's retained Arc is released even when live embedded rows fall below the training minimum. Concurrent readers can retain their already captured owners until those reads finish.

## Cryptographic review inputs

Review `security.rs`, `durable_file.rs`, `entangled.rs`, the encryption/AAD call sites, Cargo.lock and the unchanged encrypted-format goldens. Capture the exact source/build hashes and platform configuration in final validation. Existing primitives remain AES-256-GCM, Argon2id and Ed25519. Nonces now come directly from fallible `rand::rngs::SysRng`; [rand documents](https://docs.rs/rand/0.10.2/rand/rngs/struct.ThreadRng.html) why its thread-local generator requires reseeding after fork. The format remains 12-byte random nonce followed by ciphertext/tag; old ciphertexts remain readable.

For independent nonce samples under one key, the union-bound collision probability is q(q−1)/2^97. Independent entropy does not bound cumulative usage. Review all arena, ANN/cache, compaction, backup-copy and direct-helper invocations against [SP 800-38D](https://csrc.nist.gov/pubs/sp/800/38/d/final). No persistent global invocation counter or automatic re-encryption/rotation is supplied. Include Argon2 parameters/salt handling, nonce collisions and domains, AAD binding, wrong-key/tag rejection, private-file ACLs and OS entropy-failure behavior in external review. No new cryptographic primitive is introduced.

For DP bundling, public D and add/remove-one-vector adjacency give L1 sensitivity m=max(1,floor(D/256)) after clipping. Replacement adjacency costs 2m; basic composition sums epsilon over all released bundles. Top-m is post-processing only in the ideal real-arithmetic argument. The f64 sampler uses a discrete uniform draw, logarithms, rounded counts and deterministic tie resolution: finite-precision privacy and unbounded repeated releases remain unverified. The new scale bound prevents an actual infinity path, and a fresh OS-seeded standard generator prevents fork-cloned noise. These changes are not a DP proof. Masking continues to expose equality, weights, coordinate frequencies and every pairwise similarity; private-search remains its stated experimental honest-but-curious protocol.

## Authored checks and final commands

Use the final repository-owned Cargo target and retained manifest features:

```sh
cargo test --offline --locked --features security --lib core::security::tests::
cargo test --offline --locked --lib core::durable_file::tests::
cargo test --offline --locked --features security --lib core::engine::compaction::tests::
cargo test --offline --locked --lib core::entangled::tests::
cargo test --offline --locked --features security --test allocation_bounds --test format_golden --test format_crash
cargo test --offline --locked --features provenance,server --test committed_sidecars
```

The new checks cover eight concurrent signing initializers, competing salts/cross-decryption, 31/32/33-byte signing files, 15/16/17-byte salts, sparse oversized files, symlink/FIFO/directory/hardlink rejection, interrupted identity publication at three transitions, and scrubbed encrypted/plain dense owners at 127/128 live rows. Existing tenant-restore transition, wrong-passphrase, AAD, provenance-rotation and binding-persistence tests remain gates. No key/golden is rewritten to satisfy a check. Windows needs real reparse-point and service-account ACL execution; the current host cannot establish its behavior from source alone.

Hardware experiment, reserved for a separately provisioned fixture host: record filesystem/mount options, device/cache/firmware, OS, source hash, and flush-return tracing. Use a fresh synthetic encrypted store and a signed sequence of acknowledged writes held by an independent controller. Cut actual device power at each publication boundary and immediately after acknowledgment; reboot and compare the full acknowledged prefix, identity bytes, ciphertext authentication and active-generation marker. Success requires every acknowledged record and unchanged identity, with either the old or new complete generation for unacknowledged publication; partial/cross-generation acceptance or silent identity replacement fails. Include unsupported full-flush volumes as a separate reported configuration. Process `exit`/kill tests do not establish this result. Do not run against production stores.

Performance/resource tradeoffs are unmeasured: OS entropy adds a call per encryption; identity creation adds file/directory syncs; scrubbing may rebuild a dense tree. These costs buy explicit lifecycle properties and belong in the final update/maintenance accounting. The existing external hardware and independent cryptographic-review requirements remain open.
