<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Encryption at rest

Requires the `security` feature to read or write. Enabled per store (`store.json` `encrypted:
true`, fixed for the store's life).

## Key

`encryption.salt` is 16 random bytes, written once before the first frame (atomically, fsynced).
The key is Argon2id (version 0x13) over the passphrase bytes with that salt: memory 19456 KiB,
2 passes, 1 lane, 32-byte output, no secret and no associated data. These are the defaults of
the `argon2` crate at 0.5 and 0.6; `tests/golden/v4-encrypted` and `v061-encrypted` fail if a
dependency changes them. The key is AES-256-GCM. A wrong passphrase or altered frame is an
authentication failure; the store is not opened.

## Journal frames

The frame payload (after LZ4, CRC) is `[nonce:12][ciphertext||tag:16]` where the plaintext is
the whole `HMS-TXN-2\0<json>` payload. 12-byte random nonce per frame, **no associated data,
no version byte**. The frame header CRC32 covers the *encrypted* payload bytes as stored
(plaintext of the frame layer), so framing is checkable without the key; authenticity needs it.
Consequence: a frame can be copied between positions of the same store undetected by the AEAD
(the CRC and replay order still apply).

## Text segment

Version 2 records are `[nonce:12][ciphertext||tag:16]` with the record's byte offset in the
segment (u64 LE) as associated data, so a record cannot be moved. Header flag bit 0 marks it.

## Index caches

`nsg_index*.bin` / `ivf_index*.bin` are sealed with the same `[nonce][ct||tag]` envelope, no
associated data; `indices.json` records the CRC32 of the sealed bytes.

## Not encrypted

`store.json`, `format.json`, `CURRENT`, `indices.json` (journal length, shard count, CRCs),
`shard_meta.json`, `encryption.salt`, `audit.bin` (ids are SHA-256 hashed, timestamps clear),
file names and sizes, frame lengths, `seq`. An encrypted store reveals its frame count and
approximate sizes.
