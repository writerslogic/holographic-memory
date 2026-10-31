# Encrypting persisted stores

Decision: snapshots are encrypted with **AES-256-GCM through WebCrypto**; the passphrase is stretched
with **PBKDF2-HMAC-SHA256 (600 000 iterations) by default** or **Argon2id inside the module** on request.
The `security` feature's AES code is not shipped in the module.

## Measurement

`bench/crypto-bench.mjs` builds the module with the `inwasm-aes` feature (RustCrypto `aes-gcm`
compiled into the wasm, SIMD build) and times both paths on identical inputs after checking their
outputs are byte-identical. Node 26 / V8 on an Apple Silicon laptop; results are in the `crypto`
section of `benchmarks/results/wasm_v1.json` (median of repeated runs, JS-to-wasm copies included):

| Payload | WebCrypto encrypt | in-wasm encrypt | WebCrypto decrypt | in-wasm decrypt |
| --- | --- | --- | --- | --- |
| 1 MiB | 0.87 ms | 17.0 ms | 1.14 ms | 25.1 ms |
| 10 MiB | 4.9 ms | 226.7 ms | 3.8 ms | 252.7 ms |
| 50 MiB | 27.0 ms | 1411 ms | 15.8 ms | 1233 ms |

WebCrypto is 20 to 90 times faster (hardware AES instructions against a table-based software cipher in
wasm), and it keeps the cipher out of the module, which saves size. Key derivation:

| KDF | Time |
| --- | --- |
| PBKDF2-HMAC-SHA256, 600 000 iterations, WebCrypto | 91 ms |
| Argon2id, 19 456 KiB, 2 passes, 1 lane, in wasm | 73 ms |

WebCrypto has no Argon2, which is why that KDF runs in the module (`argon2idKey`); its AES step still
uses WebCrypto. Not measured: Firefox and Safari WebCrypto against in-wasm AES (the Node figures are V8
only). The gap is large enough that a different engine is unlikely to reverse it, but that is
inference, not measurement.

## Format

```
"WARME" | version u8 | kdf u8 | p1 u32 | p2 u32 | p3 u32 | salt[16] | nonce[12] | ciphertext+tag
```

kdf 1 = PBKDF2 (`p1` = iterations); kdf 2 = Argon2id (`p1` = KiB, `p2` = passes, `p3` = lanes). All
integers are big-endian and the full header is the GCM additional data, so edited parameters fail
authentication. Parameters read from a stored header are bounded (PBKDF2 at most 10 000 000
iterations, Argon2id at most 1 GiB, 64 passes, 16 lanes) before any work is done.

The key is derived once per open. Each save uses a fresh random 96-bit nonce under that key (the salt is
per store, not per save). Opening an encrypted snapshot without a passphrase, or an unencrypted one
with a passphrase, is an error: no silent downgrade. `setEncryption()` changes or removes encryption
explicitly.

## What this does and does not protect

It protects the snapshot at rest in IndexedDB, OPFS or a file. It does not protect a running page: the
decrypted store and the derived key live in memory, and the module does not zeroize them. Argon2id's
memory use (19 MiB by default) stays in the module's linear memory until the page is closed. The
vector-mask utilities (`WasmVectorMask`) are obfuscation, not encryption.
