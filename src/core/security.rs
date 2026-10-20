// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "security")]
use anyhow::{anyhow, Result};
#[cfg(feature = "security")]
use std::path::Path;

#[cfg(feature = "security")]
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
#[cfg(feature = "security")]
use zeroize::Zeroize;

#[cfg(feature = "security")]
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};

#[cfg(feature = "security")]
pub struct SigningManager {
    signing_key: SigningKey,
    #[allow(dead_code)]
    verifying_key: VerifyingKey,
}

#[cfg(feature = "security")]
impl SigningManager {
    /// Load or generate an Ed25519 keypair.
    /// If `key_path` exists, loads from it. Otherwise generates a new keypair
    /// and saves it to `key_path`.
    pub fn new(key_path: &Path) -> Result<Self> {
        if key_path.exists() {
            let mut key_bytes = std::fs::read(key_path)?;
            if key_bytes.len() != 32 {
                key_bytes.zeroize();
                return Err(anyhow!(
                    "Invalid signing key file: expected 32 bytes, got {}",
                    key_bytes.len()
                ));
            }
            let mut bytes: [u8; 32] = key_bytes[..32].try_into().expect("length checked above");
            key_bytes.zeroize();
            let signing_key = SigningKey::from_bytes(&bytes);
            bytes.zeroize();
            let verifying_key = signing_key.verifying_key();
            Ok(Self {
                signing_key,
                verifying_key,
            })
        } else {
            let signing_key = SigningKey::from_bytes(&rand::random());
            let verifying_key = signing_key.verifying_key();
            if let Some(parent) = key_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            write_private_key(key_path, &signing_key)?;
            Ok(Self {
                signing_key,
                verifying_key,
            })
        }
    }

    /// Sign arbitrary data, returning a 64-byte Ed25519 signature.
    pub fn sign(&self, data: &[u8]) -> [u8; 64] {
        self.signing_key.sign(data).to_bytes()
    }

    /// Verify a signature against data.
    #[allow(dead_code)]
    pub fn verify(&self, data: &[u8], signature: &[u8; 64]) -> Result<()> {
        let sig = ed25519_dalek::Signature::from_bytes(signature);
        self.verifying_key
            .verify(data, &sig)
            .map_err(|e| anyhow!("Signature verification failed: {}", e))
    }

    /// Export the public verifying key (32 bytes).
    #[allow(dead_code)]
    pub fn verifying_key_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }

    /// Move the signing key out for use by other managers. Only consumed by the
    /// provenance layer today, so it is dead under `security`-without-`provenance`.
    #[allow(dead_code)]
    pub fn into_signing_key(self) -> SigningKey {
        self.signing_key
    }
}

/// Write an Ed25519 private key to disk. On unix the file is created with
/// owner-only permissions (0600) so it is not world-readable on shared hosts;
/// the serialized key bytes are zeroized immediately after writing.
#[cfg(feature = "security")]
fn write_private_key(key_path: &Path, signing_key: &SigningKey) -> Result<()> {
    use std::io::Write;

    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts.open(key_path)?;
    let mut bytes = signing_key.to_bytes();
    let result = file.write_all(&bytes);
    bytes.zeroize();
    result.map_err(|e| anyhow!("failed to write signing key: {e}"))
}

/// The AES-256-GCM cipher behind one heap allocation: moving the manager moves a pointer, not
/// the key schedule (whose first two round keys are the key), and the `aes-gcm/zeroize`
/// feature zeroizes the schedule when the box drops.
#[cfg(feature = "security")]
pub struct EncryptionManager {
    cipher: Box<Aes256Gcm>,
}

#[cfg(feature = "security")]
/// Bytes of stack `scrub_stack` overwrites below its caller's frame.
const SCRUB_BYTES: usize = 256 * 1024;

/// Overwrites the stack region below the caller's frame so secrets left there by callees
/// (Argon2 temporaries, a cipher value before it moved to the heap) do not linger. Best
/// effort: the language guarantees nothing about frame layout, so the key-zeroization test
/// in this module measures the result rather than assuming it.
#[cfg(feature = "security")]
#[inline(never)]
pub fn scrub_stack() {
    let mut pad = [0u64; SCRUB_BYTES / 8];
    for word in pad.iter_mut() {
        // SAFETY: `word` is a valid, aligned, exclusive reference into `pad`; the volatile
        // write keeps the store from being elided as dead.
        unsafe { std::ptr::write_volatile(word, 0) };
    }
    std::hint::black_box(&pad);
}

#[cfg(feature = "security")]
impl EncryptionManager {
    /// Derive an AES-256 key from a passphrase using Argon2id and create the cipher.
    /// The salt is stored at `{storage_path}/encryption.salt`. If it doesn't exist,
    /// a new random salt is generated and saved.
    pub fn new(passphrase: &str, storage_path: &Path) -> Result<Self> {
        use argon2::Argon2;

        let salt_path = storage_path.join("encryption.salt");
        let salt = if salt_path.exists() {
            let s = std::fs::read(&salt_path)?;
            if s.len() != 16 {
                return Err(anyhow!(
                    "Invalid salt file: expected 16 bytes, got {}",
                    s.len()
                ));
            }
            s
        } else {
            let mut s = vec![0u8; 16];
            use rand::Rng;
            rand::rng().fill_bytes(&mut s);
            std::fs::write(&salt_path, &s)?;
            s
        };

        let mut key = [0u8; 32]; // codeql[rust/hard-coded-cryptographic-value] False positive: buffer is overwritten by Argon2
        Argon2::default()
            .hash_password_into(passphrase.as_bytes(), &salt, &mut key)
            .map_err(|e| anyhow!("Argon2 key derivation failed: {}", e))?;

        let cipher = Self::build_cipher(&key);
        key.zeroize();
        // Argon2's finalization and the cipher value before it moved into its box left copies
        // of the key in callee frames.
        scrub_stack();
        Ok(Self { cipher: cipher? })
    }

    /// Builds the cipher in its own frame so the pre-box copy of the key schedule lies in
    /// the region `scrub_stack` overwrites.
    #[inline(never)]
    fn build_cipher(key: &[u8; 32]) -> Result<Box<Aes256Gcm>> {
        Ok(Box::new(
            Aes256Gcm::new_from_slice(key)
                .map_err(|e| anyhow!("AES-256-GCM init failed: {}", e))?,
        ))
    }

    /// Encrypt data. Returns `[nonce:12][ciphertext+tag]`.
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let mut nonce_bytes = [0u8; 12];
        use rand::Rng;
        rand::rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::try_from(nonce_bytes.as_slice())
            .map_err(|_| anyhow!("AES-256-GCM nonce construction failed"))?;

        let ciphertext = self
            .cipher
            .encrypt(&nonce, plaintext)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;

        let mut output = Vec::with_capacity(12 + ciphertext.len());
        output.extend_from_slice(&nonce_bytes);
        output.extend_from_slice(&ciphertext);
        Ok(output)
    }

    /// Like [`encrypt`](Self::encrypt), binding `aad` into the authentication tag.
    pub fn encrypt_with_aad(&self, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
        use aes_gcm::aead::Payload;
        let mut nonce_bytes = [0u8; 12];
        use rand::Rng;
        rand::rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::try_from(nonce_bytes.as_slice())
            .map_err(|_| anyhow!("AES-256-GCM nonce construction failed"))?;
        let ciphertext = self
            .cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: plaintext,
                    aad,
                },
            )
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;
        let mut output = Vec::with_capacity(12 + ciphertext.len());
        output.extend_from_slice(&nonce_bytes);
        output.extend_from_slice(&ciphertext);
        Ok(output)
    }

    /// Decrypt data produced by `encrypt_with_aad()` with the same `aad`.
    pub fn decrypt_with_aad(&self, encrypted: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
        use aes_gcm::aead::Payload;
        if encrypted.len() < 12 {
            return Err(anyhow!(
                "Encrypted data too short: {} bytes",
                encrypted.len()
            ));
        }
        let nonce = Nonce::try_from(&encrypted[..12])
            .map_err(|_| anyhow!("AES-256-GCM nonce construction failed"))?;
        self.cipher
            .decrypt(
                &nonce,
                Payload {
                    msg: &encrypted[12..],
                    aad,
                },
            )
            .map_err(|e| anyhow!("Decryption failed: {}", e))
    }

    /// Decrypt data produced by `encrypt()`. Input: `[nonce:12][ciphertext+tag]`.
    pub fn decrypt(&self, encrypted: &[u8]) -> Result<Vec<u8>> {
        if encrypted.len() < 12 {
            return Err(anyhow!(
                "Encrypted data too short: {} bytes (minimum 12 for nonce)",
                encrypted.len()
            ));
        }
        let nonce = Nonce::try_from(&encrypted[..12])
            .map_err(|_| anyhow!("AES-256-GCM nonce construction failed"))?;
        let ciphertext = &encrypted[12..];

        self.cipher
            .decrypt(&nonce, ciphertext)
            .map_err(|e| anyhow!("Decryption failed: {}", e))
    }
}

/// Hash an ID using SHA-256 for audit log entries (avoids storing raw IDs).
#[cfg(feature = "security")]
pub fn hash_id(id: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(id.as_bytes());
    hasher.finalize().into()
}

/// Stub for non-security builds -- hash_id using the built-in FxHash (non-crypto, 8 bytes zero-padded).
#[cfg(not(feature = "security"))]
pub fn hash_id(id: &str) -> [u8; 32] {
    use fxhash::FxHasher;
    use std::hash::Hasher;
    let mut h = FxHasher::default();
    h.write(id.as_bytes());
    let hash = h.finish().to_le_bytes();
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&hash);
    out
}

#[cfg(all(test, feature = "security"))]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// The AES key schedule lives in one heap allocation that is zeroized when the manager
    /// drops: afterwards no readable heap or anonymous mapping holds the derived key. The
    /// needle is kept XOR-masked so the test's own copy never matches, and each scan buffer
    /// is zeroized before release so a scan cannot find its own earlier copy.
    #[cfg(target_os = "linux")]
    #[test]
    fn derived_key_leaves_the_heap_when_the_manager_drops() -> Result<()> {
        use argon2::Argon2;
        use std::io::{Read, Seek, SeekFrom};

        /// Where the masked needle occurs in the process's writable anonymous mappings, as
        /// `range perms name +offset` lines.
        fn heap_copies(masked: &[u8], mask: &[u8]) -> Result<Vec<String>> {
            let maps = std::fs::read_to_string("/proc/self/maps")?;
            let mut mem = std::fs::File::open("/proc/self/mem")?;
            let mut hits = Vec::new();
            for line in maps.lines() {
                let mut parts = line.split_whitespace();
                let (Some(range), Some(perms)) = (parts.next(), parts.next()) else {
                    continue;
                };
                let name = parts.nth(3).unwrap_or("");
                if !perms.starts_with("rw") || !(name.is_empty() || name == "[heap]") {
                    continue;
                }
                let (lo, hi) = range
                    .split_once('-')
                    .ok_or_else(|| anyhow!("maps range {range}"))?;
                let (lo, hi) = (u64::from_str_radix(lo, 16)?, u64::from_str_radix(hi, 16)?);
                if hi <= lo || hi - lo > 1 << 32 {
                    continue;
                }
                // Read in chunks so a page another test thread unmaps mid-scan costs one
                // chunk, not the whole mapping; chunks overlap by the needle length so no
                // occurrence straddles a boundary unseen.
                const CHUNK: u64 = 1 << 16;
                let overlap = masked.len() as u64 - 1;
                let mut buf = vec![0u8; (CHUNK + overlap) as usize];
                let mut at = lo;
                while at < hi {
                    let len = (CHUNK + overlap).min(hi - at) as usize;
                    let chunk = &mut buf[..len];
                    if mem.seek(SeekFrom::Start(at)).is_ok() && mem.read_exact(chunk).is_ok() {
                        for (offset, w) in chunk.windows(masked.len()).enumerate() {
                            if offset as u64 >= CHUNK && at + CHUNK < hi {
                                break;
                            }
                            if w.iter()
                                .zip(masked)
                                .zip(mask)
                                .all(|((b, n), m)| b ^ m == *n)
                            {
                                hits.push(format!(
                                    "{range} {perms} {name:?} +{:#x}",
                                    at - lo + offset as u64
                                ));
                            }
                        }
                    }
                    chunk.zeroize();
                    at += CHUNK;
                }
                buf.zeroize();
            }
            Ok(hits)
        }

        let dir = tempdir()?;
        let passphrase = "zeroization probe passphrase";
        let manager = EncryptionManager::new(passphrase, dir.path())?;
        let salt = std::fs::read(dir.path().join("encryption.salt"))?;
        let mask: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(0x9d) ^ 0x5a);
        let mut key = [0u8; 32];
        Argon2::default()
            .hash_password_into(passphrase.as_bytes(), &salt, &mut key)
            .map_err(|e| anyhow!("argon2: {e}"))?;
        let masked: Vec<u8> = key.iter().zip(&mask).map(|(k, m)| k ^ m).collect();
        key.zeroize();
        // The test's own derivation left the key in callee frames, like the manager's did.
        scrub_stack();
        let ciphertext = manager.encrypt(b"payload")?;
        // encrypt leaves key-schedule temporaries in the frames it used; like the derivation
        // above, they are stack residue of this thread, not state the manager owns.
        scrub_stack();
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("aes") {
            let mut live = false;
            for _ in 0..20 {
                if !heap_copies(&masked, &mask)?.is_empty() {
                    live = true;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            assert!(
                live,
                "a live AES-NI cipher keeps round key 0 (the key) in its heap schedule"
            );
        }
        drop(manager);
        let survivors = heap_copies(&masked, &mask)?;
        assert!(
            survivors.is_empty(),
            "derived key survives the manager's drop at {survivors:?}"
        );
        assert!(!ciphertext.is_empty());
        Ok(())
    }

    #[test]
    fn test_signing_roundtrip() {
        let dir = tempdir().unwrap();
        let key_path = dir.path().join("test.key");
        let mgr = SigningManager::new(&key_path).unwrap();

        let data = b"hello world";
        let sig = mgr.sign(data);
        mgr.verify(data, &sig).unwrap();
    }

    #[test]
    fn test_signing_tamper_detection() {
        let dir = tempdir().unwrap();
        let key_path = dir.path().join("test.key");
        let mgr = SigningManager::new(&key_path).unwrap();

        let data = b"hello world";
        let sig = mgr.sign(data);
        assert!(mgr.verify(b"tampered", &sig).is_err());
    }

    #[test]
    fn test_signing_key_persistence() {
        let dir = tempdir().unwrap();
        let key_path = dir.path().join("test.key");

        let vk1 = {
            let mgr = SigningManager::new(&key_path).unwrap();
            mgr.verifying_key_bytes()
        };
        let vk2 = {
            let mgr = SigningManager::new(&key_path).unwrap();
            mgr.verifying_key_bytes()
        };
        assert_eq!(vk1, vk2, "Reloaded key should match");
    }

    #[test]
    fn test_encryption_roundtrip() {
        let dir = tempdir().unwrap();
        let mgr = EncryptionManager::new("test-passphrase", dir.path()).unwrap();

        let plaintext = b"sensitive data here";
        let encrypted = mgr.encrypt(plaintext).unwrap();
        assert_ne!(&encrypted[12..], plaintext);

        let decrypted = mgr.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encryption_salt_persistence() {
        let dir = tempdir().unwrap();

        let encrypted = {
            let mgr = EncryptionManager::new("passphrase", dir.path()).unwrap();
            mgr.encrypt(b"test data").unwrap()
        };

        let mgr = EncryptionManager::new("passphrase", dir.path()).unwrap();
        let decrypted = mgr.decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, b"test data");
    }

    #[test]
    fn test_encryption_wrong_passphrase() {
        let dir = tempdir().unwrap();

        let encrypted = {
            let mgr = EncryptionManager::new("correct", dir.path()).unwrap();
            mgr.encrypt(b"secret").unwrap()
        };

        // Remove salt so second manager generates fresh key
        std::fs::remove_file(dir.path().join("encryption.salt")).unwrap();
        let mgr = EncryptionManager::new("wrong", dir.path()).unwrap();
        assert!(mgr.decrypt(&encrypted).is_err());
    }

    #[test]
    fn test_hash_id_deterministic() {
        let h1 = hash_id("test-id");
        let h2 = hash_id("test-id");
        assert_eq!(h1, h2);
        assert_ne!(hash_id("other"), h1);
    }
}
