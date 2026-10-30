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
        let bytes = load_or_create::<32>(key_path, true)?;
        let signing_key = SigningKey::from_bytes(&bytes);
        let verifying_key = signing_key.verifying_key();
        Ok(Self {
            signing_key,
            verifying_key,
        })
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

#[cfg(feature = "security")]
fn read_fixed<const N: usize>(
    path: &Path,
    private: bool,
) -> Result<Option<zeroize::Zeroizing<[u8; N]>>> {
    use std::io::Read;
    let mut file = match super::durable_file::open_regular(path, false) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    anyhow::ensure!(
        file.metadata()?.len() == N as u64,
        "invalid identity file length: expected {N} bytes"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        anyhow::ensure!(
            !private || file.metadata()?.permissions().mode() & 0o077 == 0,
            "signing key permissions must exclude group and other access"
        );
    }
    #[cfg(not(unix))]
    let _ = private;
    let mut bytes = zeroize::Zeroizing::new([0; N]);
    file.read_exact(&mut *bytes)?;
    let mut extra = zeroize::Zeroizing::new([0; 1]);
    anyhow::ensure!(
        file.read(&mut *extra)? == 0,
        "identity file changed while reading"
    );
    Ok(Some(bytes))
}

#[cfg(feature = "security")]
fn load_or_create<const N: usize>(
    path: &Path,
    private: bool,
) -> Result<zeroize::Zeroizing<[u8; N]>> {
    if let Some(bytes) = read_fixed(path, private)? {
        super::durable_file::sync_directory(
            path.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new(".")),
        )?;
        return Ok(bytes);
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        super::durable_file::create_identity_parent(parent)?;
    }
    let mut proposed = zeroize::Zeroizing::new([0; N]);
    secure_random(&mut *proposed)?;
    super::durable_file::atomic_create(path, &*proposed)?;
    read_fixed(path, private)?.ok_or_else(|| anyhow!("identity disappeared after publication"))
}

#[cfg(feature = "security")]
fn secure_random(bytes: &mut [u8]) -> Result<()> {
    use rand::TryRng;
    rand::rngs::SysRng
        .try_fill_bytes(bytes)
        .map_err(|_| anyhow!("operating system randomness unavailable"))
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
        let salt = load_or_create::<16>(&salt_path, false)?;

        let mut key = zeroize::Zeroizing::new([0u8; 32]);
        Argon2::default()
            .hash_password_into(passphrase.as_bytes(), &*salt, &mut *key)
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
        // GCM with empty associated data is exactly GCM without it.
        self.encrypt_with_aad(plaintext, b"")
    }

    /// Like [`encrypt`](Self::encrypt), binding `aad` into the authentication tag.
    pub fn encrypt_with_aad(&self, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>> {
        use aes_gcm::aead::Payload;
        let mut nonce_bytes = [0u8; 12];
        secure_random(&mut nonce_bytes)?;
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

/// Stub for non-security builds -- hash_id using Fx hashing (non-crypto, 8 bytes zero-padded).
#[cfg(not(feature = "security"))]
pub fn hash_id(id: &str) -> [u8; 32] {
    let hash = crate::core::hash::fx64(id.as_bytes()).to_le_bytes();
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
    fn signature_verifies_only_the_signed_data() {
        let dir = tempdir().unwrap();
        let key_path = dir.path().join("test.key");
        let mgr = SigningManager::new(&key_path).unwrap();

        let data = b"hello world";
        let sig = mgr.sign(data);
        mgr.verify(data, &sig).unwrap();
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
    fn concurrent_identity_creation_converges_without_replacement() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("nested/key");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let keys: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let path = &path;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        SigningManager::new(path).unwrap().verifying_key_bytes()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(keys.iter().all(|key| *key == keys[0]));
        assert_eq!(
            SigningManager::new(&path).unwrap().verifying_key_bytes(),
            keys[0]
        );
        let before = std::fs::read(&path).unwrap();
        assert!(!super::super::durable_file::atomic_create(&path, &[0; 32]).unwrap());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn concurrent_salt_creation_produces_interoperable_ciphertexts() {
        let directory = tempdir().unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let encrypted: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    let directory = &directory;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        EncryptionManager::new("shared", directory.path())
                            .unwrap()
                            .encrypt_with_aad(b"stored", b"domain")
                            .unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let reopened = EncryptionManager::new("shared", directory.path()).unwrap();
        for ciphertext in &encrypted {
            assert_eq!(
                reopened.decrypt_with_aad(ciphertext, b"domain").unwrap(),
                b"stored"
            );
            assert!(reopened
                .decrypt_with_aad(ciphertext, b"other-domain")
                .is_err());
        }
        assert_ne!(encrypted[0][..12], encrypted[1][..12]);
    }

    #[test]
    fn identity_lengths_and_links_fail_without_rewriting_input() {
        let directory = tempdir().unwrap();
        let key = directory.path().join("key");
        for size in [31, 32, 33, 1 << 20] {
            let file = std::fs::File::create(&key).unwrap();
            file.set_len(size).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                file.set_permissions(std::fs::Permissions::from_mode(0o600))
                    .unwrap();
            }
            assert_eq!(SigningManager::new(&key).is_ok(), size == 32);
            assert_eq!(std::fs::metadata(&key).unwrap().len(), size);
        }
        let salt = directory.path().join("encryption.salt");
        for size in [15, 16, 17] {
            let file = std::fs::File::create(&salt).unwrap();
            file.set_len(size).unwrap();
            assert_eq!(read_fixed::<16>(&salt, false).is_ok(), size == 16);
        }
        std::fs::remove_file(&key).unwrap();
        std::fs::create_dir(&key).unwrap();
        assert!(SigningManager::new(&key).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::{symlink, PermissionsExt};
            std::fs::remove_dir(&key).unwrap();
            symlink(&salt, &key).unwrap();
            assert!(SigningManager::new(&key).is_err());
            assert!(key.symlink_metadata().unwrap().file_type().is_symlink());
            std::fs::remove_file(&key).unwrap();
            std::fs::write(&key, [0; 32]).unwrap();
            std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o644)).unwrap();
            assert!(SigningManager::new(&key).is_err());
        }
    }

    #[test]
    fn identity_publication_crash_child() {
        let Some(path) = std::env::var_os("HMS_IDENTITY_TEST_PATH") else {
            return;
        };
        SigningManager::new(Path::new(&path)).unwrap();
        panic!("expected publication interruption");
    }

    #[test]
    fn identity_reopens_after_each_publication_interruption() {
        for phase in ["synced", "published", "durable"] {
            let directory = tempdir().unwrap();
            let path = directory.path().join("key");
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "core::security::tests::identity_publication_crash_child",
                ])
                .env("HMS_IDENTITY_TEST_PATH", &path)
                .env("HMS_IDENTITY_CRASH_AT", phase)
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(84), "{phase}: {output:?}");
            let published = std::fs::read(&path).ok();
            assert_eq!(published.is_some(), phase != "synced");
            let manager = SigningManager::new(&path).unwrap();
            if let Some(bytes) = published {
                assert_eq!(std::fs::read(&path).unwrap(), bytes);
            }
            let next = SigningManager::new(&path).unwrap();
            next.verify(b"recovered", &manager.sign(b"recovered"))
                .unwrap();
        }
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
        let same_salt = EncryptionManager::new("wrong", dir.path()).unwrap();
        assert!(same_salt.decrypt(&encrypted).is_err());

        // A fresh salt derives yet another key.
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
