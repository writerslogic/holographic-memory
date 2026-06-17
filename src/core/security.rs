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
            let bytes: [u8; 32] = key_bytes[..32].try_into().unwrap();
            key_bytes.zeroize();
            let signing_key = SigningKey::from_bytes(&bytes);
            let verifying_key = signing_key.verifying_key();
            Ok(Self {
                signing_key,
                verifying_key,
            })
        } else {
            let signing_key = SigningKey::generate(&mut rand::thread_rng());
            let verifying_key = signing_key.verifying_key();
            if let Some(parent) = key_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(key_path, signing_key.to_bytes())?;
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
    pub fn verify(&self, data: &[u8], signature: &[u8; 64]) -> Result<()> {
        let sig = ed25519_dalek::Signature::from_bytes(signature);
        self.verifying_key
            .verify(data, &sig)
            .map_err(|e| anyhow!("Signature verification failed: {}", e))
    }

    /// Export the public verifying key (32 bytes).
    pub fn verifying_key_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }
}

#[cfg(feature = "security")]
impl Drop for SigningManager {
    fn drop(&mut self) {
        // SigningKey implements Zeroize via ed25519-dalek's zeroize feature
    }
}

#[cfg(feature = "security")]
pub struct EncryptionManager {
    cipher: Aes256Gcm,
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
            use rand::RngCore;
            rand::thread_rng().fill_bytes(&mut s);
            std::fs::write(&salt_path, &s)?;
            s
        };

        let mut key = [0u8; 32];
        Argon2::default()
            .hash_password_into(passphrase.as_bytes(), &salt, &mut key)
            .map_err(|e| anyhow!("Argon2 key derivation failed: {}", e))?;

        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|e| anyhow!("AES-256-GCM init failed: {}", e))?;
        key.zeroize();

        Ok(Self { cipher })
    }

    /// Encrypt data. Returns `[nonce:12][ciphertext+tag]`.
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        let mut nonce_bytes = [0u8; 12];
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;

        let mut output = Vec::with_capacity(12 + ciphertext.len());
        output.extend_from_slice(&nonce_bytes);
        output.extend_from_slice(&ciphertext);
        Ok(output)
    }

    /// Decrypt data produced by `encrypt()`. Input: `[nonce:12][ciphertext+tag]`.
    pub fn decrypt(&self, encrypted: &[u8]) -> Result<Vec<u8>> {
        if encrypted.len() < 12 {
            return Err(anyhow!(
                "Encrypted data too short: {} bytes (minimum 12 for nonce)",
                encrypted.len()
            ));
        }
        let nonce = Nonce::from_slice(&encrypted[..12]);
        let ciphertext = &encrypted[12..];

        self.cipher
            .decrypt(nonce, ciphertext)
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
