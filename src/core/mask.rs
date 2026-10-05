// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Client-side keyed obfuscation of sparse hypervectors.
//!
//! A [`VectorMask`] is a secret permutation of the coordinate space, derived
//! from a passphrase with Argon2id. Applying it before vectors leave the client
//! lets a remote store index and rank them without learning which coordinates
//! were originally active.
//!
//! This is obfuscation, **not encryption**, and it is not homomorphic
//! encryption. It is deterministic and preserves every pairwise similarity by
//! construction, so a store holding masked vectors still learns the full
//! similarity structure of the collection, equality of vectors, and coordinate
//! activation frequencies. Each known (plain, masked) pair reveals part of the
//! permutation. Use AES-GCM storage encryption for confidentiality at rest.
//!
//! Set algebra (`bind`, bundling, `similarity`) commutes with the mask. The
//! cyclic `permute` used for ordered binding does not, so sequence operations
//! must be composed before masking.

use anyhow::{anyhow, ensure, Result};
use sha2::{Digest, Sha256};
use zeroize::Zeroize;

use super::entangled::EntangledHVec;

const DOMAIN: &[u8] = b"hms-vector-mask-v1";

/// Largest coordinate space a mask may be derived for (bounds the allocation).
pub const MAX_MASK_DIMENSIONS: usize = 1 << 20;

/// Minimum salt length accepted by [`VectorMask::derive`].
pub const MIN_SALT_LEN: usize = 16;

/// A secret permutation of `[0, dim)`.
pub struct VectorMask {
    forward: Vec<u32>,
}

impl VectorMask {
    /// Derive the mask for `dim` coordinates from a passphrase and salt.
    ///
    /// The same `(passphrase, salt, dim)` always yields the same mask, so every
    /// client of one collection must share the salt. The salt is not secret.
    pub fn derive(passphrase: &[u8], salt: &[u8], dim: usize) -> Result<Self> {
        ensure!(!passphrase.is_empty(), "mask passphrase must not be empty");
        ensure!(
            salt.len() >= MIN_SALT_LEN,
            "mask salt must be at least {MIN_SALT_LEN} bytes"
        );
        ensure!(
            (1..=MAX_MASK_DIMENSIONS).contains(&dim),
            "mask dimensions must be in 1..={MAX_MASK_DIMENSIONS}"
        );

        let mut key = [0u8; 32];
        argon2::Argon2::default()
            .hash_password_into(passphrase, salt, &mut key)
            .map_err(|e| anyhow!("Argon2 key derivation failed: {e}"))?;

        // Rank every coordinate by a keyed PRF; the sort order is the permutation.
        let mut prefix = Sha256::new();
        prefix.update(DOMAIN);
        prefix.update(key.as_slice());
        prefix.update((dim as u64).to_le_bytes());
        key.zeroize();

        let mut ranked: Vec<([u8; 32], u32)> = (0..dim as u32)
            .map(|idx| {
                let mut h = prefix.clone();
                h.update(idx.to_le_bytes());
                (h.finalize().into(), idx)
            })
            .collect();
        ranked.sort_unstable();

        let mut forward = vec![0u32; dim];
        for (rank, (_, idx)) in ranked.iter().enumerate() {
            forward[*idx as usize] = rank as u32;
        }
        for (tag, _) in ranked.iter_mut() {
            tag.zeroize();
        }
        Ok(Self { forward })
    }

    pub fn dimensions(&self) -> usize {
        self.forward.len()
    }

    /// Mask a vector. Fails if its dimension differs from the mask's.
    pub fn apply(&self, vector: &EntangledHVec) -> Result<EntangledHVec> {
        self.map(vector, &self.forward)
    }

    /// Recover the original vector from a masked one.
    pub fn invert(&self, masked: &EntangledHVec) -> Result<EntangledHVec> {
        let mut inverse = vec![0u32; self.forward.len()];
        for (idx, &target) in self.forward.iter().enumerate() {
            inverse[target as usize] = idx as u32;
        }
        let result = self.map(masked, &inverse);
        inverse.zeroize();
        result
    }

    fn map(&self, vector: &EntangledHVec, table: &[u32]) -> Result<EntangledHVec> {
        ensure!(
            vector.dim == table.len(),
            "vector dimension {} does not match mask dimension {}",
            vector.dim,
            table.len()
        );
        let mut indices = Vec::with_capacity(vector.indices.len());
        for &idx in &vector.indices {
            let mapped = table
                .get(idx as usize)
                .ok_or_else(|| anyhow!("vector index {idx} is out of range"))?;
            indices.push(*mapped);
        }
        indices.sort_unstable();
        Ok(EntangledHVec::from_indices(indices, vector.dim))
    }
}

impl Drop for VectorMask {
    fn drop(&mut self) {
        self.forward.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 16384;
    const SALT: &[u8] = b"0123456789abcdef";

    fn mask(passphrase: &str) -> VectorMask {
        VectorMask::derive(passphrase.as_bytes(), SALT, DIM).unwrap()
    }

    #[test]
    fn forward_table_is_a_permutation() {
        let m = mask("correct horse");
        let mut seen = m.forward.clone();
        seen.sort_unstable();
        assert!(seen.iter().enumerate().all(|(i, &v)| v == i as u32));
    }

    #[test]
    fn similarity_and_binding_are_preserved_exactly() {
        let m = mask("correct horse");
        let a = EntangledHVec::new_deterministic(DIM, 1);
        let b = EntangledHVec::new_deterministic(DIM, 2);
        // Overlapping pair so the preserved similarity is non-trivial.
        let c = EntangledHVec::bundle(&[a.clone(), b.clone()]);
        let (ma, mb, mc) = (
            m.apply(&a).unwrap(),
            m.apply(&b).unwrap(),
            m.apply(&c).unwrap(),
        );

        assert!(a.similarity(&c) > 0.1);
        assert_eq!(a.similarity(&c), ma.similarity(&mc));
        assert_eq!(a.similarity(&b), ma.similarity(&mb));
        assert_eq!(ma.indices.len(), a.indices.len());
        assert_eq!(m.apply(&a.bind(&b)).unwrap().indices, ma.bind(&mb).indices);
    }

    #[test]
    fn round_trips_and_hides_original_coordinates() {
        let m = mask("correct horse");
        let a = EntangledHVec::new_deterministic(DIM, 7);
        let masked = m.apply(&a).unwrap();
        assert_eq!(m.invert(&masked).unwrap().indices, a.indices);
        assert!(a.similarity(&masked) < 0.1);
    }

    #[test]
    fn derivation_is_deterministic_and_key_and_salt_dependent() {
        let a = EntangledHVec::new_deterministic(DIM, 7);
        let base = mask("correct horse").apply(&a).unwrap();
        assert_eq!(
            mask("correct horse").apply(&a).unwrap().indices,
            base.indices
        );
        assert_ne!(mask("wrong horse").apply(&a).unwrap().indices, base.indices);
        let other_salt = VectorMask::derive(b"correct horse", b"fedcba9876543210", DIM).unwrap();
        assert_ne!(other_salt.apply(&a).unwrap().indices, base.indices);
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(VectorMask::derive(b"", SALT, DIM).is_err());
        assert!(VectorMask::derive(b"k", &SALT[..MIN_SALT_LEN - 1], DIM).is_err());
        assert!(VectorMask::derive(b"k", SALT, 0).is_err());
        assert!(VectorMask::derive(b"k", SALT, MAX_MASK_DIMENSIONS + 1).is_err());
        assert!(VectorMask::derive(b"k", SALT, MAX_MASK_DIMENSIONS).is_ok());

        let wrong_dim = EntangledHVec::new_deterministic(DIM / 2, 1);
        assert!(mask("k").apply(&wrong_dim).is_err());
    }
}
