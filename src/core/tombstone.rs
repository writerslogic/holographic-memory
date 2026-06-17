// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

/// Bitset tracking deleted vector IDs. Bit-packed, 64 IDs per u64.
pub struct TombstoneMap {
    bits: Vec<u64>,
}

impl TombstoneMap {
    pub fn new() -> Self {
        Self { bits: Vec::new() }
    }

    pub fn mark_deleted(&mut self, vec_id: u32) {
        let word = vec_id as usize / 64;
        let bit = vec_id as usize % 64;
        if word >= self.bits.len() {
            self.bits.resize(word + 1, 0);
        }
        self.bits[word] |= 1u64 << bit;
    }

    pub fn is_deleted(&self, vec_id: u32) -> bool {
        let word = vec_id as usize / 64;
        let bit = vec_id as usize % 64;
        if word >= self.bits.len() {
            return false;
        }
        (self.bits[word] >> bit) & 1 == 1
    }

    pub fn count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.bits.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tombstone_basic() {
        let mut t = TombstoneMap::new();
        assert!(!t.is_deleted(0));
        assert!(!t.is_deleted(1000));

        t.mark_deleted(42);
        assert!(t.is_deleted(42));
        assert!(!t.is_deleted(41));
        assert_eq!(t.count(), 1);
    }

    #[test]
    fn test_tombstone_multiple() {
        let mut t = TombstoneMap::new();
        t.mark_deleted(0);
        t.mark_deleted(63);
        t.mark_deleted(64);
        t.mark_deleted(1000);
        assert_eq!(t.count(), 4);
        assert!(t.is_deleted(0));
        assert!(t.is_deleted(63));
        assert!(t.is_deleted(64));
        assert!(t.is_deleted(1000));
        assert!(!t.is_deleted(1));
    }

    #[test]
    fn test_tombstone_clear() {
        let mut t = TombstoneMap::new();
        t.mark_deleted(5);
        t.clear();
        assert!(!t.is_deleted(5));
        assert_eq!(t.count(), 0);
    }
}
