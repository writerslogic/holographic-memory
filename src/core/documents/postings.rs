// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compact lexical postings: a two-level term dictionary and delta-varint posting lists.
//!
//! The dictionary keeps a sorted base of concatenated term bytes with one `u32` offset and
//! one `u32` list id per term, and a small overlay map for terms added since the base was
//! built; the overlay folds into the base when it outgrows a quarter of it. A posting list is
//! one byte vector of LEB128 varints: the slot's delta to the previous posting, then the
//! frequency. Against a `BTreeMap<String, Vec<{usize, u32}>>` this replaces 24 bytes of key
//! header, 24 of vector header and 12 per posting with about 9 bytes per term and 2 to 3 per
//! posting.

use std::collections::BTreeMap;

const NONE: u32 = u32::MAX;

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Decodes one varint at `*at`, advancing it. Lists are written by this module, so a
/// malformed byte string is a broken invariant.
#[inline(always)]
fn get_varint(bytes: &[u8], at: &mut usize) -> u64 {
    let first = bytes[*at];
    *at += 1;
    if first < 0x80 {
        return u64::from(first);
    }
    let mut value = u64::from(first & 0x7f);
    let mut shift = 7;
    loop {
        let b = bytes[*at];
        *at += 1;
        value |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return value;
        }
        shift += 7;
        assert!(shift < 64, "posting varint overflow");
    }
}

/// Term dictionary: sorted base plus overlay, mapping a term to its list id.
#[derive(Default)]
pub(super) struct TermDictionary {
    bytes: Vec<u8>,
    /// `base_ids.len() + 1` offsets into `bytes`.
    offsets: Vec<u32>,
    /// List id per base term, `NONE` once removed (dropped at the next rebuild).
    base_ids: Vec<u32>,
    tombstones: usize,
    overlay: BTreeMap<Box<str>, u32>,
}

impl TermDictionary {
    fn base_term(&self, i: usize) -> &[u8] {
        &self.bytes[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }

    fn base_position(&self, term: &str) -> Result<usize, usize> {
        let (mut lo, mut hi) = (0usize, self.base_ids.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.base_term(mid).cmp(term.as_bytes()) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Err(lo)
    }

    pub(super) fn get(&self, term: &str) -> Option<u32> {
        if let Some(&id) = self.overlay.get(term) {
            return Some(id);
        }
        self.base_position(term)
            .ok()
            .map(|i| self.base_ids[i])
            .filter(|&id| id != NONE)
    }

    /// Adds a term absent from the dictionary.
    pub(super) fn insert(&mut self, term: &str, id: u32) {
        debug_assert!(self.get(term).is_none(), "term already in the dictionary");
        if let Ok(i) = self.base_position(term) {
            // A tombstoned base term returns.
            self.base_ids[i] = id;
            self.tombstones -= 1;
            return;
        }
        self.overlay.insert(term.into(), id);
        if self.overlay.len() > self.base_ids.len() / 4 + 1024 {
            self.rebuild();
        }
    }

    pub(super) fn remove(&mut self, term: &str) -> Option<u32> {
        if let Some(id) = self.overlay.remove(term) {
            return Some(id);
        }
        let i = self.base_position(term).ok()?;
        let id = std::mem::replace(&mut self.base_ids[i], NONE);
        if id == NONE {
            return None;
        }
        self.tombstones += 1;
        if self.tombstones > self.base_ids.len() / 2 + 1024 {
            self.rebuild();
        }
        Some(id)
    }

    /// Live terms.
    pub(super) fn len(&self) -> usize {
        self.base_ids.len() - self.tombstones + self.overlay.len()
    }

    /// Folds the overlay into a new sorted base without tombstones.
    pub(super) fn rebuild(&mut self) {
        let mut merged: Vec<(&[u8], u32)> = Vec::with_capacity(self.len());
        let overlay: Vec<(&[u8], u32)> = self
            .overlay
            .iter()
            .map(|(t, &id)| (t.as_bytes(), id))
            .collect();
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.base_ids.len() || j < overlay.len() {
            if i < self.base_ids.len() && self.base_ids[i] == NONE {
                i += 1;
                continue;
            }
            let take_base = match (i < self.base_ids.len(), j < overlay.len()) {
                (true, true) => self.base_term(i) < overlay[j].0,
                (true, false) => true,
                (false, _) => false,
            };
            if take_base {
                merged.push((self.base_term(i), self.base_ids[i]));
                i += 1;
            } else {
                merged.push(overlay[j]);
                j += 1;
            }
        }
        let mut bytes = Vec::with_capacity(merged.iter().map(|(t, _)| t.len()).sum());
        let mut offsets = Vec::with_capacity(merged.len() + 1);
        let mut ids = Vec::with_capacity(merged.len());
        offsets.push(0);
        for (term, id) in merged {
            bytes.extend_from_slice(term);
            offsets.push(u32::try_from(bytes.len()).expect("term dictionary over 4 GiB"));
            ids.push(id);
        }
        self.bytes = bytes;
        self.offsets = offsets;
        self.base_ids = ids;
        self.tombstones = 0;
        self.overlay = BTreeMap::new();
    }
}

/// One term's postings, sorted by slot, as varint slot deltas and frequencies.
#[derive(Default)]
pub(super) struct PostingList {
    data: Vec<u8>,
    count: u32,
}

impl PostingList {
    pub(super) fn len(&self) -> usize {
        self.count as usize
    }

    pub(super) fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// `(slot, frequency)` pairs in slot order.
    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, u32)> + '_ {
        let mut at = 0usize;
        let mut slot = 0u64;
        (0..self.count).map(move |i| {
            let delta = get_varint(&self.data, &mut at);
            slot = if i == 0 { delta } else { slot + delta };
            let frequency = get_varint(&self.data, &mut at);
            (
                usize::try_from(slot).expect("posting slot exceeds usize"),
                u32::try_from(frequency).expect("posting frequency exceeds u32"),
            )
        })
    }

    /// Calls `f(slot, frequency)` for every posting in slot order; the loop form keeps the
    /// decoder state in registers where the iterator cannot.
    pub(super) fn for_each(&self, mut f: impl FnMut(usize, u32)) {
        let bytes = &self.data[..];
        let mut at = 0usize;
        let mut slot = 0u64;
        for _ in 0..self.count {
            slot += get_varint(bytes, &mut at);
            let frequency = get_varint(bytes, &mut at);
            debug_assert!(slot <= usize::MAX as u64 && frequency <= u64::from(u32::MAX));
            f(slot as usize, frequency as u32);
        }
    }

    fn decode(&self) -> Vec<(usize, u32)> {
        self.iter().collect()
    }

    fn encode(&mut self, postings: &[(usize, u32)]) {
        self.data.clear();
        let mut previous = 0u64;
        for (i, &(slot, frequency)) in postings.iter().enumerate() {
            let slot = slot as u64;
            put_varint(&mut self.data, if i == 0 { slot } else { slot - previous });
            put_varint(&mut self.data, u64::from(frequency));
            previous = slot;
        }
        self.data.shrink_to_fit();
        self.count = u32::try_from(postings.len()).expect("posting list over u32::MAX");
    }

    /// Adds a slot absent from the list.
    fn insert(&mut self, slot: usize, frequency: u32) {
        let mut postings = self.decode();
        let at = postings
            .binary_search_by_key(&slot, |p| p.0)
            .expect_err("duplicate document slot in posting");
        postings.insert(at, (slot, frequency));
        self.encode(&postings);
    }

    /// Drops a slot present in the list.
    fn remove(&mut self, slot: usize) {
        let mut postings = self.decode();
        let at = postings
            .binary_search_by_key(&slot, |p| p.0)
            .expect("document missing from posting");
        postings.remove(at);
        self.encode(&postings);
    }

    fn set_frequency(&mut self, slot: usize, frequency: u32) {
        let mut postings = self.decode();
        let at = postings
            .binary_search_by_key(&slot, |p| p.0)
            .expect("document missing from posting");
        postings[at].1 = frequency;
        self.encode(&postings);
    }
}

/// The postings of one term form (stemmed or not).
#[derive(Default)]
pub(super) struct PostingLists {
    dictionary: TermDictionary,
    lists: Vec<PostingList>,
    free: Vec<u32>,
}

impl PostingLists {
    pub(super) fn get(&self, term: &str) -> Option<&PostingList> {
        self.dictionary.get(term).map(|id| &self.lists[id as usize])
    }

    /// Number of terms with postings.
    #[cfg(test)]
    pub(super) fn term_count(&self) -> usize {
        self.dictionary.len()
    }

    pub(super) fn insert(&mut self, term: &str, slot: usize, frequency: u32) {
        let id = match self.dictionary.get(term) {
            Some(id) => id,
            None => {
                let id = match self.free.pop() {
                    Some(id) => id,
                    None => {
                        self.lists.push(PostingList::default());
                        u32::try_from(self.lists.len() - 1).expect("too many terms")
                    }
                };
                self.dictionary.insert(term, id);
                id
            }
        };
        self.lists[id as usize].insert(slot, frequency);
    }

    pub(super) fn remove(&mut self, term: &str, slot: usize) {
        let id = self
            .dictionary
            .get(term)
            .expect("document term missing posting");
        let list = &mut self.lists[id as usize];
        list.remove(slot);
        if list.is_empty() {
            *list = PostingList::default();
            self.dictionary.remove(term);
            self.free.push(id);
        }
    }

    pub(super) fn set_frequency(&mut self, term: &str, slot: usize, frequency: u32) {
        let id = self
            .dictionary
            .get(term)
            .expect("document term missing posting");
        self.lists[id as usize].set_frequency(slot, frequency);
    }

    /// Every live term has a non-empty, strictly slot-sorted list; every free id an empty one.
    #[cfg(test)]
    pub(super) fn assert_valid(&self) {
        let mut live = vec![false; self.lists.len()];
        let ids = self.dictionary.overlay.values().copied().chain(
            self.dictionary
                .base_ids
                .iter()
                .copied()
                .filter(|&id| id != NONE),
        );
        for id in ids {
            assert!(!live[id as usize], "list id shared by two terms");
            live[id as usize] = true;
            let list = &self.lists[id as usize];
            assert!(!list.is_empty(), "live term with an empty list");
            let slots: Vec<usize> = list.iter().map(|(slot, _)| slot).collect();
            assert!(slots.windows(2).all(|pair| pair[0] < pair[1]));
        }
        for &id in &self.free {
            assert!(!live[id as usize] && self.lists[id as usize].is_empty());
        }
        assert_eq!(
            live.iter().filter(|&&l| l).count() + self.free.len(),
            self.lists.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Insert, update and remove against a map model over the full slot and frequency
    /// ranges, with every lookup checked through a rebuilt dictionary too.
    #[test]
    fn posting_lists_round_trip_against_a_map_model() {
        let mut lists = PostingLists::default();
        let mut model: BTreeMap<String, BTreeMap<usize, u32>> = BTreeMap::new();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let slots = [
            0usize,
            1,
            127,
            128,
            16_383,
            16_384,
            u32::MAX as usize,
            usize::MAX - 1,
            usize::MAX,
        ];
        let frequencies = [0u32, 1, 127, 128, u32::MAX - 1, u32::MAX];
        let check = |lists: &PostingLists, model: &BTreeMap<String, BTreeMap<usize, u32>>| {
            assert_eq!(lists.term_count(), model.len());
            for (term, postings) in model {
                let list = lists.get(term).expect("term present");
                assert_eq!(list.len(), postings.len());
                assert_eq!(
                    list.iter().collect::<Vec<_>>(),
                    postings.iter().map(|(&s, &f)| (s, f)).collect::<Vec<_>>()
                );
            }
            assert!(lists.get("absent").is_none());
        };
        for round in 0..4000 {
            let term = format!("t{}", next() % 300);
            let slot = if next() % 4 == 0 {
                slots[(next() % slots.len() as u64) as usize]
            } else {
                (next() % 5000) as usize
            };
            let frequency = frequencies[(next() % frequencies.len() as u64) as usize];
            let entry = model.entry(term.clone()).or_default();
            match next() % 3 {
                0 if entry.contains_key(&slot) => {
                    lists.set_frequency(&term, slot, frequency);
                    entry.insert(slot, frequency);
                }
                1 if entry.contains_key(&slot) => {
                    lists.remove(&term, slot);
                    entry.remove(&slot);
                    if entry.is_empty() {
                        model.remove(&term);
                    }
                }
                _ if !entry.contains_key(&slot) => {
                    lists.insert(&term, slot, frequency);
                    entry.insert(slot, frequency);
                }
                _ => {
                    if entry.is_empty() {
                        model.remove(&term);
                    }
                }
            }
            if round % 500 == 499 {
                lists.dictionary.rebuild();
                check(&lists, &model);
            }
        }
        check(&lists, &model);
    }

    #[test]
    fn dictionary_lookups_survive_overlay_base_and_tombstones() {
        let mut d = TermDictionary::default();
        for i in 0..3000u32 {
            d.insert(&format!("w{i:05}"), i);
        }
        assert!(d.overlay.len() < 3000, "overlay folded into the base");
        for i in 0..3000u32 {
            assert_eq!(d.get(&format!("w{i:05}")), Some(i));
        }
        assert_eq!(d.get("w"), None);
        assert_eq!(d.get("w03000"), None);
        assert_eq!(d.get("w0150"), None);
        for i in (0..3000u32).step_by(2) {
            assert_eq!(d.remove(&format!("w{i:05}")), Some(i));
        }
        assert_eq!(d.len(), 1500);
        assert_eq!(d.remove("w00000"), None);
        assert_eq!(d.get("w00000"), None);
        assert_eq!(d.get("w00001"), Some(1));
        d.insert("w00000", 9);
        assert_eq!(d.get("w00000"), Some(9));
        d.rebuild();
        assert_eq!(d.len(), 1501);
        assert_eq!(d.get("w00000"), Some(9));
        assert_eq!(d.get("w02999"), Some(2999));
        assert_eq!(d.get("w02998"), None);
    }

    #[test]
    fn varints_round_trip_at_the_boundaries() {
        for &v in &[
            0u64,
            1,
            127,
            128,
            16_383,
            16_384,
            u32::MAX as u64,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let mut out = Vec::new();
            put_varint(&mut out, v);
            let mut at = 0;
            assert_eq!(get_varint(&out, &mut at), v);
            assert_eq!(at, out.len());
        }
    }
}
