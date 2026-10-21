// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Compact lexical postings: a two-level term dictionary and delta-varint posting lists.
//!
//! The dictionary keeps a sorted base of concatenated term bytes with one `u32` offset and
//! one `u32` list id per term, and a small overlay map for terms added since the base was
//! built; the overlay folds into the base when it outgrows a quarter of it. A posting list is
//! one byte vector of LEB128 varints: the slot's delta to the previous posting, then the
//! frequency; it also remembers the slot of its final posting, so a run of higher slots
//! appends without decoding what is there. Against a `BTreeMap<String, Vec<{usize, u32}>>`
//! this replaces 24 bytes of key header, 24 of vector header and 12 per posting with about 9
//! bytes per term and 2 to 3 per posting.
//!
//! Bulk indexing ([`index_chunks`]) takes a batch of chunks in ascending slot order and
//! builds every term's new postings once:
//!
//! 1. Contiguous chunk ranges are scanned in parallel into per-range hash maps keyed by the
//!    chunk's own term bytes (no term is copied or allocated), one map per term-hash bucket.
//! 2. The maps of one bucket merge, in parallel across buckets, into one slot-ordered run per
//!    term: ranges are ordered, so a later range's postings go after an earlier range's.
//! 3. Each run is looked up once in the dictionary (parallel, read-only), then appended to
//!    its list in one delta-varint pass; nothing stored is re-encoded unless the run starts
//!    at or below the list's final slot (a slot reused after deletes), which merges that one
//!    list once.
//! 4. Terms new to the dictionary are merged into a fresh sorted base in one pass together
//!    with the overlay when there are many, or added to the overlay when there are few.
//!
//! Single-chunk inserts and the per-term replacement path take the same append-or-merge step
//! through [`PostingLists::insert`], so one document at a time and one batch reach the same
//! lists.

use std::collections::{hash_map::Entry, BTreeMap};

use fxhash::FxHashMap;
use rayon::prelude::*;

use super::StoredChunk;

const NONE: u32 = u32::MAX;

/// `(slot, frequency)`.
pub(super) type Posting = (usize, u32);

/// Term-hash buckets the per-range maps are merged by; the parallelism of that merge.
const MERGE_BUCKETS: usize = 16;
/// Batches below this many chunks are scanned as one range into one bucket.
const PARALLEL_CHUNKS: usize = 64;
/// Smallest chunk range one thread scans.
const MIN_RANGE: usize = 32;
/// Run counts below this look their terms up on the calling thread.
const PARALLEL_LOOKUPS: usize = 256;

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

    /// Overlay entries the base can absorb before the next fold.
    fn overlay_budget(&self) -> usize {
        self.base_ids.len() / 4 + 1024
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
        if self.overlay.len() > self.overlay_budget() {
            self.rebuild();
        }
    }

    /// Adds terms absent from the dictionary (and from each other). Few go through
    /// [`Self::insert`]; more than the overlay would take merge into a new base in one pass.
    pub(super) fn insert_many(&mut self, mut terms: Vec<(&str, u32)>) {
        if self.overlay.len() + terms.len() <= self.overlay_budget() {
            for (term, id) in terms {
                self.insert(term, id);
            }
            return;
        }
        terms.par_sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        debug_assert!(
            terms.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "duplicate term in a dictionary batch"
        );
        debug_assert!(
            terms.iter().all(|(term, _)| self.get(term).is_none()),
            "term already in the dictionary"
        );
        // The overlay and the new terms are sorted and disjoint: one merge gives the sorted
        // extra run the base absorbs.
        let mut extra: Vec<(&[u8], u32)> = Vec::with_capacity(self.overlay.len() + terms.len());
        let mut overlay = self
            .overlay
            .iter()
            .map(|(term, &id)| (term.as_bytes(), id))
            .peekable();
        let mut new = terms
            .iter()
            .map(|&(term, id)| (term.as_bytes(), id))
            .peekable();
        loop {
            let take_overlay = match (overlay.peek(), new.peek()) {
                (Some(a), Some(b)) => a.0 < b.0,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            let next = if take_overlay {
                overlay.next()
            } else {
                new.next()
            };
            extra.push(next.expect("peeked entry"));
        }
        let (bytes, offsets, ids) = self.merged_base(&extra);
        self.bytes = bytes;
        self.offsets = offsets;
        self.base_ids = ids;
        self.tombstones = 0;
        self.overlay = BTreeMap::new();
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
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.base_ids.len() - self.tombstones + self.overlay.len()
    }

    /// Folds the overlay into a new sorted base without tombstones.
    pub(super) fn rebuild(&mut self) {
        let overlay: Vec<(&[u8], u32)> = self
            .overlay
            .iter()
            .map(|(t, &id)| (t.as_bytes(), id))
            .collect();
        let (bytes, offsets, ids) = self.merged_base(&overlay);
        self.bytes = bytes;
        self.offsets = offsets;
        self.base_ids = ids;
        self.tombstones = 0;
        self.overlay = BTreeMap::new();
    }

    /// The live base merged with `extra` (sorted, disjoint from the live base; a tombstoned
    /// base term equal to an extra term yields to it) as new base bytes, offsets and ids.
    fn merged_base(&self, extra: &[(&[u8], u32)]) -> (Vec<u8>, Vec<u32>, Vec<u32>) {
        let count = self.base_ids.len() - self.tombstones + extra.len();
        let mut bytes = Vec::with_capacity(
            self.bytes.len() + extra.iter().map(|(t, _)| t.len()).sum::<usize>(),
        );
        let mut offsets = Vec::with_capacity(count + 1);
        let mut ids = Vec::with_capacity(count);
        offsets.push(0);
        let mut push = |term: &[u8], id: u32| {
            bytes.extend_from_slice(term);
            offsets.push(u32::try_from(bytes.len()).expect("term dictionary over 4 GiB"));
            ids.push(id);
        };
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.base_ids.len() || j < extra.len() {
            if i < self.base_ids.len() && self.base_ids[i] == NONE {
                i += 1;
                continue;
            }
            let take_base = match (i < self.base_ids.len(), j < extra.len()) {
                (true, true) => {
                    debug_assert_ne!(self.base_term(i), extra[j].0, "live term merged twice");
                    self.base_term(i) < extra[j].0
                }
                (true, false) => true,
                (false, _) => false,
            };
            if take_base {
                push(self.base_term(i), self.base_ids[i]);
                i += 1;
            } else {
                push(extra[j].0, extra[j].1);
                j += 1;
            }
        }
        bytes.shrink_to_fit();
        (bytes, offsets, ids)
    }
}

/// One term's postings, sorted by slot, as varint slot deltas and frequencies.
#[derive(Default)]
pub(super) struct PostingList {
    data: Vec<u8>,
    count: u32,
    /// Slot of the final posting; meaningful while `count > 0`.
    last: u64,
}

/// Two strictly slot-sorted runs merged into one; a shared slot is a broken invariant.
fn merge_sorted(a: &[Posting], b: &[Posting]) -> Vec<Posting> {
    let mut merged = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => {
                merged.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                merged.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => panic!("duplicate document slot in posting"),
        }
    }
    merged.extend_from_slice(&a[i..]);
    merged.extend_from_slice(&b[j..]);
    merged
}

impl PostingList {
    pub(super) fn len(&self) -> usize {
        self.count as usize
    }

    pub(super) fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// `(slot, frequency)` pairs in slot order.
    pub(super) fn iter(&self) -> impl Iterator<Item = Posting> + '_ {
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
    /// decoder state in registers where the iterator cannot. Four postings whose eight varints
    /// are one byte each (gaps and frequencies below 128, the common case in long lists)
    /// decode from one 64-bit load.
    pub(super) fn for_each(&self, mut f: impl FnMut(usize, u32)) {
        let bytes = &self.data[..];
        let mut at = 0usize;
        let mut slot = 0u64;
        let mut left = self.count as usize;
        while left > 0 {
            if left >= 4 {
                if let Some(word) = bytes.get(at..at + 8) {
                    let word = u64::from_le_bytes(word.try_into().expect("eight bytes"));
                    if word & 0x8080_8080_8080_8080 == 0 {
                        for pair in 0..4 {
                            slot += (word >> (16 * pair)) & 0xff;
                            f(slot as usize, ((word >> (16 * pair + 8)) & 0xff) as u32);
                        }
                        at += 8;
                        left -= 4;
                        continue;
                    }
                }
            }
            slot += get_varint(bytes, &mut at);
            let frequency = get_varint(bytes, &mut at);
            debug_assert!(slot <= usize::MAX as u64 && frequency <= u64::from(u32::MAX));
            f(slot as usize, frequency as u32);
            left -= 1;
        }
    }

    fn decode(&self) -> Vec<Posting> {
        self.iter().collect()
    }

    fn encode(&mut self, postings: &[Posting]) {
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
        self.last = previous;
    }

    /// Adds `run`, strictly slot-sorted and disjoint from the list. A run above the list's
    /// final slot is delta-encoded straight onto the list (through `scratch`, a buffer the
    /// caller reuses across lists); one that starts at or below it merges the two in one
    /// decode and encode.
    pub(super) fn append_run(&mut self, run: &[Posting], scratch: &mut Vec<u8>) {
        debug_assert!(
            run.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "unsorted posting run"
        );
        let Some(&(first, _)) = run.first() else {
            return;
        };
        if self.count > 0 && first as u64 <= self.last {
            let merged = merge_sorted(&self.decode(), run);
            self.encode(&merged);
            return;
        }
        scratch.clear();
        let mut previous = if self.count == 0 { 0 } else { self.last };
        for &(slot, frequency) in run {
            let slot = slot as u64;
            put_varint(scratch, slot - previous);
            put_varint(scratch, u64::from(frequency));
            previous = slot;
        }
        if self.count == 0 {
            // A fresh list is sized exactly; a growing one amortizes its reallocations.
            self.data = scratch.clone();
        } else {
            self.data.extend_from_slice(scratch);
        }
        self.count =
            u32::try_from(self.count as usize + run.len()).expect("posting list over u32::MAX");
        self.last = previous;
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

/// One term's new postings for a batch, in strictly ascending slot order.
pub(super) struct TermRun<'a> {
    term: &'a str,
    postings: Vec<Posting>,
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

    /// Adds a `(term, slot)` posting absent from the lists.
    pub(super) fn insert(&mut self, term: &str, slot: usize, frequency: u32) {
        let id = match self.dictionary.get(term) {
            Some(id) => id,
            None => {
                let id = self.allocate_list();
                self.dictionary.insert(term, id);
                id
            }
        };
        self.lists[id as usize].append_run(&[(slot, frequency)], &mut Vec::new());
    }

    /// Adds one run per distinct term, each absent from that term's list.
    pub(super) fn insert_runs(&mut self, runs: &[TermRun<'_>]) {
        if runs.is_empty() {
            return;
        }
        let dictionary = &self.dictionary;
        let ids: Vec<Option<u32>> = if runs.len() >= PARALLEL_LOOKUPS {
            runs.par_iter()
                .map(|run| dictionary.get(run.term))
                .collect()
        } else {
            runs.iter().map(|run| dictionary.get(run.term)).collect()
        };
        let mut new_terms = Vec::new();
        let mut scratch = Vec::new();
        for (run, id) in runs.iter().zip(ids) {
            let id = id.unwrap_or_else(|| {
                let id = self.allocate_list();
                new_terms.push((run.term, id));
                id
            });
            self.lists[id as usize].append_run(&run.postings, &mut scratch);
        }
        self.dictionary.insert_many(new_terms);
    }

    /// The id of an empty list for a new term.
    fn allocate_list(&mut self) -> u32 {
        match self.free.pop() {
            Some(id) => id,
            None => {
                self.lists.push(PostingList::default());
                u32::try_from(self.lists.len() - 1).expect("too many terms")
            }
        }
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

    /// Every live term has a non-empty, strictly slot-sorted list whose recorded final slot
    /// is its last posting; every free id an empty one.
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
            assert_eq!(slots.last().copied(), Some(list.last as usize));
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

/// Per-range accumulator: a term's postings in the order its chunks were scanned.
type RunMap<'a> = FxHashMap<&'a str, Vec<Posting>>;

/// Merge bucket of a term: FNV-1a over its bytes, deliberately not the maps' Fx hash, so the
/// keys that share a bucket do not also share the low hash bits their map probes by.
fn bucket_of(term: &str, buckets: usize) -> usize {
    let hash = term.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    (hash % buckets as u64) as usize
}

/// Scans one chunk range into per-form, per-bucket maps.
fn scan_range<'a>(range: &[(usize, &'a StoredChunk)], buckets: usize) -> [Vec<RunMap<'a>>; 2] {
    let mut maps: [Vec<RunMap<'a>>; 2] =
        std::array::from_fn(|_| (0..buckets).map(|_| RunMap::default()).collect());
    for &(slot, chunk) in range {
        let form = &mut maps[usize::from(chunk.stemmed)];
        for (term, &frequency) in &chunk.terms {
            let bucket = if buckets == 1 {
                0
            } else {
                bucket_of(term, buckets)
            };
            form[bucket]
                .entry(term)
                .or_default()
                .push((slot, frequency));
        }
    }
    maps
}

/// Merges the maps of one bucket, given in range order, into one run per term, sorted by
/// term so list ids are assigned the same way whatever the thread count.
fn merge_bucket<'a>(maps: Vec<RunMap<'a>>) -> Vec<TermRun<'a>> {
    let mut maps = maps.into_iter();
    let Some(mut merged) = maps.next() else {
        return Vec::new();
    };
    for map in maps {
        for (term, run) in map {
            match merged.entry(term) {
                Entry::Occupied(entry) => {
                    let postings = entry.into_mut();
                    debug_assert!(
                        postings.last().is_some_and(|last| last.0 < run[0].0),
                        "chunk ranges merged out of slot order"
                    );
                    postings.extend(run);
                }
                Entry::Vacant(entry) => {
                    entry.insert(run);
                }
            }
        }
    }
    let mut runs: Vec<TermRun<'a>> = merged
        .into_iter()
        .map(|(term, postings)| TermRun { term, postings })
        .collect();
    runs.sort_unstable_by(|a, b| a.term.cmp(b.term));
    runs
}

/// Indexes `chunks`, given as `(slot, chunk)` in strictly ascending slot order with every
/// slot absent from the lists, into the unstemmed (`postings[0]`) and stemmed (`postings[1]`)
/// lists. See the module head for the phases.
pub(super) fn index_chunks<'a>(
    postings: &mut [PostingLists; 2],
    chunks: &[(usize, &'a StoredChunk)],
) {
    debug_assert!(
        chunks.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "chunks out of slot order"
    );
    if chunks.is_empty() {
        return;
    }
    let buckets = if chunks.len() < PARALLEL_CHUNKS {
        1
    } else {
        MERGE_BUCKETS
    };
    let range = chunks
        .len()
        .div_ceil(rayon::current_num_threads().max(1))
        .max(MIN_RANGE);
    let partials: Vec<[Vec<RunMap<'a>>; 2]> = chunks
        .par_chunks(range)
        .map(|slice| scan_range(slice, buckets))
        .collect();
    // Every range's map of one (form, bucket) goes to that bucket's merge, in range order.
    let mut by_bucket: Vec<Vec<RunMap<'a>>> = (0..2 * buckets)
        .map(|_| Vec::with_capacity(partials.len()))
        .collect();
    for partial in partials {
        for (form, maps) in partial.into_iter().enumerate() {
            for (bucket, map) in maps.into_iter().enumerate() {
                by_bucket[form * buckets + bucket].push(map);
            }
        }
    }
    let merged: Vec<Vec<TermRun<'a>>> = by_bucket.into_par_iter().map(merge_bucket).collect();
    let mut merged = merged.into_iter();
    for lists in postings.iter_mut() {
        let runs: Vec<TermRun<'a>> = merged.by_ref().take(buckets).flatten().collect();
        lists.insert_runs(&runs);
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
            lists.assert_valid();
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

    /// Runs appended in bulk (above the final slot, and below it after removals) reach the
    /// lists single inserts build, through both the overlay and the bulk dictionary merge.
    #[test]
    fn bulk_runs_match_single_inserts() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let term_names: Vec<String> = (0..6000).map(|i| format!("term{i:05}")).collect();
        let mut bulk = PostingLists::default();
        let mut single = PostingLists::default();
        let mut next_slot = 0usize;
        // Slots a batch freed from every list, offered to the next batch for reuse.
        let mut freed: Vec<usize> = Vec::new();
        for batch in 0..6 {
            // Batches 0 and 3 are large (bulk dictionary merge), the others small (overlay).
            let batch_terms = if batch % 3 == 0 { 3000 } else { 40 };
            let mut runs: Vec<TermRun<'_>> = Vec::new();
            let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
            let mut slots: Vec<usize> = std::mem::take(&mut freed);
            for _ in 0..200 {
                slots.push(next_slot);
                next_slot += 1 + (next() % 3) as usize;
            }
            slots.sort_unstable();
            for t in 0..batch_terms {
                let term = term_names[(next() % 6000) as usize % (t + 1)].as_str();
                if !seen.insert(term) {
                    continue;
                }
                let mut postings: Vec<Posting> = Vec::new();
                for &slot in &slots {
                    if next() % 3 == 0 {
                        postings.push((slot, 1 + (next() % 7) as u32));
                    }
                }
                if postings.is_empty() {
                    continue;
                }
                runs.push(TermRun { term, postings });
            }
            for run in &runs {
                for &(slot, frequency) in &run.postings {
                    single.insert(run.term, slot, frequency);
                }
            }
            bulk.insert_runs(&runs);
            // Free a tenth of this batch's slots from every list holding them (only this
            // batch's runs do), so the next batch's runs start below some final slots.
            freed = slots.iter().copied().filter(|_| next() % 10 == 0).collect();
            for run in &runs {
                for &(slot, _) in &run.postings {
                    if freed.contains(&slot) {
                        bulk.remove(run.term, slot);
                        single.remove(run.term, slot);
                    }
                }
            }
            bulk.assert_valid();
            single.assert_valid();
            assert_eq!(bulk.term_count(), single.term_count());
            for term in &term_names {
                let a = bulk.get(term).map(|l| l.iter().collect::<Vec<_>>());
                let b = single.get(term).map(|l| l.iter().collect::<Vec<_>>());
                assert_eq!(a, b, "{term} after batch {batch}");
            }
        }
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

    /// A bulk insert revives tombstoned base terms, keeps the overlay, and lands every term
    /// in the base in sorted order.
    #[test]
    fn bulk_dictionary_merge_keeps_overlay_and_revives_tombstones() {
        let mut d = TermDictionary::default();
        for i in 0..2000u32 {
            d.insert(&format!("b{i:05}"), i);
        }
        d.rebuild();
        for i in (0..2000u32).step_by(3) {
            assert_eq!(d.remove(&format!("b{i:05}")), Some(i));
        }
        d.insert("a00001", 5000);
        d.insert("c00001", 5001);
        // 667 of these revive tombstones, the rest are new; together they exceed the overlay
        // budget (2000 / 4 + 1024), so the batch merges into the base.
        let names: Vec<String> = (0..2000u32).map(|i| format!("b{:05}", i * 3)).collect();
        let mut batch: Vec<(&str, u32)> = names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), 6000 + i as u32))
            .collect();
        batch.reverse();
        d.insert_many(batch);
        assert!(d.overlay.is_empty(), "bulk merge folds the overlay");
        assert_eq!(d.tombstones, 0);
        assert_eq!(d.get("a00001"), Some(5000));
        assert_eq!(d.get("c00001"), Some(5001));
        for (i, name) in names.iter().enumerate() {
            assert_eq!(d.get(name), Some(6000 + i as u32), "{name}");
        }
        assert_eq!(d.get("b00001"), Some(1));
        assert_eq!(d.get("b00002"), Some(2));
        assert_eq!(d.len(), 2000 - 667 + 2 + 2000);
        let terms: Vec<&[u8]> = (0..d.base_ids.len()).map(|i| d.base_term(i)).collect();
        assert!(terms.windows(2).all(|pair| pair[0] < pair[1]));
    }

    /// `for_each` (four one-byte postings per load where it can) visits exactly what `iter`
    /// decodes, for lists of every length up to past two loads, with a multi-byte gap or
    /// frequency at every position.
    #[test]
    fn for_each_matches_iter_across_varint_widths() {
        for length in 0..=13usize {
            for wide in 0..=length {
                for wide_frequency in [false, true] {
                    let mut run = Vec::new();
                    let mut slot = 5usize;
                    for i in 0..length {
                        slot += if i == wide && !wide_frequency {
                            300
                        } else {
                            1 + i % 3
                        };
                        let frequency = if i == wide && wide_frequency {
                            70_000
                        } else {
                            1 + i as u32 % 127
                        };
                        run.push((slot, frequency));
                    }
                    let mut list = PostingList::default();
                    list.append_run(&run, &mut Vec::new());
                    let mut visited = Vec::new();
                    list.for_each(|slot, frequency| visited.push((slot, frequency)));
                    assert_eq!(visited, list.iter().collect::<Vec<_>>());
                    assert_eq!(visited, run, "length {length} wide {wide}");
                }
            }
        }
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
