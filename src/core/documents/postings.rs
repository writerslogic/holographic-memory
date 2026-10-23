// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Compact lexical postings: a two-level term dictionary and independently bitpacked blocks.
//!
//! The dictionary keeps a sorted base of concatenated term bytes with one `u32` offset and
//! one `u32` list id per term, and a small overlay map for terms added since the base was
//! built; the overlay folds into the base when it outgrows a quarter of it. A posting list is
//! a sequence of blocks of at most 128 slots, with independent bit widths for gaps and
//! frequencies, block endpoints, maximum frequency and a conservative minimum word count.
//! Postings are rebuilt from journal term tables, so these bytes are never persisted.
//!
//! Bulk indexing ([`index_chunks`]) takes a batch of chunks in ascending slot order and
//! builds every term's new postings once:
//!
//! 1. Contiguous chunk ranges are scanned in parallel into per-range hash maps keyed by the
//!    chunk's own term bytes (no term is copied or allocated), one map per term-hash bucket.
//! 2. The maps of one bucket merge, in parallel across buckets, into one slot-ordered run per
//!    term: ranges are ordered, so a later range's postings go after an earlier range's.
//! 3. Each run is looked up once in the dictionary (parallel, read-only), then appended to
//!    its list in one bitpacking pass; nothing stored is re-encoded unless the run starts
//!    at or below the list's final slot (a slot reused after deletes), which merges that one
//!    list once.
//! 4. Terms new to the dictionary are merged into a fresh sorted base in one pass together
//!    with the overlay when there are many, or added to the overlay when there are few.
//!
//! Single-chunk inserts and the per-term replacement path take the same append-or-merge step
//! through [`PostingLists::insert`], so one document at a time and one batch reach the same
//! lists.

use std::collections::{hash_map::Entry, BTreeMap, HashMap};

use foldhash::fast::FixedState;
use rayon::prelude::*;

use super::binary::put_varint;
use super::StoredChunk;
use std::sync::OnceLock;

const NONE: u32 = u32::MAX;

/// `(slot, frequency)`.
pub(super) type Posting = (usize, u32);

/// One restored list: its term and postings in slot order.
pub(super) type RestoredList = (String, Vec<Posting>);

/// Decodes one varint at `*at`, advancing it, from checkpoint postings the open checked.
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

/// Term-hash buckets the per-range maps are merged by; the parallelism of that merge.
const MERGE_BUCKETS: usize = 16;
/// Batches below this many chunks are scanned as one range into one bucket.
const PARALLEL_CHUNKS: usize = 64;
/// Smallest chunk range one thread scans.
const MIN_RANGE: usize = 32;
/// Run counts below this look their terms up on the calling thread.
const PARALLEL_LOOKUPS: usize = 256;

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

pub(super) const BLOCK_LEN: usize = 128;
/// The fewest postings a block other than the last keeps after in-place removals.
const MIN_BLOCK_LEN: usize = BLOCK_LEN / 4;

#[derive(Clone, Debug)]
struct Block {
    offset: usize,
    first: usize,
    last: usize,
    count: u16,
    doc_bits: u8,
    frequency_bits: u8,
    max_frequency: u32,
    min_words: u32,
    frontier: Option<Box<[(u32, u32)]>>,
}

/// One term's postings in the checkpoint encoding, read from a mapped checkpoint: per posting
/// one varint `gap << 1 | (frequency != 1)`, followed by the frequency as a varint only when
/// it is not 1 (the first gap is the slot itself).
#[derive(Clone, Copy)]
pub(super) struct Postings<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) count: u32,
}

/// Appends `list`'s postings to `out` in the checkpoint encoding.
pub(super) fn pack_checkpoint(list: &PostingList, out: &mut Vec<u8>) {
    let mut previous = 0usize;
    let mut first = true;
    list.for_each(|slot, frequency| {
        let gap = if first { slot } else { slot - previous };
        first = false;
        previous = slot;
        put_varint(out, (gap as u64) << 1 | u64::from(frequency != 1));
        if frequency != 1 {
            put_varint(out, u64::from(frequency));
        }
    });
}

/// [`get_varint`] for bytes read from a file: fails instead of panicking on a malformed varint.
fn checked_varint(bytes: &[u8], at: &mut usize) -> anyhow::Result<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *bytes
            .get(*at)
            .ok_or_else(|| anyhow::anyhow!("posting list ends inside a varint"))?;
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    anyhow::bail!("posting varint overflow")
}

impl Postings<'_> {
    pub(super) fn len(&self) -> usize {
        self.count as usize
    }

    /// Calls `f(slot, frequency)` for every posting in slot order, checking every varint and
    /// that the list ends exactly after `count` postings; `f` returning false stops early.
    pub(super) fn try_for_each(&self, mut f: impl FnMut(usize, u32) -> bool) -> anyhow::Result<()> {
        let (mut at, mut slot) = (0usize, 0u64);
        for _ in 0..self.count {
            let word = checked_varint(self.bytes, &mut at)?;
            let frequency = if word & 1 == 0 {
                1
            } else {
                let frequency = checked_varint(self.bytes, &mut at)?;
                anyhow::ensure!(frequency != 1, "packed posting frequency");
                frequency
            };
            slot = slot
                .checked_add(word >> 1)
                .ok_or_else(|| anyhow::anyhow!("posting slot overflow"))?;
            let frequency = u32::try_from(frequency)?;
            if !f(usize::try_from(slot)?, frequency) {
                return Ok(());
            }
        }
        anyhow::ensure!(at == self.bytes.len(), "posting list has trailing bytes");
        Ok(())
    }

    /// Calls `f(slot, frequency)` for every posting in slot order; the loop form keeps the
    /// decoder state in registers where the iterator cannot. Eight postings whose words are
    /// one byte each with frequency 1 (gaps below 64, the common case in long lists) decode
    /// from one 64-bit load.
    pub(super) fn for_each(&self, mut f: impl FnMut(usize, u32)) {
        let bytes = self.bytes;
        let mut at = 0usize;
        let mut slot = 0u64;
        let mut left = self.count as usize;
        while left > 0 {
            if left >= 8 {
                if let Some(word) = bytes.get(at..at + 8) {
                    let word = u64::from_le_bytes(word.try_into().expect("eight bytes"));
                    // Bytes before the first with a continuation or frequency flag are
                    // whole postings of frequency 1; the slow path takes the rest.
                    let flags = word & 0x8181_8181_8181_8181;
                    let clean = if flags == 0 {
                        8
                    } else {
                        flags.trailing_zeros() as usize / 8
                    };
                    for byte in 0..clean {
                        slot += (word >> (8 * byte + 1)) & 0x7f;
                        f(slot as usize, 1);
                    }
                    at += clean;
                    left -= clean;
                    if clean == 8 {
                        continue;
                    }
                }
            }
            let word = get_varint(bytes, &mut at);
            slot += word >> 1;
            let frequency = if word & 1 == 0 {
                1
            } else {
                get_varint(bytes, &mut at)
            };
            debug_assert!(slot <= usize::MAX as u64 && frequency <= u64::from(u32::MAX));
            f(slot as usize, frequency as u32);
            left -= 1;
        }
    }
}
#[derive(Default)]
pub(super) struct PostingList {
    data: Vec<u8>,
    blocks: Vec<Block>,
    count: u32,
    positive_count: u32,
    last: u64,
    /// For a list of at least [`SUMMARY_MIN_BLOCKS`] blocks, the frontier of each run of
    /// [`SUPER_LEN`] blocks, built by the first query that needs it after a change.
    summary: Option<Box<OnceLock<Box<[Frontier]>>>>,
}

/// Undominated `(frequency, words)` points, by frequency descending.
pub(super) type Frontier = Box<[(u32, u32)]>;

/// Blocks per run that one summary frontier covers.
pub(super) const SUPER_LEN: usize = 32;
/// The fewest blocks a list summarizes: shorter lists are scanned block by block.
const SUMMARY_MIN_BLOCKS: usize = 2 * SUPER_LEN;

/// The points of `points` no other point dominates (a larger frequency at no greater
/// length), by frequency descending; never empty for a nonempty input.
fn pareto(mut points: Vec<(u32, u32)>) -> Box<[(u32, u32)]> {
    points.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut kept: Vec<(u32, u32)> = Vec::new();
    for point in points {
        if kept.last().is_none_or(|last| point.1 < last.1) {
            kept.push(point);
        }
    }
    kept.into_boxed_slice()
}

fn merge_sorted(a: &[Posting], b: &[Posting]) -> Vec<Posting> {
    let mut merged = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
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

/// Appends `values`, each below `2^bits`, as a contiguous little-endian bit string.
pub(super) fn pack(out: &mut Vec<u8>, values: impl Iterator<Item = u64>, bits: u8) {
    let bits = u32::from(bits);
    let (mut buffer, mut used) = (0u64, 0u32);
    for value in values {
        buffer |= value << used;
        let total = used + bits;
        if total >= 64 {
            out.extend_from_slice(&buffer.to_le_bytes());
            buffer = if used == 0 { 0 } else { value >> (64 - used) };
            used = total - 64;
        } else {
            used = total;
        }
    }
    out.extend_from_slice(&buffer.to_le_bytes()[..used.div_ceil(8) as usize]);
}

/// Widths that one unaligned 64-bit load holds together with its byte shift.
const DIRECT_BITS: u8 = 56;

/// Calls `emit(index, value)` for the `count` values of `bits` bits at the start of `bytes`.
fn unpack_into(bytes: &[u8], count: usize, bits: u8, mut emit: impl FnMut(usize, u64)) {
    if bits == 0 {
        for index in 0..count {
            emit(index, 0);
        }
        return;
    }
    if bits > DIRECT_BITS {
        let mut reader = BitReader::new(bytes, bits);
        for index in 0..count {
            emit(index, reader.next());
        }
        return;
    }
    let mask = (1u64 << bits) - 1;
    let mut bit = 0usize;
    for index in 0..count {
        let byte = bit >> 3;
        let word = match bytes.get(byte..byte + 8) {
            Some(window) => u64::from_le_bytes(window.try_into().expect("eight bytes")),
            None => {
                let mut window = [0u8; 8];
                let tail = &bytes[byte..];
                window[..tail.len()].copy_from_slice(tail);
                u64::from_le_bytes(window)
            }
        };
        emit(index, (word >> (bit & 7)) & mask);
        bit += usize::from(bits);
    }
}

pub(super) fn unpack(bytes: &[u8], index: usize, bits: u8) -> u64 {
    if bits == 0 {
        return 0;
    }
    let bit = index * usize::from(bits);
    let start = bit / 8;
    let shift = bit % 8;
    if bits <= DIRECT_BITS {
        if let Some(window) = bytes.get(start..start + 8) {
            let word = u64::from_le_bytes(window.try_into().expect("eight bytes"));
            return (word >> shift) & ((1u64 << bits) - 1);
        }
    }
    let size = (shift + usize::from(bits)).div_ceil(8);
    let mut buffer = [0u8; 16];
    buffer[..size].copy_from_slice(&bytes[start..start + size]);
    let value = u128::from_le_bytes(buffer) >> shift;
    (value & ((1u128 << bits) - 1)) as u64
}

struct BitReader<'a> {
    bytes: &'a [u8],
    buffer: u128,
    used: u8,
    bits: u8,
    mask: u128,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8], bits: u8) -> Self {
        Self {
            bytes,
            buffer: 0,
            used: 0,
            bits,
            mask: (1u128 << bits) - 1,
        }
    }
    fn next(&mut self) -> u64 {
        while self.used < self.bits {
            self.buffer |= u128::from(self.bytes[0]) << self.used;
            self.bytes = &self.bytes[1..];
            self.used += 8;
        }
        let value = (self.buffer & self.mask) as u64;
        self.buffer >>= self.bits;
        self.used -= self.bits;
        value
    }
}

/// The `(frequency, words)` points of a block no other point dominates (a larger frequency
/// at no greater length), by frequency descending, when there are at least two.
fn frontier(points: impl Iterator<Item = (u32, u32)>) -> Option<Box<[(u32, u32)]>> {
    // The least length per frequency: a table for the frequencies a block usually holds and
    // a short sorted spill for the rest.
    let mut least = [u32::MAX; BLOCK_LEN + 1];
    let mut spill = [(0u32, 0u32); BLOCK_LEN];
    let mut spilled = 0;
    for (used, (frequency, words)) in points.filter(|p| p.0 > 0).enumerate() {
        assert!(used < BLOCK_LEN, "frontier exceeds posting block length");
        if let Some(entry) = least.get_mut(frequency as usize) {
            *entry = (*entry).min(words);
        } else {
            spill[spilled] = (frequency, words);
            spilled += 1;
        }
    }
    let spill = &mut spill[..spilled];
    spill.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let table = (1..=BLOCK_LEN)
        .rev()
        .filter(|&frequency| least[frequency] != u32::MAX)
        .map(|frequency| (frequency as u32, least[frequency]));
    let mut kept = [(0u32, 0u32); BLOCK_LEN];
    let mut count = 0;
    let mut shortest = u32::MAX;
    for point in spill.iter().copied().chain(table) {
        if count == 0 || point.1 < shortest {
            shortest = point.1;
            kept[count] = point;
            count += 1;
        }
    }
    (count > 1).then(|| kept[..count].into())
}

impl PostingList {
    pub(super) fn len(&self) -> usize {
        self.count as usize
    }
    pub(super) fn positive_len(&self) -> usize {
        debug_assert!(self.positive_count == 0 || self.blocks.iter().any(|b| b.max_frequency > 0));
        self.positive_count as usize
    }

    pub(super) fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub(super) fn iter(&self) -> PostingCursor<'_> {
        PostingCursor::new(self)
    }
    pub(super) fn for_each(&self, mut f: impl FnMut(usize, u32)) {
        let mut decoded = [(0, 0); BLOCK_LEN];
        for block in &self.blocks {
            self.decode_values(block, &mut decoded);
            for &(slot, frequency) in &decoded[..usize::from(block.count)] {
                f(slot, frequency);
            }
        }
    }
    pub(super) fn mark_slots(&self, matched: &mut [u64]) {
        if self.positive_count != self.count {
            self.for_each(|slot, frequency| {
                if frequency > 0 {
                    matched[slot / 64] |= 1 << (slot % 64);
                }
            });
            return;
        }
        for block in &self.blocks {
            let mut slot = block.first;
            unpack_into(
                &self.data[block.offset..],
                usize::from(block.count),
                block.doc_bits,
                |_, gap| {
                    slot = slot
                        .checked_add(gap as usize)
                        .expect("posting gap overflow");
                    matched[slot / 64] |= 1 << (slot % 64);
                },
            );
        }
    }

    /// The frequency of a block's first posting, read without decoding the block.
    fn first_frequency(&self, block: &Block) -> u32 {
        let gap_bytes = (usize::from(block.count) * usize::from(block.doc_bits)).div_ceil(8);
        unpack(
            &self.data[block.offset + gap_bytes..],
            0,
            block.frequency_bits,
        ) as u32
    }

    /// The largest block frequency and least block length bound of the list, which bound
    /// every posting's weight.
    pub(super) fn maxima(&self) -> (u32, u32) {
        self.blocks
            .iter()
            .fold((0, u32::MAX), |(frequency, words), block| {
                (
                    frequency.max(block.max_frequency),
                    words.min(block.min_words),
                )
            })
    }
    fn decode(&self) -> Vec<Posting> {
        self.iter().collect()
    }

    /// The number of blocks.
    pub(super) fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Block `index`'s largest frequency, least length and frontier, which bound its
    /// postings' weights.
    pub(super) fn block_bounds(&self, index: usize) -> ((u32, u32), &[(u32, u32)]) {
        let block = &self.blocks[index];
        (
            (block.max_frequency, block.min_words),
            block.frontier.as_deref().unwrap_or(&[]),
        )
    }

    /// Decodes block `index` into `out` and returns its posting count.
    pub(super) fn decode_block_into(&self, index: usize, out: &mut [Posting; BLOCK_LEN]) -> usize {
        self.decode_block(index, out);
        usize::from(self.blocks[index].count)
    }

    fn decode_block(&self, index: usize, out: &mut [Posting; BLOCK_LEN]) {
        self.decode_values(&self.blocks[index], out);
    }

    fn decode_values(&self, block: &Block, out: &mut [Posting; BLOCK_LEN]) {
        let count = usize::from(block.count);
        let bytes = &self.data[block.offset..];
        let gap_bytes = (count * usize::from(block.doc_bits)).div_ceil(8);
        let mut slot = block.first;
        unpack_into(bytes, count, block.doc_bits, |index, gap| {
            slot = slot
                .checked_add(gap as usize)
                .expect("posting gap overflow");
            out[index].0 = slot;
        });
        unpack_into(
            &bytes[gap_bytes..],
            count,
            block.frequency_bits,
            |index, frequency| out[index].1 = frequency as u32,
        );
    }

    fn append_block(
        &mut self,
        postings: &[Posting],
        min_words: u32,
        frontier: Option<Box<[(u32, u32)]>>,
    ) {
        let mut gap = 0u64;
        for pair in postings.windows(2) {
            gap = gap.max((pair[1].0 - pair[0].0) as u64);
        }
        let max_frequency = postings.iter().map(|p| p.1).max().expect("nonempty block");
        let doc_bits = (64 - gap.leading_zeros()) as u8;
        let frequency_bits = (32 - max_frequency.leading_zeros()) as u8;
        let offset = self.data.len();
        let mut previous = postings[0].0;
        pack(
            &mut self.data,
            postings.iter().map(|p| {
                let delta = p.0 - previous;
                previous = p.0;
                delta as u64
            }),
            doc_bits,
        );
        pack(
            &mut self.data,
            postings.iter().map(|p| u64::from(p.1)),
            frequency_bits,
        );
        self.blocks.push(Block {
            offset,
            first: postings[0].0,
            last: postings.last().unwrap().0,
            count: postings.len() as u16,
            doc_bits,
            frequency_bits,
            max_frequency,
            min_words,
            frontier,
        });
        self.touch();
    }

    /// Rewrites the list as `postings`, with each block's length bounds taken from `words`
    /// as appends take them, so a rewrite keeps block-max pruning as tight as a fresh build.
    fn encode(&mut self, postings: &[Posting], words: impl Fn(usize) -> u32) {
        self.data.clear();
        self.blocks.clear();
        self.touch();
        for block in postings.chunks(BLOCK_LEN) {
            self.append_block(
                block,
                block.iter().map(|p| words(p.0)).min().unwrap_or(0),
                frontier(block.iter().map(|p| (p.1, words(p.0)))),
            );
        }
        self.data.shrink_to_fit();
        self.count = u32::try_from(postings.len()).expect("posting list over u32::MAX");
        self.positive_count = postings.iter().filter(|p| p.1 > 0).count() as u32;
        self.last = postings.last().map_or(0, |p| p.0 as u64);
    }

    #[cfg(test)]
    pub(super) fn append_run(&mut self, run: &[Posting]) {
        self.append_run_with_lengths(run, |_| 0);
    }

    fn append_run_with_lengths(&mut self, run: &[Posting], words: impl Fn(usize) -> u32) {
        debug_assert!(
            run.windows(2).all(|p| p[0].0 < p[1].0),
            "unsorted posting run"
        );
        let Some(&(first, _)) = run.first() else {
            return;
        };
        if self.blocks.capacity() == 0 {
            self.blocks.reserve_exact(run.len().div_ceil(BLOCK_LEN));
        }
        if self.count > 0 && first as u64 <= self.last {
            if let [posting] = run {
                // A replacement's posting returns to its slot: one block takes it.
                self.insert(*posting, words);
                return;
            }
            let merged = merge_sorted(&self.decode(), run);
            self.encode(&merged, words);
            return;
        }
        let mut start = 0;
        if let Some(block) = self.blocks.pop_if(|b| usize::from(b.count) < BLOCK_LEN) {
            let mut decoded = [(0, 0); BLOCK_LEN];
            self.decode_values(&block, &mut decoded);
            self.data.truncate(block.offset);
            let old_len = usize::from(block.count);
            let take = (BLOCK_LEN - old_len).min(run.len());
            decoded[old_len..old_len + take].copy_from_slice(&run[..take]);
            let min_words = run[..take]
                .iter()
                .map(|p| words(p.0))
                .min()
                .unwrap_or(0)
                .min(block.min_words);
            let fallback = (block.max_frequency, block.min_words);
            let old_points = block
                .frontier
                .as_deref()
                .unwrap_or(std::slice::from_ref(&fallback));
            let frontier = frontier(
                old_points
                    .iter()
                    .copied()
                    .chain(run[..take].iter().map(|p| (p.1, words(p.0)))),
            );
            self.append_block(&decoded[..old_len + take], min_words, frontier);
            start = take;
        }
        for block in run[start..].chunks(BLOCK_LEN) {
            self.append_block(
                block,
                block.iter().map(|p| words(p.0)).min().unwrap_or(0),
                frontier(block.iter().map(|p| (p.1, words(p.0)))),
            );
        }
        self.count = u32::try_from(self.len() + run.len()).expect("posting list over u32::MAX");
        self.positive_count += run.iter().filter(|p| p.1 > 0).count() as u32;
        self.last = run.last().unwrap().0 as u64;
    }

    /// The block holding `slot`, decoded into `out`, and its posting count.
    fn block_of(&self, slot: usize, out: &mut [Posting; BLOCK_LEN]) -> (usize, usize) {
        let block = self.blocks.partition_point(|b| b.last < slot);
        assert!(block < self.blocks.len(), "document missing from posting");
        (block, self.decode_block_into(block, out))
    }

    /// Re-encodes block `index` as `postings`: no block when they are empty, two halves when
    /// they overflow a block. The later blocks keep their encoding and bounds and only move,
    /// so an edit recomputes one block's bounds rather than the list's; blocks other than the
    /// last may then hold fewer than [`BLOCK_LEN`] postings.
    fn rewrite_block(&mut self, index: usize, postings: &[Posting], words: impl Fn(usize) -> u32) {
        let start = self.blocks[index].offset;
        let end = self
            .blocks
            .get(index + 1)
            .map_or(self.data.len(), |block| block.offset);
        let mut scratch = PostingList::default();
        let size = if postings.len() > BLOCK_LEN {
            postings.len().div_ceil(2)
        } else {
            BLOCK_LEN
        };
        for run in postings.chunks(size) {
            scratch.append_block(
                run,
                run.iter().map(|p| words(p.0)).min().unwrap_or(0),
                frontier(run.iter().map(|p| (p.1, words(p.0)))),
            );
        }
        let length = scratch.data.len();
        self.data.splice(start..end, scratch.data);
        for block in &mut self.blocks[index + 1..] {
            block.offset = block.offset - (end - start) + length;
        }
        for block in &mut scratch.blocks {
            block.offset += start;
        }
        self.blocks.splice(index..=index, scratch.blocks);
        self.touch();
    }

    /// Inserts one posting before the list's last into its block.
    fn insert(&mut self, posting: Posting, words: impl Fn(usize) -> u32) {
        let mut decoded = [(0, 0); BLOCK_LEN + 1];
        let block = self.blocks.partition_point(|b| b.last < posting.0);
        let count = {
            let out: &mut [Posting; BLOCK_LEN] =
                (&mut decoded[..BLOCK_LEN]).try_into().expect("block");
            self.decode_block_into(block, out)
        };
        let at = match decoded[..count].binary_search_by_key(&posting.0, |p| p.0) {
            Ok(_) => panic!("duplicate document slot in posting"),
            Err(at) => at,
        };
        decoded.copy_within(at..count, at + 1);
        decoded[at] = posting;
        self.rewrite_block(block, &decoded[..count + 1], words);
        self.count += 1;
        self.positive_count += u32::from(posting.1 > 0);
    }

    fn remove(&mut self, slot: usize, words: impl Fn(usize) -> u32) {
        let mut decoded = [(0, 0); BLOCK_LEN];
        let (block, count) = self.block_of(slot, &mut decoded);
        if count <= MIN_BLOCK_LEN && block + 1 < self.blocks.len() {
            // A block this short re-packs the list rather than fragment it further.
            let mut postings = self.decode();
            let index = postings
                .binary_search_by_key(&slot, |p| p.0)
                .expect("document missing from posting");
            postings.remove(index);
            self.encode(&postings, words);
            return;
        }
        let index = decoded[..count]
            .binary_search_by_key(&slot, |p| p.0)
            .expect("document missing from posting");
        let (_, frequency) = decoded[index];
        decoded.copy_within(index + 1..count, index);
        self.rewrite_block(block, &decoded[..count - 1], words);
        self.count -= 1;
        self.positive_count -= u32::from(frequency > 0);
        self.last = self.blocks.last().map_or(0, |b| b.last as u64);
        if self.count == 0 {
            self.data = Vec::new();
            self.blocks = Vec::new();
            self.touch();
        }
    }

    fn set_frequency(&mut self, slot: usize, frequency: u32, words: impl Fn(usize) -> u32) {
        let mut decoded = [(0, 0); BLOCK_LEN];
        let (block, count) = self.block_of(slot, &mut decoded);
        let index = decoded[..count]
            .binary_search_by_key(&slot, |p| p.0)
            .expect("document missing from posting");
        let previous = std::mem::replace(&mut decoded[index].1, frequency);
        self.positive_count =
            self.positive_count - u32::from(previous > 0) + u32::from(frequency > 0);
        self.rewrite_block(block, &decoded[..count], words);
    }

    /// Whether every block's bounds hold for its postings under the lengths `words`: the
    /// largest frequency, the least length, and a frontier point dominating each posting.
    #[cfg(test)]
    pub(super) fn bounds_cover(&self, words: &[u32]) -> bool {
        self.blocks.iter().enumerate().all(|(index, block)| {
            let mut postings = [(0, 0); BLOCK_LEN];
            self.decode_block(index, &mut postings);
            postings[..usize::from(block.count)]
                .iter()
                .all(|&(slot, frequency)| {
                    let length = words[slot];
                    frequency <= block.max_frequency
                        && block.min_words <= length
                        && (frequency == 0
                            || block.frontier.as_deref().is_none_or(|points| {
                                points.iter().any(|&(f, w)| frequency <= f && w <= length)
                            }))
                })
        })
    }

    /// Whether every block's length bounds are exactly those a fresh build from `words`
    /// gives: no rewrite loosened them.
    #[cfg(test)]
    pub(super) fn bounds_are_tight(&self, words: &[u32]) -> bool {
        self.blocks.iter().enumerate().all(|(index, block)| {
            let mut postings = [(0, 0); BLOCK_LEN];
            self.decode_block(index, &mut postings);
            let postings = &postings[..usize::from(block.count)];
            block.min_words == postings.iter().map(|p| words[p.0]).min().unwrap_or(0)
                && block.frontier == frontier(postings.iter().map(|p| (p.1, words[p.0])))
        })
    }

    pub(super) fn invalidate_length(&mut self, slot: usize) {
        let block = self.blocks.partition_point(|b| b.last < slot);
        self.blocks[block].min_words = 0;
        self.blocks[block].frontier = None;
        self.touch();
    }

    /// Drops the summary after a change to the blocks; a long list gets an empty one.
    fn touch(&mut self) {
        if self.blocks.len() < SUMMARY_MIN_BLOCKS {
            self.summary = None;
        } else if let Some(summary) = &mut self.summary {
            if summary.get().is_some() {
                **summary = OnceLock::new();
            }
        } else {
            self.summary = Some(Box::default());
        }
    }

    /// The frontier of each run of [`SUPER_LEN`] blocks (a block without a frontier
    /// contributes its largest frequency at its least length), or `None` for a short list.
    pub(super) fn summary(&self) -> Option<&[Frontier]> {
        let summary = self.summary.as_deref()?;
        Some(summary.get_or_init(|| {
            self.blocks
                .chunks(SUPER_LEN)
                .map(|run| {
                    let mut points = Vec::new();
                    for block in run {
                        match block.frontier.as_deref() {
                            Some(frontier) => points.extend_from_slice(frontier),
                            None => points.push((block.max_frequency, block.min_words)),
                        }
                    }
                    pareto(points)
                })
                .collect()
        }))
    }
}

/// A forward cursor over a list. A block is decoded only when the cursor needs a posting
/// after its first, so skipping over blocks by their bounds costs their metadata alone.
pub(super) struct PostingCursor<'a> {
    list: &'a PostingList,
    block: usize,
    index: usize,
    decoded: [Posting; BLOCK_LEN],
    length: usize,
    filled: bool,
}

impl<'a> PostingCursor<'a> {
    fn new(list: &'a PostingList) -> Self {
        let mut cursor = Self {
            list,
            block: 0,
            index: 0,
            decoded: [(0, 0); BLOCK_LEN],
            length: 0,
            filled: false,
        };
        cursor.load(0);
        cursor
    }

    fn load(&mut self, block: usize) {
        self.block = block;
        self.index = 0;
        self.filled = false;
        match self.list.blocks.get(block) {
            Some(b) => {
                self.length = usize::from(b.count);
                self.decoded[0] = (b.first, self.list.first_frequency(b));
            }
            None => self.length = 0,
        }
    }

    fn fill(&mut self) {
        if !self.filled {
            self.list.decode_block(self.block, &mut self.decoded);
            self.filled = true;
        }
    }

    pub(super) fn peek(&self) -> Option<Posting> {
        (self.index < self.length).then(|| self.decoded[self.index])
    }

    /// Moves to the first block that can hold `target` without decoding it, so the block's
    /// bounds can be read first; `false` once no block can. The cursor's posting may then
    /// still precede `target` until [`Self::advance`].
    pub(super) fn locate(&mut self, target: usize) -> bool {
        if self.index >= self.length {
            return false;
        }
        if self.list.blocks[self.block].last < target {
            let after = self.block + 1;
            let block = after + self.list.blocks[after..].partition_point(|b| b.last < target);
            self.load(block);
        }
        self.length > 0
    }

    /// The largest frequency and least length over the blocks at or after the cursor's that
    /// overlap slots `first..=last`, read without decoding; `None` when none overlaps.
    pub(super) fn range_maximum(&self, first: usize, last: usize) -> Option<(u32, u32)> {
        if self.index >= self.length {
            return None;
        }
        let blocks = &self.list.blocks[self.block..];
        let start = blocks.partition_point(|b| b.last < first);
        blocks[start..]
            .iter()
            .take_while(|b| b.first <= last)
            .map(|b| (b.max_frequency, b.min_words))
            .reduce(|(f, w), (g, v)| (f.max(g), w.min(v)))
    }

    pub(super) fn block_end(&self) -> Option<usize> {
        self.peek().map(|_| self.list.blocks[self.block].last)
    }

    /// The index of the cursor's block, which changes only when the cursor leaves it.
    pub(super) fn block_index(&self) -> usize {
        self.block
    }

    /// Whether the cursor stands on a block it has not decoded.
    pub(super) fn at_block_start(&self) -> bool {
        self.index == 0 && !self.filled && self.length > 0
    }

    #[cfg(test)]
    pub(super) fn block_impacts(&self) -> &[(u32, u32)] {
        let block = &self.list.blocks[self.block];
        block.frontier.as_deref().unwrap_or(&[])
    }

    pub(super) fn block_maximum(&self) -> (u32, u32) {
        let block = &self.list.blocks[self.block];
        (block.max_frequency, block.min_words)
    }

    pub(super) fn advance(&mut self, target: usize) -> Option<Posting> {
        if self.peek().is_none_or(|p| p.0 >= target) {
            return self.peek();
        }
        // Blocks before the current one end below the target; only those after are searched.
        if !self.locate(target) || self.decoded[0].0 >= target && self.index == 0 {
            return self.peek();
        }
        self.fill();
        self.index += self.decoded[self.index..self.length].partition_point(|p| p.0 < target);
        self.peek()
    }
}

impl Iterator for PostingCursor<'_> {
    type Item = Posting;
    fn next(&mut self) -> Option<Posting> {
        let value = self.peek()?;
        self.index += 1;
        if self.index == self.length {
            self.load(self.block + 1);
        } else {
            self.fill();
        }
        Some(value)
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
    #[cfg(test)]
    pub(super) fn record_memory(&self, census: &mut super::census::MemoryCensus) {
        use std::mem::{size_of, size_of_val};
        census
            .counts
            .insert("postings.block_size", size_of::<Block>());
        census
            .counts
            .insert("postings.list_size", size_of::<PostingList>());
        census.bytes(
            "postings.lists.capacity",
            self.lists.capacity() * size_of::<PostingList>(),
        );
        census.bytes(
            "postings.free.capacity",
            self.free.capacity() * size_of::<u32>(),
        );
        census.bytes(
            "postings.dictionary.text.capacity",
            self.dictionary.bytes.capacity(),
        );
        census.bytes(
            "postings.dictionary.offsets.capacity",
            self.dictionary.offsets.capacity() * size_of::<u32>(),
        );
        census.bytes(
            "postings.dictionary.ids.capacity",
            self.dictionary.base_ids.capacity() * size_of::<u32>(),
        );
        census.bytes(
            "postings.dictionary.overlay.payload",
            self.dictionary
                .overlay
                .iter()
                .map(|(term, id)| term.len() + size_of_val(term) + size_of_val(id))
                .sum(),
        );
        for list in &self.lists {
            *census
                .histogram
                .entry((list.len(), list.blocks.len(), list.blocks.capacity()))
                .or_default() += 1;
            census.count("postings.postings", list.len());
            census.count("postings.lists", 1);
            census.count("postings.blocks", list.blocks.len());
            census.bytes("postings.data.length", list.data.len());
            census.bytes("postings.data.capacity", list.data.capacity());
            census.bytes(
                "postings.data.unused",
                list.data.capacity() - list.data.len(),
            );
            census.bytes(
                "postings.blocks.length",
                list.blocks.len() * size_of::<Block>(),
            );
            census.bytes(
                "postings.blocks.capacity",
                list.blocks.capacity() * size_of::<Block>(),
            );
            census.bytes(
                "postings.blocks.unused",
                (list.blocks.capacity() - list.blocks.len()) * size_of::<Block>(),
            );
            for block in &list.blocks {
                census.bytes(
                    "postings.frontiers",
                    block.frontier.as_deref().map_or(0, size_of_val),
                );
            }
        }
    }

    /// Lists for `entries`: `(term, postings)` in strictly ascending term order, every list
    /// non-empty and sorted by slot; `words` holds each slot's word count, which bounds the
    /// blocks as a fresh build bounds them.
    pub(super) fn from_sorted(entries: Vec<RestoredList>, words: &[u32]) -> Self {
        let mut dictionary = TermDictionary::default();
        let mut lists = Vec::with_capacity(entries.len());
        dictionary.offsets.reserve_exact(entries.len() + 1);
        dictionary.offsets.push(0);
        for (id, (term, postings)) in entries.into_iter().enumerate() {
            debug_assert!(!postings.is_empty(), "empty posting list");
            dictionary.bytes.extend_from_slice(term.as_bytes());
            dictionary
                .offsets
                .push(u32::try_from(dictionary.bytes.len()).expect("term dictionary over 4 GiB"));
            dictionary
                .base_ids
                .push(u32::try_from(id).expect("term count over u32"));
            let mut list = PostingList::default();
            list.encode(&postings, |slot| words.get(slot).copied().unwrap_or(0));
            lists.push(list);
        }
        debug_assert!(
            (1..dictionary.base_ids.len())
                .all(|i| dictionary.base_term(i - 1) < dictionary.base_term(i)),
            "unsorted posting terms"
        );
        dictionary.bytes.shrink_to_fit();
        Self {
            dictionary,
            lists,
            free: Vec::new(),
        }
    }

    pub(super) fn get(&self, term: &str) -> Option<&PostingList> {
        self.dictionary.get(term).map(|id| &self.lists[id as usize])
    }

    /// Every term with postings in byte order (`offsets` delimit the terms in `bytes`) and
    /// its list, as a serving image stores them.
    pub(super) fn sorted_terms(&self) -> (Vec<u8>, Vec<u32>, Vec<&PostingList>) {
        let overlay: Vec<(&[u8], u32)> = self
            .dictionary
            .overlay
            .iter()
            .map(|(term, &id)| (term.as_bytes(), id))
            .collect();
        let (bytes, offsets, ids) = self.dictionary.merged_base(&overlay);
        let lists = ids.iter().map(|&id| &self.lists[id as usize]).collect();
        (bytes, offsets, lists)
    }

    /// Number of terms with postings.
    #[cfg(test)]
    pub(super) fn term_count(&self) -> usize {
        self.dictionary.len()
    }

    /// Adds a `(term, slot)` posting absent from the lists; `words` holds each slot's length.
    pub(super) fn insert(&mut self, term: &str, slot: usize, frequency: u32, words: &[u32]) {
        let id = match self.dictionary.get(term) {
            Some(id) => id,
            None => {
                let id = self.allocate_list();
                self.dictionary.insert(term, id);
                id
            }
        };
        self.lists[id as usize].append_run_with_lengths(&[(slot, frequency)], |slot| {
            words.get(slot).copied().unwrap_or(0)
        });
    }

    /// Adds one run per distinct term, each absent from that term's list.
    #[cfg(test)]
    pub(super) fn insert_runs(&mut self, runs: &[TermRun<'_>]) {
        self.insert_runs_with_lengths(runs, &[]);
    }

    fn insert_runs_with_lengths(&mut self, runs: &[TermRun<'_>], word_counts: &[u32]) {
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
        for (run, id) in runs.iter().zip(ids) {
            let id = id.unwrap_or_else(|| {
                let id = self.allocate_list();
                new_terms.push((run.term, id));
                id
            });
            self.lists[id as usize].append_run_with_lengths(&run.postings, |slot| {
                word_counts.get(slot).copied().unwrap_or(0)
            });
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

    pub(super) fn remove(&mut self, term: &str, slot: usize, words: &[u32]) {
        let id = self
            .dictionary
            .get(term)
            .expect("document term missing posting");
        let list = &mut self.lists[id as usize];
        list.remove(slot, |slot| words.get(slot).copied().unwrap_or(0));
        if list.is_empty() {
            *list = PostingList::default();
            self.dictionary.remove(term);
            self.free.push(id);
        }
    }

    pub(super) fn invalidate_length(&mut self, term: &str, slot: usize) {
        let id = self
            .dictionary
            .get(term)
            .expect("document term missing posting");
        self.lists[id as usize].invalidate_length(slot);
    }

    pub(super) fn set_frequency(&mut self, term: &str, slot: usize, frequency: u32, words: &[u32]) {
        let id = self
            .dictionary
            .get(term)
            .expect("document term missing posting");
        self.lists[id as usize].set_frequency(slot, frequency, |slot| {
            words.get(slot).copied().unwrap_or(0)
        });
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
            // In-place edits keep every block but the last at least a quarter full.
            let (last, earlier) = list.blocks.split_last().expect("nonempty list");
            assert!(last.count > 0);
            assert!(earlier
                .iter()
                .all(|b| usize::from(b.count) >= MIN_BLOCK_LEN));
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
type RunMap<'a> = HashMap<&'a str, Vec<Posting>, FixedState>;

/// Merge bucket of a term: FNV-1a over its bytes, deliberately not the maps' hash, so the
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
    word_counts: &[u32],
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
        lists.insert_runs_with_lengths(&runs, word_counts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packed checkpoint decoder agrees with the checked one for every mix of one-byte
    /// postings, wide gaps and frequencies at every position in the eight-byte window.
    #[test]
    fn checkpoint_fast_decoding_matches_checked_decoding() {
        for pattern in 0u32..256 {
            let mut postings = Vec::new();
            let mut slot = 0usize;
            for i in 0..67u32 {
                let wide = (pattern >> (i % 8)) & 1 == 1;
                slot += if wide {
                    70 + (i as usize) * 31
                } else {
                    1 + (i as usize) % 60
                };
                let frequency = if i % 11 == 3 { 2 + i } else { 1 };
                postings.push((slot, frequency));
            }
            let mut list = PostingList::default();
            list.append_run(&postings);
            let mut bytes = Vec::new();
            pack_checkpoint(&list, &mut bytes);
            let packed = Postings {
                bytes: &bytes,
                count: postings.len() as u32,
            };
            let mut fast = Vec::new();
            packed.for_each(|slot, frequency| fast.push((slot, frequency)));
            let mut checked = Vec::new();
            packed
                .try_for_each(|slot, frequency| {
                    checked.push((slot, frequency));
                    true
                })
                .unwrap();
            assert_eq!(fast, postings, "pattern {pattern}");
            assert_eq!(checked, postings, "pattern {pattern}");
        }
    }

    #[test]
    fn initial_block_capacity_matches_the_first_run() {
        for length in [0, 1, 127, 128, 129, 255, 256, 257, 511, 512, 513] {
            let run: Vec<Posting> = (0..length).map(|slot| (slot, 1)).collect();
            let mut list = PostingList::default();
            list.append_run(&run);
            assert_eq!(list.blocks.capacity(), length.div_ceil(BLOCK_LEN));
            assert_eq!(list.decode(), run);
            list.append_run(&[(length, 2)]);
            assert_eq!(list.len(), length + 1);
            assert_eq!(list.iter().last(), Some((length, 2)));
        }
    }

    #[test]
    fn frontiers_match_pairwise_dominance() {
        let check = |points: &[(u32, u32)]| {
            let mut expected: Vec<_> = points
                .iter()
                .copied()
                .filter(|&(frequency, words)| {
                    frequency > 0
                        && !points.iter().any(|&(other_frequency, other_words)| {
                            other_frequency >= frequency
                                && other_words <= words
                                && (other_frequency > frequency || other_words < words)
                        })
                })
                .collect();
            expected.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            expected.dedup();
            let expected = (expected.len() > 1).then_some(expected);
            assert_eq!(
                frontier(points.iter().copied()).as_deref(),
                expected.as_deref()
            );
        };
        for points in [
            vec![],
            vec![(0, 0), (0, u32::MAX)],
            vec![(1, u32::MAX)],
            vec![(1, 0), (1, 0), (u32::MAX, u32::MAX)],
            vec![(0, 0), (2, 5), (2, 7), (1, 5), (3, 8), (1, 1)],
        ] {
            check(&points);
        }
        for length in [BLOCK_LEN - 1, BLOCK_LEN] {
            check(
                &(0..length)
                    .map(|i| (i as u32 + 1, i as u32))
                    .collect::<Vec<_>>(),
            );
            for seed in 0..64 {
                let points: Vec<_> = (0..length)
                    .map(|i| (((i * 37 + seed) % 13) as u32, ((i * 17 + seed) % 23) as u32))
                    .collect();
                check(&points);
            }
        }
    }

    #[test]
    #[should_panic(expected = "frontier exceeds posting block length")]
    fn frontiers_reject_more_points_than_a_block_can_hold() {
        frontier(std::iter::repeat_n((1, 1), BLOCK_LEN + 1));
    }

    #[test]
    fn partial_blocks_preserve_full_frontiers_and_invalidated_fallbacks() {
        let run: Vec<_> = (0..BLOCK_LEN).map(|i| (i, i as u32 + 1)).collect();
        let mut list = PostingList::default();
        for posting in &run {
            list.append_run_with_lengths(std::slice::from_ref(posting), |slot| slot as u32 + 1);
        }
        let expected: Vec<_> = run
            .iter()
            .map(|&(slot, frequency)| (frequency, slot as u32 + 1))
            .rev()
            .collect();
        assert_eq!(
            list.blocks[0].frontier.as_deref(),
            Some(expected.as_slice())
        );
        assert_eq!(list.decode(), run);

        let mut list = PostingList::default();
        list.append_run_with_lengths(&[(0, 5)], |_| 10);
        list.invalidate_length(0);
        list.append_run_with_lengths(&[(1, 6), (2, 0)], |_| 20);
        assert_eq!(
            list.blocks[0].frontier.as_deref(),
            Some([(6, 20), (5, 0)].as_slice())
        );
        assert_eq!(list.blocks[0].min_words, 0);
        assert_eq!(list.blocks[0].max_frequency, 6);
    }

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
                    lists.set_frequency(&term, slot, frequency, &[]);
                    entry.insert(slot, frequency);
                }
                1 if entry.contains_key(&slot) => {
                    lists.remove(&term, slot, &[]);
                    entry.remove(&slot);
                    if entry.is_empty() {
                        model.remove(&term);
                    }
                }
                _ if !entry.contains_key(&slot) => {
                    lists.insert(&term, slot, frequency, &[]);
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
                    single.insert(run.term, slot, frequency, &[]);
                }
            }
            bulk.insert_runs(&runs);
            // Free a tenth of this batch's slots from every list holding them (only this
            // batch's runs do), so the next batch's runs start below some final slots.
            freed = slots.iter().copied().filter(|_| next() % 10 == 0).collect();
            for run in &runs {
                for &(slot, _) in &run.postings {
                    if freed.contains(&slot) {
                        bulk.remove(run.term, slot, &[]);
                        single.remove(run.term, slot, &[]);
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
    fn for_each_matches_iter_across_packed_widths() {
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
                    list.append_run(&run);
                    let mut visited = Vec::new();
                    list.for_each(|slot, frequency| visited.push((slot, frequency)));
                    assert_eq!(visited, list.iter().collect::<Vec<_>>());
                    assert_eq!(visited, run, "length {length} wide {wide}");
                }
            }
        }
    }
}

#[cfg(test)]
mod cursor_tests {
    use super::*;

    #[test]
    fn next_and_advance_match_sorted_runs_at_block_and_integer_limits() {
        for length in [0, 1, 127, 128, 129, 255, 256, 257] {
            for gap in [1usize, 127, 128, 16_384] {
                let mut run: Vec<Posting> = (0..length)
                    .map(|i| (i * gap, if i % 2 == 0 { 1 } else { u32::MAX }))
                    .collect();
                run.push((usize::MAX, 0));
                let mut list = PostingList::default();
                for batch in run.chunks(17) {
                    list.append_run(batch);
                }
                assert_eq!(list.iter().collect::<Vec<_>>(), run);
                assert_eq!(list.positive_len(), length);
                for target in [
                    0,
                    1,
                    126 * gap,
                    127 * gap,
                    128 * gap,
                    129 * gap,
                    usize::MAX - 1,
                    usize::MAX,
                ] {
                    let mut cursor = list.iter();
                    let expected = run.iter().copied().find(|p| p.0 >= target);
                    assert_eq!(cursor.advance(target), expected);
                    assert_eq!(cursor.advance(target), expected);
                    assert_eq!(cursor.next(), expected);
                    assert_eq!(
                        cursor.collect::<Vec<_>>(),
                        run.iter()
                            .copied()
                            .filter(|p| p.0 > expected.unwrap().0)
                            .collect::<Vec<_>>()
                    );
                }
                let mut cursor = list.iter();
                assert_eq!(cursor.advance(usize::MAX), Some((usize::MAX, 0)));
                assert_eq!(cursor.next(), Some((usize::MAX, 0)));
                assert_eq!(cursor.advance(usize::MAX), None);
            }
        }
    }
}
