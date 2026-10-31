// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! A document store served from a checkpoint and written to without rebuilding it. Writes (in
//! any encode mode: the checkpoint has no vector state of its own) go to the live
//! [`DocumentStore`] (which starts empty), and a written or deleted checkpoint
//! document is only marked dead here. The journal frames after the checkpoint record those
//! writes, so replaying them rebuilds the same dead set. Lexical statistics count the live
//! checkpoint chunks plus the live store's, and posting lists skip dead slots as they decode, so
//! scores equal those of the store rebuilt in full.

use std::sync::{Arc, OnceLock};

use super::image::ServingImage;
use super::index::DocumentStore;
use super::postings::Postings;
use super::search::{LexicalSource, PostingSet};

/// The checkpoint under a live store and which of its documents are dead.
pub(crate) struct Base {
    pub(super) image: Arc<ServingImage>,
    dead_slots: Vec<u64>,
    dead_documents: Vec<u64>,
    dead_document_count: usize,
    dead_chunks: usize,
    dead_words: usize,
    /// Each document's slots: offsets by ordinal into a slot list, built on the first kill.
    document_slots: OnceLock<(Vec<u32>, Vec<u32>)>,
}

fn bit(bits: &[u64], i: usize) -> bool {
    bits[i / 64] & (1 << (i % 64)) != 0
}

impl Base {
    pub(crate) fn new(image: Arc<ServingImage>) -> Self {
        Self {
            dead_slots: vec![0; image.slot_total().div_ceil(64)],
            dead_documents: vec![0; image.document_total().div_ceil(64)],
            image,
            dead_document_count: 0,
            dead_chunks: 0,
            dead_words: 0,
            document_slots: OnceLock::new(),
        }
    }

    /// Whether no checkpoint document is dead.
    pub(super) fn intact(&self) -> bool {
        self.dead_document_count == 0
    }

    /// The ordinal of checkpoint document `id`, when it is live.
    pub(super) fn live(&self, id: &str) -> Option<usize> {
        self.image
            .ordinal(id)
            .filter(|&ordinal| !bit(&self.dead_documents, ordinal))
    }

    pub(super) fn live_documents(&self) -> usize {
        self.image.document_total() - self.dead_document_count
    }

    pub(super) fn live_ordinals_from(
        &self,
        from: Option<&str>,
    ) -> impl Iterator<Item = usize> + '_ {
        let start = from.map_or(0, |id| self.image.lower_bound(id));
        (start..self.image.document_total()).filter(|&ordinal| !bit(&self.dead_documents, ordinal))
    }

    pub(super) fn slot_dead(&self, slot: usize) -> bool {
        bit(&self.dead_slots, slot)
    }

    /// Ordinals of the dead checkpoint documents, ascending.
    pub(super) fn dead_ordinals(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.image.document_total()).filter(|&ordinal| bit(&self.dead_documents, ordinal))
    }

    pub(super) fn live_chunks(&self) -> usize {
        self.image.chunk_count() - self.dead_chunks
    }

    pub(super) fn live_words(&self) -> usize {
        self.image.word_count() - self.dead_words
    }

    /// Marks live checkpoint document `ordinal` dead.
    pub(super) fn kill(&mut self, ordinal: usize) {
        debug_assert!(!bit(&self.dead_documents, ordinal));
        let (offsets, slots) = self.document_slots.get_or_init(|| {
            let documents = self.image.document_total();
            let mut offsets = vec![0u32; documents + 1];
            for slot in 0..self.image.slot_total() {
                if let Some(owner) = self.image.slot_owner(slot) {
                    offsets[owner + 1] += 1;
                }
            }
            for i in 0..documents {
                offsets[i + 1] += offsets[i];
            }
            let mut next = offsets.clone();
            let mut slots = vec![0u32; offsets[documents] as usize];
            for slot in 0..self.image.slot_total() {
                if let Some(owner) = self.image.slot_owner(slot) {
                    slots[next[owner] as usize] = slot as u32;
                    next[owner] += 1;
                }
            }
            (offsets, slots)
        });
        for &slot in &slots[offsets[ordinal] as usize..offsets[ordinal + 1] as usize] {
            let slot = slot as usize;
            self.dead_slots[slot / 64] |= 1 << (slot % 64);
            self.dead_chunks += 1;
            self.dead_words += self.image.slot_word_count(slot) as usize;
        }
        self.dead_documents[ordinal / 64] |= 1 << (ordinal % 64);
        self.dead_document_count += 1;
    }
}

/// Postings gathered from the checkpoint's live slots and the live store's, in slot order
/// (checkpoint slots first, then the live store's after them).
pub(super) struct Gathered(Vec<(u32, u32)>);

impl PostingSet for Gathered {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn for_each(&self, mut f: impl FnMut(usize, u32)) {
        for &(slot, frequency) in &self.0 {
            f(slot as usize, frequency);
        }
    }
}

/// The checkpoint's live documents and the live store's as one lexical source; slot `s` below
/// the checkpoint's slot count is the checkpoint's, slot `s` above it the live store's
/// `s - count`.
pub(super) struct Overlaid<'a> {
    pub(super) base: &'a Base,
    pub(super) store: &'a DocumentStore,
}

impl Overlaid<'_> {
    pub(super) fn split(&self) -> usize {
        self.base.image.slot_total()
    }
}

impl LexicalSource for Overlaid<'_> {
    type List<'b>
        = Gathered
    where
        Self: 'b;

    fn slot_count(&self) -> usize {
        self.split() + self.store.slot_count
    }

    fn chunk_count(&self) -> usize {
        self.base.live_chunks() + self.store.chunk_count
    }

    fn word_count(&self) -> usize {
        self.base.live_words() + self.store.word_count
    }

    fn state(&self) -> (super::index::StoreIdentity, u64) {
        (self.store.identity, self.store.generation)
    }

    fn postings(&self, form: usize, term: &str) -> Option<Gathered> {
        let base: Option<Postings<'_>> = self.base.image.postings(form, term);
        let live = self.store.postings[form].get(term);
        let capacity = base.map_or(0, |list| list.len()) + live.map_or(0, |list| list.len());
        let mut gathered = Vec::with_capacity(capacity);
        if let Some(list) = base {
            list.for_each(|slot, frequency| {
                if !self.base.slot_dead(slot) {
                    gathered.push((slot as u32, frequency));
                }
            });
        }
        if let Some(list) = live {
            let split = self.split();
            list.for_each(|slot, frequency| gathered.push(((split + slot) as u32, frequency)));
        }
        Some(Gathered(gathered))
    }

    fn slot_word_count(&self, slot: usize) -> u32 {
        match slot.checked_sub(self.split()) {
            None => self.base.image.slot_word_count(slot),
            Some(slot) => self.store.slot_word_count(slot),
        }
    }

    fn slot_order(&self, slot: usize) -> (&str, usize) {
        match slot.checked_sub(self.split()) {
            None => self.base.image.slot_order(slot),
            Some(slot) => self.store.slot_order(slot),
        }
    }
}
