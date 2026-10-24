// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cmp::Ordering;
use std::collections::{btree_map, BTreeMap};
use std::sync::Arc;

#[derive(Clone, Default)]
pub(crate) struct StoredTerms {
    terms: Option<Box<TermData>>,
    pub(super) positions: Option<Arc<super::positions::Positions>>,
}

#[derive(Clone)]
enum TermData {
    Packed(PackedTerms),
    Shared(SharedTerms),
    Tree(BTreeMap<String, u32>),
}

#[derive(Clone)]
struct PackedTerms {
    text: Box<str>,
    starts: Box<[u32]>,
    frequencies: Box<[u32]>,
}

pub(super) struct SharedDictionary {
    text: Box<str>,
    starts: Box<[u32]>,
}

impl SharedDictionary {
    pub(super) fn new(words: &[&str]) -> Self {
        let mut text = String::new();
        let mut starts = Vec::with_capacity(words.len());
        for word in words {
            starts.push(u32::try_from(text.len()).expect("dictionary exceeds u32"));
            text.push_str(word);
        }
        Self::from_parts(text, starts)
    }
    pub(super) fn from_parts(text: String, starts: Vec<u32>) -> Self {
        Self {
            text: text.into_boxed_str(),
            starts: starts.into_boxed_slice(),
        }
    }
    pub(super) fn len(&self) -> usize {
        self.starts.len()
    }
    pub(super) fn term(&self, index: usize) -> &str {
        let end = self
            .starts
            .get(index + 1)
            .map_or(self.text.len(), |&v| v as usize);
        &self.text[self.starts[index] as usize..end]
    }

    /// The id of `key`; the terms are strictly ascending (built sorted, and decoding
    /// rejects any other order).
    fn find(&self, key: &str) -> Option<u32> {
        let (mut low, mut high) = (0, self.starts.len());
        while low < high {
            let middle = low + (high - low) / 2;
            match self.term(middle).cmp(key) {
                Ordering::Less => low = middle + 1,
                Ordering::Equal => return u32::try_from(middle).ok(),
                Ordering::Greater => high = middle,
            }
        }
        None
    }
}

#[derive(Clone)]
struct SharedTerms {
    dictionary: Arc<SharedDictionary>,
    ids: Box<[u8]>,
    bits: u8,
    frequencies: SharedFrequencies,
}

#[derive(Clone)]
enum SharedFrequencies {
    Byte(Box<[u8]>),
    Short(Box<[u16]>),
    Word(Box<[u32]>),
}

static FREQUENCIES: [u32; 65_536] = {
    let mut values = [0; 65_536];
    let mut i = 0;
    while i < values.len() {
        values[i] = i as u32;
        i += 1;
    }
    values
};

impl From<Vec<u32>> for SharedFrequencies {
    fn from(values: Vec<u32>) -> Self {
        let maximum = values.iter().copied().max().unwrap_or(0);
        if maximum <= u32::from(u8::MAX) {
            Self::Byte(values.into_iter().map(|v| v as u8).collect())
        } else if maximum <= u32::from(u16::MAX) {
            Self::Short(values.into_iter().map(|v| v as u16).collect())
        } else {
            Self::Word(values.into_boxed_slice())
        }
    }
}

impl SharedFrequencies {
    fn len(&self) -> usize {
        match self {
            Self::Byte(v) => v.len(),
            Self::Short(v) => v.len(),
            Self::Word(v) => v.len(),
        }
    }
    fn get(&self, index: usize) -> Option<&u32> {
        match self {
            Self::Byte(v) => v.get(index).map(|&f| &FREQUENCIES[usize::from(f)]),
            Self::Short(v) => v.get(index).map(|&f| &FREQUENCIES[usize::from(f)]),
            Self::Word(v) => v.get(index),
        }
    }
}

impl SharedTerms {
    fn key(&self, index: usize) -> &str {
        self.dictionary
            .term(super::postings::unpack(&self.ids, index, self.bits) as usize)
    }
    fn index(&self, key: &str) -> Option<usize> {
        let (mut low, mut high) = (0, self.frequencies.len());
        while low < high {
            let middle = low + (high - low) / 2;
            match self.key(middle).cmp(key) {
                Ordering::Less => low = middle + 1,
                Ordering::Equal => return Some(middle),
                Ordering::Greater => high = middle,
            }
        }
        None
    }

    /// [`Self::index`] by the key's dictionary id: the ids ascend with the keys, so the
    /// search compares integers.
    fn index_cached(&self, key: &str, cache: &mut WordId) -> Option<usize> {
        let id = match &cache.dictionary {
            Some((dictionary, id)) if Arc::ptr_eq(dictionary, &self.dictionary) => *id,
            _ => {
                let id = self.dictionary.find(key);
                cache.dictionary = Some((Arc::clone(&self.dictionary), id));
                id
            }
        };
        let id = u64::from(id?);
        let (mut low, mut high) = self.bracket(id)?;
        while low < high {
            let middle = low + (high - low) / 2;
            match super::postings::unpack(&self.ids, middle, self.bits).cmp(&id) {
                Ordering::Less => low = middle + 1,
                Ordering::Equal => return Some(middle),
                Ordering::Greater => high = middle,
            }
        }
        None
    }
}

impl SharedTerms {
    /// A range of indices that holds `id` if any does, from a guess interpolated between the
    /// first and last ids and widened by doubling steps: the probes of a search over a cold
    /// table stay few.
    fn bracket(&self, id: u64) -> Option<(usize, usize)> {
        let count = self.frequencies.len();
        let at = |index: usize| super::postings::unpack(&self.ids, index, self.bits);
        let (first, last) = (at(0), at(count.checked_sub(1)?));
        if id < first || id > last {
            return None;
        }
        let guess = if last == first {
            0
        } else {
            ((u128::from(id - first) * (count - 1) as u128) / u128::from(last - first)) as usize
        };
        let value = at(guess);
        if value == id {
            return Some((guess, guess + 1));
        }
        let mut step = 1;
        if value < id {
            let mut low = guess + 1;
            loop {
                let probe = guess + step;
                if probe >= count {
                    return Some((low, count));
                }
                if at(probe) >= id {
                    return Some((low, probe + 1));
                }
                low = probe + 1;
                step *= 2;
            }
        }
        let mut high = guess;
        loop {
            let Some(probe) = guess.checked_sub(step) else {
                return Some((0, high));
            };
            if at(probe) <= id {
                return Some((probe, high));
            }
            high = probe;
            step *= 2;
        }
    }
}

/// One word's id in the last shared dictionary a lookup met. The chunks of one ingest frame
/// share a dictionary, so a scan over many chunks resolves the word about once per frame.
#[derive(Default)]
pub(super) struct WordId {
    dictionary: Option<(Arc<SharedDictionary>, Option<u32>)>,
}

impl PackedTerms {
    fn key(&self, index: usize) -> &str {
        let end = self
            .starts
            .get(index + 1)
            .map(|&at| at as usize)
            .unwrap_or(self.text.len());
        &self.text[self.starts[index] as usize..end]
    }

    fn get(&self, key: &str) -> Option<&u32> {
        let (mut low, mut high) = (0, self.frequencies.len());
        while low < high {
            let middle = low + (high - low) / 2;
            match self.key(middle).cmp(key) {
                Ordering::Less => low = middle + 1,
                Ordering::Equal => return Some(&self.frequencies[middle]),
                Ordering::Greater => high = middle,
            }
        }
        None
    }
}

impl StoredTerms {
    #[cfg(test)]
    pub(super) fn record_memory(&self, census: &mut super::census::MemoryCensus) {
        use std::mem::{size_of, size_of_val};
        if let Some(terms) = self.terms.as_deref() {
            census.bytes("forward.term_records", size_of::<TermData>());
            census.count("forward.terms", self.len());
            match terms {
                TermData::Packed(table) => {
                    census.count("forward.packed_tables", 1);
                    census.bytes("forward.term_text", table.text.len());
                    census.bytes("forward.term_offsets", size_of_val(&*table.starts));
                    census.bytes("forward.frequencies", size_of_val(&*table.frequencies));
                }
                TermData::Shared(table) => {
                    census.count("forward.shared_tables", 1);
                    census.bytes("forward.term_ids", table.ids.len());
                    census.bytes(
                        "forward.frequencies",
                        match &table.frequencies {
                            SharedFrequencies::Byte(v) => size_of_val(&**v),
                            SharedFrequencies::Short(v) => size_of_val(&**v),
                            SharedFrequencies::Word(v) => size_of_val(&**v),
                        },
                    );
                    if census
                        .dictionaries
                        .insert(Arc::as_ptr(&table.dictionary) as usize)
                    {
                        census.count("forward.shared_dictionaries", 1);
                        census.bytes(
                            "forward.dictionary_records",
                            size_of::<SharedDictionary>() + 2 * size_of::<usize>(),
                        );
                        census.bytes("forward.dictionary_text", table.dictionary.text.len());
                        census.bytes(
                            "forward.dictionary_offsets",
                            size_of_val(&*table.dictionary.starts),
                        );
                    }
                }
                TermData::Tree(table) => {
                    census.count("forward.tree_tables", 1);
                    census.bytes(
                        "forward.tree.payload",
                        table
                            .iter()
                            .map(|(term, frequency)| {
                                size_of_val(term) + size_of_val(frequency) + term.capacity()
                            })
                            .sum(),
                    );
                }
            }
        }
        if let Some(positions) = self.positions.as_deref() {
            positions.record_memory(census);
        }
    }

    fn len(&self) -> usize {
        match self.terms.as_deref() {
            None => 0,
            Some(TermData::Packed(table)) => table.frequencies.len(),
            Some(TermData::Shared(table)) => table.frequencies.len(),
            Some(TermData::Tree(table)) => table.len(),
        }
    }

    pub(crate) fn get(&self, key: &str) -> Option<&u32> {
        match self.terms.as_deref() {
            None => None,
            Some(TermData::Packed(table)) => table.get(key),
            Some(TermData::Shared(table)) => table
                .index(key)
                .and_then(|index| table.frequencies.get(index)),
            Some(TermData::Tree(table)) => table.get(key),
        }
    }

    pub(crate) fn is_whole_document(&self) -> bool {
        self.positions
            .as_ref()
            .is_some_and(|p| p.version == super::positions::DOCUMENT_VERSION)
    }

    pub(super) fn index(&self, key: &str) -> Option<usize> {
        match self.terms.as_deref() {
            None => None,
            Some(TermData::Packed(table)) => {
                let (mut low, mut high) = (0, table.starts.len());
                while low < high {
                    let middle = low + (high - low) / 2;
                    match table.key(middle).cmp(key) {
                        Ordering::Less => low = middle + 1,
                        Ordering::Equal => return Some(middle),
                        Ordering::Greater => high = middle,
                    }
                }
                None
            }
            Some(TermData::Tree(table)) => table.keys().position(|k| k == key),
            Some(TermData::Shared(table)) => table.index(key),
        }
    }

    /// [`Self::index`] for a word looked up in many chunks, with its dictionary id in `cache`.
    pub(super) fn index_cached(&self, key: &str, cache: &mut WordId) -> Option<usize> {
        match self.terms.as_deref() {
            Some(TermData::Shared(table)) => table.index_cached(key, cache),
            _ => self.index(key),
        }
    }

    pub(super) fn frequency_at(&self, index: usize) -> u32 {
        match self.terms.as_deref().expect("nonempty term table") {
            TermData::Packed(table) => table.frequencies[index],
            TermData::Shared(table) => *table.frequencies.get(index).expect("term index"),
            TermData::Tree(table) => *table.values().nth(index).expect("term index"),
        }
    }

    pub(crate) fn contains_key(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub(crate) fn keys(&self) -> impl ExactSizeIterator<Item = &str> {
        self.into_iter().map(|(key, _)| key)
    }

    #[cfg(test)]
    pub(crate) fn assert_valid_storage(&self) {
        match self.terms.as_deref() {
            None => assert_eq!(self.len(), 0),
            Some(TermData::Tree(table)) => {
                assert!(!table.is_empty());
                assert!(table.keys().all(|key| key.capacity() == key.len()));
            }
            Some(TermData::Shared(table)) => {
                assert_eq!(
                    table.ids.len(),
                    (table.frequencies.len() * usize::from(table.bits)).div_ceil(8)
                );
                assert!(self.keys().zip(self.keys().skip(1)).all(|(a, b)| a < b));
            }
            Some(TermData::Packed(table)) => {
                assert_eq!(table.starts.len(), table.frequencies.len());
                assert_eq!(table.starts.first(), Some(&0));
                assert!(table
                    .starts
                    .iter()
                    .all(|&at| table.text.is_char_boundary(at as usize)));
                assert_eq!(self.keys().map(str::len).sum::<usize>(), table.text.len());
                assert!(self.keys().zip(self.keys().skip(1)).all(|(a, b)| a < b));
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn insert(&mut self, key: String, frequency: u32) -> Option<u32> {
        let mut map: BTreeMap<_, _> = (&*self)
            .into_iter()
            .map(|(key, value)| (key.to_owned(), *value))
            .collect();
        let old = map.insert(key, frequency);
        *self = map.into();
        old
    }

    #[cfg(test)]
    pub(crate) fn remove(&mut self, key: &str) -> Option<u32> {
        let mut map: BTreeMap<_, _> = (&*self)
            .into_iter()
            .map(|(key, value)| (key.to_owned(), *value))
            .collect();
        let old = map.remove(key);
        *self = map.into();
        old
    }
}

impl StoredTerms {
    pub(super) fn from_dictionary(
        dictionary: Arc<SharedDictionary>,
        ids: &[u32],
        frequencies: Vec<u32>,
    ) -> Self {
        assert_eq!(ids.len(), frequencies.len(), "shared term count mismatch");
        if ids.is_empty() {
            return Self::default();
        }
        debug_assert!(ids.windows(2).all(|p| p[0] < p[1]));
        debug_assert!((ids[ids.len() - 1] as usize) < dictionary.len());
        let bits = (32 - ids[ids.len() - 1].leading_zeros()) as u8;
        let mut packed = Vec::new();
        super::postings::pack(&mut packed, ids.iter().map(|&id| u64::from(id)), bits);
        Self {
            terms: Some(Box::new(TermData::Shared(SharedTerms {
                dictionary,
                ids: packed.into_boxed_slice(),
                bits,
                frequencies: frequencies.into(),
            }))),
            positions: None,
        }
    }

    pub(super) fn share_documents(documents: &mut [Arc<super::StoredDocument>]) {
        if documents.iter().all(|d| {
            d.chunks
                .iter()
                .all(|c| matches!(c.terms.terms.as_deref(), None | Some(TermData::Shared(_))))
        }) {
            return;
        }
        let (dictionary, ids) =
            super::binary::FrameTerms::new(documents.iter().map(Arc::as_ref)).into_shared();
        let mut at = 0;
        for document in documents {
            for chunk in &mut Arc::make_mut(document).chunks {
                let count = chunk.terms.len();
                let frequencies = (&chunk.terms).into_iter().map(|(_, &f)| f).collect();
                let mut shared = Self::from_dictionary(
                    Arc::clone(&dictionary),
                    &ids[at..at + count],
                    frequencies,
                );
                shared.positions = chunk.terms.positions.take();
                chunk.terms = shared;
                at += count;
            }
        }
        assert_eq!(at, ids.len(), "shared dictionary cursor mismatch");
    }
}

fn can_pack(count: usize, bytes: usize) -> bool {
    let allowance = std::mem::size_of::<(String, u32)>() - 2 * std::mem::size_of::<u32>();
    u32::try_from(bytes).is_ok()
        && count
            .checked_mul(allowance)
            .is_some_and(|limit| bytes <= limit)
}

impl From<BTreeMap<String, u32>> for StoredTerms {
    fn from(map: BTreeMap<String, u32>) -> Self {
        Self::from_sorted(map.into_iter().collect())
    }
}

impl StoredTerms {
    /// Builds the table from entries whose keys are strictly ascending (the order of a
    /// `BTreeMap` and of every serialized table), without an intermediate tree.
    fn from_sorted(entries: Vec<(String, u32)>) -> Self {
        debug_assert!(entries.windows(2).all(|pair| pair[0].0 < pair[1].0));
        if entries.is_empty() {
            return Self::default();
        }
        let count = entries.len();
        let bytes = entries
            .iter()
            .try_fold(0usize, |total, (key, _)| total.checked_add(key.len()))
            .expect("term byte length overflow");
        let data = if can_pack(count, bytes) {
            let mut text = String::with_capacity(bytes);
            let mut starts = Vec::with_capacity(count);
            let mut frequencies = Vec::with_capacity(count);
            for (key, frequency) in entries {
                starts.push(u32::try_from(text.len()).expect("packed term offset overflow"));
                text.push_str(&key);
                frequencies.push(frequency);
            }
            TermData::Packed(PackedTerms {
                text: text.into_boxed_str(),
                starts: starts.into_boxed_slice(),
                frequencies: frequencies.into_boxed_slice(),
            })
        } else {
            TermData::Tree(
                entries
                    .into_iter()
                    .map(|(mut key, frequency)| {
                        super::compact_string(&mut key);
                        (key, frequency)
                    })
                    .collect(),
            )
        };
        Self {
            terms: Some(Box::new(data)),
            positions: None,
        }
    }

    /// [`Self::from_sorted`] from `count` borrowed entries whose keys total `bytes`, without
    /// an owned string per key unless the table cannot be packed.
    pub(crate) fn from_sorted_refs<'a>(
        count: usize,
        bytes: usize,
        entries: impl Iterator<Item = (&'a str, u32)>,
    ) -> Self {
        if count == 0 {
            return Self::default();
        }
        if !can_pack(count, bytes) {
            return Self::from_sorted(entries.map(|(key, f)| (key.to_owned(), f)).collect());
        }
        let mut text = String::with_capacity(bytes);
        let mut starts = Vec::with_capacity(count);
        let mut frequencies = Vec::with_capacity(count);
        for (key, frequency) in entries {
            debug_assert!(text.is_empty() || &text[*starts.last().unwrap() as usize..] < key);
            starts.push(u32::try_from(text.len()).expect("packed term offset overflow"));
            text.push_str(key);
            frequencies.push(frequency);
        }
        debug_assert_eq!((text.len(), frequencies.len()), (bytes, count));
        Self {
            terms: Some(Box::new(TermData::Packed(PackedTerms {
                text: text.into_boxed_str(),
                starts: starts.into_boxed_slice(),
                frequencies: frequencies.into_boxed_slice(),
            }))),
            positions: None,
        }
    }
}

pub(crate) struct TermIter<'a>(TermIterData<'a>);

enum TermIterData<'a> {
    Empty,
    Tree(btree_map::Iter<'a, String, u32>),
    Packed { table: &'a PackedTerms, next: usize },
    Shared { table: &'a SharedTerms, next: usize },
}

impl<'a> Iterator for TermIter<'a> {
    type Item = (&'a str, &'a u32);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            TermIterData::Empty => None,
            TermIterData::Tree(iter) => iter.next().map(|(key, value)| (key.as_str(), value)),
            TermIterData::Shared { table, next } => {
                let frequency = table.frequencies.get(*next)?;
                let key = table.key(*next);
                *next += 1;
                Some((key, frequency))
            }
            TermIterData::Packed { table, next } => {
                let frequency = table.frequencies.get(*next)?;
                let key = table.key(*next);
                *next += 1;
                Some((key, frequency))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = match &self.0 {
            TermIterData::Empty => 0,
            TermIterData::Tree(iter) => iter.len(),
            TermIterData::Packed { table, next } => table.frequencies.len() - *next,
            TermIterData::Shared { table, next } => table.frequencies.len() - *next,
        };
        (count, Some(count))
    }
}

impl ExactSizeIterator for TermIter<'_> {}

impl<'a> IntoIterator for &'a StoredTerms {
    type Item = (&'a str, &'a u32);
    type IntoIter = TermIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        TermIter(match self.terms.as_deref() {
            None => TermIterData::Empty,
            Some(TermData::Tree(table)) => TermIterData::Tree(table.iter()),
            Some(TermData::Packed(table)) => TermIterData::Packed { table, next: 0 },
            Some(TermData::Shared(table)) => TermIterData::Shared { table, next: 0 },
        })
    }
}

impl Serialize for StoredTerms {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map =
            serializer.serialize_map(Some(self.len() + usize::from(self.positions.is_some())))?;
        if let Some(positions) = &self.positions {
            map.serialize_entry("$hms_positions", positions)?;
        }
        for (key, frequency) in self {
            map.serialize_entry(key, frequency)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for StoredTerms {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TermsVisitor;
        impl<'de> serde::de::Visitor<'de> for TermsVisitor {
            type Value = StoredTerms;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a map")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut access: A,
            ) -> Result<StoredTerms, A::Error> {
                // Stored tables are written in key order, so the common case is a straight
                // append; any other order (or a repeated key) takes the map's insert
                // semantics, where the last value for a key wins.
                let mut entries: Vec<(String, u32)> =
                    Vec::with_capacity(access.size_hint().unwrap_or(0).min(1 << 16));
                let mut ascending = true;
                let mut positions = None;
                while let Some(key) = access.next_key::<String>()? {
                    if key == "$hms_positions" {
                        if positions.is_some() {
                            return Err(serde::de::Error::custom("duplicate positional payload"));
                        }
                        positions = Some(access.next_value::<Arc<super::positions::Positions>>()?);
                        continue;
                    }
                    let frequency = access.next_value::<u32>()?;
                    ascending &= entries.last().is_none_or(|(last, _)| *last < key);
                    entries.push((key, frequency));
                }
                let mut table = if ascending {
                    StoredTerms::from_sorted(entries)
                } else {
                    let mut map = BTreeMap::new();
                    for (key, frequency) in entries {
                        map.insert(key, frequency);
                    }
                    StoredTerms::from(map)
                };
                if let Some(positions) = positions {
                    positions
                        .validate(&table)
                        .map_err(serde::de::Error::custom)?;
                    table.positions = Some(positions);
                }
                Ok(table)
            }
        }
        deserializer.deserialize_map(TermsVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The id-based lookup finds exactly the words the string search finds, over subsets
    /// with clustered, sparse and dense ids and every width.
    #[test]
    fn id_lookup_matches_string_lookup() {
        let words: Vec<String> = (0..700).map(|i| format!("w{i:04}")).collect();
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let dictionary = Arc::new(SharedDictionary::new(&refs));
        for seed in 0..60usize {
            let ids: Vec<u32> = (0..700u32)
                .filter(|&i| match seed % 4 {
                    0 => (i as usize * 31 + seed).is_multiple_of(seed + 2),
                    1 => !(3..=690).contains(&i) || (i as usize + seed).is_multiple_of(97),
                    2 => (i as usize / 50 + seed).is_multiple_of(3),
                    _ => i as usize == seed * 11 % 700,
                })
                .collect();
            if ids.is_empty() {
                continue;
            }
            let frequencies = ids.iter().map(|&i| i + 1).collect();
            let terms = StoredTerms::from_dictionary(Arc::clone(&dictionary), &ids, frequencies);
            for word in words
                .iter()
                .map(String::as_str)
                .chain(["a", "w9999", "w0350x"])
            {
                // A cache belongs to one word; the second lookup reuses its id.
                let mut cache = WordId::default();
                assert_eq!(
                    terms.index_cached(word, &mut cache),
                    terms.index(word),
                    "{word}"
                );
                assert_eq!(
                    terms.index_cached(word, &mut cache),
                    terms.index(word),
                    "{word}"
                );
            }
        }
    }

    fn parity(map: BTreeMap<String, u32>) {
        let expected = serde_json::to_vec(&map).unwrap();
        let table = StoredTerms::from(map.clone());
        table.assert_valid_storage();
        assert_eq!(serde_json::to_vec(&table).unwrap(), expected);
        assert_eq!(serde_json::to_vec(&table.clone()).unwrap(), expected);
        assert_eq!(
            table.keys().collect::<Vec<_>>(),
            map.keys().map(String::as_str).collect::<Vec<_>>()
        );
        for (key, value) in &map {
            assert_eq!(table.get(key), Some(value));
            assert!(table.contains_key(key));
        }
        for key in ["absent:outside", "\0absent", "µ-absent"] {
            assert_eq!(table.get(key), map.get(key));
            assert_eq!(table.contains_key(key), map.contains_key(key));
        }
        let old = bincode::serde::encode_to_vec(&map, bincode::config::standard()).unwrap();
        assert_eq!(
            bincode::serde::encode_to_vec(&table, bincode::config::standard()).unwrap(),
            old
        );
        let (decoded, consumed): (StoredTerms, usize) =
            bincode::serde::decode_from_slice(&old, bincode::config::standard()).unwrap();
        decoded.assert_valid_storage();
        assert_eq!(consumed, old.len());
        assert_eq!(serde_json::to_vec(&decoded).unwrap(), expected);
    }

    #[test]
    fn offset_limit_and_budget_overflow_choose_the_full_range_fallback() {
        let allowance = std::mem::size_of::<(String, u32)>() - 2 * std::mem::size_of::<u32>();
        let limit = usize::try_from(u32::MAX).unwrap();
        for bytes in [limit - 1, limit] {
            let count = bytes.div_ceil(allowance);
            assert_eq!(can_pack(count, bytes), usize::BITS > u32::BITS);
            assert!(!can_pack(count - 1, bytes));
        }
        if let Some(bytes) = limit.checked_add(1) {
            assert!(!can_pack(bytes.div_ceil(allowance), bytes));
        }
        assert!(!can_pack(usize::MAX, 1));
        assert!(!can_pack(0, 1));
    }

    #[test]
    fn packing_boundaries_preserve_legacy_bytes_lookups_and_clones() {
        parity(BTreeMap::new());
        let allowance = std::mem::size_of::<(String, u32)>()
            - std::mem::size_of::<u32>()
            - std::mem::size_of::<u32>();
        for count in [1usize, 2, 10, 11, 12, 127, 128, 129, 4095, 4096, 4097] {
            for delta in [-1i64, 0, 1] {
                let target = (count * allowance) as i64 + delta;
                let mut map = BTreeMap::new();
                for i in 0..count {
                    map.insert(
                        format!("k{i:x}"),
                        match i % 3 {
                            0 => 0,
                            1 => u32::MAX - 1,
                            _ => u32::MAX,
                        },
                    );
                }
                let existing = map.keys().map(String::len).sum::<usize>();
                let (mut key, frequency) = map.pop_last().unwrap();
                key.push_str(&"x".repeat(target as usize - existing));
                map.insert(key, frequency);
                let table = StoredTerms::from(map.clone());
                assert_eq!(
                    matches!(table.terms.as_deref(), Some(TermData::Packed(_))),
                    delta <= 0
                );
                parity(map);
            }
        }
        parity(
            [
                ("".into(), 0),
                ("a\0é".into(), u32::MAX),
                ("µ😀".into(), 1),
                ("文書".into(), u32::MAX - 1),
            ]
            .into(),
        );
    }

    #[test]
    fn large_keys_retain_the_legacy_map_path_and_complete_bytes() {
        for length in [65535usize, 65536, 65537, 2 * 1024 * 1024] {
            let map: BTreeMap<_, _> = [("x".repeat(length), u32::MAX)].into();
            let table = StoredTerms::from(map.clone());
            assert!(matches!(table.terms.as_deref(), Some(TermData::Tree(_))));
            parity(map);
        }
    }

    #[test]
    fn legacy_duplicate_keys_frequency_limits_and_errors_stay_exact() {
        for input in [
            r#"{"z":1,"a":0,"z":4294967295,"µ":2}"#,
            r#"{"k":4294967294}"#,
            r#"{"k":4294967295}"#,
            r#"{"k":4294967296}"#,
            r#"{"k":-1}"#,
            r#"{"k":1.5}"#,
            r#"{"k":null}"#,
            r#"{"k":[]}"#,
            "[]",
            "null",
            r#"{"\u0000":0}"#,
            r#"{"k":-1,"k":1}"#,
        ] {
            let old = serde_json::from_str::<BTreeMap<String, u32>>(input);
            let new = serde_json::from_str::<StoredTerms>(input);
            match (old, new) {
                (Ok(old), Ok(new)) => {
                    new.assert_valid_storage();
                    assert_eq!(
                        serde_json::to_vec(&old).unwrap(),
                        serde_json::to_vec(&new).unwrap()
                    );
                }
                (Err(old), Err(new)) => assert_eq!(old.to_string(), new.to_string()),
                _ => panic!("legacy term parsing changed"),
            }
        }
    }

    /// Deserialization matches a `BTreeMap` for ordered, unordered and repeated keys.
    #[test]
    fn deserialize_matches_btreemap_for_any_key_order() {
        for json in [
            r#"{"a":1,"b":2,"c":3}"#,
            r#"{"c":3,"a":1,"b":2}"#,
            r#"{"a":1,"a":5,"b":2}"#,
            r#"{}"#,
            r#"{"µ":1,"z":2}"#,
        ] {
            let tree: BTreeMap<String, u32> = serde_json::from_str(json).unwrap();
            let table: StoredTerms = serde_json::from_str(json).unwrap();
            table.assert_valid_storage();
            assert_eq!(
                serde_json::to_vec(&table).unwrap(),
                serde_json::to_vec(&tree).unwrap(),
                "{json}"
            );
        }
    }
}

#[cfg(test)]
mod shared_tests {
    use super::*;

    #[test]
    fn compact_frequencies_keep_references_and_integer_boundaries() {
        for maximum in [0, 254, 255, 256, 65_534, 65_535, 65_536, u32::MAX] {
            let expected = vec![0, maximum];
            let values = SharedFrequencies::from(expected.clone());
            assert_eq!(values.len(), expected.len());
            for (i, frequency) in expected.iter().enumerate() {
                assert_eq!(values.get(i), Some(frequency));
            }
            assert_eq!(values.get(expected.len()), None);
            assert_eq!(matches!(values, SharedFrequencies::Byte(_)), maximum <= 255);
            assert_eq!(
                matches!(values, SharedFrequencies::Word(_)),
                maximum > 65_535
            );
        }
    }

    #[test]
    fn shared_dictionary_ids_preserve_keys_frequencies_and_legacy_json() {
        for count in [1, 127, 128, 129, 255, 256, 257] {
            let names: Vec<String> = (0..count).map(|i| format!("term{i:05}東京")).collect();
            let dictionary = Arc::new(SharedDictionary::new(
                &names.iter().map(String::as_str).collect::<Vec<_>>(),
            ));
            let ids: Vec<u32> = (0..count).step_by(2).map(|i| i as u32).collect();
            let frequencies: Vec<u32> = ids
                .iter()
                .map(|&i| if i == 0 { 0 } else { u32::MAX - i })
                .collect();
            let expected: BTreeMap<_, _> = ids
                .iter()
                .zip(&frequencies)
                .map(|(&i, &f)| (names[i as usize].clone(), f))
                .collect();
            let table = StoredTerms::from_dictionary(Arc::clone(&dictionary), &ids, frequencies);
            table.assert_valid_storage();
            assert_eq!(
                serde_json::to_vec(&table).unwrap(),
                serde_json::to_vec(&expected).unwrap()
            );
            for (index, (key, &frequency)) in expected.iter().enumerate() {
                assert_eq!(table.get(key), Some(&frequency));
                assert_eq!(table.index(key), Some(index));
            }
            let held = table.clone();
            drop(table);
            drop(dictionary);
            assert_eq!(
                serde_json::to_vec(&held).unwrap(),
                serde_json::to_vec(&expected).unwrap()
            );
        }
    }
}
