// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cmp::Ordering;
use std::collections::{btree_map, BTreeMap};

#[derive(Clone, Default)]
pub(crate) struct StoredTerms(Option<Box<TermData>>);

#[derive(Clone)]
enum TermData {
    Packed(PackedTerms),
    Tree(BTreeMap<String, u32>),
}

#[derive(Clone)]
struct PackedTerms {
    text: Box<str>,
    starts: Box<[u32]>,
    frequencies: Box<[u32]>,
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
    fn len(&self) -> usize {
        match self.0.as_deref() {
            None => 0,
            Some(TermData::Packed(table)) => table.frequencies.len(),
            Some(TermData::Tree(table)) => table.len(),
        }
    }

    pub(crate) fn get(&self, key: &str) -> Option<&u32> {
        match self.0.as_deref() {
            None => None,
            Some(TermData::Packed(table)) => table.get(key),
            Some(TermData::Tree(table)) => table.get(key),
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
        match self.0.as_deref() {
            None => assert_eq!(self.len(), 0),
            Some(TermData::Tree(table)) => {
                assert!(!table.is_empty());
                assert!(table.keys().all(|key| key.capacity() == key.len()));
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

fn can_pack(count: usize, bytes: usize) -> bool {
    let allowance = std::mem::size_of::<(String, u32)>() - 2 * std::mem::size_of::<u32>();
    u32::try_from(bytes).is_ok()
        && count
            .checked_mul(allowance)
            .is_some_and(|limit| bytes <= limit)
}

impl From<BTreeMap<String, u32>> for StoredTerms {
    fn from(map: BTreeMap<String, u32>) -> Self {
        if map.is_empty() {
            return Self::default();
        }
        let count = map.len();
        let bytes = map
            .keys()
            .try_fold(0usize, |total, key| total.checked_add(key.len()))
            .expect("term byte length overflow");
        let pack = can_pack(count, bytes);
        let data = if pack {
            let mut text = String::with_capacity(bytes);
            let mut starts = Vec::with_capacity(count);
            let mut frequencies = Vec::with_capacity(count);
            for (key, frequency) in map {
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
                map.into_iter()
                    .map(|(mut key, frequency)| {
                        super::compact_string(&mut key);
                        (key, frequency)
                    })
                    .collect(),
            )
        };
        Self(Some(Box::new(data)))
    }
}

pub(crate) struct TermIter<'a>(TermIterData<'a>);

enum TermIterData<'a> {
    Empty,
    Tree(btree_map::Iter<'a, String, u32>),
    Packed { table: &'a PackedTerms, next: usize },
}

impl<'a> Iterator for TermIter<'a> {
    type Item = (&'a str, &'a u32);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            TermIterData::Empty => None,
            TermIterData::Tree(iter) => iter.next().map(|(key, value)| (key.as_str(), value)),
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
        };
        (count, Some(count))
    }
}

impl ExactSizeIterator for TermIter<'_> {}

impl<'a> IntoIterator for &'a StoredTerms {
    type Item = (&'a str, &'a u32);
    type IntoIter = TermIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        TermIter(match self.0.as_deref() {
            None => TermIterData::Empty,
            Some(TermData::Tree(table)) => TermIterData::Tree(table.iter()),
            Some(TermData::Packed(table)) => TermIterData::Packed { table, next: 0 },
        })
    }
}

impl Serialize for StoredTerms {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (key, frequency) in self {
            map.serialize_entry(key, frequency)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for StoredTerms {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        BTreeMap::<String, u32>::deserialize(deserializer).map(Self::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    matches!(table.0.as_deref(), Some(TermData::Packed(_))),
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
            assert!(matches!(table.0.as_deref(), Some(TermData::Tree(_))));
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
}
