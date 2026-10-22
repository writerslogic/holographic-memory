// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::binary::{put_varint, Reader};
use super::packed_terms::StoredTerms;
use super::{stemmer, MAX_DOCUMENT_BYTES};

pub(super) const VERSION: u8 = 1;
pub(super) const DOCUMENT_VERSION: u8 = 2;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Positions {
    pub(super) version: u8,
    pub(super) offsets: Box<[u32]>,
    pub(super) data: Box<[u8]>,
}

pub(super) fn tokens(text: &str, stem: bool) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|word| {
            let lower = word.to_lowercase();
            if stem {
                stemmer().stem(&lower).into_owned()
            } else {
                lower
            }
        })
        .collect()
}

impl Positions {
    pub(super) fn from_text(
        text: &str,
        stem: bool,
        terms: &StoredTerms,
        whole_document: bool,
    ) -> Self {
        let mut table = BTreeMap::<String, Vec<u32>>::new();
        for (position, token) in tokens(text, stem).into_iter().enumerate() {
            table.entry(token).or_default().push(position as u32);
        }
        let mut offsets = Vec::with_capacity(terms.keys().len() + 1);
        let mut data = Vec::new();
        offsets.push(0);
        for word in terms.keys() {
            let mut previous = 0;
            for &position in &table[word] {
                put_varint(&mut data, u64::from(position - previous));
                previous = position;
            }
            offsets.push(u32::try_from(data.len()).expect("positions exceed u32"));
        }
        Self {
            version: if whole_document {
                DOCUMENT_VERSION
            } else {
                VERSION
            },
            offsets: offsets.into_boxed_slice(),
            data: data.into_boxed_slice(),
        }
    }

    pub(super) fn validate(&self, terms: &StoredTerms) -> Result<()> {
        ensure!(
            matches!(self.version, VERSION | DOCUMENT_VERSION),
            "positional postings version {} is newer than supported",
            self.version
        );
        ensure!(
            self.offsets.len() == terms.keys().len() + 1
                && self.offsets.first() == Some(&0)
                && self.offsets.last().copied().map(|v| v as usize) == Some(self.data.len())
                && self.offsets.windows(2).all(|p| p[0] < p[1])
                && self.data.len() <= MAX_DOCUMENT_BYTES,
            "invalid positional posting offsets or length"
        );
        let mut count = 0usize;
        let mut maximum = None;
        for (index, (_, &frequency)) in terms.into_iter().enumerate() {
            let mut reader = Reader::new(self.bytes(index));
            let mut previous = None::<u32>;
            let mut term_count = 0;
            while !reader.is_empty() {
                let gap = u32::try_from(reader.varint()?).context("position exceeds u32")?;
                ensure!(
                    previous.is_none() || gap > 0,
                    "positions are not strictly ascending"
                );
                let position = previous
                    .unwrap_or(0)
                    .checked_add(gap)
                    .context("position overflows")?;
                ensure!(
                    (position as usize) < MAX_DOCUMENT_BYTES,
                    "position exceeds document limit"
                );
                previous = Some(position);
                maximum = Some(maximum.map_or(position, |m: u32| m.max(position)));
                term_count += 1;
            }
            ensure!(
                term_count > 0 && term_count <= frequency as usize,
                "position count exceeds term frequency"
            );
            count += term_count;
        }
        ensure!(
            count <= MAX_DOCUMENT_BYTES && maximum.map_or(0, |p| p as usize + 1) == count,
            "positions do not cover the token sequence"
        );
        let mut seen = vec![0u64; count.div_ceil(64)];
        for index in 0..terms.keys().len() {
            for position in self.iter(index) {
                let position = position as usize;
                ensure!(position < count, "position exceeds token count");
                let bit = 1 << (position % 64);
                ensure!(
                    seen[position / 64] & bit == 0,
                    "token position occurs in multiple terms"
                );
                seen[position / 64] |= bit;
            }
        }
        Ok(())
    }

    fn bytes(&self, index: usize) -> &[u8] {
        &self.data[self.offsets[index] as usize..self.offsets[index + 1] as usize]
    }

    pub(super) fn iter(&self, index: usize) -> impl Iterator<Item = u32> + '_ {
        let mut reader = Reader::new(self.bytes(index));
        let mut position = 0u32;
        std::iter::from_fn(move || {
            if reader.is_empty() {
                return None;
            }
            position += u32::try_from(reader.varint().expect("validated position"))
                .expect("validated u32 position");
            Some(position)
        })
    }
}
