// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashSet};

#[derive(Default)]
pub(crate) struct MemoryCensus {
    pub(super) bytes: BTreeMap<&'static str, usize>,
    pub(super) counts: BTreeMap<&'static str, usize>,
    pub(super) histogram: BTreeMap<(usize, usize, usize), usize>,
    pub(super) dictionaries: HashSet<usize>,
}

impl MemoryCensus {
    pub(super) fn bytes(&mut self, name: &'static str, bytes: usize) {
        *self.bytes.entry(name).or_default() += bytes;
    }

    pub(super) fn count(&mut self, name: &'static str, count: usize) {
        *self.counts.entry(name).or_default() += count;
    }

    pub(crate) fn report(&self) -> serde_json::Value {
        let histogram: Vec<_> = self
            .histogram
            .iter()
            .map(|(&(postings, length, capacity), &lists)| {
                serde_json::json!({
                    "posting_count": postings,
                    "block_length": length,
                    "block_capacity": capacity,
                    "lists": lists,
                })
            })
            .collect();
        serde_json::json!({
            "bytes": self.bytes,
            "counts": self.counts,
            "posting_list_histogram": histogram,
            "accounting": "Requested capacities, excluding allocator rounding, tree node overhead, and engine state outside DocumentStore. The *.unused entries are subsets of *.capacity; *.payload entries are lower bounds. Shared dictionaries are counted once. Census scratch is excluded.",
        })
    }
}
