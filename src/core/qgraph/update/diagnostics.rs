// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::State;

#[derive(Default)]
pub(super) struct Diagnostics {
    pub(super) enabled: bool,
    queries: AtomicU64,
    delta_searches: AtomicU64,
    delta_visits: AtomicU64,
    delta_candidates: AtomicU64,
}

impl Diagnostics {
    pub(super) fn from_env() -> Self {
        Self {
            enabled: std::env::var_os("HMS_STREAM_DIAGNOSTICS").is_some_and(|v| v == "1"),
            ..Self::default()
        }
    }

    pub(super) fn query(&self) {
        if self.enabled {
            self.queries.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn delta_search(&self, visits: usize, candidates: usize) {
        self.delta_searches.fetch_add(1, Ordering::Relaxed);
        self.delta_visits
            .fetch_add(visits as u64, Ordering::Relaxed);
        self.delta_candidates
            .fetch_add(candidates as u64, Ordering::Relaxed);
    }

    pub(super) fn snapshot(&self, state: &State) -> Option<BTreeMap<String, f64>> {
        if !self.enabled {
            return None;
        }
        let mut out = state.delta.topology();
        for (name, value) in [
            ("live", state.len()),
            ("base_slots", state.base.rows.ids.len()),
            ("base_dead", state.deleted),
            ("base_topology_available", 0),
            ("pending_ops", state.pending.len()),
        ] {
            out.insert(name.into(), value as f64);
        }
        for (name, counter) in [
            ("queries", &self.queries),
            ("delta_searches", &self.delta_searches),
            ("delta_visits", &self.delta_visits),
            ("delta_candidates", &self.delta_candidates),
        ] {
            out.insert(name.into(), counter.load(Ordering::Relaxed) as f64);
        }
        Some(out)
    }
}
