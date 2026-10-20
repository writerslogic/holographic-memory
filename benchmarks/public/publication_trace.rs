// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use parking_lot::Mutex;
use serde::Serialize;
use std::{sync::OnceLock, time::Instant};

struct Collector {
    origin: Instant,
    rows: Mutex<Vec<Row>>,
}

static COLLECTOR: OnceLock<Collector> = OnceLock::new();

#[derive(Serialize)]
pub struct Row {
    pub kind: &'static str,
    pub count: usize,
    pub wait_ns: u64,
    pub acquired_ns: u64,
    pub append_start_ns: u64,
    pub append_end_ns: u64,
    pub exclusion_start_ns: u64,
    pub publication_end_ns: u64,
    pub released_ns: u64,
}

pub fn enable(origin: Instant) {
    assert!(COLLECTOR
        .set(Collector {
            origin,
            rows: Mutex::new(Vec::with_capacity(1024)),
        })
        .is_ok());
}

pub fn now() -> u64 {
    COLLECTOR
        .get()
        .map_or(0, |collector| collector.origin.elapsed().as_nanos() as u64)
}

pub fn record(row: Row) {
    if let Some(collector) = COLLECTOR.get() {
        let mut rows = collector.rows.lock();
        assert!(rows.len() < 1024, "diagnostic trace capacity exhausted");
        rows.push(row);
    }
}

pub fn take() -> Vec<Row> {
    std::mem::take(&mut *COLLECTOR.get().expect("collector disabled").rows.lock())
}
