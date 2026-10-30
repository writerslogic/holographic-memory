// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// Per-call allocations and admission shared by every caller of one embedder.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct EmbedLimits {
    #[serde(alias = "max_texts")]
    pub max_texts: usize,
    #[serde(alias = "max_input_bytes")]
    pub max_input_bytes: usize,
    #[serde(alias = "max_tokens")]
    pub max_tokens: usize,
    /// Includes both the flat inference output and returned row copies.
    #[serde(alias = "max_output_bytes")]
    pub max_output_bytes: usize,
    /// Calls above this limit fail immediately rather than queue inside the embedder.
    #[serde(alias = "max_concurrent_calls")]
    pub max_concurrent_calls: usize,
}

impl Default for EmbedLimits {
    fn default() -> Self {
        Self {
            max_texts: 4096,
            max_input_bytes: 64 * 1024 * 1024,
            max_tokens: 1_048_576,
            max_output_bytes: 64 * 1024 * 1024,
            max_concurrent_calls: 4,
        }
    }
}

impl EmbedLimits {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.max_texts > 0
                && self.max_input_bytes > 0
                && self.max_tokens > 0
                && self.max_output_bytes > 0
                && self.max_concurrent_calls > 0,
            "embedding limits must be positive"
        );
        Ok(())
    }

    pub(super) fn output_values(&self, count: usize, dimensions: usize) -> Result<usize> {
        ensure!(
            count <= self.max_texts,
            "embedding text count exceeds max_texts"
        );
        let values = count
            .checked_mul(dimensions)
            .context("embedding output size overflow")?;
        let bytes = values
            .checked_mul(2 * size_of::<f32>())
            .context("embedding output size overflow")?;
        ensure!(
            bytes <= self.max_output_bytes,
            "embedding output exceeds max_output_bytes"
        );
        Ok(values)
    }
}

pub(super) fn reserve(
    counter: &AtomicUsize,
    amount: usize,
    limit: usize,
    name: &str,
) -> Result<()> {
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current
            .checked_add(amount)
            .filter(|&next| next <= limit)
            .with_context(|| format!("embedding {name} limit exceeded"))?;
        match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return Ok(()),
            Err(actual) => current = actual,
        }
    }
}

pub(super) struct CallLease<'a>(&'a AtomicUsize);

impl<'a> CallLease<'a> {
    pub(super) fn acquire(active: &'a AtomicUsize, limit: usize) -> Result<Self> {
        reserve(active, 1, limit, "concurrent calls")?;
        Ok(Self(active))
    }
}

impl Drop for CallLease<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}
