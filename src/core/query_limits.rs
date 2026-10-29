// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

/// Bounds for optional reasoning and concept queries. Exhaustion returns an error, never
/// a partial answer. Byte accounting includes retained inputs and intermediate frontiers.
#[derive(Clone, Copy, Debug)]
pub struct QueryLimits {
    pub max_work: usize,
    pub max_frontier: usize,
    pub max_bytes: usize,
}

impl Default for QueryLimits {
    fn default() -> Self {
        Self {
            max_work: 10_000_000,
            max_frontier: 16_384,
            max_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("query {resource} limit {limit} exceeded")]
pub struct QueryLimitExceeded {
    pub resource: &'static str,
    pub limit: usize,
}

pub(crate) struct QueryBudget {
    limits: QueryLimits,
    work: usize,
    bytes: usize,
}

impl QueryBudget {
    pub(crate) fn new(limits: QueryLimits) -> Self {
        Self {
            limits,
            work: 0,
            bytes: 0,
        }
    }

    pub(crate) fn work(&mut self, amount: usize) -> Result<(), QueryLimitExceeded> {
        Self::charge(&mut self.work, amount, self.limits.max_work, "work")
    }

    pub(crate) fn reserve(&mut self, amount: usize) -> Result<(), QueryLimitExceeded> {
        Self::charge(&mut self.bytes, amount, self.limits.max_bytes, "bytes")
    }

    pub(crate) fn release(&mut self, amount: usize) {
        self.bytes = self
            .bytes
            .checked_sub(amount)
            .expect("query budget released unreserved bytes");
    }

    pub(crate) fn frontier(&self, count: usize) -> Result<(), QueryLimitExceeded> {
        if count > self.limits.max_frontier {
            Err(QueryLimitExceeded {
                resource: "frontier",
                limit: self.limits.max_frontier,
            })
        } else {
            Ok(())
        }
    }

    fn charge(
        used: &mut usize,
        amount: usize,
        limit: usize,
        resource: &'static str,
    ) -> Result<(), QueryLimitExceeded> {
        *used = used
            .checked_add(amount)
            .filter(|&total| total <= limit)
            .ok_or(QueryLimitExceeded { resource, limit })?;
        Ok(())
    }
}
