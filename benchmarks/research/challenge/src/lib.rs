// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Research round 2, package `challenge`: answer-and-challenge retrieval with scoped
//! erase-units for agent memory, prototyped on the WARM memory API and kill-tested against
//! ranking baselines.
//!
//! # Mechanism
//!
//! The retrieval unit is `{primary_claim, supporting_evidence, strongest_defeater, status}`
//! with `status` either `Resolved` or `UnresolvedConflict` ([`contract`]). A query carries
//! `(entity, attribute)`, a scope, an `as_of` time and text. The support pass ranks the
//! family's current claims lexically within scope; the challenge pass runs four exact,
//! similarity-independent lookups on the same family (newer overlapping-scope claims,
//! explicit retractions, narrower still-valid exceptions, concurrent branches). When the
//! scope cannot decide, every branch is returned instead of a blended ranking.
//!
//! # Data layout
//!
//! Records live in the root crate's memory API as ordinary facts (subject = family,
//! validity window, explicit supersession, structured caller metadata; [`store`]). The
//! challenge index is an in-process arena of one contiguous block per family, rebuilt from
//! `Memory::get` records; a family is erased by dropping its block and forgetting its subject
//! in one memory API call.
//!
//! # Experiment
//!
//! [`synth`] generates 1,000 families x 10 observations with gold labels derived from the
//! construction; [`baselines`] implements global BM25, scope-filtered latest-first BM25 (two
//! variants), the scoped lookup without the challenge pass, and the challenge method behind
//! one answer shape; [`eval`] scores answer-and-evidence correctness at a 4-item budget,
//! stale-answer and missed-exception rates, p95 latency, the erase check and the PASS/FAIL
//! verdict. [`audit`] measures what fraction of a real query trace carries a usable key.
//!
//! # Invariants
//!
//! - Nothing here mutates the root crate; every root API used is listed in `RUN.md`.
//! - Every random choice comes from a fixed-seed SplitMix64 stream; runs are reproducible.
//! - Timing is single-caller-thread wall clock with warmups and medians over repeats.

#![forbid(unsafe_code)]

pub mod audit;
pub mod baselines;
pub mod contract;
pub mod eval;
pub mod store;
pub mod synth;
