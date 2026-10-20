// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cognition layer: background discovery of implicit knowledge.
//!
//! All components operate with read-only access to meaning memory stores.
//! Discovered insights require explicit promotion before becoming stored facts.

pub mod abstraction;
pub mod analogy;
pub mod gaps;
pub mod governor;
pub mod hypothesis;
pub mod r#loop;
pub mod patterns;
pub mod refiner;
