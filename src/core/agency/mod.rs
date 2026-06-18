// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Agency layer: goal-directed behavior.
//!
//! Gives the system the ability to maintain goals, plan actions via
//! backward-chaining, generate questions from knowledge gaps, and
//! propose self-modifications that require explicit user approval.

pub mod goals;
pub mod planner;
pub mod questions;
pub mod self_modify;
