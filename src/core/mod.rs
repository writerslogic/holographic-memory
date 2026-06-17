// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod admission;
pub(crate) mod atom_memory;
pub(crate) mod audit;
pub(crate) mod composite_memory;
pub(crate) mod config;
pub(crate) mod decompose;
pub(crate) mod graph;
pub(crate) mod idf;
pub(crate) mod indexed_memory;
pub(crate) mod posting;
pub(crate) mod role;
pub(crate) mod rules;
pub(crate) mod triple_store;
pub(crate) mod tombstone;
pub(crate) mod diffusion;
pub(crate) mod encoding;
pub mod engine;
pub mod entangled;
pub(crate) mod error;
pub(crate) mod index;
pub mod intersection;
pub(crate) mod ivf;
pub(crate) mod nsg;
pub(crate) mod security;
pub(crate) mod storage;
pub(crate) mod text;
pub(crate) mod types;

pub use config::HmsConfig;
pub use engine::HmsCore;
