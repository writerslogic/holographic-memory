// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

pub mod admin;
pub(crate) mod admission;
pub mod agency;
pub(crate) mod durable_file;
pub(crate) mod schema;
// Experimental VSA research modules. Reachable only from the `src/bin/*`
// experiment harnesses (or fully orphaned) and not used by `HmsCore`. Gated
// behind the `experimental` feature so they are not part of the crate's
// published public API. See CONTRIBUTING for the codebook-first roadmap.
#[cfg(feature = "experimental")]
pub mod algebra;
pub(crate) mod atom_memory;
pub(crate) mod audit;
#[cfg(feature = "experimental")]
pub mod block_codes;
#[cfg(feature = "experimental")]
pub mod bloom_memory;
#[cfg(feature = "experimental")]
pub mod bundle_subset;
#[cfg(feature = "experimental")]
pub mod cls_memory;
pub mod cognition;
#[cfg(feature = "experimental")]
pub mod compose;
#[cfg(feature = "experimental")]
pub mod counting_bloom_memory;
// Living connection-graph: plastic, event-sourced holographic relation store.
// Experimental and opt-in; does not touch the sparse-binary core.
pub(crate) mod composite_memory;
pub(crate) mod config;
#[cfg(feature = "experimental")]
pub mod connection_graph;
pub(crate) mod decompose;
// Phasor relational memory: rotation-typed relations over a quantized-phase
// histogram field -- relation algebra + retrieval the sparse core cannot do.
pub(crate) mod diffusion;
pub mod documents;
pub mod encoding;
pub mod engine;
pub mod entangled;
pub(crate) mod error;
pub(crate) mod graph;
pub mod hardware;
pub mod hopfield;
pub(crate) mod idf;
pub(crate) mod index;
pub(crate) mod indexed_memory;
pub mod intersection;
pub(crate) mod ivf;
#[cfg(feature = "security")]
pub mod mask;
pub(crate) mod nsg;
#[cfg(feature = "experimental")]
pub mod phase_graph;
pub mod phase_hvec;
pub mod phase_resonator;
pub(crate) mod posting;
#[cfg(feature = "provenance")]
pub mod provenance;
pub(crate) mod role;
pub(crate) mod rules;
pub(crate) mod security;
#[cfg(target_arch = "x86_64")]
pub(crate) mod simd_math;
#[cfg(feature = "experimental")]
pub mod sparse_autoencoder;
pub(crate) mod storage;
pub(crate) mod store_lock;
#[cfg(feature = "experimental")]
pub mod ternary;
pub(crate) mod text;
pub(crate) mod tombstone;
pub(crate) mod triple_store;
pub(crate) mod types;

pub use config::HmsConfig;
pub use engine::HmsCore;
pub use phase_hvec::PhaseHVec;
pub use phase_resonator::{
    phase_resonator_factorize, FactorResult, PhaseResonator, ResonatorConfig,
};
pub use schema::EmbeddingSpace;
