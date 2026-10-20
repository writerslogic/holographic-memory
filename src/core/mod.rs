// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#[cfg(not(target_arch = "wasm32"))]
pub mod admin;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod admission;
#[cfg(not(target_arch = "wasm32"))]
pub mod agency;
pub(crate) mod durable_file;
pub(crate) mod schema;
// Experimental VSA research modules. Reachable only from the `src/bin/*`
// experiment harnesses (or fully orphaned) and not used by `HmsCore`. Gated
// behind the `experimental` feature so they are not part of the crate's
// published public API. See CONTRIBUTING for the codebook-first roadmap.
#[cfg(feature = "experimental")]
pub mod algebra;
#[cfg(feature = "experimental")]
pub mod algebraic_codebook;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod atom_memory;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod audit;
#[cfg(feature = "experimental")]
pub mod block_codes;
#[cfg(feature = "experimental")]
pub mod bloom_memory;
#[cfg(feature = "experimental")]
pub mod bundle_subset;
#[cfg(feature = "experimental")]
pub mod cls_memory;
#[cfg(not(target_arch = "wasm32"))]
pub mod cognition;
#[cfg(feature = "experimental")]
pub mod compose;
#[cfg(feature = "experimental")]
pub mod counting_bloom_memory;
// Living connection-graph: plastic, event-sourced holographic relation store.
// Experimental and opt-in; does not touch the sparse-binary core.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod composite_memory;
pub(crate) mod config;
#[cfg(feature = "experimental")]
pub mod connection_graph;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod decompose;
// Phasor relational memory: rotation-typed relations over a quantized-phase
// histogram field -- relation algebra + retrieval the sparse core cannot do.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod diffusion;
pub mod documents;
pub mod encoding;
#[cfg(not(target_arch = "wasm32"))]
pub mod engine;
pub mod entangled;
pub(crate) mod error;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod graph;
pub mod hardware;
pub mod hopfield;
pub(crate) mod idf;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod index;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod indexed_memory;
pub mod intersection;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod ivf;
#[cfg(feature = "security")]
pub mod mask;
pub mod memory_api;
#[cfg(feature = "local-models")]
pub mod models;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod nsg;
#[cfg(feature = "experimental")]
pub mod operand_retriever;
#[cfg(feature = "experimental")]
pub mod phase_graph;
pub mod phase_hvec;
pub mod phase_resonator;
pub(crate) mod posting;
#[cfg(feature = "private-search")]
pub mod private_query;
#[cfg(feature = "provenance")]
pub mod provenance;
#[cfg(not(target_arch = "wasm32"))]
pub mod qgraph;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod role;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod rules;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod security;
#[cfg(feature = "security")]
pub use security::scrub_stack;
#[cfg(target_arch = "x86_64")]
pub(crate) mod simd_math;
pub mod sparse;
#[cfg(feature = "experimental")]
pub mod sparse_autoencoder;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod storage;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod store_lock;
#[cfg(feature = "experimental")]
pub mod ternary;
pub(crate) mod text;
pub(crate) mod tombstone;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod triple_store;
pub(crate) mod types;

pub use config::HmsConfig;
#[cfg(not(target_arch = "wasm32"))]
pub use engine::HmsCore;
pub use phase_hvec::PhaseHVec;
pub use phase_resonator::{
    phase_resonator_factorize, FactorResult, PhaseResonator, ResonatorConfig,
};
pub use schema::EmbeddingSpace;
