// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default)]
pub struct HmsConfig {
    pub ivf: IVFConfig,
    pub nsg: NSGConfig,
    pub shard: ShardConfig,
    pub query: QueryConfig,
    pub concepts: ConceptsConfig,
    pub diffusion: DiffusionConfig,
    pub security: SecurityConfig,
    pub privacy: PrivacyConfig,
    pub meaning: MeaningConfig,
}

#[derive(Clone, Debug)]
pub struct MeaningConfig {
    pub enabled: bool,
    pub beta: f64,
    pub algebraic_max_fanout: usize,
    pub auto_decompose: bool,
    pub max_hop_depth: usize,
    pub max_rule_depth: usize,
    pub idf_clip_factor: f64,
}

impl Default for MeaningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            beta: 24.0,
            algebraic_max_fanout: 40,
            auto_decompose: false,
            max_hop_depth: 10,
            max_rule_depth: 10,
            idf_clip_factor: 3.0,
        }
    }
}

#[derive(Clone, Debug)]
#[derive(Default)]
pub struct SecurityConfig {
    pub signing_enabled: bool,
    pub key_path: Option<String>,
    pub encryption_enabled: bool,
    pub encryption_passphrase: Option<String>,
    pub audit_enabled: bool,
}

#[derive(Clone, Debug)]
pub struct PrivacyConfig {
    /// Enable epsilon-differential privacy in bundle operations.
    pub dp_enabled: bool,
    /// Privacy budget epsilon. Smaller = more private, noisier.
    /// Typical range: 0.1 (strong) to 10.0 (weak).
    pub epsilon: f64,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            dp_enabled: false,
            epsilon: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QueryConfig {
    pub component_similarity_threshold: f64,
    pub component_max_neighbors: u32,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            component_similarity_threshold: 0.05,
            component_max_neighbors: 20,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConceptsConfig {
    pub similarity_threshold: f64,
    pub min_cluster_size: usize,
}

impl Default for ConceptsConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.3,
            min_cluster_size: 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DiffusionConfig {
    pub steps: usize,
    pub sigma_max: f64,
    pub sigma_min: f64,
    pub step_size: f64,
    pub n_langevin: usize,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            steps: 10,
            sigma_max: 0.5,
            sigma_min: 0.01,
            step_size: 0.1,
            n_langevin: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShardConfig {
    pub enabled: bool,
    pub shard_count: usize,
    pub auto_threshold: usize,
    pub target_shard_size: usize,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            shard_count: 0,
            auto_threshold: 1_000_000,
            target_shard_size: 250_000,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NSGConfig {
    pub max_degree: usize,
    pub ef_construction: usize,
    pub auto_threshold: usize,
    pub seed: u64,
}

impl Default for NSGConfig {
    fn default() -> Self {
        Self {
            max_degree: 32,
            ef_construction: 128,
            auto_threshold: 10_000,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IVFConfig {
    /// Controls auto-training only. Manual `train_ivf()` works regardless.
    pub enabled: bool,
    pub n_clusters: usize,
    pub n_landmarks: usize,
    pub d_reduced: usize,
    pub n_probe: usize,
    pub auto_threshold: usize,
}

impl Default for IVFConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            n_clusters: 256,
            n_landmarks: 1024,
            d_reduced: 128,
            n_probe: 8,
            auto_threshold: 10_000,
        }
    }
}
