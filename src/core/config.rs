// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

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
    pub provenance: ProvenanceConfig,
    pub privacy: PrivacyConfig,
    pub meaning: MeaningConfig,
    pub cognition: CognitionConfig,
    pub hopfield: super::hopfield::HopfieldConfig,
    pub embedding_space: Option<super::schema::EmbeddingSpace>,
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

#[derive(Clone, Debug, Default)]
pub struct SecurityConfig {
    pub signing_enabled: bool,
    pub key_path: Option<String>,
    pub encryption_enabled: bool,
    pub encryption_passphrase: Option<String>,
    /// Name of an environment variable containing the encryption passphrase.
    /// Prefer this over embedding a secret in application configuration.
    pub encryption_passphrase_env: Option<String>,
    pub audit_enabled: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ProvenanceConfig {
    pub enabled: bool,
    pub key_path: Option<String>,
    pub auto_sign: bool,
    #[cfg(feature = "provenance-scitt")]
    pub scitt_endpoint: Option<String>,
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

#[derive(Clone, Debug)]
pub struct CognitionConfig {
    pub enabled: bool,
    pub interval_secs: u64,
    pub min_pattern_freq: usize,
    pub min_abstraction_members: usize,
    pub min_shared_relations: usize,
    pub min_peer_coverage: f64,
    pub hypothesis_beta: f64,
    pub min_hypothesis_confidence: f64,
    pub min_analogy_relations: usize,
    pub governor_duplicate_threshold: f64,
    pub governor_max_scan_size: usize,
    pub governor_forget_unreferenced: bool,
    pub refine_atoms: bool,
}

impl Default for CognitionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_secs: 60,
            min_pattern_freq: 3,
            min_abstraction_members: 3,
            min_shared_relations: 2,
            min_peer_coverage: 0.5,
            hypothesis_beta: 24.0,
            min_hypothesis_confidence: 0.3,
            min_analogy_relations: 2,
            governor_duplicate_threshold: 0.95,
            governor_max_scan_size: 1000,
            governor_forget_unreferenced: false,
            refine_atoms: false,
        }
    }
}

impl HmsConfig {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            cfg!(feature = "security")
                || !(self.security.signing_enabled || self.security.encryption_enabled),
            "signing/encryption requires a build with the security feature"
        );
        anyhow::ensure!(
            cfg!(feature = "provenance") || !self.provenance.enabled,
            "provenance requires a build with the provenance feature"
        );
        anyhow::ensure!(
            self.privacy.epsilon.is_finite() && self.privacy.epsilon > 0.0,
            "privacy epsilon must be finite and positive"
        );
        anyhow::ensure!(
            self.shard.target_shard_size > 0 && self.shard.shard_count <= 1024,
            "invalid shard configuration"
        );
        anyhow::ensure!(
            self.meaning.beta.is_finite() && self.meaning.beta > 0.0,
            "meaning beta must be finite and positive"
        );
        if let Some(space) = &self.embedding_space {
            anyhow::ensure!(!space.model.is_empty() && !space.revision.is_empty() && (1..=65536).contains(&space.dimensions)
                && space.metric == "cosine" && space.normalization == "l2",
                "embedding space requires model, revision, dimensions 1..=65536, normalization l2 and metric cosine");
        }
        Ok(())
    }
}
