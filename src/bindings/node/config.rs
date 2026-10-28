// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use napi_derive::napi;

use crate::EmbeddingSpace;

#[napi(object)]
pub struct HmsConfigJs {
    pub embedding_model: Option<String>,
    pub embedding_revision: Option<String>,
    pub embedding_dimensions: Option<u32>,
    pub nsg_max_degree: Option<u32>,
    pub nsg_ef_construction: Option<u32>,
    pub nsg_auto_threshold: Option<u32>,
    pub ivf_enabled: Option<bool>,
    pub ivf_n_clusters: Option<u32>,
    pub ivf_n_landmarks: Option<u32>,
    pub ivf_d_reduced: Option<u32>,
    pub ivf_n_probe: Option<u32>,
    pub ivf_auto_threshold: Option<u32>,
    pub shard_enabled: Option<bool>,
    pub shard_count: Option<u32>,
    pub shard_auto_threshold: Option<u32>,
    pub shard_target_size: Option<u32>,
    pub component_similarity_threshold: Option<f64>,
    pub component_max_neighbors: Option<u32>,
    pub concept_similarity_threshold: Option<f64>,
    pub concept_min_cluster_size: Option<u32>,
    pub diffusion_steps: Option<u32>,
    pub diffusion_sigma_max: Option<f64>,
    pub diffusion_sigma_min: Option<f64>,
    pub diffusion_step_size: Option<f64>,
    pub diffusion_n_langevin: Option<u32>,
    pub signing_enabled: Option<bool>,
    pub signing_key_path: Option<String>,
    pub encryption_enabled: Option<bool>,
    pub encryption_passphrase: Option<String>,
    pub encryption_passphrase_env: Option<String>,
    pub audit_enabled: Option<bool>,
    pub dp_enabled: Option<bool>,
    pub dp_epsilon: Option<f64>,
    pub meaning_enabled: Option<bool>,
    pub meaning_beta: Option<f64>,
    pub meaning_max_fanout: Option<u32>,
    pub meaning_auto_decompose: Option<bool>,
    pub meaning_max_hop_depth: Option<u32>,
    pub cognition_enabled: Option<bool>,
    pub cognition_interval_secs: Option<u32>,
    pub cognition_min_pattern_freq: Option<u32>,
    pub cognition_min_abstraction_members: Option<u32>,
    pub cognition_min_hypothesis_confidence: Option<f64>,
}

impl HmsConfigJs {
    pub(super) fn into_config(self) -> crate::core::config::HmsConfig {
        use crate::core::config::*;
        let mut cfg = HmsConfig::default();
        if self.embedding_model.is_some()
            || self.embedding_revision.is_some()
            || self.embedding_dimensions.is_some()
        {
            cfg.embedding_space = Some(EmbeddingSpace {
                model: self.embedding_model.unwrap_or_default(),
                revision: self.embedding_revision.unwrap_or_default(),
                dimensions: self.embedding_dimensions.unwrap_or_default() as usize,
                normalization: "l2".into(),
                metric: "cosine".into(),
            });
        }
        if let Some(v) = self.nsg_max_degree {
            cfg.nsg.max_degree = v as usize;
        }
        if let Some(v) = self.nsg_ef_construction {
            cfg.nsg.ef_construction = v as usize;
        }
        if let Some(v) = self.nsg_auto_threshold {
            cfg.nsg.auto_threshold = v as usize;
        }
        if let Some(v) = self.ivf_enabled {
            cfg.ivf.enabled = v;
        }
        if let Some(v) = self.ivf_n_clusters {
            cfg.ivf.n_clusters = v as usize;
        }
        if let Some(v) = self.ivf_n_landmarks {
            cfg.ivf.n_landmarks = v as usize;
        }
        if let Some(v) = self.ivf_d_reduced {
            cfg.ivf.d_reduced = v as usize;
        }
        if let Some(v) = self.ivf_n_probe {
            cfg.ivf.n_probe = v as usize;
        }
        if let Some(v) = self.ivf_auto_threshold {
            cfg.ivf.auto_threshold = v as usize;
        }
        if let Some(v) = self.shard_enabled {
            cfg.shard.enabled = v;
        }
        if let Some(v) = self.shard_count {
            cfg.shard.shard_count = v as usize;
        }
        if let Some(v) = self.shard_auto_threshold {
            cfg.shard.auto_threshold = v as usize;
        }
        if let Some(v) = self.shard_target_size {
            cfg.shard.target_shard_size = v as usize;
        }
        if let Some(v) = self.component_similarity_threshold {
            cfg.query.component_similarity_threshold = v;
        }
        if let Some(v) = self.component_max_neighbors {
            cfg.query.component_max_neighbors = v;
        }
        if let Some(v) = self.concept_similarity_threshold {
            cfg.concepts.similarity_threshold = v;
        }
        if let Some(v) = self.concept_min_cluster_size {
            cfg.concepts.min_cluster_size = v as usize;
        }
        if let Some(v) = self.diffusion_steps {
            cfg.diffusion.steps = v as usize;
        }
        if let Some(v) = self.diffusion_sigma_max {
            cfg.diffusion.sigma_max = v;
        }
        if let Some(v) = self.diffusion_sigma_min {
            cfg.diffusion.sigma_min = v;
        }
        if let Some(v) = self.diffusion_step_size {
            cfg.diffusion.step_size = v;
        }
        if let Some(v) = self.diffusion_n_langevin {
            cfg.diffusion.n_langevin = v as usize;
        }
        if let Some(v) = self.signing_enabled {
            cfg.security.signing_enabled = v;
        }
        if let Some(v) = self.signing_key_path {
            cfg.security.key_path = Some(v);
        }
        if let Some(v) = self.encryption_enabled {
            cfg.security.encryption_enabled = v;
        }
        if let Some(v) = self.encryption_passphrase {
            cfg.security.encryption_passphrase = Some(v);
        }
        if let Some(v) = self.encryption_passphrase_env {
            cfg.security.encryption_passphrase_env = Some(v);
        }
        if let Some(v) = self.audit_enabled {
            cfg.security.audit_enabled = v;
        }
        if let Some(v) = self.dp_enabled {
            cfg.privacy.dp_enabled = v;
        }
        if let Some(v) = self.dp_epsilon {
            cfg.privacy.epsilon = v;
        }
        if let Some(v) = self.meaning_enabled {
            cfg.meaning.enabled = v;
        }
        if let Some(v) = self.meaning_beta {
            cfg.meaning.beta = v;
        }
        if let Some(v) = self.meaning_max_fanout {
            cfg.meaning.algebraic_max_fanout = v as usize;
        }
        if let Some(v) = self.meaning_auto_decompose {
            cfg.meaning.auto_decompose = v;
        }
        if let Some(v) = self.meaning_max_hop_depth {
            cfg.meaning.max_hop_depth = v as usize;
        }
        if let Some(v) = self.cognition_enabled {
            cfg.cognition.enabled = v;
        }
        if let Some(v) = self.cognition_interval_secs {
            cfg.cognition.interval_secs = v as u64;
        }
        if let Some(v) = self.cognition_min_pattern_freq {
            cfg.cognition.min_pattern_freq = v as usize;
        }
        if let Some(v) = self.cognition_min_abstraction_members {
            cfg.cognition.min_abstraction_members = v as usize;
        }
        if let Some(v) = self.cognition_min_hypothesis_confidence {
            cfg.cognition.min_hypothesis_confidence = v;
        }
        cfg
    }
}
