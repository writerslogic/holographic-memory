use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default)]
pub struct HmsConfig {
    pub ivf: IVFConfig,
    pub nsg: NSGConfig,
    pub shard: ShardConfig,
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
