use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HmsConfig {
    pub ivf: IVFConfig,
    pub nsg: NSGConfig,
    pub sharding: ShardConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NSGConfig {
    pub max_degree: usize,
    pub ef_search: usize,
    pub ef_construction: usize,
    pub auto_threshold: usize,
    pub seed: u64,
}

impl Default for NSGConfig {
    fn default() -> Self {
        Self {
            max_degree: 32,
            ef_search: 64,
            ef_construction: 128,
            auto_threshold: 10_000,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardConfig {
    pub enabled: bool,
    pub max_shard_size: usize,
    pub num_shards: usize,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_shard_size: 100_000,
            num_shards: 1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
