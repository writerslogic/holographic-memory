use anyhow::Result;

use super::graph;
use super::NSGIndex;
use crate::core::config::NSGConfig;
use crate::core::entangled::EntangledHVec;

pub fn train(
    vectors: &[EntangledHVec],
    ids: &[String],
    config: &NSGConfig,
) -> Result<NSGIndex> {
    let k_build = config.ef_construction.min(vectors.len().saturating_sub(1));

    let knn_graph = graph::build_knn_graph(vectors, k_build, config.seed);
    let neighbors = graph::prune_edges(vectors, &knn_graph, config.max_degree);
    let navigating_node = graph::select_navigating_node(vectors);

    Ok(NSGIndex {
        neighbors,
        vectors: vectors.to_vec(),
        id_map: ids.to_vec(),
        navigating_node,
        trained: true,
        config: config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_basic() {
        let vectors: Vec<EntangledHVec> = (0..30)
            .map(|i| EntangledHVec::new_deterministic(1000, i))
            .collect();
        let ids: Vec<String> = (0..30).map(|i| format!("v_{}", i)).collect();
        let config = NSGConfig {
            max_degree: 8,
            ef_construction: 16,
            auto_threshold: 0,
            seed: 42,
        };

        let index = train(&vectors, &ids, &config).unwrap();
        assert!(index.is_trained());
        assert!((index.navigating_node as usize) < vectors.len());
        for neighbors in &index.neighbors {
            assert!(neighbors.len() <= 8, "Degree exceeds max_degree");
        }
    }

    #[test]
    fn online_insert() {
        let vectors: Vec<EntangledHVec> = (0..20)
            .map(|i| EntangledHVec::new_deterministic(1000, i))
            .collect();
        let ids: Vec<String> = (0..20).map(|i| format!("v_{}", i)).collect();
        let config = NSGConfig {
            max_degree: 8,
            ef_construction: 16,
            auto_threshold: 0,
            seed: 42,
        };

        let mut index = train(&vectors, &ids, &config).unwrap();
        let new_vec = EntangledHVec::new_deterministic(1000, 999);
        index.insert("new_vec", &new_vec).unwrap();

        assert_eq!(index.vectors.len(), 21);
        assert_eq!(index.id_map.len(), 21);
        assert_eq!(index.neighbors.len(), 21);
        assert!(
            !index.neighbors[20].is_empty(),
            "New node should have neighbors"
        );
    }
}
