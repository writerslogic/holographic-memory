// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hopfield-in-the-loop composition: interpose associative cleanup between
//! bind steps to maintain fidelity across deep composition chains.
//!
//! Standard VSA composition degrades rapidly with depth because each bind
//! spreads energy across the space. By cleaning up through Hopfield retrieval
//! after each step, the composed vector stays close to a known pattern,
//! allowing arbitrarily deep chains limited only by the pattern bank size.

use crate::core::entangled::EntangledHVec;
use crate::core::hopfield::{hopfield_query, HopfieldConfig};

pub struct ComposeConfig {
    pub hopfield: HopfieldConfig,
    pub cleanup_every: usize,
}

impl Default for ComposeConfig {
    fn default() -> Self {
        Self {
            hopfield: HopfieldConfig {
                beta: 100.0,
                alpha: 2.0,
                max_iter: 1,
            },
            cleanup_every: 1,
        }
    }
}

/// Bind a sequence of keys onto a target, with Hopfield cleanup between steps.
///
/// Without cleanup: target ⊗ k1 ⊗ k2 ⊗ ... ⊗ kN (fidelity decays with N)
/// With cleanup: each intermediate result is projected back to the nearest
/// pattern in the memory bank before the next bind.
///
/// Returns the final composed vector after all binds and cleanup.
pub fn compose_with_cleanup(
    target: &EntangledHVec,
    keys: &[EntangledHVec],
    patterns: &[(String, EntangledHVec)],
    config: &ComposeConfig,
) -> EntangledHVec {
    let mut current = target.clone();

    for (i, key) in keys.iter().enumerate() {
        current = current.bind(key);

        if config.cleanup_every > 0 && (i + 1) % config.cleanup_every == 0 {
            current = cleanup(&current, patterns, &config.hopfield);
        }
    }

    current
}

/// Unbind a sequence of keys (in reverse) with cleanup between steps.
pub fn decompose_with_cleanup(
    composed: &EntangledHVec,
    keys: &[EntangledHVec],
    patterns: &[(String, EntangledHVec)],
    config: &ComposeConfig,
) -> EntangledHVec {
    let mut current = composed.clone();

    for (i, key) in keys.iter().rev().enumerate() {
        current = current.bind(key);

        if config.cleanup_every > 0 && (i + 1) % config.cleanup_every == 0 {
            current = cleanup(&current, patterns, &config.hopfield);
        }
    }

    current
}

/// Deep composition roundtrip: bind keys forward, then unbind in reverse,
/// with cleanup at each step. Returns the recovered vector.
pub fn roundtrip_with_cleanup(
    target: &EntangledHVec,
    keys: &[EntangledHVec],
    patterns: &[(String, EntangledHVec)],
    config: &ComposeConfig,
) -> EntangledHVec {
    let composed = compose_with_cleanup(target, keys, patterns, config);
    decompose_with_cleanup(&composed, keys, patterns, config)
}

fn cleanup(
    query: &EntangledHVec,
    patterns: &[(String, EntangledHVec)],
    config: &HopfieldConfig,
) -> EntangledHVec {
    if patterns.is_empty() {
        return query.clone();
    }
    let results = hopfield_query(query, patterns, config, 1);
    match results.first() {
        Some(r) => {
            patterns.iter()
                .find(|(id, _)| *id == r.id)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| query.clone())
        }
        None => query.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cleanup_preserves_identity() {
        let dim = 16384;
        let patterns: Vec<(String, EntangledHVec)> = (0..20)
            .map(|i| (format!("p{}", i), EntangledHVec::new_deterministic(dim, i * 100)))
            .collect();

        let target = &patterns[5].1;
        let config = ComposeConfig::default();
        let result = compose_with_cleanup(target, &[], &patterns, &config);
        assert!((result.similarity(target) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_deep_roundtrip_with_cleanup() {
        let dim = 16384;
        let patterns: Vec<(String, EntangledHVec)> = (0..50)
            .map(|i| (format!("p{}", i), EntangledHVec::new_deterministic(dim, i * 100)))
            .collect();

        let target = &patterns[10].1;
        let depth = 10;
        let keys: Vec<EntangledHVec> = (0..depth)
            .map(|d| EntangledHVec::new_deterministic(dim, 50000 + d as u64))
            .collect();

        let config = ComposeConfig::default();
        let recovered = roundtrip_with_cleanup(target, &keys, &patterns, &config);

        let sim_with = recovered.similarity(target);
        assert!(sim_with > 0.5,
            "Cleanup roundtrip at depth {} should recover target, got sim={:.4}", depth, sim_with);
    }

    #[test]
    fn test_cleanup_beats_raw_at_depth() {
        let dim = 16384;
        let patterns: Vec<(String, EntangledHVec)> = (0..50)
            .map(|i| (format!("p{}", i), EntangledHVec::new_deterministic(dim, i * 100)))
            .collect();

        let target = &patterns[3].1;
        let depth = 5;
        let keys: Vec<EntangledHVec> = (0..depth)
            .map(|d| EntangledHVec::new_deterministic(dim, 60000 + d as u64))
            .collect();

        // Raw roundtrip (no cleanup)
        let mut raw = target.clone();
        for k in &keys { raw = raw.bind(k); }
        for k in keys.iter().rev() { raw = raw.bind(k); }
        let raw_sim = raw.similarity(target);

        // With cleanup
        let config = ComposeConfig::default();
        let cleaned = roundtrip_with_cleanup(target, &keys, &patterns, &config);
        let clean_sim = cleaned.similarity(target);

        assert!(clean_sim >= raw_sim,
            "Cleanup ({:.4}) should beat or match raw ({:.4}) at depth {}",
            clean_sim, raw_sim, depth);
    }
}
