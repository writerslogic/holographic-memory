// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::core::entangled::EntangledHVec;

const BRUTE_FORCE_THRESHOLD: usize = 1000;
const HOPFIELD_MAX_PATTERNS: usize = 10_000;
const SPARSE_INDEX_THRESHOLD: usize = 4;
// A floor of 128 materially improves self-recall on the seeded quality corpus
// 1,200-vector quality corpus while remaining bounded for interactive queries.
const EF_SEARCH_MIN: usize = 128;
const EF_SEARCH_MAX: usize = 512;
const N_PROBE_MIN: usize = 4;
const N_PROBE_MAX: usize = 64;
const INVERTED_SPARSITY_DENOM: usize = 32;

#[derive(Debug, PartialEq, Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum IndexRoute {
    NSG,
    Inverted,
    IVF,
    Hopfield,
    BruteForce,
}

/// A structured plan for executing a query, including dynamic parameters.
pub(crate) struct QueryPlan {
    pub route: IndexRoute,
    pub ef_search: usize,
    pub n_probe: usize,
    pub rationale: &'static str,
}

impl IndexRoute {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NSG => "nsg",
            Self::Inverted => "inverted",
            Self::IVF => "ivf",
            Self::Hopfield => "hopfield",
            Self::BruteForce => "brute_force",
        }
    }
}

/// Adaptive Query Planner that selects the optimal retrieval strategy
/// based on collection statistics and query complexity.
pub(crate) struct QueryPlanner {
    pub nsg_available: bool,
    pub inverted_available: bool,
    pub ivf_available: bool,
    pub vector_count: usize,
    pub dimensions: usize,
}

impl QueryPlanner {
    pub fn new(
        nsg_available: bool,
        inverted_available: bool,
        ivf_available: bool,
        vector_count: usize,
        dimensions: usize,
    ) -> Self {
        Self {
            nsg_available,
            inverted_available,
            ivf_available,
            vector_count,
            dimensions,
        }
    }

    /// Select the best route and search parameters for a given query vector and requested k.
    pub fn plan(&self, query_vec: &EntangledHVec, k: u32) -> QueryPlan {
        let n = self.vector_count;
        let s = query_vec.indices.len(); // Query sparsity
        let k_idx = k as usize;

        // 1. Calculate dynamic parameters based on k and N
        // For NSG, ef_search should generally be >= k. SOTA engines use ef_search = k * multiplier + additive_constant.
        let ef_search = (k_idx * 2).clamp(EF_SEARCH_MIN, EF_SEARCH_MAX);
        // For IVF, n_probe should increase with k and total collection size.
        let n_probe = (k_idx / 8).clamp(N_PROBE_MIN, N_PROBE_MAX);

        // 2. Select Route

        // Brute-force is almost always faster for very small collections
        if n < BRUTE_FORCE_THRESHOLD {
            return QueryPlan {
                route: IndexRoute::BruteForce,
                ef_search,
                n_probe,
                rationale: "small collections are faster with an exact scan",
            };
        }

        // Exact inverted scoring costs about N*s^2/D counter increments. On
        // benchmarks/results/route_sweep.json (Apple M4, N = 1e3..1e6, D = 4096 and
        // 16384, s = D/256, random and clustered codes) it had recall@10 = 1.0 at every
        // point. NSG and IVF were faster only on clustered codes at N >= 1e5, with
        // recall@10 <= 0.37, so no usable crossover was reached in that range.
        // Denser queries are unmeasured and fall through to the approximate indexes.
        if self.inverted_available
            && (s <= SPARSE_INDEX_THRESHOLD || s < self.dimensions / INVERTED_SPARSITY_DENOM)
        {
            return QueryPlan {
                route: IndexRoute::Inverted,
                ef_search,
                n_probe,
                rationale: "exact inverted scoring is exact and fast for sparse queries",
            };
        }

        if self.nsg_available {
            return QueryPlan {
                route: IndexRoute::NSG,
                ef_search,
                n_probe,
                rationale: "a dense query with a trained NSG index uses graph retrieval",
            };
        }

        if self.ivf_available {
            return QueryPlan {
                route: IndexRoute::IVF,
                ef_search,
                n_probe,
                rationale: "a dense query with a trained IVF index uses IVF retrieval",
            };
        }

        // Hopfield associative retrieval for mid-sized collections without trained indices.
        // Entmax attention gives better ranking than brute-force similarity scan.
        if n <= HOPFIELD_MAX_PATTERNS {
            return QueryPlan {
                route: IndexRoute::Hopfield,
                ef_search,
                n_probe,
                rationale: "mid-sized unindexed collections use associative retrieval",
            };
        }

        QueryPlan {
            route: IndexRoute::BruteForce,
            ef_search,
            n_probe,
            rationale: "no trained index is available, so HMS falls back to exact search",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::entangled::EntangledHVec;

    #[test]
    fn plan_small_collection_uses_brute_force() {
        let planner = QueryPlanner::new(true, true, true, 500, 1000);
        let q = EntangledHVec::from_indices(vec![1, 2, 3], 1000);
        let plan = planner.plan(&q, 10);
        assert_eq!(plan.route, IndexRoute::BruteForce);
    }

    #[test]
    fn plan_high_sparsity_uses_inverted() {
        let planner = QueryPlanner::new(true, true, true, 5000, 1000);
        let q = EntangledHVec::from_indices(vec![1, 2], 1000);
        let plan = planner.plan(&q, 10);
        assert_eq!(plan.route, IndexRoute::Inverted);
    }

    #[test]
    fn plan_mid_collection_no_indices_uses_hopfield() {
        let planner = QueryPlanner::new(false, false, false, 5000, 1000);
        let q = EntangledHVec::from_indices((0..10).collect(), 1000);
        let plan = planner.plan(&q, 10);
        assert_eq!(plan.route, IndexRoute::Hopfield);
    }

    #[test]
    fn plan_large_collection_no_indices_uses_brute_force() {
        let planner = QueryPlanner::new(false, false, false, 20_000, 1000);
        let q = EntangledHVec::from_indices((0..10).collect(), 1000);
        let plan = planner.plan(&q, 10);
        assert_eq!(plan.route, IndexRoute::BruteForce);
    }

    #[test]
    fn plan_adjusts_ef_search_for_large_k() {
        let planner = QueryPlanner::new(true, true, true, 5000, 1000);
        let q = EntangledHVec::from_indices((0..40).collect(), 1000);
        let plan = planner.plan(&q, 100);
        assert!(plan.ef_search >= 100);
        assert_eq!(plan.route, IndexRoute::NSG);
    }

    #[test]
    fn plan_default_density_prefers_inverted_over_trained_indexes() {
        // D/256 active indices, the density the sweep measured.
        let q = EntangledHVec::from_indices((0..64).collect(), 16384);
        for n in [1_000, 100_000, 1_000_000] {
            let plan = QueryPlanner::new(true, true, true, n, 16384).plan(&q, 10);
            assert_eq!(plan.route, IndexRoute::Inverted, "n = {n}");
        }
    }

    #[test]
    fn plan_dense_query_falls_back_to_trained_indexes() {
        let q = EntangledHVec::from_indices((0..40).collect(), 1000);
        let nsg = QueryPlanner::new(true, true, true, 5000, 1000).plan(&q, 10);
        assert_eq!(nsg.route, IndexRoute::NSG);
        let ivf = QueryPlanner::new(false, true, true, 5000, 1000).plan(&q, 10);
        assert_eq!(ivf.route, IndexRoute::IVF);
    }

    #[test]
    fn plan_brute_force_boundary() {
        let q = EntangledHVec::from_indices((0..64).collect(), 16384);
        let below = QueryPlanner::new(true, true, true, BRUTE_FORCE_THRESHOLD - 1, 16384);
        assert_eq!(below.plan(&q, 10).route, IndexRoute::BruteForce);
        let at = QueryPlanner::new(true, true, true, BRUTE_FORCE_THRESHOLD, 16384);
        assert_eq!(at.plan(&q, 10).route, IndexRoute::Inverted);
    }
}
