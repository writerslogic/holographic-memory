use crate::core::entangled::EntangledHVec;

const BRUTE_FORCE_THRESHOLD: usize = 1000;
const SPARSE_INDEX_THRESHOLD: usize = 4;
const EF_SEARCH_MIN: usize = 32;
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
    BruteForce,
}

/// A structured plan for executing a query, including dynamic parameters.
pub(crate) struct QueryPlan {
    pub route: IndexRoute,
    pub ef_search: usize,
    pub n_probe: usize,
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
            };
        }

        // High-sparsity queries are ideal for Sparse Inverted Index
        if self.inverted_available && s <= SPARSE_INDEX_THRESHOLD {
            return QueryPlan {
                route: IndexRoute::Inverted,
                ef_search,
                n_probe,
            };
        }

        // Preferred indexed routes
        if self.nsg_available {
            return QueryPlan {
                route: IndexRoute::NSG,
                ef_search,
                n_probe,
            };
        }

        if self.ivf_available {
            return QueryPlan {
                route: IndexRoute::IVF,
                ef_search,
                n_probe,
            };
        }

        // Default to Inverted if it's the only thing we have and s isn't too high.
        if self.inverted_available && s < (self.dimensions / INVERTED_SPARSITY_DENOM) {
            return QueryPlan {
                route: IndexRoute::Inverted,
                ef_search,
                n_probe,
            };
        }

        QueryPlan {
            route: IndexRoute::BruteForce,
            ef_search,
            n_probe,
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
    fn plan_adjusts_ef_search_for_large_k() {
        let planner = QueryPlanner::new(true, true, true, 5000, 1000);
        let q = EntangledHVec::from_indices((0..10).collect(), 1000);
        let plan = planner.plan(&q, 100);
        assert!(plan.ef_search >= 100);
        assert_eq!(plan.route, IndexRoute::NSG);
    }
}
