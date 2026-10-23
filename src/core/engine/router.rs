// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::entangled::EntangledHVec;

const BRUTE_FORCE_THRESHOLD: usize = 1000;
const HOPFIELD_MAX_PATTERNS: usize = 10_000;
const SPARSE_INDEX_THRESHOLD: usize = 4;
// A floor of 128 materially improves self-recall on the seeded 1,200-vector quality corpus
// while remaining bounded for interactive queries.
const EF_SEARCH_MIN: usize = 128;
const EF_SEARCH_MAX: usize = 512;
const N_PROBE_MIN: usize = 4;
const N_PROBE_MAX: usize = 64;
const INVERTED_SPARSITY_DENOM: usize = 32;

#[derive(Debug, PartialEq, Clone, Copy)]
#[expect(clippy::upper_case_acronyms, reason = "the index names")]
pub(crate) enum IndexRoute {
    NSG,
    Inverted,
    IVF,
    Hopfield,
    BruteForce,
}

/// A structured plan for executing a query, including dynamic parameters.
pub(crate) struct QueryPlan {
    pub(crate) route: IndexRoute,
    pub(crate) ef_search: usize,
    pub(crate) n_probe: usize,
    pub(crate) rationale: &'static str,
}

impl IndexRoute {
    pub(crate) fn as_str(self) -> &'static str {
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
    pub(crate) nsg_available: bool,
    pub(crate) inverted_available: bool,
    pub(crate) ivf_available: bool,
    pub(crate) vector_count: usize,
    pub(crate) dimensions: usize,
}

impl QueryPlanner {
    pub(crate) fn new(
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
    pub(crate) fn plan(&self, query_vec: &EntangledHVec, k: u32) -> QueryPlan {
        let n = self.vector_count;
        let s = query_vec.indices.len();
        let k_idx = k as usize;

        // A graph beam narrower than k cannot return k results.
        let ef_search = (k_idx * 2).clamp(EF_SEARCH_MIN, EF_SEARCH_MAX);
        let n_probe = (k_idx / 8).clamp(N_PROBE_MIN, N_PROBE_MAX);

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

    #[test]
    fn plan_routes_at_each_threshold() {
        use IndexRoute::*;
        const ALL: [bool; 3] = [true; 3];
        const NONE: [bool; 3] = [false; 3];
        let (exact, hopfield) = (BRUTE_FORCE_THRESHOLD, HOPFIELD_MAX_PATTERNS);
        let sparse = SPARSE_INDEX_THRESHOLD;
        // (name, [nsg, inverted, ivf] available, vectors, dimensions, query indices, route).
        // 64 active of 16384 is D/256, the density benchmarks/results/route_sweep.json
        // measured; at D = 64 only the absolute sparsity threshold applies.
        let cases = [
            ("exact - 1", ALL, exact - 1, 16384, 64, BruteForce),
            ("exact", ALL, exact, 16384, 64, Inverted),
            ("exact + 1", ALL, exact + 1, 16384, 64, Inverted),
            ("1e5 at D/256", ALL, 100_000, 16384, 64, Inverted),
            ("1e6 at D/256", ALL, 1_000_000, 16384, 64, Inverted),
            ("sparse - 1", ALL, 5000, 64, sparse - 1, Inverted),
            ("sparse", ALL, 5000, 64, sparse, Inverted),
            ("sparse + 1", ALL, 5000, 64, sparse + 1, NSG),
            ("D/32 - 1", ALL, 5000, 1000, 1000 / 32 - 1, Inverted),
            ("D/32", ALL, 5000, 1000, 1000 / 32, NSG),
            ("no NSG", [false, true, true], 5000, 1000, 40, IVF),
            ("no graph", [false, true, false], 5000, 1000, 40, Hopfield),
            ("hopfield - 1", NONE, hopfield - 1, 1000, 10, Hopfield),
            ("hopfield", NONE, hopfield, 1000, 10, Hopfield),
            ("hopfield + 1", NONE, hopfield + 1, 1000, 10, BruteForce),
        ];
        for (name, [nsg, inverted, ivf], vectors, dimensions, active, route) in cases {
            let query = EntangledHVec::from_indices((0..active as u32).collect(), dimensions);
            let planner = QueryPlanner::new(nsg, inverted, ivf, vectors, dimensions);
            assert_eq!(planner.plan(&query, 10).route, route, "{name}");
        }
    }

    #[test]
    fn graph_beam_covers_k_up_to_its_cap() {
        let planner = QueryPlanner::new(true, true, true, 5000, 1000);
        let query = EntangledHVec::from_indices((0..40).collect(), 1000);
        let cap = EF_SEARCH_MAX as u32;
        // 2k reaches EF_SEARCH_MIN at k = 64 and EF_SEARCH_MAX at k = 256.
        for k in [1, 63, 64, 65, 255, 256, 257, cap, cap + 1] {
            let plan = planner.plan(&query, k);
            assert_eq!(plan.route, IndexRoute::NSG, "k = {k}");
            assert!(
                (k.min(cap) as usize..=EF_SEARCH_MAX).contains(&plan.ef_search),
                "k = {k}: ef_search = {}",
                plan.ef_search
            );
        }
    }
}
