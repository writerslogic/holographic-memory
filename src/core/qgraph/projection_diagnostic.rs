// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Experimental access to the production query-neighborhood projection primitive.

use super::Graph;
use crate::core::error::WarmError;

pub use super::projection::ProjectionStats;

impl Graph {
    /// Applies the production training routine to a cached, untrained diagnostic graph.
    pub fn train_diagnostic(&mut self, queries: &[f32]) -> Result<(), WarmError> {
        super::projection::validate_queries(queries, self.dim, self.metric)?;
        self.apply_training_queries(queries)
    }

    /// Projects construction-only neighbor lists using the production primitive.
    /// Queries use graph coordinates; each neighbor row starts at its nearest base vector.
    pub fn project_diagnostic(
        &mut self,
        training_queries: Option<&[f32]>,
        query_neighbor_ids: &[u32],
        neighbors_per_query: usize,
        projected_degree: usize,
    ) -> Result<ProjectionStats, WarmError> {
        self.project_neighborhoods(
            training_queries,
            query_neighbor_ids,
            neighbors_per_query,
            projected_degree,
        )
    }
}
