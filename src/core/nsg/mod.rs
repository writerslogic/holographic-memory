// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

pub(crate) mod graph;
pub(crate) mod search;
pub(crate) mod training;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::core::config::NSGConfig;
use crate::core::entangled::EntangledHVec;
use crate::core::types::RetrievalResult;

#[derive(Serialize, Deserialize)]
pub(crate) struct NSGIndex {
    pub(crate) neighbors: Vec<Vec<u32>>,
    pub(crate) vectors: Vec<EntangledHVec>,
    pub(crate) id_map: Vec<String>,
    pub(crate) navigating_node: u32,
    pub(crate) trained: bool,
    pub(crate) config: NSGConfig,
}

impl NSGIndex {
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    pub fn insert(&mut self, id: &str, vector: &EntangledHVec) -> Result<()> {
        graph::insert_online(self, id, vector)
    }

    pub fn query(&self, query: &EntangledHVec, k: usize, ef_search: usize) -> Vec<RetrievalResult> {
        search::greedy_search(self, query, k, ef_search)
    }
}
