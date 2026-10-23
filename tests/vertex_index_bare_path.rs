// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! `VGraph::save` to a bare file name (relative to the working directory) succeeds and the file
//! opens. Its own test binary, since it changes the process's working directory.

use std::path::Path;

use holographic_memory::core::qgraph::{BuildParams, Graph, Residency, VGraph};

#[test]
fn save_to_a_bare_file_name_succeeds() {
    let dim = 16;
    let data: Vec<f32> = (0..200 * dim)
        .map(|i| ((i * 7919) % 101) as f32 / 101.0 - 0.5)
        .collect();
    let graph = Graph::build(&data, dim, &BuildParams::default()).unwrap();
    let index = VGraph::from_graph(&graph, false).unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::env::set_current_dir(dir.path()).unwrap();
    index.save(Path::new("index")).unwrap();
    let opened = VGraph::open(Path::new("index"), Residency::Heap).unwrap();
    assert_eq!(opened.dim(), dim);
    assert_eq!(opened.index_bytes(), index.index_bytes());
}
