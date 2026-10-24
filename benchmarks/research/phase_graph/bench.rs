// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use crate::core::phase_graph::PhaseGraph;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let facts: usize = args[1].parse().unwrap();
    let dim: usize = args[2].parse().unwrap();
    let n_phases: u32 = args[3].parse().unwrap();
    let names: Vec<String> = (0..=facts)
        .map(|i| format!("episode/entity/{i:08}/canonical-record"))
        .collect();
    let mut graph = PhaseGraph::new(dim, n_phases);
    let start = Instant::now();
    for i in 0..facts {
        graph.relate(&names[i], "precedes", &names[i + 1]);
    }
    let build_ns = start.elapsed().as_nanos();
    let queries: Vec<usize> = (0..32).map(|i| 1 + i * (facts - 4) / 32).collect();
    let mut forward = 0;
    let mut backward = 0;
    let mut paths = 0;
    let mut reverse_score = 0.0;
    for &i in &queries {
        forward +=
            usize::from(graph.retrieve_object(&names[i], "precedes") == Some(&*names[i + 1]));
        backward +=
            usize::from(graph.retrieve_subject("precedes", &names[i + 1]) == Some(&*names[i]));
        paths += usize::from(
            graph
                .retrieve_path(&names[i], &["precedes", "precedes", "precedes"])
                .as_deref()
                == Some(&*names[i + 3]),
        );
        reverse_score += graph.score(&names[i + 1], "precedes", &names[i]);
    }
    let start = Instant::now();
    for &i in &queries {
        black_box(graph.retrieve_object(black_box(&names[i]), "precedes"));
        black_box(graph.retrieve_subject("precedes", black_box(&names[i + 1])));
    }
    let query_ns = start.elapsed().as_nanos() as f64 / (queries.len() * 2) as f64;
    let start = Instant::now();
    for _ in 0..8 {
        for &i in &queries {
            black_box(graph.score(black_box(&names[i]), "precedes", &names[i + 1]));
        }
    }
    let score_ns = start.elapsed().as_nanos() as f64 / (queries.len() * 8) as f64;
    assert!(graph.verify());
    println!(
        "{{\"facts\":{facts},\"dim\":{dim},\"n_phases\":{n_phases},\"queries\":{},\"forward_correct\":{forward},\"backward_correct\":{backward},\"three_hop_correct\":{paths},\"mean_reverse_score\":{},\"build_ns\":{build_ns},\"query_ns\":{query_ns},\"score_ns\":{score_ns}}}",
        queries.len(), reverse_score / queries.len() as f64,
    );
}
