use anyhow::{ensure, Context, Result};
use serde_json::json;
use std::{fs, path::Path, time::Instant};
use writerslogic_warm::core::qgraph::{
    BuildParams, Graph, Metric, SearchKnobs, VGraph, VSearchParams,
};

fn rows(path: &Path, count: usize) -> Result<Vec<f32>> {
    ensure!(
        fs::metadata(path)?.len() == (count * 100 * 4) as u64,
        "wrong matrix size"
    );
    let bytes = fs::read(path)?;
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    ensure!(values.iter().all(|x| x.is_finite()), "nonfinite matrix");
    ensure!(
        values
            .as_chunks::<100>()
            .0
            .iter()
            .all(|r| r.iter().any(|&x| x != 0.0)),
        "zero angular row"
    );
    Ok(values)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(
        args.len() == 4,
        "usage: vector_glove_diagnosis INPUT OUTPUT plain|profiled"
    );
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    let profiled = match args[3].to_str() {
        Some("plain") => false,
        Some("profiled") => true,
        _ => anyhow::bail!("invalid mode"),
    };
    ensure!(!output.exists(), "output already exists");
    ensure!(
        std::env::var_os("WARM_QGRAPH_BUILD_PROFILE").is_some() == profiled,
        "profile flag mismatch"
    );
    let declaration: serde_json::Value =
        serde_json::from_slice(&fs::read(input.join("declaration.json"))?)?;
    ensure!(
        declaration["official_test_arrays_read"] == false
            && declaration["construction_count"] == 65536
            && declaration["development_count"] == 256,
        "input declaration mismatch"
    );
    let base = rows(&input.join("construction.f32"), 65536)?;
    let queries = rows(&input.join("development.f32"), 256)?;
    let params = BuildParams {
        metric: Metric::Angular,
        degree: 64,
        build_ef: 500,
        quantized_search: true,
        ..Default::default()
    };
    eprintln!("BUILD_STARTED");
    let start = Instant::now();
    let graph = Graph::build_owned(base, 100, &params)?;
    let build_s = start.elapsed().as_secs_f64();
    let graph_hash = graph.structure_hash();
    let start = Instant::now();
    let index = VGraph::from_graph_with(&graph, 8, 8, true, true, 3, false)?;
    let encoding_s = start.elapsed().as_secs_f64();
    drop(graph);
    let index_path = output.with_extension("warm");
    ensure!(!index_path.exists(), "index output already exists");
    index.save(&index_path).context("save diagnostic index")?;
    eprintln!("BUILD_FINISHED");
    let mut searcher = index.searcher_with(SearchKnobs::OFF);
    let mut out = Vec::new();
    let mut settings = Vec::new();
    for ef in [80, 240, 640] {
        let params = VSearchParams {
            ef,
            rerank: 64,
            ..Default::default()
        };
        let mut ids = Vec::new();
        let mut work = Vec::new();
        for query in queries.as_chunks::<100>().0 {
            searcher.search(query, 10, params, &mut out)?;
            ensure!(
                out.len() == 10 && out.iter().all(|&x| x < 65536),
                "invalid IDs"
            );
            let mut unique = out.clone();
            unique.sort_unstable();
            unique.dedup();
            ensure!(unique.len() == 10, "duplicate IDs");
            let expected = out.clone();
            if profiled {
                let profile = searcher.profile_search(query, 10, params, &mut out)?;
                ensure!(out == expected, "profile changed ordered results");
                work.push(serde_json::to_value(profile)?);
            }
            ids.push(expected);
        }
        settings.push(json!({"ef": ef, "rerank": 64, "ids": ids, "profiles": work}));
    }
    fs::write(
        output,
        serde_json::to_vec_pretty(&json!({
            "source": "12c80decb094d57ee13ba187a01bf1de8788ca2f", "profiled": profiled,
            "diagnostic_only": true, "graph_hash": format!("{graph_hash:016x}"),
            "build_s": build_s, "encoding_s": encoding_s, "index_bytes": index.index_bytes(),
            "settings": settings
        }))?,
    )?;
    Ok(())
}
