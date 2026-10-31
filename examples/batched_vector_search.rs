use anyhow::{ensure, Result};
use writerslogic_warm::core::qgraph::{
    BuildParams, Graph, Metric, Residency, SearchKnobs, VGraph, VSearchParams,
};

fn main() -> Result<()> {
    let dim = 32;
    let rows: Vec<f32> = (0..512)
        .flat_map(|row| {
            (0..dim).map(move |column| {
                ((row * 29 + column * 11) as f32).sin() + (((row + 3) * (column + 7)) as f32).cos()
            })
        })
        .collect();
    let query = rows[17 * dim..18 * dim].to_vec();
    let graph = Graph::build_owned(
        rows,
        dim,
        &BuildParams {
            metric: Metric::Angular,
            degree: 24,
            build_ef: 64,
            quantized_search: true,
            ..Default::default()
        },
    )?;
    let index = VGraph::from_graph_with(&graph, 8, 8, true, true, 3, false)?;
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("vectors.warm");
    index.save(&path)?;
    let reopened = VGraph::open(&path, Residency::Mapped)?;
    let params = VSearchParams {
        ef: 80,
        rerank: 64,
        ..Default::default()
    };
    let mut reference = Vec::new();
    let reference_work =
        index
            .searcher_with(SearchKnobs::OFF)
            .search(&query, 10, params, &mut reference)?;
    let mut searcher = reopened.searcher_with(SearchKnobs::OFF);
    let mut neighbors = Vec::new();
    let work = searcher.search(&query, 10, params, &mut neighbors)?;
    ensure!(neighbors == reference && work == reference_work);
    println!("Nearest row IDs: {neighbors:?}");
    println!("Reopened search preserved the results and all {work} distance evaluations.");
    Ok(())
}
