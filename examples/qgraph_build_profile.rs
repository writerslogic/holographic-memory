// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Allocation and time profile of one vector-index build, per construction phase.
//!
//! A counting global allocator records allocations, bytes allocated, live bytes and the peak
//! of live bytes; the construction calls back at the end of each of its phases (`setup`, which
//! includes the row normalization or norms, `pass` x2, `fill`, `upper`), and this program
//! brackets `encode` (`VGraph::from_graph_with`), `drop` (the build graph), `save` and `open`. The counters cost an atomic per allocation, so the
//! times here are for attribution; build time is measured without this allocator.
//!
//! ```sh
//! cargo run --release --features experimental --example qgraph_build_profile -- \
//!     --data DIR --dim 128 --metric euclidean          # DIR/train.f32, row-major f32
//! cargo run --release --features experimental --example qgraph_build_profile -- \
//!     --synthetic 200000 --dim 128 --metric euclidean  # clustered Gaussian rows
//! ```
//! Options: `--rows N` (the first N rows), `--orders` (locality of candidate vertex
//! orders), `--degree 64 --build-ef 500 --quantized-search --save PATH`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::Instant;

use writerslogic_warm::core::qgraph::order_diagnostic::order_locality;
use writerslogic_warm::core::qgraph::{
    set_build_phase_hook, BuildParams, Graph, Metric, Residency, VGraph,
};

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static MAX_PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(size: usize) {
    ALLOCS.fetch_add(1, Relaxed);
    BYTES.fetch_add(size as u64, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: every call forwards to `System` with the caller's layout unchanged; the counters
// only observe sizes and never touch the memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            grow(new_size);
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Counter values at the end of the previous phase.
struct Mark {
    at: Instant,
    allocs: u64,
    bytes: u64,
}

static MARK: Mutex<Option<Mark>> = Mutex::new(None);

/// Prints the phase's wall time, allocations, bytes allocated, peak live bytes during the
/// phase and live bytes at its end, then starts the next phase.
fn end_phase(name: &str) {
    let mib = |b: usize| b as f64 / f64::from(1 << 20);
    let mut mark = MARK.lock().unwrap_or_else(|e| e.into_inner());
    let (allocs, bytes, live) = (
        ALLOCS.load(Relaxed),
        BYTES.load(Relaxed),
        LIVE.load(Relaxed),
    );
    let peak = PEAK.swap(live, Relaxed);
    MAX_PEAK.fetch_max(peak, Relaxed);
    if let Some(m) = mark.as_ref() {
        println!(
            "{name:<8} {:>8.2} s {:>12} allocs {:>10.1} MiB alloc'd {:>9.1} MiB peak {:>9.1} MiB live",
            m.at.elapsed().as_secs_f64(),
            allocs - m.allocs,
            (bytes - m.bytes) as f64 / f64::from(1 << 20),
            mib(peak),
            mib(live),
        );
    }
    *mark = Some(Mark {
        at: Instant::now(),
        allocs,
        bytes,
    });
}

/// Deterministic clustered Gaussian rows: 1,000 centres, unit-variance noise around each.
fn synthetic(n: usize, dim: usize) -> Vec<f32> {
    let mut s = 0x5EED_u64;
    let mut next = move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f32 / (1u64 << 53) as f32
    };
    // Sum of four uniforms: close enough to Gaussian for a graph workload.
    let mut gauss = move || (next() + next() + next() + next() - 2.0) * 1.7;
    let centres: Vec<f32> = (0..1000 * dim).map(|_| gauss() * 8.0).collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = &centres[(i * 7919 % 1000) * dim..][..dim];
        out.extend(c.iter().map(|&m| m + gauss()));
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let dim: usize = opt("--dim").ok_or("--dim is required")?.parse()?;
    let metric = match opt("--metric").as_deref() {
        Some("angular") => Metric::Angular,
        Some("euclidean") | None => Metric::Euclidean,
        Some(m) => return Err(format!("unknown metric {m}").into()),
    };
    let params = BuildParams {
        metric,
        degree: opt("--degree").map_or(Ok(64), |v| v.parse())?,
        build_ef: opt("--build-ef").map_or(Ok(500), |v| v.parse())?,
        quantized_search: args.iter().any(|a| a == "--quantized-search"),
        ..BuildParams::default()
    };
    let data = match (opt("--data"), opt("--synthetic")) {
        (Some(dir), _) => {
            let raw = std::fs::read(std::path::Path::new(&dir).join("train.f32"))?;
            let rows: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            drop(raw);
            rows
        }
        (None, Some(n)) => synthetic(n.parse()?, dim),
        (None, None) => return Err("--data DIR or --synthetic N is required".into()),
    };
    let mut data = data;
    if let Some(rows) = opt("--rows") {
        data.truncate(rows.parse::<usize>()? * dim);
        data.shrink_to_fit();
    }
    println!(
        "rows {} dim {dim} {metric:?} degree {} build_ef {} quantized {} threads {}",
        data.len() / dim,
        params.degree,
        params.build_ef,
        params.quantized_search,
        rayon::current_num_threads()
    );
    set_build_phase_hook(end_phase);
    let total = Instant::now();
    end_phase("input");
    let graph = Graph::build_owned(data, dim, &params)?;
    let build = total.elapsed().as_secs_f64();
    let v = VGraph::from_graph_with(&graph, 8, 8, true, true, 3, false)?;
    end_phase("encode");
    println!("graph_hash {:016x}", graph.structure_hash());
    if args.iter().any(|a| a == "--orders") {
        for o in order_locality(&graph) {
            println!(
                "order {:<12} {:>7.3} s {:.3} B/edge log2gap {:.2} within32 {:.4} within512 {:.4} pages/list {:.2}",
                o.name,
                o.order_secs,
                o.varint_bytes_per_edge,
                o.mean_log2_gap,
                o.within_32,
                o.within_512,
                o.pages_per_list
            );
        }
        end_phase("orders");
    }
    drop(graph);
    end_phase("drop");
    if let Some(path) = opt("--save") {
        let path = std::path::PathBuf::from(path);
        v.save(&path)?;
        drop(v);
        end_phase("save");
        let w = VGraph::open(&path, Residency::Heap)?;
        end_phase("open");
        drop(w);
    }
    println!(
        "build {build:.2} s, total {:.2} s, process peak live {:.1} MiB",
        total.elapsed().as_secs_f64(),
        MAX_PEAK.load(Relaxed) as f64 / f64::from(1 << 20)
    );
    Ok(())
}
