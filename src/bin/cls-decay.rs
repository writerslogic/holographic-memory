use holographic_memory::core::cls_memory::{ClsConfig, ClsMemory};
use holographic_memory::core::entangled::EntangledHVec;

fn measure_flat(
    dim: usize,
    denom: usize,
    items: &[EntangledHVec],
    probes: &[EntangledHVec],
    load_points: &[usize],
) {
    println!("# FLAT BLOOM dim={} denom={}", dim, denom);
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    for &n in load_points {
        if n > items.len() {
            break;
        }
        let bundle = EntangledHVec::bundle_bloom(&items[..n]);
        let density = bundle.indices().len() as f64 / dim as f64;

        let ms: Vec<f64> = items[..n]
            .iter()
            .map(|i| i.corrected_containment(&bundle))
            .collect();
        let ns: Vec<f64> = probes
            .iter()
            .map(|i| i.corrected_containment(&bundle))
            .collect();

        let mm = mean(&ms);
        let mi = fmin(&ms);
        let nm = mean(&ns);
        let nx = fmax(&ns);

        println!(
            "flat\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            n,
            mm,
            mi,
            nm,
            nx,
            mi - nx
        );
        if density > 0.995 {
            break;
        }
    }
    println!();
}

fn measure_cls(
    dim: usize,
    denom: usize,
    items: &[EntangledHVec],
    probes: &[EntangledHVec],
    load_points: &[usize],
    consol_interval: usize,
    heart1_cap: usize,
) {
    println!(
        "# CLS dim={} denom={} consolidation_interval={} heart1_capacity={}",
        dim, denom, consol_interval, heart1_cap
    );
    println!("scheme\tn_items\th1_count\th2_count\th1_density\th2_density\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    let mut cfg = ClsConfig::new(dim, denom);
    cfg.consolidation_interval = consol_interval;
    cfg.heart1_capacity = heart1_cap;
    cfg.resonance_threshold = 0.15;
    let mut mem = ClsMemory::new(cfg);

    let mut next_lp = 0;

    for i in 0..items.len() {
        mem.add(items[i].clone());
        let n = i + 1;

        if next_lp < load_points.len() && n == load_points[next_lp] {
            next_lp += 1;

            let ms: Vec<f64> = items[..n].iter().map(|item| mem.query(item)).collect();
            let ns: Vec<f64> = probes.iter().map(|item| mem.query(item)).collect();

            let mm = mean(&ms);
            let mi = fmin(&ms);
            let nm = mean(&ns);
            let nx = fmax(&ns);

            println!(
                "cls\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                n,
                mem.heart1_count(),
                mem.heart2_count(),
                mem.heart1_density(),
                mem.heart2_density(),
                mm,
                mi,
                nm,
                nx,
                mi - nx
            );
        }

        if n >= *load_points.last().unwrap_or(&0) {
            break;
        }
    }
    println!();
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn fmin(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}
fn fmax(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

fn main() {
    let dim = 16384;
    let denom = 256;
    let max_items = 5000;
    let n_probes = 200;

    let items: Vec<EntangledHVec> = (0..max_items)
        .map(|i| EntangledHVec::new_with_density(dim, denom, i as u64 * 37 + 1))
        .collect();
    let probes: Vec<EntangledHVec> = (0..n_probes)
        .map(|i| EntangledHVec::new_with_density(dim, denom, (max_items + i) as u64 * 37 + 9999))
        .collect();

    let load_points: Vec<usize> = {
        let mut pts = Vec::new();
        let mut n = 10;
        while n <= max_items {
            pts.push(n);
            if n < 100 {
                n += 10;
            } else if n < 1000 {
                n += 100;
            } else {
                n += 500;
            }
        }
        pts
    };

    println!("# CLS-VSA vs Flat Bloom: same dimension, same items, same probes");
    println!("# D={} denom={} probes={}", dim, denom, n_probes);
    println!();

    measure_flat(dim, denom, &items, &probes, &load_points);
    measure_cls(dim, denom, &items, &probes, &load_points, 50, 200);
    measure_cls(dim, denom, &items, &probes, &load_points, 25, 100);
}
