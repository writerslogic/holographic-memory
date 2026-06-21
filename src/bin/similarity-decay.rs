use holographic_memory::core::entangled::EntangledHVec;
use holographic_memory::core::block_codes::{BlockCodeConfig, BlockCodeVec, BlockCodeBundle};

fn measure_bloom(dim: usize, density_denom: usize, max_items: usize, n_probes: usize) {
    println!("# BLOOM dim={} denom={}", dim, density_denom);
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    let items: Vec<EntangledHVec> = (0..max_items)
        .map(|i| EntangledHVec::new_with_density(dim, density_denom, i as u64 * 37 + 1))
        .collect();
    let non_members: Vec<EntangledHVec> = (0..n_probes)
        .map(|i| EntangledHVec::new_with_density(dim, density_denom, (max_items + i) as u64 * 37 + 9999))
        .collect();

    for &n_items in &load_points(max_items) {
        let bundle = EntangledHVec::bundle_bloom(&items[..n_items]);
        let density = bundle.indices().len() as f64 / dim as f64;

        let member_sims: Vec<f64> = items[..n_items]
            .iter()
            .map(|item| item.corrected_containment(&bundle))
            .collect();
        let member_mean = mean(&member_sims);
        let member_min = fmin(&member_sims);

        let nonmember_sims: Vec<f64> = non_members
            .iter()
            .map(|item| item.corrected_containment(&bundle))
            .collect();
        let nonmember_mean = mean(&nonmember_sims);
        let nonmember_max = fmax(&nonmember_sims);

        println!("bloom\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                 n_items, member_mean, member_min, nonmember_mean, nonmember_max,
                 member_min - nonmember_max);

        if density > 0.995 { break; }
    }
    println!();
}

fn measure_sbc(n_blocks: usize, block_len: usize, max_items: usize, n_probes: usize) {
    let cfg = BlockCodeConfig::new(n_blocks, block_len);
    println!("# SBC dim={} blocks={} block_len={}", cfg.dim(), n_blocks, block_len);
    println!("scheme\tn_items\tmember_mean\tmember_min\tnonmember_mean\tnonmember_max\tgap");

    let items: Vec<BlockCodeVec> = (0..max_items)
        .map(|i| BlockCodeVec::new_deterministic(&cfg, i as u64 * 37 + 1))
        .collect();
    let non_members: Vec<BlockCodeVec> = (0..n_probes)
        .map(|i| BlockCodeVec::new_deterministic(&cfg, (max_items + i) as u64 * 37 + 9999))
        .collect();

    let mut bundle = BlockCodeBundle::new(&cfg);

    for &n_items in &load_points(max_items) {
        while bundle.n_items() < n_items {
            bundle.add(&items[bundle.n_items()]);
        }

        let member_scores: Vec<f64> = items[..n_items]
            .iter()
            .map(|item| bundle.count_score(item))
            .collect();
        let nonmember_scores: Vec<f64> = non_members
            .iter()
            .map(|item| bundle.count_score(item))
            .collect();

        let expected_nonmember = n_items as f64 / block_len as f64;
        let member_mean = mean(&member_scores) - expected_nonmember;
        let member_min = fmin(&member_scores) - expected_nonmember;
        let nonmember_mean = mean(&nonmember_scores) - expected_nonmember;
        let nonmember_max = fmax(&nonmember_scores) - expected_nonmember;

        println!("sbc\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
                 n_items, member_mean, member_min, nonmember_mean, nonmember_max,
                 member_min - nonmember_max);
    }
    println!();
}

fn load_points(max_items: usize) -> Vec<usize> {
    let mut points = Vec::new();
    let mut n = 1;
    while n <= max_items {
        points.push(n);
        if n < 10 { n += 1; }
        else if n < 100 { n += 10; }
        else if n < 1000 { n += 100; }
        else if n < 10000 { n += 1000; }
        else { n += 5000; }
    }
    if *points.last().unwrap_or(&0) != max_items {
        points.push(max_items);
    }
    points
}

fn mean(vals: &[f64]) -> f64 { vals.iter().sum::<f64>() / vals.len() as f64 }
fn fmin(vals: &[f64]) -> f64 { vals.iter().cloned().fold(f64::INFINITY, f64::min) }
fn fmax(vals: &[f64]) -> f64 { vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }

fn main() {
    println!("# Similarity decay: Bloom vs SBC at D=16384");
    println!("# Bloom: corrected containment (member=1.0 until cliff)");
    println!("# SBC: count_score - expected_noise (member≈1.0, nonmember≈0)");
    println!("# gap = member_min - nonmember_max (>0 means perfect separation)");
    println!();

    measure_bloom(16384, 256, 5000, 200);

    measure_sbc(256, 64, 5000, 200);
    measure_sbc(64, 256, 5000, 200);
    measure_sbc(16, 1024, 10000, 200);
}
