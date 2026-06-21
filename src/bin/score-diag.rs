fn hash_pair(seed: u64) -> (usize, usize) {
    let a = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let b = a
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (a as usize, b as usize)
}

fn main() {
    let n_concepts = 200;
    let mut seen = std::collections::HashSet::new();
    let mut country_counts = std::collections::HashMap::new();
    let mut seed: u64 = 0;
    let mut facts = Vec::new();
    while facts.len() < 200 {
        let (a, b) = hash_pair(seed);
        seed += 1;
        let ci = a % n_concepts;
        let ki = b % n_concepts;
        if ci == ki {
            continue;
        }
        if !seen.insert((ci, ki)) {
            continue;
        }
        *country_counts.entry(ci).or_insert(0usize) += 1;
        facts.push((ci, ki));
    }
    let multi: Vec<_> = country_counts.iter().filter(|(_, &v)| v > 1).collect();
    let multi_facts: usize = multi.iter().map(|(_, &v)| v).sum();
    println!("unique countries: {}", country_counts.len());
    println!(
        "countries with 2+ facts: {} (covering {} facts)",
        multi.len(),
        multi_facts
    );
    let mut hist = std::collections::HashMap::new();
    for &v in country_counts.values() {
        *hist.entry(v).or_insert(0usize) += 1;
    }
    let mut hist_sorted: Vec<_> = hist.into_iter().collect();
    hist_sorted.sort();
    println!("frequency histogram: {:?}", hist_sorted);
}
