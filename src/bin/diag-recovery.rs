use holographic_memory::core::block_codes::{BlockCodeConfig, BlockCodeVec, BlockCodeBundle};

fn main() {
    let cfg = BlockCodeConfig::default_config(); // 1024x16
    
    // Same setup as test_recover_from_codebook
    let codebook: Vec<BlockCodeVec> = (0..50)
        .map(|i| BlockCodeVec::new_deterministic(&cfg, i * 31 + 7))
        .collect();
    let relation = BlockCodeVec::new_deterministic(&cfg, 999);

    let mut bundle = BlockCodeBundle::new(&cfg);
    for ci in 0..20 {
        let ki = (ci + 1) % 50;
        let fact = codebook[ci].bind(&relation).bind(&codebook[ki]);
        bundle.add(&fact);
    }

    println!("fact_idx\tcorrect_ki\tdecoded_sim\tbest_idx\tbest_sim\t2nd_sim\trecovered");
    for ci in 0..20 {
        let ki = (ci + 1) % 50;
        let cue = codebook[ci].bind(&relation);
        let decoded = bundle.unbind_and_decode(&cue);
        let correct_sim = decoded.similarity(&codebook[ki]);
        
        let mut sims: Vec<(usize, f64)> = (0..50)
            .map(|j| (j, decoded.similarity(&codebook[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("{}\t{}\t{:.4}\t{}\t{:.4}\t{:.4}\t{}",
                 ci, ki, correct_sim, sims[0].0, sims[0].1, sims[1].1,
                 if sims[0].0 == ki { "YES" } else { "NO" });
    }

    // Now same but with benchmark-style facts (ci=i, ki=0)
    println!("\n# Benchmark-style facts (all ki=0)");
    let codebook2: Vec<BlockCodeVec> = (0..200)
        .map(|i| BlockCodeVec::new_deterministic(&cfg, i as u64 * 31 + 7))
        .collect();
    let relation2 = BlockCodeVec::new_deterministic(&cfg, 999999);
    
    let mut bundle2 = BlockCodeBundle::new(&cfg);
    // First 20 non-self facts from benchmark pattern
    let mut facts2 = Vec::new();
    let mut sc: u64 = 0;
    while facts2.len() < 20 {
        let ci = (sc as usize) % 200;
        let ki = ((sc as usize) / 200) % 200;
        sc += 1;
        if ci == ki { continue; }
        facts2.push((ci, ki));
        let fact = codebook2[ci].bind(&relation2).bind(&codebook2[ki]);
        bundle2.add(&fact);
    }
    
    println!("fact_idx\tci\tki\tdecoded_sim\tbest_idx\tbest_sim\trecovered");
    for (fi, &(ci, ki)) in facts2.iter().enumerate() {
        let cue = codebook2[ci].bind(&relation2);
        let decoded = bundle2.unbind_and_decode(&cue);
        let correct_sim = decoded.similarity(&codebook2[ki]);
        
        let mut sims: Vec<(usize, f64)> = (0..200)
            .map(|j| (j, decoded.similarity(&codebook2[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{}",
                 fi, ci, ki, correct_sim, sims[0].0, sims[0].1,
                 if sims[0].0 == ki { "YES" } else { "NO" });
    }
}
