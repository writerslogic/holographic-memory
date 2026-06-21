use holographic_memory::core::block_codes::{BlockCodeConfig, BlockCodeStore};
use std::collections::HashMap;

struct TestResult {
    n_facts: usize,
    n_symbols: usize,
    correct: usize,
    crosstalk: usize,
    confound: usize,
}

fn run_test(
    cfg: BlockCodeConfig,
    n_chars: usize,
    n_rels: usize,
    n_objs: usize,
    max_facts: usize,
    preregister_all: bool,
) -> TestResult {
    let chars: Vec<String> = (0..n_chars).map(|i| format!("c{}", i)).collect();
    let rels: Vec<String> = (0..n_rels).map(|i| format!("r{}", i)).collect();
    let objs: Vec<String> = (0..n_objs).map(|i| format!("o{}", i)).collect();

    let mut store = BlockCodeStore::new(cfg);

    if preregister_all {
        for s in &chars {
            store.register(s);
        }
        for r in &rels {
            store.register(r);
        }
        for o in &objs {
            store.register(o);
        }
    }

    let mut facts: Vec<(String, String, String)> = Vec::new();
    let mut stored: HashMap<(String, String), Vec<String>> = HashMap::new();

    let mut fi = 0usize;
    for i in 0..n_chars {
        for j in 0..n_rels {
            if fi >= max_facts {
                break;
            }
            let s = &chars[i];
            let r = &rels[j];
            let o = &objs[(i * 3 + j * 7 + i * j) % n_objs];
            store.memorize_triplet(s, r, o);
            stored
                .entry((s.clone(), r.clone()))
                .or_default()
                .push(o.clone());
            facts.push((s.clone(), r.clone(), o.clone()));
            fi += 1;
        }
        if fi >= max_facts {
            break;
        }
    }

    let mut correct = 0;
    let mut crosstalk = 0;
    let mut confound = 0;

    for (s, r, expected) in &facts {
        match store.query_triplet(s, r) {
            Some(got) if got == expected => correct += 1,
            Some(got) => {
                let key = (s.clone(), r.clone());
                let all_stored = stored.get(&key).unwrap();
                if all_stored.iter().any(|o| o == got) {
                    confound += 1;
                } else {
                    crosstalk += 1;
                }
            }
            None => crosstalk += 1,
        }
    }

    TestResult {
        n_facts: facts.len(),
        n_symbols: store.n_symbols(),
        correct,
        crosstalk,
        confound,
    }
}

fn main() {
    println!("# BlockCodeStore capacity: isolated 1D sweeps, failure-mode logging");
    println!("# All symbols pre-registered before memorizing. Codebook size is fixed per sweep.");
    println!("# confound = wrong answer was a valid stored object for same (s,r) pair");
    println!("# crosstalk = wrong answer from a different (s,r) pair (genuine interference)");
    println!();

    let cfg = BlockCodeConfig::default_config();

    // SWEEP 1: Fixed vocabulary, vary fact count
    // 100 chars × 15 rels × 10 objs = 125 symbols, all pre-registered
    // Each (char, rel) pair maps to exactly one object, max 1500 unique facts
    println!("# SWEEP 1: Fixed vocab (125 symbols pre-registered), vary fact count, D=16384");
    println!("facts\tsymbols\ttop1\tcrosstalk\tconfound");
    for &target in &[
        50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500,
    ] {
        let r = run_test(cfg, 100, 15, 10, target, true);
        println!(
            "{}\t{}\t{:.4}\t{}\t\t{}",
            r.n_facts,
            r.n_symbols,
            r.correct as f64 / r.n_facts as f64,
            r.crosstalk,
            r.confound
        );
    }
    println!();

    // SWEEP 2: Fixed facts (500), vary codebook size via pre-registration
    // More registered symbols = harder search problem for score decoder
    println!("# SWEEP 2: Fixed facts (500), vary codebook size (pre-registered), D=16384");
    println!("# Extra symbols are distractors: registered but never appear in any fact");
    println!("chars\trels\tobjs\tfacts\tsymbols\ttop1\tcrosstalk\tconfound");
    for &(nc, nr, no) in &[
        (34, 15, 10),
        (50, 15, 10),
        (100, 15, 10),
        (200, 15, 10),
        (500, 15, 10),
        (1000, 15, 10),
        (2000, 15, 10),
    ] {
        let r = run_test(cfg, nc, nr, no, 500, true);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t\t{}",
            nc,
            nr,
            no,
            r.n_facts,
            r.n_symbols,
            r.correct as f64 / r.n_facts as f64,
            r.crosstalk,
            r.confound
        );
    }
    println!();

    // SWEEP 3: Fixed vocab and facts, vary D
    println!("# SWEEP 3: Fixed vocab (125 symbols), fixed facts (500), vary D");
    println!("B\tL\tD\tfacts\tsymbols\ttop1\tcrosstalk\tconfound");
    for &(b, l) in &[(256, 16), (512, 16), (1024, 16), (2048, 16), (4096, 16)] {
        let c = BlockCodeConfig::new(b, l);
        let r = run_test(c, 100, 15, 10, 500, true);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{:.4}\t{}\t\t{}",
            b,
            l,
            c.dim(),
            r.n_facts,
            r.n_symbols,
            r.correct as f64 / r.n_facts as f64,
            r.crosstalk,
            r.confound
        );
    }
    println!();

    // SWEEP 4: D-scaling exponent measurement
    // Fix facts at 1000 (89.3% at D=16384 — squarely in the informative band)
    // Fine-grained D sweep to read error counts directly
    println!("# SWEEP 4: D-scaling exponent (125 symbols, 1000 facts, vary D)");
    println!("# Goal: pin whether error ~ 1/D, 1/(D/logD), or steeper");
    println!("B\tL\tD\terrors\terror_rate\tlog2D\tlog2err");
    for &(b, l) in &[
        (128, 16),
        (192, 16),
        (256, 16),
        (384, 16),
        (512, 16),
        (768, 16),
        (1024, 16),
        (1536, 16),
        (2048, 16),
        (3072, 16),
        (4096, 16),
    ] {
        let c = BlockCodeConfig::new(b, l);
        let r = run_test(c, 100, 15, 10, 1000, true);
        let errors = r.crosstalk + r.confound;
        let d = c.dim() as f64;
        let err_rate = errors as f64 / r.n_facts as f64;
        if errors > 0 {
            println!(
                "{}\t{}\t{}\t{}\t{:.4}\t\t{:.2}\t{:.2}",
                b,
                l,
                c.dim(),
                errors,
                err_rate,
                d.log2(),
                (errors as f64).log2()
            );
        } else {
            println!(
                "{}\t{}\t{}\t{}\t{:.4}\t\t{:.2}\t-inf",
                b,
                l,
                c.dim(),
                errors,
                err_rate,
                d.log2()
            );
        }
    }
}
