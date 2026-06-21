// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! CLS (Complementary Learning Systems) with STRUCTURED data.
//!
//! Key insight: codebook-composed facts (concept_a XOR concept_b) share
//! binding-algebraic structure. Facts sharing a concept have detectable
//! overlap, enabling structure-aware consolidation from Heart1 (fast/volatile)
//! to Heart2 (slow/stable).
//!
//! Heart1: Bloom bundle (fast containment query), FIFO capped at 200 items.
//! Heart2: Individual item store (exact Jaccard similarity search).
//! Consolidation: every 50 items, unbind heart1 items against heart2 items
//! to detect shared codebook concepts. Promote structurally compatible items.
//!
//! Also compares structured vs IID capacity in flat Bloom bundles.

use holographic_memory::core::entangled::EntangledHVec;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}
fn fmin(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::INFINITY, f64::min)
}
fn fmax(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Simple deterministic PRNG (splitmix64).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Codebook + structured fact generation
// ---------------------------------------------------------------------------

struct Codebook {
    concepts: Vec<EntangledHVec>,
}

impl Codebook {
    fn new(n_concepts: usize, dim: usize, density_denom: usize, seed: u64) -> Self {
        let concepts: Vec<EntangledHVec> = (0..n_concepts)
            .map(|i| EntangledHVec::new_with_density(dim, density_denom, seed + i as u64 * 997))
            .collect();
        Self { concepts }
    }

    /// Generate a structured fact: concept_a XOR concept_b.
    /// Returns (fact, index_a, index_b).
    fn make_fact(&self, rng: &mut Rng) -> (EntangledHVec, usize, usize) {
        let a = rng.next_usize(self.concepts.len());
        let mut b = rng.next_usize(self.concepts.len() - 1);
        if b >= a { b += 1; }
        let fact = self.concepts[a].bind(&self.concepts[b]);
        (fact, a, b)
    }

    /// Check if a fact decomposes into two codebook concepts.
    /// Unbind by each concept; if residual matches another concept, we found structure.
    /// Returns max similarity of best (concept_i, concept_j) decomposition.
    fn detect_structure(&self, fact: &EntangledHVec) -> f64 {
        let mut best = 0.0_f64;
        for i in 0..self.concepts.len() {
            let residual = fact.bind(&self.concepts[i]);
            for j in 0..self.concepts.len() {
                if j == i { continue; }
                let sim = self.concepts[j].similarity(&residual);
                best = best.max(sim);
                if best > 0.9 { return best; } // early exit
            }
        }
        best
    }

    /// Check if two facts share a codebook concept.
    /// fact1 = A^B, fact2 = A^C => fact1^fact2 = B^C.
    /// If we then unbind B^C by each concept and check, we can detect shared structure.
    /// Simpler: unbind fact1 by each concept, check if residual matches any concept,
    /// then do same for fact2, see if they share a concept index.
    fn shared_concepts(&self, fact: &EntangledHVec) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..self.concepts.len() {
            let residual = fact.bind(&self.concepts[i]);
            for j in 0..self.concepts.len() {
                if j == i { continue; }
                let sim = self.concepts[j].similarity(&residual);
                if sim > 0.8 {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }
}

// ---------------------------------------------------------------------------
// CLS with structure-aware consolidation
// ---------------------------------------------------------------------------

struct StructuredCls {
    dim: usize,
    heart1_items: Vec<EntangledHVec>,
    heart1_bundle: Option<EntangledHVec>,
    heart1_capacity: usize,

    heart2_items: Vec<EntangledHVec>,
    /// For each heart2 item, the set of concept indices it decomposes into.
    heart2_concepts: Vec<(usize, usize)>,

    consolidation_interval: usize,
    total_added: usize,
    total_consolidated: usize,
}

impl StructuredCls {
    fn new(dim: usize, heart1_capacity: usize, consolidation_interval: usize) -> Self {
        Self {
            dim,
            heart1_items: Vec::new(),
            heart1_bundle: None,
            heart1_capacity,
            heart2_items: Vec::new(),
            heart2_concepts: Vec::new(),
            consolidation_interval,
            total_added: 0,
            total_consolidated: 0,
        }
    }

    fn add(&mut self, item: EntangledHVec, codebook: &Codebook) {
        self.heart1_items.push(item);
        self.rebuild_heart1();
        self.total_added += 1;

        if self.total_added % self.consolidation_interval == 0 {
            self.consolidate(codebook);
        }
    }

    fn consolidate(&mut self, codebook: &Codebook) {
        if self.heart1_items.is_empty() {
            return;
        }

        let mut promoted = Vec::new();
        let mut promoted_concepts = Vec::new();
        let mut retained = Vec::new();

        for item in self.heart1_items.drain(..) {
            // Decompose item into codebook concepts
            let pairs = codebook.shared_concepts(&item);

            if pairs.is_empty() {
                // Not codebook-structured, keep in heart1
                retained.push(item);
                continue;
            }

            // Check if item shares a concept with any heart2 item
            let (ci, cj) = pairs[0]; // primary decomposition
            let mut shares_concept = false;

            if self.heart2_items.is_empty() {
                // Bootstrap: promote first structured items unconditionally
                shares_concept = true;
            } else {
                for &(h2_ci, h2_cj) in &self.heart2_concepts {
                    if ci == h2_ci || ci == h2_cj || cj == h2_ci || cj == h2_cj {
                        shares_concept = true;
                        break;
                    }
                }
            }

            if shares_concept {
                promoted.push(item);
                promoted_concepts.push((ci, cj));
            } else {
                retained.push(item);
            }
        }

        self.total_consolidated += promoted.len();
        self.heart2_items.extend(promoted);
        self.heart2_concepts.extend(promoted_concepts);

        // Enforce heart1 capacity (FIFO eviction of oldest)
        if retained.len() > self.heart1_capacity {
            let drain = retained.len() - self.heart1_capacity;
            retained.drain(..drain);
        }

        self.heart1_items = retained;
        self.rebuild_heart1();
    }

    fn rebuild_heart1(&mut self) {
        if self.heart1_items.is_empty() {
            self.heart1_bundle = None;
        } else {
            self.heart1_bundle = Some(EntangledHVec::bundle_bloom(&self.heart1_items));
        }
    }

    fn query(&self, item: &EntangledHVec) -> f64 {
        // Heart1: corrected containment against Bloom bundle
        let h1_score = match &self.heart1_bundle {
            Some(b) => item.corrected_containment(b),
            None => 0.0,
        };

        // Heart2: exact similarity search (max Jaccard across individual items)
        let h2_score = self.heart2_items.iter()
            .map(|h2| item.similarity(h2))
            .fold(0.0_f64, f64::max);

        h1_score.max(h2_score)
    }

    fn heart1_count(&self) -> usize { self.heart1_items.len() }
    fn heart2_count(&self) -> usize { self.heart2_items.len() }

    fn heart1_density(&self) -> f64 {
        match &self.heart1_bundle {
            Some(b) => b.indices().len() as f64 / self.dim as f64,
            None => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

struct Stats {
    member_mean: f64,
    member_min: f64,
    nonmember_mean: f64,
    nonmember_max: f64,
    gap: f64,
}

fn measure_flat_bloom(items: &[EntangledHVec], probes: &[EntangledHVec]) -> Stats {
    let bundle = EntangledHVec::bundle_bloom(items);
    let ms: Vec<f64> = items.iter().map(|i| i.corrected_containment(&bundle)).collect();
    let ns: Vec<f64> = probes.iter().map(|i| i.corrected_containment(&bundle)).collect();
    let mm = mean(&ms);
    let mi = fmin(&ms);
    let nm = mean(&ns);
    let nx = fmax(&ns);
    Stats { member_mean: mm, member_min: mi, nonmember_mean: nm, nonmember_max: nx, gap: mi - nx }
}

fn measure_cls(cls: &StructuredCls, items: &[EntangledHVec], probes: &[EntangledHVec]) -> Stats {
    let ms: Vec<f64> = items.iter().map(|i| cls.query(i)).collect();
    let ns: Vec<f64> = probes.iter().map(|i| cls.query(i)).collect();
    let mm = mean(&ms);
    let mi = fmin(&ms);
    let nm = mean(&ns);
    let nx = fmax(&ns);
    Stats { member_mean: mm, member_min: mi, nonmember_mean: nm, nonmember_max: nx, gap: mi - nx }
}

fn bloom_density(items: &[EntangledHVec], dim: usize) -> f64 {
    EntangledHVec::bundle_bloom(items).indices().len() as f64 / dim as f64
}

// ---------------------------------------------------------------------------
// Part 1: Structural analysis -- verify the algebra
// ---------------------------------------------------------------------------

fn run_structural_analysis() {
    let dim = 16384;
    let density_denom = 256;
    let n_concepts = 100;

    println!("================================================================");
    println!("Part 1: Structural Overlap Analysis");
    println!("================================================================");
    println!();

    let codebook = Codebook::new(n_concepts, dim, density_denom, 42);

    // Pairwise Jaccard among codebook concepts
    let mut concept_sims: Vec<f64> = Vec::new();
    for i in 0..n_concepts.min(50) {
        for j in (i+1)..n_concepts.min(50) {
            concept_sims.push(codebook.concepts[i].similarity(&codebook.concepts[j]));
        }
    }
    println!("Codebook concept pairwise Jaccard (50 concepts sampled):");
    println!("  mean={:.6}  min={:.6}  max={:.6}",
             mean(&concept_sims), fmin(&concept_sims), fmax(&concept_sims));
    println!();

    // Bind involution: (A^B)^B = A
    let fact_01 = codebook.concepts[0].bind(&codebook.concepts[1]);
    let recovered = fact_01.bind(&codebook.concepts[1]);
    println!("Bind involution: (C0^C1)^C1 vs C0");
    println!("  similarity = {:.6} (expected 1.0)", recovered.similarity(&codebook.concepts[0]));
    println!();

    // Facts sharing a concept vs not
    let fact_02 = codebook.concepts[0].bind(&codebook.concepts[2]);
    let fact_34 = codebook.concepts[3].bind(&codebook.concepts[4]);
    println!("Facts sharing concept 0:");
    println!("  (C0^C1) vs (C0^C2) Jaccard = {:.6}", fact_01.similarity(&fact_02));
    println!("Facts with no shared concept:");
    println!("  (C0^C1) vs (C3^C4) Jaccard = {:.6}", fact_01.similarity(&fact_34));
    println!();

    // Unbinding test: (A^B)^A should recover B
    let residual = fact_01.bind(&codebook.concepts[0]);
    println!("Unbinding: (C0^C1)^C0 should recover C1");
    println!("  similarity to C1 = {:.6} (expected 1.0)", residual.similarity(&codebook.concepts[1]));
    println!();

    // Structure detection via codebook decomposition
    let test_fact = codebook.concepts[5].bind(&codebook.concepts[17]);
    let pairs = codebook.shared_concepts(&test_fact);
    println!("Structure detection: fact = C5^C17");
    println!("  Detected pairs (sim > 0.8):");
    for &(i, j) in &pairs {
        println!("    unbind by C[{}] -> matches C[{}]", i, j);
    }
    println!();

    // Detection rate on random structured facts
    let mut rng = Rng::new(12345);
    let mut detected = 0;
    let n_test = 50;
    for _ in 0..n_test {
        let (fact, _, _) = codebook.make_fact(&mut rng);
        let max_sim = codebook.detect_structure(&fact);
        if max_sim > 0.8 { detected += 1; }
    }
    println!("Structure detection rate: {}/{} facts detected (threshold=0.8)",
             detected, n_test);
    println!();
}

// ---------------------------------------------------------------------------
// Part 2: Structured vs IID capacity in flat Bloom
// ---------------------------------------------------------------------------

fn run_structured_vs_iid() {
    let dim = 16384;
    let density_denom = 256;
    let n_concepts = 100;
    let n_items = 5000;
    let n_probes = 200;

    println!("================================================================");
    println!("Part 2: Structured vs IID in Flat Bloom");
    println!("================================================================");
    println!("dim={}  density=1/{}  items={}  probes={}", dim, density_denom, n_items, n_probes);
    println!();

    let codebook = Codebook::new(n_concepts, dim, density_denom, 42);
    let mut rng = Rng::new(77777);

    // Structured facts (codebook-composed)
    let structured: Vec<EntangledHVec> = (0..n_items)
        .map(|_| { let (f, _, _) = codebook.make_fact(&mut rng); f })
        .collect();

    // IID random facts (no codebook structure)
    let iid: Vec<EntangledHVec> = (0..n_items)
        .map(|i| EntangledHVec::new_with_density(dim, density_denom, i as u64 * 37 + 80000))
        .collect();

    // Random non-member probes
    let probes: Vec<EntangledHVec> = (0..n_probes)
        .map(|i| EntangledHVec::new_with_density(dim, density_denom, i as u64 * 53 + 999000))
        .collect();

    let load_points: Vec<usize> = vec![50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000];

    println!("{:<12} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
             "data_type", "n", "mem_mean", "mem_min", "nm_mean", "nm_max", "gap", "density");

    for &n in &load_points {
        if n > n_items { break; }

        let s_struct = measure_flat_bloom(&structured[..n], &probes);
        let struct_density = bloom_density(&structured[..n], dim);

        let s_iid = measure_flat_bloom(&iid[..n], &probes);
        let iid_density = bloom_density(&iid[..n], dim);

        println!("{:<12} {:>6} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>8.4} {:>8.4}",
                 "structured", n, s_struct.member_mean, s_struct.member_min,
                 s_struct.nonmember_mean, s_struct.nonmember_max, s_struct.gap, struct_density);
        println!("{:<12} {:>6} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>8.4} {:>8.4}",
                 "iid_random", n, s_iid.member_mean, s_iid.member_min,
                 s_iid.nonmember_mean, s_iid.nonmember_max, s_iid.gap, iid_density);
        println!();

        if struct_density > 0.995 && iid_density > 0.995 { break; }
    }
}

// ---------------------------------------------------------------------------
// Part 3: Full CLS with structured consolidation
// ---------------------------------------------------------------------------

fn run_cls_structured() {
    let dim = 16384;
    let density_denom = 256;
    let n_concepts = 100;
    let n_facts = 5000;
    let n_probes = 200;
    let heart1_cap = 200;
    let consol_interval = 50;

    println!("================================================================");
    println!("Part 3: CLS with Structure-Aware Consolidation");
    println!("================================================================");
    println!("dim={}  density=1/{}  concepts={}  facts={}  probes={}",
             dim, density_denom, n_concepts, n_facts, n_probes);
    println!("heart1_cap={}  consolidation_interval={}", heart1_cap, consol_interval);
    println!();

    let codebook = Codebook::new(n_concepts, dim, density_denom, 42);
    let mut rng = Rng::new(12345);

    // Generate structured facts, tracking concept indices for analysis
    let facts: Vec<(EntangledHVec, usize, usize)> = (0..n_facts)
        .map(|_| codebook.make_fact(&mut rng))
        .collect();
    let fact_vecs: Vec<EntangledHVec> = facts.iter().map(|(f, _, _)| f.clone()).collect();

    // Non-member probes: random vectors (no codebook structure)
    let probes: Vec<EntangledHVec> = (0..n_probes)
        .map(|i| EntangledHVec::new_with_density(dim, density_denom, i as u64 * 53 + 999000))
        .collect();

    let load_points: Vec<usize> = vec![50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000];

    // --- Flat Bloom baseline ---
    println!("--- Flat Bloom Baseline ---");
    println!("{:<8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
             "scheme", "n", "mem_mean", "mem_min", "nm_mean", "nm_max", "gap", "density");
    for &n in &load_points {
        if n > fact_vecs.len() { break; }
        let s = measure_flat_bloom(&fact_vecs[..n], &probes);
        let density = bloom_density(&fact_vecs[..n], dim);
        println!("{:<8} {:>6} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>8.4} {:>8.4}",
                 "flat", n, s.member_mean, s.member_min, s.nonmember_mean, s.nonmember_max, s.gap, density);
        if density > 0.995 { break; }
    }
    println!();

    // --- CLS ---
    println!("--- CLS with Structure-Aware Consolidation ---");
    println!("{:<8} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
             "scheme", "n", "h1_n", "h2_n", "mem_mean", "mem_min", "nm_mean", "nm_max", "gap", "h1_dens");

    let mut cls = StructuredCls::new(dim, heart1_cap, consol_interval);
    let mut next_lp = 0;

    for i in 0..fact_vecs.len() {
        cls.add(fact_vecs[i].clone(), &codebook);
        let n = i + 1;

        if next_lp < load_points.len() && n == load_points[next_lp] {
            next_lp += 1;
            let s = measure_cls(&cls, &fact_vecs[..n], &probes);
            println!("{:<8} {:>6} {:>6} {:>6} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>8.4} {:>8.4}",
                     "cls", n, cls.heart1_count(), cls.heart2_count(),
                     s.member_mean, s.member_min, s.nonmember_mean, s.nonmember_max, s.gap,
                     cls.heart1_density());
        }
    }
    println!();

    // --- Early fact retrieval ---
    let early_count = 100;
    println!("--- Early Fact Retrieval (first {} facts) ---", early_count);
    println!("Fraction of first {} facts still retrievable at each load point.", early_count);
    println!("Retrieval = query score > 0.05 (minimal signal above noise).");
    println!();

    let retrieval_threshold = 0.05;
    let retrieval_points: Vec<usize> = vec![100, 500, 1000, 2000, 5000];

    // Re-run CLS for retrieval measurement
    let mut cls2 = StructuredCls::new(dim, heart1_cap, consol_interval);
    let mut next_rp = 0;

    println!("{:<10} {:>6} {:>12} {:>12}", "metric", "n", "cls_frac", "flat_frac");

    for i in 0..fact_vecs.len() {
        cls2.add(fact_vecs[i].clone(), &codebook);
        let n = i + 1;

        if next_rp < retrieval_points.len() && n == retrieval_points[next_rp] {
            next_rp += 1;

            let check_n = early_count.min(n);

            // CLS retrieval of early facts
            let cls_retrieved = (0..check_n)
                .filter(|&j| cls2.query(&fact_vecs[j]) > retrieval_threshold)
                .count();
            let cls_frac = cls_retrieved as f64 / check_n as f64;

            // Flat Bloom retrieval of early facts
            let flat_bundle = EntangledHVec::bundle_bloom(&fact_vecs[..n]);
            let flat_density = flat_bundle.indices().len() as f64 / dim as f64;
            let flat_retrieved = if flat_density < 0.995 {
                (0..check_n)
                    .filter(|&j| fact_vecs[j].corrected_containment(&flat_bundle) > retrieval_threshold)
                    .count()
            } else {
                0
            };
            let flat_frac = flat_retrieved as f64 / check_n as f64;

            println!("{:<10} {:>6} {:>12.4} {:>12.4}", "early_ret", n, cls_frac, flat_frac);
        }
    }
    println!();

    // --- Capacity summary ---
    println!("--- Capacity Summary ---");
    println!("CLS total consolidated to heart2: {}", cls.total_consolidated);
    println!("CLS heart1 final count: {}", cls.heart1_count());
    println!("CLS heart2 final count: {}", cls.heart2_count());
    println!();

    // Find the load point where flat Bloom gap drops below 0
    let mut flat_capacity = 0;
    for &n in &load_points {
        if n > fact_vecs.len() { break; }
        let s = measure_flat_bloom(&fact_vecs[..n], &probes);
        if s.gap <= 0.0 {
            flat_capacity = n;
            break;
        }
    }
    // Find CLS capacity
    let mut cls3 = StructuredCls::new(dim, heart1_cap, consol_interval);
    let mut cls_capacity = 0;
    for i in 0..fact_vecs.len() {
        cls3.add(fact_vecs[i].clone(), &codebook);
        let n = i + 1;
        if load_points.contains(&n) {
            let s = measure_cls(&cls3, &fact_vecs[..n], &probes);
            if s.gap <= 0.0 {
                cls_capacity = n;
                break;
            }
        }
    }
    if flat_capacity > 0 {
        println!("Flat Bloom gap hits 0 at n={}", flat_capacity);
    } else {
        println!("Flat Bloom gap positive through n={}", load_points.last().unwrap_or(&0));
    }
    if cls_capacity > 0 {
        println!("CLS gap hits 0 at n={}", cls_capacity);
    } else {
        println!("CLS gap positive through n={}", load_points.last().unwrap_or(&0));
    }
    if flat_capacity > 0 && cls_capacity > 0 {
        println!("Capacity ratio (CLS/flat): {:.1}x", cls_capacity as f64 / flat_capacity as f64);
    } else if flat_capacity > 0 && cls_capacity == 0 {
        println!("CLS still has positive gap at max load -- capacity > {:.1}x flat",
                 *load_points.last().unwrap_or(&1) as f64 / flat_capacity as f64);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    println!("CLS-Structured: Complementary Learning Systems with Codebook-Composed Facts");
    println!("============================================================================");
    println!();

    // Part 1: verify the algebra works
    run_structural_analysis();
    println!();

    // Part 2: structured vs IID capacity in flat Bloom
    run_structured_vs_iid();
    println!();

    // Part 3: full CLS with structured consolidation
    run_cls_structured();
}
