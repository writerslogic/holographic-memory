use crate::core::entangled::hash_u64;

#[derive(Clone, Copy)]
pub struct BlockCodeConfig {
    pub n_blocks: usize,
    pub block_len: usize,
}

impl BlockCodeConfig {
    pub fn new(n_blocks: usize, block_len: usize) -> Self {
        Self {
            n_blocks,
            block_len,
        }
    }

    pub fn default_config() -> Self {
        Self {
            n_blocks: 1024,
            block_len: 16,
        }
    }

    pub fn dim(&self) -> usize {
        self.n_blocks * self.block_len
    }

    pub fn active_count(&self) -> usize {
        self.n_blocks
    }
}

#[derive(Clone, Debug)]
pub struct BlockCodeVec {
    n_blocks: usize,
    block_len: usize,
    winners: Vec<u32>,
}

impl BlockCodeVec {
    pub fn new_deterministic(cfg: &BlockCodeConfig, seed: u64) -> Self {
        let mut winners = Vec::with_capacity(cfg.n_blocks);
        for b in 0..cfg.n_blocks {
            let h = hash_u64(seed, b as u64);
            winners.push((h % cfg.block_len as u64) as u32);
        }
        Self {
            n_blocks: cfg.n_blocks,
            block_len: cfg.block_len,
            winners,
        }
    }

    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    pub fn block_len(&self) -> usize {
        self.block_len
    }

    pub fn winner(&self, block: usize) -> u32 {
        self.winners[block]
    }

    pub fn to_global_indices(&self) -> Vec<u32> {
        self.winners
            .iter()
            .enumerate()
            .map(|(b, &w)| (b * self.block_len) as u32 + w)
            .collect()
    }

    pub fn similarity(&self, other: &Self) -> f64 {
        let matching = self
            .winners
            .iter()
            .zip(&other.winners)
            .filter(|(&a, &b)| a == b)
            .count();
        matching as f64 / self.n_blocks as f64
    }

    pub fn bind(&self, other: &Self) -> Self {
        let winners = self
            .winners
            .iter()
            .zip(&other.winners)
            .map(|(&a, &b)| (a + b) % self.block_len as u32)
            .collect();
        Self {
            n_blocks: self.n_blocks,
            block_len: self.block_len,
            winners,
        }
    }

    pub fn unbind(&self, other: &Self) -> Self {
        let bl = self.block_len as u32;
        let winners = self
            .winners
            .iter()
            .zip(&other.winners)
            .map(|(&a, &b)| (a + bl - b) % bl)
            .collect();
        Self {
            n_blocks: self.n_blocks,
            block_len: self.block_len,
            winners,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BlockCodeBundle {
    n_blocks: usize,
    block_len: usize,
    counts: Vec<Vec<u32>>,
    n_items: usize,
}

impl BlockCodeBundle {
    pub fn new(cfg: &BlockCodeConfig) -> Self {
        Self {
            n_blocks: cfg.n_blocks,
            block_len: cfg.block_len,
            counts: vec![vec![0u32; cfg.block_len]; cfg.n_blocks],
            n_items: 0,
        }
    }

    pub fn add(&mut self, vec: &BlockCodeVec) {
        for (b, &w) in vec.winners.iter().enumerate() {
            self.counts[b][w as usize] += 1;
        }
        self.n_items += 1;
    }

    pub fn n_items(&self) -> usize {
        self.n_items
    }

    pub fn to_winner_vec(&self) -> BlockCodeVec {
        let winners: Vec<u32> = self
            .counts
            .iter()
            .map(|block| {
                block
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, &count)| count)
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0)
            })
            .collect();
        BlockCodeVec {
            n_blocks: self.n_blocks,
            block_len: self.block_len,
            winners,
        }
    }

    pub fn count_score(&self, vec: &BlockCodeVec) -> f64 {
        let raw: u64 = vec
            .winners
            .iter()
            .enumerate()
            .map(|(b, &w)| self.counts[b][w as usize] as u64)
            .sum();
        raw as f64 / self.n_blocks as f64
    }

    pub fn corrected_similarity(&self, vec: &BlockCodeVec) -> f64 {
        let raw_per_block = self.count_score(vec);
        let expected = self.n_items as f64 / self.block_len as f64;
        let denom = 1.0 - expected;
        if denom <= 0.0 {
            return 0.0;
        }
        ((raw_per_block - expected) / denom).clamp(0.0, 1.0)
    }

    pub fn rotated_counts(&self, cue: &BlockCodeVec) -> Vec<Vec<u32>> {
        let bl = self.block_len;
        self.counts
            .iter()
            .enumerate()
            .map(|(b, block_counts)| {
                let shift = cue.winners[b] as usize;
                (0..bl)
                    .map(|pos| {
                        let src = (pos + shift) % bl;
                        block_counts[src]
                    })
                    .collect()
            })
            .collect()
    }

    pub fn unbind_and_decode(&self, cue: &BlockCodeVec) -> BlockCodeVec {
        let bl = self.block_len;
        let winners: Vec<u32> = self
            .counts
            .iter()
            .enumerate()
            .map(|(b, block_counts)| {
                let shift = cue.winners[b] as usize;
                (0..bl)
                    .max_by_key(|&pos| {
                        let src = (pos + shift) % bl;
                        block_counts[src]
                    })
                    .unwrap_or(0) as u32
            })
            .collect();
        BlockCodeVec {
            n_blocks: self.n_blocks,
            block_len: self.block_len,
            winners,
        }
    }

    pub fn recover(&self, cue: &BlockCodeVec, codebook: &[BlockCodeVec]) -> Option<usize> {
        if codebook.is_empty() {
            return None;
        }
        let decoded = self.unbind_and_decode(cue);
        let (best_idx, _) = codebook
            .iter()
            .enumerate()
            .map(|(i, v)| (i, decoded.similarity(v)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-empty codebook");
        Some(best_idx)
    }

    pub fn recover_by_score(&self, cue: &BlockCodeVec, codebook: &[BlockCodeVec]) -> Option<usize> {
        if codebook.is_empty() {
            return None;
        }
        let (best_idx, _) = codebook
            .iter()
            .enumerate()
            .map(|(i, candidate)| {
                let composed = cue.bind(candidate);
                (i, self.count_score(&composed))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("non-empty codebook");
        Some(best_idx)
    }

    pub fn resonator_recover(
        &self,
        cue: &BlockCodeVec,
        codebook: &[BlockCodeVec],
        max_iters: usize,
    ) -> ResonatorResult {
        if codebook.is_empty() {
            return ResonatorResult {
                best_idx: None,
                iterations: 0,
                converged: false,
            };
        }

        let bl = self.block_len;
        let mut estimate = self.unbind_and_decode(cue);
        let mut prev_idx = None;
        let mut stable_count = 0u32;

        for iter in 0..max_iters {
            let (best_idx, best_sim) = codebook
                .iter()
                .enumerate()
                .map(|(i, v)| (i, estimate.similarity(v)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            if Some(best_idx) == prev_idx {
                stable_count += 1;
                if stable_count >= 2 {
                    return ResonatorResult {
                        best_idx: Some(best_idx),
                        iterations: iter + 1,
                        converged: true,
                    };
                }
            } else {
                stable_count = 0;
            }
            prev_idx = Some(best_idx);

            if (best_sim - 1.0).abs() < f64::EPSILON {
                return ResonatorResult {
                    best_idx: Some(best_idx),
                    iterations: iter + 1,
                    converged: true,
                };
            }

            let rebound = codebook[best_idx].bind(cue);
            let rotated_counts: Vec<Vec<u32>> = self
                .counts
                .iter()
                .enumerate()
                .map(|(b, block_counts)| {
                    let shift = rebound.winners[b] as usize;
                    let mut rotated = vec![0u32; bl];
                    for pos in 0..bl {
                        rotated[pos] = block_counts[(pos + shift) % bl];
                    }
                    rotated
                })
                .collect();

            let new_winners: Vec<u32> = rotated_counts
                .iter()
                .map(|block| {
                    block
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, &c)| c)
                        .map(|(idx, _)| idx as u32)
                        .unwrap_or(0)
                })
                .collect();

            let argmax_from_rotated = BlockCodeVec {
                n_blocks: self.n_blocks,
                block_len: self.block_len,
                winners: new_winners,
            };

            let combined_winners: Vec<u32> = estimate
                .winners
                .iter()
                .zip(&argmax_from_rotated.winners)
                .enumerate()
                .map(|(b, (&est, &fresh))| {
                    let est_count = rotated_counts[b][est as usize];
                    let fresh_count = rotated_counts[b][fresh as usize];
                    if fresh_count > est_count {
                        fresh
                    } else {
                        est
                    }
                })
                .collect();

            estimate = BlockCodeVec {
                n_blocks: self.n_blocks,
                block_len: self.block_len,
                winners: combined_winners,
            };
        }

        ResonatorResult {
            best_idx: prev_idx,
            iterations: max_iters,
            converged: false,
        }
    }
}

pub struct ResonatorResult {
    pub best_idx: Option<usize>,
    pub iterations: usize,
    pub converged: bool,
}

fn hash_name(name: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in name.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub struct BlockCodeStore {
    cfg: BlockCodeConfig,
    names: Vec<String>,
    name_to_idx: std::collections::HashMap<String, usize>,
    codebook: Vec<BlockCodeVec>,
    bundle: BlockCodeBundle,
}

impl BlockCodeStore {
    pub fn new(cfg: BlockCodeConfig) -> Self {
        let bundle = BlockCodeBundle::new(&cfg);
        Self {
            cfg,
            names: Vec::new(),
            name_to_idx: std::collections::HashMap::new(),
            codebook: Vec::new(),
            bundle,
        }
    }

    pub fn register(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.name_to_idx.get(name) {
            return idx;
        }
        let idx = self.codebook.len();
        let seed = hash_name(name);
        let vec = BlockCodeVec::new_deterministic(&self.cfg, seed);
        self.codebook.push(vec);
        self.names.push(name.to_string());
        self.name_to_idx.insert(name.to_string(), idx);
        idx
    }

    pub fn memorize_triplet(&mut self, head: &str, relation: &str, tail: &str) {
        let hi = self.register(head);
        let ri = self.register(relation);
        let ti = self.register(tail);
        let composed = self.codebook[hi]
            .bind(&self.codebook[ri])
            .bind(&self.codebook[ti]);
        self.bundle.add(&composed);
    }

    pub fn query_triplet(&self, head: &str, relation: &str) -> Option<&str> {
        let hi = *self.name_to_idx.get(head)?;
        let ri = *self.name_to_idx.get(relation)?;
        let cue = self.codebook[hi].bind(&self.codebook[ri]);
        let best = self.bundle.recover_by_score(&cue, &self.codebook)?;
        Some(&self.names[best])
    }

    pub fn n_facts(&self) -> usize {
        self.bundle.n_items()
    }

    pub fn n_symbols(&self) -> usize {
        self.codebook.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_code_self_similarity() {
        let cfg = BlockCodeConfig::new(256, 64);
        let v = BlockCodeVec::new_deterministic(&cfg, 42);
        assert!((v.similarity(&v) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_block_code_random_similarity() {
        let cfg = BlockCodeConfig::new(256, 64);
        let a = BlockCodeVec::new_deterministic(&cfg, 1);
        let b = BlockCodeVec::new_deterministic(&cfg, 2);
        let sim = a.similarity(&b);
        assert!(
            sim < 0.05,
            "Random pair similarity {:.4} should be near 1/L = {:.4}",
            sim,
            1.0 / 64.0
        );
    }

    #[test]
    fn test_block_code_bind_unbind() {
        let cfg = BlockCodeConfig::new(256, 64);
        let a = BlockCodeVec::new_deterministic(&cfg, 1);
        let b = BlockCodeVec::new_deterministic(&cfg, 2);
        let ab = a.bind(&b);
        let recovered = ab.unbind(&b);
        assert!(
            (a.similarity(&recovered) - 1.0).abs() < f64::EPSILON,
            "Unbind should perfectly recover original"
        );
    }

    #[test]
    fn test_block_code_bundle_small() {
        let cfg = BlockCodeConfig::new(256, 64);
        let mut bundle = BlockCodeBundle::new(&cfg);
        let v = BlockCodeVec::new_deterministic(&cfg, 42);
        bundle.add(&v);
        assert!((bundle.corrected_similarity(&v) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_block_code_bundle_capacity() {
        let cfg = BlockCodeConfig::new(64, 256);
        let items: Vec<BlockCodeVec> = (0..100)
            .map(|i| BlockCodeVec::new_deterministic(&cfg, i * 37 + 1))
            .collect();
        let mut bundle = BlockCodeBundle::new(&cfg);
        for item in &items {
            bundle.add(item);
        }
        let min_sim = items
            .iter()
            .map(|item| bundle.corrected_similarity(item))
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_sim > 0.5,
            "100 items in 64x256: min corrected similarity {:.4} should be >0.5",
            min_sim
        );
    }

    #[test]
    fn test_default_config() {
        let cfg = BlockCodeConfig::default_config();
        assert_eq!(cfg.n_blocks, 1024);
        assert_eq!(cfg.block_len, 16);
        assert_eq!(cfg.dim(), 16384);
    }

    #[test]
    fn test_unbind_and_decode_single_fact() {
        let cfg = BlockCodeConfig::default_config();
        let country = BlockCodeVec::new_deterministic(&cfg, 1);
        let relation = BlockCodeVec::new_deterministic(&cfg, 2);
        let capital = BlockCodeVec::new_deterministic(&cfg, 3);

        let fact = country.bind(&relation).bind(&capital);
        let mut bundle = BlockCodeBundle::new(&cfg);
        bundle.add(&fact);

        let cue = country.bind(&relation);
        let recovered = bundle.unbind_and_decode(&cue);
        assert!(
            (recovered.similarity(&capital) - 1.0).abs() < f64::EPSILON,
            "Single fact recovery must be perfect"
        );
    }

    #[test]
    fn test_recover_from_codebook() {
        let cfg = BlockCodeConfig::default_config();
        let codebook: Vec<BlockCodeVec> = (0..100)
            .map(|i| BlockCodeVec::new_deterministic(&cfg, i as u64 * 31 + 7))
            .collect();
        let relation = BlockCodeVec::new_deterministic(&cfg, 999);

        let n_facts = 5;
        let mut bundle = BlockCodeBundle::new(&cfg);
        let mut fact_pairs = Vec::new();
        for i in 0..n_facts {
            let ci = i * 2;
            let ki = i * 2 + 1;
            let fact = codebook[ci].bind(&relation).bind(&codebook[ki]);
            bundle.add(&fact);
            fact_pairs.push((ci, ki));
        }

        let mut correct = 0;
        for &(ci, ki) in &fact_pairs {
            let cue = codebook[ci].bind(&relation);
            if let Some(recovered_idx) = bundle.recover(&cue, &codebook) {
                if recovered_idx == ki {
                    correct += 1;
                }
            }
        }
        let accuracy = correct as f64 / n_facts as f64;
        assert!(
            (accuracy - 1.0).abs() < f64::EPSILON,
            "5 non-overlapping facts in 1024x16: recovery accuracy {:.0}% should be 100%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_recovery_at_capacity() {
        let cfg = BlockCodeConfig::default_config();
        let codebook: Vec<BlockCodeVec> = (0..200)
            .map(|i| BlockCodeVec::new_deterministic(&cfg, i as u64 * 31 + 7))
            .collect();
        let relation = BlockCodeVec::new_deterministic(&cfg, 999999);

        let n_facts = 10;
        let mut bundle = BlockCodeBundle::new(&cfg);
        let mut fact_pairs = Vec::new();
        for i in 0..n_facts {
            let ci = i * 2;
            let ki = i * 2 + 1;
            let fact = codebook[ci].bind(&relation).bind(&codebook[ki]);
            bundle.add(&fact);
            fact_pairs.push((ci, ki));
        }

        let mut correct = 0;
        for &(ci, ki) in &fact_pairs {
            let cue = codebook[ci].bind(&relation);
            if let Some(idx) = bundle.recover(&cue, &codebook) {
                if idx == ki {
                    correct += 1;
                }
            }
        }
        let accuracy = correct as f64 / n_facts as f64;
        assert!(
            accuracy >= 0.90,
            "10 facts in 1024x16: recovery accuracy {:.0}% should be >= 90%",
            accuracy * 100.0
        );
    }

    #[test]
    fn test_block_code_store_basic() {
        let cfg = BlockCodeConfig::default_config();
        let mut store = BlockCodeStore::new(cfg);

        store.memorize_triplet("Sarah", "loves", "Marcus");
        store.memorize_triplet("Marcus", "caused", "fire");
        store.memorize_triplet("Sarah", "lives_in", "Chicago");
        store.memorize_triplet("Marcus", "works_at", "factory");
        store.memorize_triplet("Sarah", "sister_of", "Elena");

        assert_eq!(store.query_triplet("Sarah", "loves"), Some("Marcus"));
        assert_eq!(store.query_triplet("Marcus", "caused"), Some("fire"));
        assert_eq!(store.query_triplet("Sarah", "lives_in"), Some("Chicago"));
        assert_eq!(store.query_triplet("Marcus", "works_at"), Some("factory"));
        assert_eq!(store.query_triplet("Sarah", "sister_of"), Some("Elena"));
        assert_eq!(store.query_triplet("nobody", "anything"), None);
        assert_eq!(store.n_facts(), 5);
    }

    #[test]
    fn test_block_code_store_shared_symbols() {
        let cfg = BlockCodeConfig::default_config();
        let mut store = BlockCodeStore::new(cfg);

        let characters = [
            "Sarah", "Marcus", "Elena", "James", "Sofia", "Dante", "Lila", "Kai", "Nora", "Victor",
        ];
        let relations = [
            "loves", "hates", "trusts", "fears", "helps", "betrayed", "forgave", "taught", "saved",
            "followed",
        ];
        let objects = [
            "truth", "fire", "knife", "letter", "house", "bridge", "secret", "money", "child",
            "ring",
        ];

        let mut facts = Vec::new();
        for (i, &subj) in characters.iter().enumerate() {
            for j in 0..relations.len() {
                let rel = relations[(i + j) % relations.len()];
                let obj = objects[(i * 3 + j * 7) % objects.len()];
                facts.push((subj, rel, obj));
                store.memorize_triplet(subj, rel, obj);
            }
        }

        let n = facts.len();
        let mut correct = 0;
        for &(subj, rel, expected_obj) in &facts {
            if store.query_triplet(subj, rel) == Some(expected_obj) {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / n as f64;
        assert!(
            accuracy >= 0.90,
            "Shared-symbol store: {}/{} = {:.1}% should be >= 90%",
            correct,
            n,
            accuracy * 100.0
        );
    }
}
