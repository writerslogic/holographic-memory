use crate::core::entangled::EntangledHVec;

pub struct ClsConfig {
    pub dim: usize,
    pub density_denom: usize,
    pub consolidation_interval: usize,
    pub resonance_threshold: f64,
    pub heart1_capacity: usize,
}

impl ClsConfig {
    pub fn new(dim: usize, density_denom: usize) -> Self {
        Self {
            dim,
            density_denom,
            consolidation_interval: 50,
            resonance_threshold: 0.3,
            heart1_capacity: 200,
        }
    }
}

pub struct ClsMemory {
    cfg: ClsConfig,
    heart1_items: Vec<EntangledHVec>,
    heart1_bundle: Option<EntangledHVec>,
    heart2_items: Vec<EntangledHVec>,
    heart2_bundle: Option<EntangledHVec>,
    total_added: usize,
    total_consolidated: usize,
}

impl ClsMemory {
    pub fn new(cfg: ClsConfig) -> Self {
        Self {
            cfg,
            heart1_items: Vec::new(),
            heart1_bundle: None,
            heart2_items: Vec::new(),
            heart2_bundle: None,
            total_added: 0,
            total_consolidated: 0,
        }
    }

    pub fn add(&mut self, item: EntangledHVec) {
        self.heart1_items.push(item);
        self.rebuild_heart1();
        self.total_added += 1;

        if self.total_added % self.cfg.consolidation_interval == 0 {
            self.consolidate();
        }
    }

    fn consolidate(&mut self) {
        if self.heart1_items.is_empty() {
            return;
        }

        let h1_bundle = self.heart1_bundle.clone();
        let mut promoted = Vec::new();
        let mut retained = Vec::new();

        for item in self.heart1_items.drain(..) {
            let resonates = match &self.heart2_bundle {
                Some(h2) => item.corrected_containment(h2) > self.cfg.resonance_threshold,
                None => true,
            };

            let reinforced = match &h1_bundle {
                Some(b) => item.corrected_containment(b) > 0.5,
                None => false,
            };

            if resonates || reinforced {
                promoted.push(item);
            } else {
                retained.push(item);
            }
        }

        for item in &promoted {
            self.heart2_items.push(item.clone());
        }
        self.total_consolidated += promoted.len();

        if retained.len() > self.cfg.heart1_capacity {
            let drain = retained.len() - self.cfg.heart1_capacity;
            retained.drain(..drain);
        }

        self.heart1_items = retained;
        self.rebuild_heart1();
        self.rebuild_heart2();
    }

    fn rebuild_heart1(&mut self) {
        if self.heart1_items.is_empty() {
            self.heart1_bundle = None;
        } else {
            self.heart1_bundle = Some(EntangledHVec::bundle_bloom(&self.heart1_items));
        }
    }

    fn rebuild_heart2(&mut self) {
        if self.heart2_items.is_empty() {
            self.heart2_bundle = None;
        } else {
            self.heart2_bundle = Some(EntangledHVec::bundle_bloom(&self.heart2_items));
        }
    }

    pub fn query(&self, item: &EntangledHVec) -> f64 {
        let h1_score = match &self.heart1_bundle {
            Some(b) => item.corrected_containment(b),
            None => 0.0,
        };
        let h2_score = match &self.heart2_bundle {
            Some(b) => item.corrected_containment(b),
            None => 0.0,
        };
        h1_score.max(h2_score)
    }

    pub fn heart1_count(&self) -> usize {
        self.heart1_items.len()
    }

    pub fn heart2_count(&self) -> usize {
        self.heart2_items.len()
    }

    pub fn heart1_density(&self) -> f64 {
        match &self.heart1_bundle {
            Some(b) => b.indices().len() as f64 / self.cfg.dim as f64,
            None => 0.0,
        }
    }

    pub fn heart2_density(&self) -> f64 {
        match &self.heart2_bundle {
            Some(b) => b.indices().len() as f64 / self.cfg.dim as f64,
            None => 0.0,
        }
    }

    pub fn total_consolidated(&self) -> usize {
        self.total_consolidated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cls_single_item() {
        let cfg = ClsConfig::new(16384, 256);
        let mut mem = ClsMemory::new(cfg);
        let item = EntangledHVec::new_deterministic(16384, 42);
        mem.add(item.clone());
        assert!(mem.query(&item) > 0.9);
    }

    #[test]
    fn test_cls_consolidation_runs() {
        let mut cfg = ClsConfig::new(16384, 256);
        cfg.consolidation_interval = 10;
        let mut mem = ClsMemory::new(cfg);
        for i in 0..20 {
            mem.add(EntangledHVec::new_deterministic(16384, i * 37 + 1));
        }
        assert!(mem.total_consolidated() > 0);
    }
}
