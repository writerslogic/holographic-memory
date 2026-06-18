// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Background cognition loop.
//!
//! Runs pattern scanning, abstraction, gap detection, hypothesis generation,
//! and analogy detection on a configurable interval. Uses READ-ONLY locks on
//! meaning memory stores. Discovered insights are collected in a separate Vec.

use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::core::atom_memory::AtomMemory;
use crate::core::cognition::abstraction::{Abstraction, AbstractionEngine};
use crate::core::cognition::analogy::{Analogy, AnalogyDetector};
use crate::core::cognition::gaps::{GapDetector, KnowledgeGap};
use crate::core::cognition::hypothesis::{Hypothesis, HypothesisEngine};
use crate::core::cognition::patterns::{PatternScanner, RelationPattern};
use crate::core::triple_store::TripleStore;

/// A single insight discovered by the cognition loop.
#[derive(Clone, Debug)]
pub enum Insight {
    Pattern(RelationPattern),
    Abstraction(Abstraction),
    Gap(KnowledgeGap),
    Hypothesis(Hypothesis),
    Analogy(Analogy),
}

/// Configuration for the cognition loop.
#[derive(Clone, Debug)]
pub struct CognitionConfig {
    /// Interval between cognition cycles.
    pub interval: Duration,
    /// Minimum frequency for a relation pattern to be reported.
    pub min_pattern_freq: usize,
    /// Minimum members to form an abstraction.
    pub min_abstraction_members: usize,
    /// Minimum shared relations to consider entities as peers (gap detection).
    pub min_shared_relations: usize,
    /// Minimum fraction of peers that must have a relation for it to be a gap.
    pub min_peer_coverage: f64,
    /// Hopfield beta for hypothesis cleanup.
    pub hypothesis_beta: f64,
    /// Minimum confidence for a hypothesis to be reported.
    pub min_hypothesis_confidence: f64,
    /// Minimum shared relations for analogy detection.
    pub min_analogy_relations: usize,
}

impl Default for CognitionConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60),
            min_pattern_freq: 3,
            min_abstraction_members: 3,
            min_shared_relations: 2,
            min_peer_coverage: 0.5,
            hypothesis_beta: 24.0,
            min_hypothesis_confidence: 0.3,
            min_analogy_relations: 2,
        }
    }
}

/// Shared state for the cognition loop: insights and run flag.
pub struct CognitionState {
    insights: RwLock<Vec<Insight>>,
    running: AtomicBool,
    stop: AtomicBool,
    cycle_count: std::sync::atomic::AtomicU64,
}

impl CognitionState {
    fn new() -> Self {
        Self {
            insights: RwLock::new(Vec::new()),
            running: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            cycle_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn cycle_count(&self) -> u64 {
        self.cycle_count.load(Ordering::SeqCst)
    }

    pub fn take_insights(&self) -> Vec<Insight> {
        std::mem::take(&mut *self.insights.write())
    }

    pub fn insight_count(&self) -> usize {
        self.insights.read().len()
    }
}

/// The cognition loop. Spawns a background thread that periodically
/// scans the triple store for patterns, gaps, and analogies.
pub struct CognitionLoop {
    state: Arc<CognitionState>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl CognitionLoop {
    /// Start the cognition loop in a background thread.
    ///
    /// The loop acquires read-only references to the atom memory and triple store.
    /// It does NOT hold locks across cycles; it snapshots and releases.
    pub fn start(
        atom_memory: Arc<AtomMemory>,
        triple_store: Arc<TripleStore>,
        config: CognitionConfig,
    ) -> Self {
        let state = Arc::new(CognitionState::new());
        let thread_state = Arc::clone(&state);

        state.running.store(true, Ordering::SeqCst);

        let handle = std::thread::Builder::new()
            .name("hms-cognition".to_string())
            .spawn(move || {
                cognition_thread(&thread_state, &atom_memory, &triple_store, &config);
            })
            .expect("failed to spawn cognition thread");

        Self {
            state,
            handle: Some(handle),
        }
    }

    /// Request the loop to stop and wait for the thread to finish.
    pub fn stop(&mut self) {
        self.state.stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.state.running.store(false, Ordering::SeqCst);
    }

    /// Access the shared state (insights, cycle count, etc.).
    pub fn state(&self) -> &Arc<CognitionState> {
        &self.state
    }

    /// Run a single cognition cycle synchronously (for testing).
    pub fn run_once(
        atom_memory: &AtomMemory,
        triple_store: &TripleStore,
        config: &CognitionConfig,
    ) -> Vec<Insight> {
        run_cycle(atom_memory, triple_store, config)
    }
}

impl Drop for CognitionLoop {
    fn drop(&mut self) {
        self.stop();
    }
}

fn cognition_thread(
    state: &CognitionState,
    atom_memory: &AtomMemory,
    triple_store: &TripleStore,
    config: &CognitionConfig,
) {
    while !state.stop.load(Ordering::SeqCst) {
        let new_insights = run_cycle(atom_memory, triple_store, config);

        if !new_insights.is_empty() {
            state.insights.write().extend(new_insights);
        }

        state.cycle_count.fetch_add(1, Ordering::SeqCst);

        // Sleep in small increments so we can respond to stop quickly
        let mut remaining = config.interval;
        let tick = Duration::from_millis(100);
        while remaining > Duration::ZERO && !state.stop.load(Ordering::SeqCst) {
            let sleep_time = remaining.min(tick);
            std::thread::sleep(sleep_time);
            remaining = remaining.saturating_sub(sleep_time);
        }
    }

    state.running.store(false, Ordering::SeqCst);
}

fn run_cycle(
    atom_memory: &AtomMemory,
    triple_store: &TripleStore,
    config: &CognitionConfig,
) -> Vec<Insight> {
    let mut insights = Vec::new();

    // 1. Pattern scanning
    let patterns = PatternScanner::scan_relation_patterns(triple_store, config.min_pattern_freq);
    for p in patterns {
        insights.push(Insight::Pattern(p));
    }

    // 2. Abstraction
    let abstractions =
        AbstractionEngine::discover(triple_store, atom_memory, config.min_abstraction_members);
    for a in abstractions {
        insights.push(Insight::Abstraction(a));
    }

    // 3. Gap detection
    let gaps = GapDetector::detect(
        triple_store,
        config.min_shared_relations,
        config.min_peer_coverage,
    );

    // 4. Hypothesis generation (from gaps)
    let hypotheses = HypothesisEngine::propose(
        &gaps,
        triple_store,
        atom_memory,
        config.hypothesis_beta,
        config.min_hypothesis_confidence,
    );
    for h in hypotheses {
        insights.push(Insight::Hypothesis(h));
    }

    // Push gaps after hypotheses so hypotheses reference valid gaps
    for g in gaps {
        insights.push(Insight::Gap(g));
    }

    // 5. Analogy detection
    let analogies = AnalogyDetector::detect(triple_store, config.min_analogy_relations);
    for a in analogies {
        insights.push(Insight::Analogy(a));
    }

    insights
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (AtomMemory, TripleStore) {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let triple_store = TripleStore::new();

        for city in &["paris", "berlin", "tokyo", "london", "madrid"] {
            atom_mem.get_or_insert(city);
        }
        for country in &["france", "germany", "japan", "uk", "spain", "europe"] {
            atom_mem.get_or_insert(country);
        }

        triple_store.add("paris", "capital_of", "france", "c1");
        triple_store.add("berlin", "capital_of", "germany", "c2");
        triple_store.add("tokyo", "capital_of", "japan", "c3");
        triple_store.add("london", "capital_of", "uk", "c4");
        triple_store.add("madrid", "capital_of", "spain", "c5");
        triple_store.add("paris", "located_in", "europe", "c6");
        triple_store.add("berlin", "located_in", "europe", "c7");
        triple_store.add("london", "located_in", "europe", "c8");
        triple_store.add("madrid", "located_in", "europe", "c9");
        // tokyo missing located_in

        (atom_mem, triple_store)
    }

    #[test]
    fn test_run_once() {
        let (atom_mem, triple_store) = make_test_data();
        let config = CognitionConfig::default();
        let insights = CognitionLoop::run_once(&atom_mem, &triple_store, &config);
        assert!(!insights.is_empty());

        // Should find patterns
        assert!(insights.iter().any(|i| matches!(i, Insight::Pattern(_))));
    }

    #[test]
    fn test_run_once_finds_gaps() {
        let (atom_mem, triple_store) = make_test_data();
        let config = CognitionConfig {
            min_shared_relations: 1,
            min_peer_coverage: 0.5,
            ..Default::default()
        };
        let insights = CognitionLoop::run_once(&atom_mem, &triple_store, &config);

        let has_gap = insights.iter().any(|i| {
            if let Insight::Gap(g) = i {
                g.entity == "tokyo" && g.missing_relation == "located_in"
            } else {
                false
            }
        });
        assert!(has_gap, "Should detect tokyo missing located_in");
    }

    #[test]
    fn test_background_loop_start_stop() {
        let (atom_mem, triple_store) = make_test_data();
        let config = CognitionConfig {
            interval: Duration::from_millis(50),
            ..Default::default()
        };

        let mut cognition =
            CognitionLoop::start(Arc::new(atom_mem), Arc::new(triple_store), config);

        assert!(cognition.state().is_running());

        // Let it run a couple cycles
        std::thread::sleep(Duration::from_millis(200));

        let cycles = cognition.state().cycle_count();
        assert!(
            cycles >= 1,
            "Should have completed at least 1 cycle, got {}",
            cycles
        );

        cognition.stop();
        assert!(!cognition.state().is_running());
    }

    #[test]
    fn test_take_insights() {
        let (atom_mem, triple_store) = make_test_data();
        let config = CognitionConfig {
            interval: Duration::from_millis(50),
            ..Default::default()
        };

        let mut cognition =
            CognitionLoop::start(Arc::new(atom_mem), Arc::new(triple_store), config);

        std::thread::sleep(Duration::from_millis(200));

        let insights = cognition.state().take_insights();
        assert!(!insights.is_empty());

        // After take, count should be 0 (or only new ones from ongoing cycle)
        // Stop first to prevent race
        cognition.stop();
        let remaining = cognition.state().take_insights();
        // This is fine - just checking take works
        let _ = remaining;
    }

    #[test]
    fn test_empty_store_no_panic() {
        let atom_mem = AtomMemory::new(16384, 3.0);
        let triple_store = TripleStore::new();
        let config = CognitionConfig::default();
        let insights = CognitionLoop::run_once(&atom_mem, &triple_store, &config);
        assert!(insights.is_empty());
    }
}
