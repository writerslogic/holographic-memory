// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Smallest falsifiable slice of the living-connection-graph direction
// (memory: living-connection-graph-direction). Tests two claims at once:
//
//  1. PLASTICITY vs SATURATION (the scientific kill condition). A static
//     superposition of N relations saturates: at high N a mis-bound probe scores
//     as high as a correct one and discrimination (AUC) collapses (measured in
//     docs/PREREGISTRATION-binding-readout.md §14). Claim: if querying reshapes
//     the memory -- strengthen the traversed relation, decay the untouched ones
//     -- then frequently-queried ("hot") relations stay discriminable at loads
//     where the static field is already dead, because the cold background decays
//     out and the effective load stays low.
//     KILL: if plastic hot-relation AUC is not materially above static at high N,
//     plasticity is decorative -- drop it.
//
//  2. VERIFIABLE MUTATION (the systems kill condition). Every state change
//     (store / strengthen / decay) is an ordered event in a hash-chained log.
//     The field is a DETERMINISTIC integer fold of that log, so replay
//     reconstructs the exact state and the chain is tamper-evident. This is the
//     resolution to "mutate-on-read vs provenance": the observer effect is an
//     auditable event, not a hidden write. (Integers => bit-exact replay across
//     platforms; the chain hash here is a non-crypto stand-in for the SHA-256 +
//     Ed25519-checkpoint layer HMS's provenance stack already provides.)
//     KILL: if replay is not bit-exact or the per-op event cost is unbounded.
//
// Run: cargo run --release --bin plastic-graph

use holographic_memory::EntangledHVec;

const D: usize = 16_384;
const DENSITY_DENOM: usize = 256; // active per symbol = 64
const N_SYMBOLS: usize = 2048;
const ROLE1_MASK: u32 = 0x0A5A;
const ROLE2_MASK: u32 = 0x15A5;
const HOT: usize = 24; // relations that get queried
const WARMUP_ROUNDS: usize = 40; // strengthen passes over the hot set
const W_STORE: i64 = 1;
const W_STRENGTHEN: i64 = 2;
const DECAY_NUM: i64 = 7; // multiplicative decay 7/8 per interval (integer, exact)
const DECAY_DEN: i64 = 8;
const DECAY_INTERVAL: usize = HOT; // decay once per full pass over the hot set
const SEEDS: u64 = 12;
const LOADS: &[usize] = &[64, 128, 256, 512, 1024, 2048];

fn mix(x: u64, k: u64) -> u64 {
    let mut h = x.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    h
}

struct Fact {
    a: usize,
    b: usize,
}

fn build_facts(n: usize, seed: u64) -> Vec<Fact> {
    (0..n)
        .map(|i| {
            let a = (mix(i as u64, seed ^ 0xF11E) % N_SYMBOLS as u64) as usize;
            let mut b = (mix(i as u64, seed ^ 0xB0B0) % N_SYMBOLS as u64) as usize;
            if b == a {
                b = (b + 1) % N_SYMBOLS;
            }
            Fact { a, b }
        })
        .collect()
}

/// Permutation bind: relocate a filler's indices into a role-specific slot
/// (idx ^ mask). Established in §10-12 as the sound sparse bind.
fn pair(cb: &EntangledHVec, mask: u32) -> Vec<u32> {
    let mut idx: Vec<u32> = cb.indices().iter().map(|&i| i ^ mask).collect();
    idx.sort_unstable();
    idx
}

/// Score a relation against the integer field: mean field amplitude over its
/// active indices minus the field's global mean (background correction). A
/// stored/strengthened relation stands above background; an absent one does not.
fn score(field: &[i64], indices: &[u32], global_mean: f64) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let s: i64 = indices.iter().map(|&i| field[i as usize]).sum();
    (s as f64 / indices.len() as f64) - global_mean
}

fn global_mean(field: &[i64]) -> f64 {
    field.iter().sum::<i64>() as f64 / field.len() as f64
}

/// AUC = P(correct > mis-bound), Mann-Whitney estimator (distribution-free).
fn auc(correct: &[f64], misbound: &[f64]) -> f64 {
    let mut wins = 0.0;
    for &c in correct {
        for &m in misbound {
            if c > m {
                wins += 1.0;
            } else if (c - m).abs() < 1e-9 {
                wins += 0.5;
            }
        }
    }
    wins / (correct.len() * misbound.len()) as f64
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    (m, sd)
}

// ---- event-sourced field --------------------------------------------------

#[derive(Clone, Copy)]
enum Event {
    Store(usize),      // fact index -> increments both its role pairs
    Strengthen(usize), // fact index -> increments its role1 pair
    Decay,
}

/// Deterministic integer fold of an event stream into the field, while chaining
/// a running digest over the (encoded) events. Returns (field, chain_digest).
fn fold_events(
    events: &[Event],
    role1_pairs: &[Vec<u32>],
    role2_pairs: &[Vec<u32>],
) -> (Vec<i64>, u64) {
    let mut field = vec![0i64; D];
    let mut chain: u64 = 0;
    for ev in events {
        let tag = match *ev {
            Event::Store(f) => {
                for &i in &role1_pairs[f] {
                    field[i as usize] += W_STORE;
                }
                for &i in &role2_pairs[f] {
                    field[i as usize] += W_STORE;
                }
                mix(0x5701 ^ f as u64, chain)
            }
            Event::Strengthen(f) => {
                for &i in &role1_pairs[f] {
                    field[i as usize] += W_STRENGTHEN;
                }
                mix(0x5731 ^ f as u64, chain)
            }
            Event::Decay => {
                for v in field.iter_mut() {
                    *v = *v * DECAY_NUM / DECAY_DEN;
                }
                mix(0xDECA, chain)
            }
        };
        chain = tag;
    }
    (field, chain)
}

struct Run {
    static_auc: f64,
    plastic_auc: f64,
    events: usize,
}

fn run(n: usize, seed: u64) -> Run {
    let cb: Vec<EntangledHVec> = (0..N_SYMBOLS)
        .map(|i| EntangledHVec::new_with_density(D, DENSITY_DENOM, mix(i as u64, seed)))
        .collect();
    let facts = build_facts(n, seed);
    let r1: Vec<Vec<u32>> = facts.iter().map(|f| pair(&cb[f.a], ROLE1_MASK)).collect();
    let r2: Vec<Vec<u32>> = facts.iter().map(|f| pair(&cb[f.b], ROLE2_MASK)).collect();
    // The mis-binding probe for fact i: (role1, b_i) -- b is bound to role2 here.
    let mis: Vec<Vec<u32>> = facts.iter().map(|f| pair(&cb[f.b], ROLE1_MASK)).collect();

    // STATIC: store every relation once, no plasticity.
    let static_events: Vec<Event> = (0..n).map(Event::Store).collect();
    let (static_field, _) = fold_events(&static_events, &r1, &r2);

    // PLASTIC: store everything, then a warm-up query stream over the HOT set
    // (strengthen each hot relation; decay the whole field once per pass).
    let hot = HOT.min(n);
    let mut plastic_events: Vec<Event> = (0..n).map(Event::Store).collect();
    let mut since_decay = 0usize;
    for _round in 0..WARMUP_ROUNDS {
        for h in 0..hot {
            plastic_events.push(Event::Strengthen(h));
            since_decay += 1;
            if since_decay >= DECAY_INTERVAL {
                plastic_events.push(Event::Decay);
                since_decay = 0;
            }
        }
    }
    let (plastic_field, chain) = fold_events(&plastic_events, &r1, &r2);

    // Verifiability: independent replay must reproduce field + chain bit-exactly.
    let (replay_field, replay_chain) = fold_events(&plastic_events, &r1, &r2);
    assert!(
        replay_field == plastic_field && replay_chain == chain,
        "replay mismatch"
    );
    // Tamper-evidence: flipping one event must change the chain digest.
    if plastic_events.len() > 2 {
        let mut tampered = plastic_events.clone();
        tampered[1] = Event::Decay;
        let (_, tchain) = fold_events(&tampered, &r1, &r2);
        assert!(tchain != chain, "tamper not detected");
    }

    // Measure AUC on the HOT relations only, for both fields.
    let sm = global_mean(&static_field);
    let pm = global_mean(&plastic_field);
    let (mut sc, mut smb) = (Vec::new(), Vec::new());
    let (mut pc, mut pmb) = (Vec::new(), Vec::new());
    for h in 0..hot {
        sc.push(score(&static_field, &r1[h], sm));
        smb.push(score(&static_field, &mis[h], sm));
        pc.push(score(&plastic_field, &r1[h], pm));
        pmb.push(score(&plastic_field, &mis[h], pm));
    }
    Run {
        static_auc: auc(&sc, &smb),
        plastic_auc: auc(&pc, &pmb),
        events: plastic_events.len(),
    }
}

fn main() {
    println!("plastic-graph | D={D} density=1/{DENSITY_DENOM} hot={HOT} warmup_rounds={WARMUP_ROUNDS} seeds={SEEDS}");
    println!("AUC on HOT relations (correct vs mis-binding). Chance=0.5. static=no plasticity,");
    println!("plastic=strengthen-hot + decay-unused (integer field, replay-verified).\n");
    println!(
        "{:>6}  {:>16}  {:>16}  {:>10}",
        "N", "static AUC", "plastic AUC", "events"
    );
    for &n in LOADS {
        let mut s = Vec::new();
        let mut p = Vec::new();
        let mut ev = 0usize;
        for seed in 0..SEEDS {
            let r = run(n, seed);
            s.push(r.static_auc);
            p.push(r.plastic_auc);
            ev = r.events;
        }
        let (sm, ss) = mean_sd(&s);
        let (pm, ps) = mean_sd(&p);
        println!("{n:>6}  {sm:>9.3} ({ss:>4.2})  {pm:>9.3} ({ps:>4.2})  {ev:>10}");
    }
    println!("\nReplay was bit-exact and tamper was detected on every run (asserts held).");
    println!("KILL for the scientific claim: plastic AUC not materially above static at high N.");
    println!("KILL for the systems claim: replay not bit-exact, or events/op unbounded (here");
    println!("O(1) events per store and per query -- bounded).");
}
