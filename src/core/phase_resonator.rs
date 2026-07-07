// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Resonator network on the quantized-phase substrate ([`super::phase_hvec`]).
//!
//! Given a composite C = bind(f1, ..., fk) of one entry from each factor codebook,
//! recovers the factor indices by the Frady/Kent (2020) resonator dynamics:
//! maintain a SUPERPOSITION estimate per factor, and each step unbind the other
//! estimates, project (similarity-weighted) onto the factor's codebook, and snap
//! back to a unit phasor at the substrate's phase resolution. The superposition
//! state is what lets a resonator escape the local minima that trap greedy
//! alternating search, giving capacity that scales with dimension rather than
//! collapsing at small codebooks.
//!
//! Unlike [`super::resonator`] (greedy hard-index alternation on the self-inverse
//! sparse-binary substrate, which only holds for tiny codebooks), this runs the
//! real resonator on a non-self-inverse phase substrate. §20 of
//! `docs/PREREGISTRATION-binding-readout.md` shows it factors 3-way products across
//! the capacity knee (search spaces to ~110k at D=1024) at the SAME accuracy as a
//! float FHRR resonator, down to 4-bit phase -- i.e. deterministic, replay-exact
//! factorization at no capacity cost. The factorization itself is a float query;
//! the stored substrate stays integer/replay-verifiable.
//!
//! Entry points: [`PhaseResonator`] builds a reusable index over a fixed set of
//! factor codebooks (phasors precomputed once, then decode many composites) and is
//! the ergonomic path for retrieval; [`phase_resonator_factorize`] is the one-shot
//! free function for a single query.

use super::phase_hvec::PhaseHVec;
use super::resonator::{FactorResult, ResonatorConfig};
use std::f64::consts::TAU;

/// Complex vector as parallel real/imaginary parts.
struct Cx {
    re: Vec<f64>,
    im: Vec<f64>,
}

impl Cx {
    fn zeros(dim: usize) -> Self {
        Self {
            re: vec![0.0; dim],
            im: vec![0.0; dim],
        }
    }

    /// Unit phasors for a phase vector.
    fn from_phases(v: &PhaseHVec) -> Self {
        let n = v.n() as f64;
        let re = v
            .phases()
            .iter()
            .map(|&p| (TAU * p as f64 / n).cos())
            .collect();
        let im = v
            .phases()
            .iter()
            .map(|&p| (TAU * p as f64 / n).sin())
            .collect();
        Self { re, im }
    }

    /// Snap each component to a unit phasor, quantizing its phase to `n` levels
    /// (the qFHRR recover step). Mirrors `PhaseHVec::bundle`'s recovery.
    fn snap(&mut self, n: u32) {
        for d in 0..self.re.len() {
            let ang = self.im[d].atan2(self.re[d]);
            let idx = (ang / TAU * n as f64).round() as i64;
            let p = TAU * idx.rem_euclid(n as i64) as f64 / n as f64;
            self.re[d] = p.cos();
            self.im[d] = p.sin();
        }
    }
}

/// Real part of the Hermitian inner product Re<a, b> = Σ (a̅ · b).
fn dot(a: &Cx, b: &Cx) -> f64 {
    let mut s = 0.0;
    for d in 0..a.re.len() {
        s += a.re[d] * b.re[d] + a.im[d] * b.im[d];
    }
    s
}

/// Project `v` onto `codebook` (similarity-weighted superposition), snap to phasors.
fn cleanup(codebook: &[Cx], v: &Cx, n: u32) -> Cx {
    let dim = v.re.len();
    let mut out = Cx::zeros(dim);
    for c in codebook {
        let s = dot(c, v);
        for d in 0..dim {
            out.re[d] += s * c.re[d];
            out.im[d] += s * c.im[d];
        }
    }
    out.snap(n);
    out
}

fn argmax(codebook: &[Cx], v: &Cx) -> (usize, f64) {
    let dim = v.re.len().max(1) as f64;
    (0..codebook.len())
        .map(|k| (k, dot(&codebook[k], v) / dim))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or((0, 0.0))
}

/// Core resonator dynamics over codebooks already converted to unit phasors.
/// Shared by the free [`phase_resonator_factorize`] and the reusable
/// [`PhaseResonator`] index; `n` is the shared phase resolution used at each snap.
fn resonate(books: &[Vec<Cx>], n: u32, comp: &Cx, config: &ResonatorConfig) -> Vec<FactorResult> {
    let k = books.len();
    if k == 0 {
        return Vec::new();
    }

    // Initialize each estimate to the (snapped) superposition of its codebook.
    let mut est: Vec<Cx> = books
        .iter()
        .map(|cb| {
            let mut acc = Cx::zeros(comp.re.len());
            for c in cb {
                for d in 0..acc.re.len() {
                    acc.re[d] += c.re[d];
                    acc.im[d] += c.im[d];
                }
            }
            acc.snap(n);
            acc
        })
        .collect();

    let mut prev = vec![usize::MAX; k];
    let mut stable = 0;
    let mut iters = 0;
    let mut converged = false;
    while iters < config.max_iter {
        iters += 1;
        for f in 0..k {
            // unbind = composite ⊙ Π_{j≠f} conj(est[j])
            let mut ub = Cx {
                re: comp.re.clone(),
                im: comp.im.clone(),
            };
            for (j, e) in est.iter().enumerate() {
                if j != f {
                    for d in 0..ub.re.len() {
                        let (r, i) = (ub.re[d], ub.im[d]);
                        // multiply by conj(e) = (e.re, -e.im)
                        ub.re[d] = r * e.re[d] + i * e.im[d];
                        ub.im[d] = i * e.re[d] - r * e.im[d];
                    }
                }
            }
            est[f] = cleanup(&books[f], &ub, n);
        }
        let cur: Vec<usize> = (0..k).map(|f| argmax(&books[f], &est[f]).0).collect();
        if cur == prev {
            stable += 1;
            if stable >= 3 {
                converged = true;
                break;
            }
        } else {
            stable = 0;
        }
        prev = cur;
    }

    (0..k)
        .map(|f| {
            let (idx, sim) = argmax(&books[f], &est[f]);
            FactorResult {
                factor_idx: f,
                codebook_entry: idx,
                similarity: sim,
                converged,
                iterations: iters,
            }
        })
        .collect()
}

/// Factorize `composite` into one entry per factor codebook via the phase resonator.
/// Returns the recovered index, similarity, convergence flag, and iteration count
/// for each factor. Converts the codebooks to phasors on every call; for repeated
/// queries against a fixed set of codebooks, build a [`PhaseResonator`] once instead.
pub fn phase_resonator_factorize(
    composite: &PhaseHVec,
    codebooks: &[Vec<PhaseHVec>],
    config: &ResonatorConfig,
) -> Vec<FactorResult> {
    let books: Vec<Vec<Cx>> = codebooks
        .iter()
        .map(|cb| cb.iter().map(Cx::from_phases).collect())
        .collect();
    let comp = Cx::from_phases(composite);
    resonate(&books, composite.n(), &comp, config)
}

/// A reusable resonator index over a fixed set of factor codebooks.
///
/// Build it once from the per-factor codebooks (the codebook phasors are
/// precomputed at construction), then call [`PhaseResonator::factorize`] for each
/// bound composite you want to decode. This is the shape of a real retrieval
/// workload: register the factor alphabets once, then recover which entries went
/// into any composite `bind(f1, ..., fk)` from the composite alone.
///
/// ```
/// use holographic_memory::core::phase_hvec::PhaseHVec;
/// use holographic_memory::core::PhaseResonator;
///
/// // Three factor codebooks (e.g. subject / relation / object alphabets),
/// // each 6 entries at 8-bit phase resolution in D = 1024.
/// let (dim, n) = (1024, 256);
/// let axis = |base: u64| -> Vec<PhaseHVec> {
///     (0..6).map(|i| PhaseHVec::new_random(dim, n, base + i)).collect()
/// };
/// let codebooks = [axis(100), axis(200), axis(300)];
///
/// // Bind one entry from each axis into a single composite hypervector ...
/// let composite = codebooks[0][4]
///     .bind(&codebooks[1][1])
///     .bind(&codebooks[2][5]);
///
/// // ... and recover all three indices from the composite alone.
/// let resonator = PhaseResonator::new(&codebooks);
/// let recovered = resonator.factorize(&composite);
/// assert_eq!(recovered[0].codebook_entry, 4);
/// assert_eq!(recovered[1].codebook_entry, 1);
/// assert_eq!(recovered[2].codebook_entry, 5);
/// ```
pub struct PhaseResonator {
    books: Vec<Vec<Cx>>,
    n: u32,
    config: ResonatorConfig,
}

impl PhaseResonator {
    /// Build a resonator over `codebooks` (one codebook per factor) with the
    /// default [`ResonatorConfig`]. The phase resolution is taken from the
    /// codebook entries.
    pub fn new(codebooks: &[Vec<PhaseHVec>]) -> Self {
        Self::with_config(codebooks, ResonatorConfig::default())
    }

    /// Build a resonator over `codebooks` with an explicit iteration `config`.
    pub fn with_config(codebooks: &[Vec<PhaseHVec>], config: ResonatorConfig) -> Self {
        let n = codebooks
            .iter()
            .flatten()
            .next()
            .map(PhaseHVec::n)
            .unwrap_or(2);
        let books = codebooks
            .iter()
            .map(|cb| cb.iter().map(Cx::from_phases).collect())
            .collect();
        Self { books, n, config }
    }

    /// Number of factors (codebooks) this resonator decodes.
    pub fn factor_count(&self) -> usize {
        self.books.len()
    }

    /// Recover one entry per factor from `composite`. `composite` must share the
    /// codebooks' phase resolution.
    pub fn factorize(&self, composite: &PhaseHVec) -> Vec<FactorResult> {
        if self.books.is_empty() {
            return Vec::new();
        }
        assert_eq!(
            composite.n(),
            self.n,
            "composite phase resolution must match the codebooks'"
        );
        let comp = Cx::from_phases(composite);
        resonate(&self.books, self.n, &comp, &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codebooks(k: usize, f: usize, n: u32, dim: usize, seed: u64) -> Vec<Vec<PhaseHVec>> {
        (0..k)
            .map(|axis| {
                (0..f)
                    .map(|i| PhaseHVec::new_random(dim, n, seed ^ ((axis as u64) << 40) ^ i as u64))
                    .collect()
            })
            .collect()
    }

    fn compose(cbs: &[Vec<PhaseHVec>], idx: &[usize]) -> PhaseHVec {
        let mut c = cbs[0][idx[0]].clone();
        for axis in 1..cbs.len() {
            c = c.bind(&cbs[axis][idx[axis]]);
        }
        c
    }

    #[test]
    fn recovers_two_factors() {
        let cbs = codebooks(2, 12, 256, 1024, 0xAB);
        let cfg = ResonatorConfig::default();
        let target = [5usize, 9];
        let res = phase_resonator_factorize(&compose(&cbs, &target), &cbs, &cfg);
        assert_eq!(res[0].codebook_entry, target[0]);
        assert_eq!(res[1].codebook_entry, target[1]);
    }

    #[test]
    fn reusable_index_matches_free_function() {
        // The precomputed PhaseResonator must return the same factorization as the
        // one-shot free function for the same codebooks and composite.
        let cbs = codebooks(3, 8, 256, 1024, 0xC0DE);
        let cfg = ResonatorConfig::default();
        let target = [4usize, 1, 5];
        let composite = compose(&cbs, &target);

        let idx = PhaseResonator::new(&cbs);
        assert_eq!(idx.factor_count(), 3);
        let via_struct = idx.factorize(&composite);
        let via_free = phase_resonator_factorize(&composite, &cbs, &cfg);

        for a in 0..3 {
            assert_eq!(via_struct[a].codebook_entry, target[a]);
            assert_eq!(via_struct[a].codebook_entry, via_free[a].codebook_entry);
        }
    }

    #[test]
    fn factors_above_greedy_capacity() {
        // F=16 -> search space 4096 at D=1024, far past what the toy greedy sparse
        // resonator (tested only at F<=10, 2 factors) reaches. The real resonator
        // recovers all three factors on the large majority of trials (§20: ~98%).
        let dim = 1024;
        let n = 256;
        let f = 16;
        let cbs = codebooks(3, f, n, dim, 0x5EED);
        let cfg = ResonatorConfig {
            max_iter: 40,
            convergence_threshold: 0.999,
        };
        let mut ok = 0;
        let trials = 20;
        for t in 0..trials {
            let idx: Vec<usize> = (0..3)
                .map(|axis| ((t * 7 + axis * 3) % f) as usize)
                .collect();
            let res = phase_resonator_factorize(&compose(&cbs, &idx), &cbs, &cfg);
            if (0..3).all(|a| res[a].codebook_entry == idx[a]) {
                ok += 1;
            }
        }
        assert!(
            ok as f64 / trials as f64 >= 0.8,
            "expected >=80% recovery, got {ok}/{trials}"
        );
    }

    #[test]
    fn quantized_4bit_still_factors() {
        // Deterministic resonator down to 4-bit phase (N=16), the §20 headline.
        let cbs = codebooks(3, 12, 16, 1024, 0x4B17);
        let cfg = ResonatorConfig {
            max_iter: 40,
            convergence_threshold: 0.999,
        };
        let target = [3usize, 7, 10];
        let res = phase_resonator_factorize(&compose(&cbs, &target), &cbs, &cfg);
        assert!(
            (0..3).all(|a| res[a].codebook_entry == target[a]),
            "4-bit phase resonator must still factor"
        );
    }
}
