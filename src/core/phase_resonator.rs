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
//! Unlike a greedy hard-index alternation resonator on the self-inverse
//! sparse-binary substrate (which only holds for tiny codebooks), this runs the
//! real resonator on a non-self-inverse phase substrate. §20 of
//! `docs/PREREGISTRATION-binding-readout.md` shows it factors 3-way products across
//! the capacity knee (search spaces to ~110k at D=1024) at the SAME accuracy as a
//! float FHRR resonator in those experiments, down to 4-bit phase. Factorization
//! is a floating-point query; the stored phase substrate stays integer and
//! replay-verifiable. Cross-platform bitwise float equivalence is not promised.
//!
//! Entry points: [`PhaseResonator`] builds a reusable index over a fixed set of
//! factor codebooks (phasors precomputed once, then decode many composites) and is
//! the ergonomic path for retrieval; [`phase_resonator_factorize`] is the one-shot
//! free function for a single query.

use super::phase_hvec::PhaseHVec;
use std::f64::consts::TAU;

/// Iteration parameters for resonator factorization.
pub struct ResonatorConfig {
    pub max_iter: usize,
    /// Reserved for compatibility. Convergence currently requires three rounds
    /// of unchanged factor indices, independent of this value.
    pub convergence_threshold: f64,
}

impl Default for ResonatorConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            convergence_threshold: 0.999,
        }
    }
}

/// One recovered factor: which codebook entry matched and how confidently.
pub struct FactorResult {
    pub factor_idx: usize,
    pub codebook_entry: usize,
    pub similarity: f64,
    pub converged: bool,
    pub iterations: usize,
}

/// Complex vector as parallel real/imaginary parts.
#[derive(Clone)]
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
    fn from_phases(v: &PhaseHVec, table: &PhaseTable) -> Self {
        if table.values.is_none() {
            let n = v.n() as f64;
            return Self {
                re: v
                    .phases()
                    .iter()
                    .map(|&p| (TAU * p as f64 / n).cos())
                    .collect(),
                im: v
                    .phases()
                    .iter()
                    .map(|&p| (TAU * p as f64 / n).sin())
                    .collect(),
            };
        }
        let mut out = Self::zeros(v.dim());
        for (d, &p) in v.phases().iter().enumerate() {
            (out.re[d], out.im[d]) = table.get(p as usize);
        }
        out
    }

    /// Snap each component to a unit phasor, quantizing its phase to `n` levels
    /// (the qFHRR recover step). Mirrors `PhaseHVec::bundle`'s recovery.
    fn snap(&mut self, table: &PhaseTable) {
        let n = table.n;
        for d in 0..self.re.len() {
            let ang = self.im[d].atan2(self.re[d]);
            let idx = (ang / TAU * n as f64).round() as i64;
            (self.re[d], self.im[d]) = table.get(idx.rem_euclid(n as i64) as usize);
        }
    }
}

#[derive(Clone, Copy)]
struct CxView<'a> {
    re: &'a [f64],
    im: &'a [f64],
}

#[derive(Clone, Copy)]
enum EntryView<'a> {
    Expanded(CxView<'a>),
    Packed4 {
        data: &'a [u8],
        start: usize,
        dim: usize,
    },
    Packed8(&'a [u8]),
    Packed16(&'a [u16]),
}

impl EntryView<'_> {
    fn for_each(self, table: &PhaseTable, mut visit: impl FnMut(usize, f64, f64)) {
        match self {
            Self::Expanded(c) => {
                for d in 0..c.re.len() {
                    visit(d, c.re[d], c.im[d]);
                }
            }
            Self::Packed4 { data, start, dim } => {
                let values = table
                    .values
                    .as_ref()
                    .expect("packed codebooks require a phase table");
                for d in 0..dim {
                    let i = start + d;
                    let p = ((data[i / 2] >> ((i % 2) * 4)) & 15) as usize;
                    visit(d, values.re[p], values.im[p]);
                }
            }
            Self::Packed8(data) => {
                let values = table
                    .values
                    .as_ref()
                    .expect("packed codebooks require a phase table");
                for (d, &p) in data.iter().enumerate() {
                    visit(d, values.re[p as usize], values.im[p as usize]);
                }
            }
            Self::Packed16(data) => {
                let values = table
                    .values
                    .as_ref()
                    .expect("packed codebooks require a phase table");
                for (d, &p) in data.iter().enumerate() {
                    visit(d, values.re[p as usize], values.im[p as usize]);
                }
            }
        }
    }
}

struct PhaseTable {
    n: u32,
    values: Option<Cx>,
}

impl PhaseTable {
    fn new(n: u32, cached: bool) -> Self {
        let values = cached.then(|| Cx {
            re: (0..n).map(|p| (TAU * p as f64 / n as f64).cos()).collect(),
            im: (0..n).map(|p| (TAU * p as f64 / n as f64).sin()).collect(),
        });
        Self { n, values }
    }

    fn get(&self, p: usize) -> (f64, f64) {
        if let Some(values) = &self.values {
            (values.re[p], values.im[p])
        } else {
            let angle = TAU * p as f64 / self.n as f64;
            (angle.cos(), angle.sin())
        }
    }
}

enum CodebookData {
    Expanded(Cx),
    Packed4(Vec<u8>),
    Packed8(Vec<u8>),
    Packed16(Vec<u16>),
}

struct Codebook {
    entries: usize,
    data: CodebookData,
}

impl Codebook {
    fn new(book: &[PhaseHVec], table: &PhaseTable, packed: bool) -> Self {
        let elements = book
            .len()
            .checked_mul(book[0].dim())
            .expect("codebook size overflow");
        let data = if packed {
            match table.n {
                2..=16 => {
                    let mut data = vec![0; elements.div_ceil(2)];
                    for (i, &p) in book.iter().flat_map(PhaseHVec::phases).enumerate() {
                        data[i / 2] |= (p as u8) << ((i % 2) * 4);
                    }
                    CodebookData::Packed4(data)
                }
                17..=256 => {
                    let mut data = Vec::with_capacity(elements);
                    for entry in book {
                        data.extend(entry.phases().iter().map(|&p| p as u8));
                    }
                    CodebookData::Packed8(data)
                }
                _ => {
                    let mut data = Vec::with_capacity(elements);
                    for entry in book {
                        data.extend_from_slice(entry.phases());
                    }
                    CodebookData::Packed16(data)
                }
            }
        } else {
            let mut values = Cx::zeros(elements);
            if let Some(table) = &table.values {
                for (i, &p) in book.iter().flat_map(PhaseHVec::phases).enumerate() {
                    values.re[i] = table.re[p as usize];
                    values.im[i] = table.im[p as usize];
                }
            } else {
                // Keep input conversion in separate passes, as in the baseline;
                // paired sin/cos calls can round differently on some libm paths.
                for (i, &p) in book.iter().flat_map(PhaseHVec::phases).enumerate() {
                    values.re[i] = (TAU * p as f64 / table.n as f64).cos();
                }
                for (i, &p) in book.iter().flat_map(PhaseHVec::phases).enumerate() {
                    values.im[i] = (TAU * p as f64 / table.n as f64).sin();
                }
            }
            CodebookData::Expanded(values)
        };
        Self {
            entries: book.len(),
            data,
        }
    }

    fn for_each(&self, dim: usize, mut visit: impl FnMut(usize, EntryView<'_>)) {
        match &self.data {
            CodebookData::Expanded(values) => {
                for entry in 0..self.entries {
                    let start = entry * dim;
                    visit(
                        entry,
                        EntryView::Expanded(CxView {
                            re: &values.re[start..start + dim],
                            im: &values.im[start..start + dim],
                        }),
                    );
                }
            }
            CodebookData::Packed4(data) => {
                for entry in 0..self.entries {
                    let start = entry * dim;
                    visit(entry, EntryView::Packed4 { data, start, dim });
                }
            }
            CodebookData::Packed8(data) => {
                for entry in 0..self.entries {
                    let start = entry * dim;
                    visit(entry, EntryView::Packed8(&data[start..start + dim]));
                }
            }
            CodebookData::Packed16(data) => {
                for entry in 0..self.entries {
                    let start = entry * dim;
                    visit(entry, EntryView::Packed16(&data[start..start + dim]));
                }
            }
        }
    }
}

/// Real part of the Hermitian inner product Re<a, b> = Σ (a̅ · b).
fn dot(a: EntryView<'_>, b: &Cx, table: &PhaseTable) -> f64 {
    let mut s = 0.0;
    a.for_each(table, |d, re, im| s += re * b.re[d] + im * b.im[d]);
    s
}

/// Project `v` onto `codebook` (similarity-weighted superposition), snap to phasors.
fn cleanup(codebook: &Codebook, v: &Cx, table: &PhaseTable, out: &mut Cx) {
    let dim = v.re.len();
    out.re.fill(0.0);
    out.im.fill(0.0);
    codebook.for_each(dim, |_, c| {
        let s = dot(c, v, table);
        c.for_each(table, |d, re, im| {
            out.re[d] += s * re;
            out.im[d] += s * im;
        });
    });
    out.snap(table);
}

fn argmax(codebook: &Codebook, v: &Cx, table: &PhaseTable) -> (usize, f64) {
    let scale = v.re.len().max(1) as f64;
    let mut best: Option<(usize, f64)> = None;
    codebook.for_each(v.re.len(), |entry, c| {
        let sim = dot(c, v, table) / scale;
        if best.is_none_or(|(_, previous)| {
            sim.partial_cmp(&previous)
                .unwrap_or(std::cmp::Ordering::Equal)
                != std::cmp::Ordering::Less
        }) {
            best = Some((entry, sim));
        }
    });
    best.unwrap_or((0, 0.0))
}

/// Core resonator dynamics over codebooks already converted to unit phasors.
/// Shared by the free [`phase_resonator_factorize`] and the reusable
/// [`PhaseResonator`] index.
fn resonate(index: &PhaseResonator, comp: &Cx, config: &ResonatorConfig) -> Vec<FactorResult> {
    let books = &index.books;
    let table = &index.table;
    let k = books.len();
    if k == 0 {
        return Vec::new();
    }

    let mut est = index.initial.clone();
    let mut ub = Cx::zeros(comp.re.len());
    let mut results: Vec<FactorResult> = (0..k)
        .map(|factor_idx| FactorResult {
            factor_idx,
            codebook_entry: usize::MAX,
            similarity: 0.0,
            converged: false,
            iterations: 0,
        })
        .collect();
    let mut stable = 0;
    let mut iters = 0;
    let mut converged = false;
    while iters < config.max_iter {
        iters += 1;
        for f in 0..k {
            // unbind = composite ⊙ Π_{j≠f} conj(est[j])
            ub.re.copy_from_slice(&comp.re);
            ub.im.copy_from_slice(&comp.im);
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
            cleanup(&books[f], &ub, table, &mut est[f]);
        }
        let mut unchanged = true;
        for f in 0..k {
            let (entry, similarity) = argmax(&books[f], &est[f], table);
            unchanged &= results[f].codebook_entry == entry;
            results[f].codebook_entry = entry;
            results[f].similarity = similarity;
        }
        if unchanged {
            stable += 1;
            if stable >= 3 {
                converged = true;
                break;
            }
        } else {
            stable = 0;
        }
    }

    for f in 0..k {
        if iters == 0 {
            let (entry, similarity) = argmax(&books[f], &est[f], table);
            results[f].codebook_entry = entry;
            results[f].similarity = similarity;
        }
        results[f].converged = converged;
        results[f].iterations = iters;
    }
    results
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
    let index = PhaseResonator::new(codebooks);
    index.factorize_with_config(composite, config)
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
    books: Vec<Codebook>,
    table: PhaseTable,
    initial: Vec<Cx>,
    dim: usize,
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
    ///
    /// # Panics
    ///
    /// Panics if a codebook is empty, or entries have different dimensions or
    /// phase resolutions. An empty outer slice represents zero factors.
    pub fn with_config(codebooks: &[Vec<PhaseHVec>], config: ResonatorConfig) -> Self {
        Self::build(codebooks, config, false)
    }

    /// Build an index with packed codebooks and an explicit iteration `config`.
    ///
    /// Codebook coordinates use 4 bits for `n <= 16`, 8 bits for `n <= 256`, and
    /// 16 bits otherwise, plus a shared `16 * n` byte phasor table and floating
    /// initial estimates. Queries decode coordinates through the shared table.
    /// This saves resident memory but may cost query throughput;
    /// the default constructors retain expanded phasors for faster decoding.
    /// Factorization arithmetic and stopping rules are identical in both modes.
    ///
    /// # Panics
    ///
    /// Panics for the same invalid codebooks as [`Self::with_config`].
    pub fn with_packed_codebooks(codebooks: &[Vec<PhaseHVec>], config: ResonatorConfig) -> Self {
        Self::build(codebooks, config, true)
    }

    fn build(codebooks: &[Vec<PhaseHVec>], config: ResonatorConfig, packed: bool) -> Self {
        let first = codebooks.iter().flatten().next();
        let (dim, n) = first.map(|v| (v.dim(), v.n())).unwrap_or((0, 2));
        let mut elements = 0usize;
        for book in codebooks {
            assert!(!book.is_empty(), "factor codebooks must not be empty");
            for entry in book {
                assert_eq!(entry.dim(), dim, "codebook dimensions must match");
                assert_eq!(entry.n(), n, "codebook phase resolutions must match");
                elements = elements.saturating_add(dim);
            }
        }
        // Bound the default cache to 64 KiB and avoid table setup for tiny books.
        let table = PhaseTable::new(n, packed || (n <= 4096 && elements >= n as usize));
        let books: Vec<Codebook> = codebooks
            .iter()
            .map(|cb| Codebook::new(cb, &table, packed))
            .collect();
        let initial = books
            .iter()
            .map(|book| {
                let mut acc = Cx::zeros(dim);
                book.for_each(dim, |_, c| {
                    c.for_each(&table, |d, re, im| {
                        acc.re[d] += re;
                        acc.im[d] += im;
                    });
                });
                acc.snap(&table);
                acc
            })
            .collect();
        Self {
            books,
            table,
            initial,
            dim,
            config,
        }
    }

    /// Number of factors (codebooks) this resonator decodes.
    pub fn factor_count(&self) -> usize {
        self.books.len()
    }

    /// Recover one entry per factor from `composite`. `composite` must share the
    /// codebooks' dimension and phase resolution.
    ///
    /// # Panics
    ///
    /// Panics if the composite dimension or phase resolution differs from the
    /// codebooks'. With zero factors, any composite yields an empty result.
    pub fn factorize(&self, composite: &PhaseHVec) -> Vec<FactorResult> {
        self.factorize_with_config(composite, &self.config)
    }

    fn factorize_with_config(
        &self,
        composite: &PhaseHVec,
        config: &ResonatorConfig,
    ) -> Vec<FactorResult> {
        if self.books.is_empty() {
            return Vec::new();
        }
        assert_eq!(
            composite.n(),
            self.table.n,
            "composite phase resolution must match the codebooks'"
        );
        assert_eq!(
            composite.dim(),
            self.dim,
            "composite dimension must match the codebooks'"
        );
        let comp = Cx::from_phases(composite, &self.table);
        resonate(self, &comp, config)
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
    fn empty_codebooks_yield_no_factors() {
        // Zero factors: both entry points return an empty result, no panic.
        let cfg = ResonatorConfig::default();
        let composite = PhaseHVec::new_random(64, 256, 1);
        assert!(phase_resonator_factorize(&composite, &[], &cfg).is_empty());

        let idx = PhaseResonator::new(&[]);
        assert_eq!(idx.factor_count(), 0);
        assert!(idx.factorize(&composite).is_empty());
    }

    #[test]
    fn single_factor_and_single_entry_boundaries() {
        // One factor axis with one entry: the only candidate must be recovered.
        let cbs = codebooks(1, 1, 256, 512, 0x11);
        let composite = compose(&cbs, &[0]);
        let res = PhaseResonator::new(&cbs).factorize(&composite);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].codebook_entry, 0);

        // One factor axis, several entries: a pure cleanup problem, must be exact.
        let cbs = codebooks(1, 8, 256, 512, 0x22);
        let composite = compose(&cbs, &[5]);
        let res = phase_resonator_factorize(&composite, &cbs, &ResonatorConfig::default());
        assert_eq!(res[0].codebook_entry, 5);
    }

    #[test]
    fn factors_at_minimum_phase_resolution() {
        // N=2 (1-bit phase) is the substrate minimum; the resonator must still run and
        // recover a small two-factor product without panicking.
        let cbs = codebooks(2, 4, 2, 1024, 0x33);
        let target = [1usize, 3];
        let res =
            phase_resonator_factorize(&compose(&cbs, &target), &cbs, &ResonatorConfig::default());
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn reusable_index_derives_resolution_from_codebooks() {
        // Precomputed index must adopt the codebooks' N; a mismatched composite is a
        // programming error and is rejected loudly.
        let cbs = codebooks(2, 6, 64, 256, 0x44);
        let idx = PhaseResonator::new(&cbs);
        let composite = compose(&cbs, &[2, 4]);
        assert_eq!(idx.factorize(&composite).len(), 2);
    }

    #[test]
    #[should_panic(expected = "phase resolution")]
    fn mismatched_resolution_panics() {
        let cbs = codebooks(2, 6, 64, 256, 0x55);
        let idx = PhaseResonator::new(&cbs);
        // Composite at a different N than the codebooks.
        let wrong = PhaseHVec::new_random(256, 256, 9);
        let _ = idx.factorize(&wrong);
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
            let idx: Vec<usize> = (0..3).map(|axis| (t * 7 + axis * 3) % f).collect();
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
