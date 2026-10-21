// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! `hms-research-forget`: kill test for exact deletion from a counter-backed
//! sparse-hypervector bundle.
//!
//! For each `(n, delete_frac)` row: generate `n` engine hypervectors at `--dim`
//! (seeded), insert them into a [`CounterBundle`], delete a seeded subset of
//! `k = max(1, round(n * frac))` with certificates, verify every certificate
//! against the post-delete state, and then
//!
//! (a) compare the counts bit-for-bit with a from-scratch rebuild of the
//!     survivors (and, at `tau = 1`, the view with `EntangledHVec::bundle_bloom`);
//! (b) score deleted items, fresh random probes and surviving items against the
//!     thresholded view (overlap fraction) and against the raw counts (cosine),
//!     and report the z-score of deleted-vs-random; do the same for the
//!     thresholded-only baseline under both delete semantics a bit-only bundle
//!     allows (`noop`: bits stay; `clear`: bits cleared, survivors damaged);
//! (c) time insert, delete, certified delete, verification, state digest and
//!     rebuild on one thread, medians over `--runs` after `--warmup`.
//!
//! Pass per row: rebuild exact AND certificate chain verified AND counter
//! deleted-vs-random |z| < `--z-threshold` on both readouts AND the baseline
//! leaks (noop z > threshold) wherever the baseline still recognises its own
//! members (otherwise the baseline is saturated and the leak check is n/a).
//! All randomness derives from `--seed`.

use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use holographic_memory::core::entangled::hash_u64;
use holographic_memory::EntangledHVec;
use serde::Serialize;

use hms_research_forget::bundle::{
    BaselineDelete, CapacityPolicy, Counter, CounterBundle, Digest32, MemoryAccounting,
    ThresholdedBundle,
};
use hms_research_forget::certificate::{
    chain_break, delete_with_certificate, verify, DeletionCertificate, DEFAULT_SIGMA,
};
use hms_research_forget::{median, sample, z_group, Sample};

const SCHEMA: &str = "hms-research-forget/v1";
const ITEM_TAG: u64 = 0x4954_454D_535F_0001;
const DELETE_TAG: u64 = 0x4445_4C45_5445_0002;
const PROBE_TAG: u64 = 0x5052_4F42_4553_0003;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum CounterKind {
    #[value(name = "i16")]
    I16,
    #[value(name = "i32")]
    I32,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum TauSpec {
    /// Fixed view threshold (1 = union, the engine's `bundle_bloom`).
    Fixed(u32),
    /// `max(1, ceil(n * k / dim))`: the Poisson rate at full load.
    Auto,
}

impl TauSpec {
    fn parse(s: &str) -> Result<Self> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        let v: u32 = s.parse().with_context(|| format!("--tau must be an integer >= 1 or 'auto', got {s:?}"))?;
        if v == 0 {
            bail!("--tau must be >= 1");
        }
        Ok(Self::Fixed(v))
    }

    fn resolve(self, n: usize, k: usize, dim: usize) -> u32 {
        match self {
            Self::Fixed(t) => t,
            Self::Auto => ((n * k) as f64 / dim as f64).ceil().max(1.0) as u32,
        }
    }

    fn label(self) -> String {
        match self {
            Self::Fixed(t) => t.to_string(),
            Self::Auto => "auto".to_string(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "hms-research-forget",
    about = "Exact deletion from a counter-backed sparse-hypervector bundle: kill test"
)]
struct Cli {
    /// Hypervector dimension (engine default 16384 gives k = 64 active indices).
    #[arg(long, default_value_t = 16384)]
    dim: usize,
    /// Item counts to test.
    #[arg(long, value_delimiter = ',', default_value = "100,1000,10000")]
    ns: Vec<usize>,
    /// Fractions of items to delete per row.
    #[arg(long, value_delimiter = ',', default_value = "0.01,0.10")]
    delete_fracs: Vec<f64>,
    /// View threshold: integer >= 1 or 'auto'. Baselines are evaluated only at 1.
    #[arg(long, default_value = "1")]
    tau: String,
    /// Counter width.
    #[arg(long, value_enum, default_value_t = CounterKind::I16)]
    counter: CounterKind,
    /// Fresh random hypervectors forming the empirical null.
    #[arg(long, default_value_t = 2000)]
    probes: usize,
    /// Maximum surviving items scored per row.
    #[arg(long, default_value_t = 2000)]
    survivor_sample: usize,
    /// Timed repeats (median reported).
    #[arg(long, default_value_t = 5)]
    runs: usize,
    /// Untimed repeats before the timed ones.
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Residual bound width in standard deviations (certificate field).
    #[arg(long, default_value_t = DEFAULT_SIGMA)]
    sigma: f64,
    /// |z| above which deleted items count as distinguishable from random.
    #[arg(long, default_value_t = 3.0)]
    z_threshold: f64,
    /// Capacity policy: maximum live items (default unbounded).
    #[arg(long)]
    max_items: Option<usize>,
    /// Capacity policy: maximum view density after an insert.
    #[arg(long, default_value_t = 1.0)]
    max_view_density: f64,
    /// Output JSON path (default benchmarks/research/forget/out/forget_d<dim>_<counter>_tau<tau>.json).
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
struct Readout {
    name: &'static str,
    deleted: Sample,
    random: Sample,
    survivors: Sample,
    z_deleted_vs_random: f64,
    z_survivors_vs_random: f64,
    /// Fraction of deleted items individually above `random.mean + z_threshold * random.sd`.
    frac_deleted_over_threshold: f64,
    /// Survivors are distinguishable from random (`z_survivors_vs_random > z_threshold`).
    informative: bool,
}

fn readout(name: &'static str, deleted: &[f64], random: &[f64], survivors: &[f64], zt: f64) -> Readout {
    let d = sample(deleted);
    let r = sample(random);
    let s = sample(survivors);
    let z_deleted_vs_random = z_group(&d, &r);
    let z_survivors_vs_random = z_group(&s, &r);
    let frac_deleted_over_threshold = if r.sd > 0.0 && !deleted.is_empty() {
        deleted.iter().filter(|&&x| (x - r.mean) / r.sd > zt).count() as f64 / deleted.len() as f64
    } else {
        0.0
    };
    Readout {
        name,
        deleted: d,
        random: r,
        survivors: s,
        z_deleted_vs_random,
        z_survivors_vs_random,
        frac_deleted_over_threshold,
        informative: z_survivors_vs_random > zt,
    }
}

#[derive(Clone, Debug, Serialize)]
struct Stat {
    median: f64,
    runs: Vec<f64>,
}

fn stat(mut runs: Vec<f64>) -> Stat {
    let median = median(&mut runs);
    Stat { median, runs }
}

#[derive(Clone, Debug, Serialize)]
struct Timing {
    insert_ns_per_item: Stat,
    delete_ns_per_item: Stat,
    delete_with_certificate_ns_per_item: Stat,
    verify_certificate_us: Stat,
    state_digest_us: Stat,
    rebuild_ms: Stat,
    baseline_insert_ns_per_item: Stat,
}

#[derive(Clone, Debug, Serialize)]
struct CertificateSummary {
    issued: usize,
    verified: usize,
    chain_consistent: bool,
    first_state_before_matches_full_state: bool,
    within_bound: usize,
    max_abs_view_z: f64,
    max_abs_count_z: f64,
    /// The first certificate of the row, as a schema example.
    example: Option<DeletionCertificate>,
}

#[derive(Clone, Debug, Serialize)]
struct CounterReport {
    counter: &'static str,
    rebuild_exact: bool,
    counts_equal: bool,
    state_digest_after: Digest32,
    rebuild_state_digest: Digest32,
    view_digest_after: Digest32,
    rebuild_view_digest: Digest32,
    /// `tau = 1` only: the view equals `EntangledHVec::bundle_bloom(survivors)`.
    view_matches_bundle_bloom: Option<bool>,
    view_density_full: f64,
    view_density_after: f64,
    lambda_full: f64,
    lambda_after: f64,
    certificates: CertificateSummary,
    readouts: Vec<Readout>,
    memory: MemoryAccounting,
    timing: Timing,
}

#[derive(Clone, Debug, Serialize)]
struct BaselineReport {
    mode: BaselineDelete,
    view_density_full: f64,
    view_density_after: f64,
    bits_cleared: usize,
    readout: Readout,
    /// `1 - mean survivor overlap` (survivors are fully present before the delete).
    survivor_bits_lost_frac: f64,
    /// Deleted items score above random by more than the threshold.
    leaks: bool,
    /// Deleted items are distinguishable from random in either direction.
    distinguishable: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Verdict {
    exact: bool,
    chain_consistent: bool,
    counter_indistinguishable: bool,
    /// `None` when the baseline is saturated (cannot recognise its own members).
    baseline_leaks: Option<bool>,
    pass: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Row {
    n: usize,
    delete_frac: f64,
    k_deleted: usize,
    k_active: usize,
    tau: u32,
    counter: CounterReport,
    baselines: Vec<BaselineReport>,
    verdict: Verdict,
}

#[derive(Clone, Debug, Serialize)]
struct Config {
    dim: usize,
    ns: Vec<usize>,
    delete_fracs: Vec<f64>,
    tau: String,
    counter: CounterKind,
    probes: usize,
    survivor_sample: usize,
    runs: usize,
    warmup: usize,
    seed: u64,
    sigma: f64,
    z_threshold: f64,
    policy: CapacityPolicy,
    threads: usize,
}

#[derive(Clone, Debug, Serialize)]
struct ShardNote {
    per_document: bool,
    deletion_already_exact: bool,
    note: &'static str,
    evidence: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
struct Overall {
    pass: bool,
    rows_pass: usize,
    rows_total: usize,
    rows_baseline_not_applicable: usize,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    generated_unix_s: u64,
    config: Config,
    shards: ShardNote,
    rows: Vec<Row>,
    overall: Overall,
}

fn gen_items(dim: usize, seed: u64, tag: u64, n: usize) -> Vec<EntangledHVec> {
    (0..n as u64)
        .map(|i| EntangledHVec::new_deterministic(dim, hash_u64(seed ^ tag, i)))
        .collect()
}

fn delete_set(seed: u64, n: usize, k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| hash_u64(seed ^ DELETE_TAG, i as u64));
    let mut del = order[..k.min(n)].to_vec();
    del.sort_unstable();
    del
}

fn weight(item: &EntangledHVec) -> f64 {
    item.indices().len().max(1) as f64
}

/// (view overlap fraction, count cosine) per item.
fn counter_scores<'a, C: Counter>(
    b: &CounterBundle<C>,
    items: impl Iterator<Item = &'a EntangledHVec>,
    count_norm: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut overlap = Vec::new();
    let mut cosine = Vec::new();
    for it in items {
        let k = weight(it);
        overlap.push(b.view_overlap(it) as f64 / k);
        let dot = b.count_dot(it) as f64;
        cosine.push(if count_norm > 0.0 { dot / (count_norm * k.sqrt()) } else { 0.0 });
    }
    (overlap, cosine)
}

fn baseline_scores<'a>(b: &ThresholdedBundle, items: impl Iterator<Item = &'a EntangledHVec>) -> Vec<f64> {
    items.map(|it| b.overlap(it) as f64 / weight(it)).collect()
}

fn survivor_subset<'a>(survivors: &'a [&'a EntangledHVec], cap: usize) -> impl Iterator<Item = &'a EntangledHVec> + 'a {
    let step = (survivors.len() / cap.max(1)).max(1);
    survivors.iter().step_by(step).take(cap.max(1)).copied()
}

fn baseline_report(
    mode: BaselineDelete,
    full: &ThresholdedBundle,
    items: &[EntangledHVec],
    del: &[usize],
    probes: &[EntangledHVec],
    survivors: &[&EntangledHVec],
    cap: usize,
    zt: f64,
) -> Result<BaselineReport> {
    let mut b = full.clone();
    let mut bits_cleared = 0usize;
    for &i in del {
        bits_cleared += b.delete(&items[i], mode)?;
    }
    let deleted = baseline_scores(&b, del.iter().map(|&i| &items[i]));
    let random = baseline_scores(&b, probes.iter());
    let surv = baseline_scores(&b, survivor_subset(survivors, cap));
    let readout = readout("bit_overlap", &deleted, &random, &surv, zt);
    let z = readout.z_deleted_vs_random;
    Ok(BaselineReport {
        mode,
        view_density_full: full.density(),
        view_density_after: b.density(),
        bits_cleared,
        survivor_bits_lost_frac: 1.0 - readout.survivors.mean,
        leaks: readout.informative && z > zt,
        distinguishable: readout.informative && z.abs() > zt,
        readout,
    })
}

#[allow(clippy::too_many_arguments)]
fn timing<C: Counter>(
    dim: usize,
    tau: u32,
    policy: CapacityPolicy,
    items: &[EntangledHVec],
    del: &[usize],
    survivors: &[&EntangledHVec],
    sigma: f64,
    runs: usize,
    warmup: usize,
) -> Result<Timing> {
    let n = items.len().max(1) as f64;
    let kd = del.len().max(1) as f64;
    let mut ins = Vec::new();
    let mut dels = Vec::new();
    let mut certs = Vec::new();
    let mut verifies = Vec::new();
    let mut digests = Vec::new();
    let mut rebuilds = Vec::new();
    let mut base_ins = Vec::new();
    for r in 0..warmup + runs {
        let mut b = CounterBundle::<C>::new(dim, tau, policy)?;
        let t = Instant::now();
        for it in items {
            b.insert(it)?;
        }
        let insert_ns = t.elapsed().as_nanos() as f64 / n;

        let t = Instant::now();
        for &i in del {
            b.delete(&items[i])?;
        }
        let delete_ns = t.elapsed().as_nanos() as f64 / kd;
        black_box(b.items());

        for &i in del {
            b.insert(&items[i])?;
        }
        let mut last: Option<DeletionCertificate> = None;
        let t = Instant::now();
        for &i in del {
            last = Some(delete_with_certificate(&mut b, &items[i], sigma)?);
        }
        let cert_ns = t.elapsed().as_nanos() as f64 / kd;

        let verify_us = match (&last, del.last()) {
            (Some(cert), Some(&i)) => {
                let t = Instant::now();
                let v = verify(cert, &items[i], &b)?;
                let us = t.elapsed().as_nanos() as f64 / 1e3;
                if !v.ok {
                    bail!("certificate verification failed during timing: {v:?}");
                }
                us
            }
            _ => 0.0,
        };

        let t = Instant::now();
        black_box(b.state_digest());
        let digest_us = t.elapsed().as_nanos() as f64 / 1e3;

        let t = Instant::now();
        let rebuilt = CounterBundle::<C>::rebuild(dim, tau, policy, survivors.iter().copied())?;
        let rebuild_ms = t.elapsed().as_nanos() as f64 / 1e6;
        black_box(rebuilt.items());

        let mut tb = ThresholdedBundle::new(dim);
        let t = Instant::now();
        for it in items {
            tb.insert(it)?;
        }
        let base_ns = t.elapsed().as_nanos() as f64 / n;
        black_box(tb.set_count());

        if r >= warmup {
            ins.push(insert_ns);
            dels.push(delete_ns);
            certs.push(cert_ns);
            verifies.push(verify_us);
            digests.push(digest_us);
            rebuilds.push(rebuild_ms);
            base_ins.push(base_ns);
        }
    }
    Ok(Timing {
        insert_ns_per_item: stat(ins),
        delete_ns_per_item: stat(dels),
        delete_with_certificate_ns_per_item: stat(certs),
        verify_certificate_us: stat(verifies),
        state_digest_us: stat(digests),
        rebuild_ms: stat(rebuilds),
        baseline_insert_ns_per_item: stat(base_ins),
    })
}

fn run_row<C: Counter>(cli: &Cli, tau_spec: TauSpec, policy: CapacityPolicy, n: usize, frac: f64) -> Result<Row> {
    let dim = cli.dim;
    let items = gen_items(dim, cli.seed, ITEM_TAG, n);
    let probes = gen_items(dim, cli.seed, PROBE_TAG, cli.probes);
    let k_active = items.first().map(|v| v.indices().len()).unwrap_or(0);
    let k_deleted = ((n as f64 * frac).round() as usize).clamp(1, n);
    let del = delete_set(cli.seed, n, k_deleted);
    let tau = tau_spec.resolve(n, k_active, dim);
    let mut is_deleted = vec![false; n];
    for &i in &del {
        is_deleted[i] = true;
    }
    let survivors: Vec<&EntangledHVec> = items
        .iter()
        .enumerate()
        .filter(|(i, _)| !is_deleted[*i])
        .map(|(_, v)| v)
        .collect();
    let zt = cli.z_threshold;

    let mut b = CounterBundle::<C>::new(dim, tau, policy)?;
    for it in &items {
        b.insert(it)?;
    }
    let full_digest = b.state_digest();
    let view_density_full = b.view_density();
    let lambda_full = b.lambda();

    let mut certs: Vec<DeletionCertificate> = Vec::with_capacity(k_deleted);
    let mut verified = 0usize;
    let mut within_bound = 0usize;
    let mut max_abs_view_z = 0.0f64;
    let mut max_abs_count_z = 0.0f64;
    for &i in &del {
        let cert = delete_with_certificate(&mut b, &items[i], cli.sigma)?;
        let v = verify(&cert, &items[i], &b)?;
        if v.ok {
            verified += 1;
        }
        if cert.residual.within_bound {
            within_bound += 1;
        }
        max_abs_view_z = max_abs_view_z.max(cert.residual.view_z.abs());
        max_abs_count_z = max_abs_count_z.max(cert.residual.count_z.abs());
        certs.push(cert);
    }
    let chain_consistent = chain_break(&certs).is_none();
    let first_state_before_matches_full_state =
        certs.first().map(|c| c.state_before == full_digest).unwrap_or(true);

    let rebuilt = CounterBundle::<C>::rebuild(dim, tau, policy, survivors.iter().copied())?;
    let counts_equal = b.counts_equal(&rebuilt);
    let state_digest_after = b.state_digest();
    let rebuild_state_digest = rebuilt.state_digest();
    let view_digest_after = b.view_digest();
    let rebuild_view_digest = rebuilt.view_digest();
    let rebuild_exact = counts_equal
        && state_digest_after == rebuild_state_digest
        && view_digest_after == rebuild_view_digest;
    let view_matches_bundle_bloom = if tau == 1 {
        let bloom = EntangledHVec::bundle_bloom(&survivors);
        Some(b.view_indices() == bloom.indices())
    } else {
        None
    };

    let (_, sum_sq) = b.count_moments();
    let count_norm = sum_sq.sqrt();
    let (d_ov, d_cos) = counter_scores(&b, del.iter().map(|&i| &items[i]), count_norm);
    let (r_ov, r_cos) = counter_scores(&b, probes.iter(), count_norm);
    let (s_ov, s_cos) = counter_scores(&b, survivor_subset(&survivors, cli.survivor_sample), count_norm);
    let readouts = vec![
        readout("view_overlap", &d_ov, &r_ov, &s_ov, zt),
        readout("count_cosine", &d_cos, &r_cos, &s_cos, zt),
    ];

    let mut baselines = Vec::new();
    if tau == 1 {
        let mut full = ThresholdedBundle::new(dim);
        for it in &items {
            full.insert(it)?;
        }
        for mode in [BaselineDelete::Noop, BaselineDelete::Clear] {
            baselines.push(baseline_report(
                mode,
                &full,
                &items,
                &del,
                &probes,
                &survivors,
                cli.survivor_sample,
                zt,
            )?);
        }
    }

    let timing = timing::<C>(dim, tau, policy, &items, &del, &survivors, cli.sigma, cli.runs, cli.warmup)?;
    let memory = b.accounting(k_active);

    let exact = rebuild_exact && verified == certs.len() && first_state_before_matches_full_state;
    let counter_indistinguishable = readouts
        .iter()
        .all(|r| r.z_deleted_vs_random.abs() < zt && r.frac_deleted_over_threshold <= 0.01);
    let baseline_leaks = baselines
        .iter()
        .find(|r| r.mode == BaselineDelete::Noop)
        .and_then(|r| if r.readout.informative { Some(r.leaks) } else { None });
    let pass = exact && chain_consistent && counter_indistinguishable && baseline_leaks != Some(false);

    Ok(Row {
        n,
        delete_frac: frac,
        k_deleted,
        k_active,
        tau,
        counter: CounterReport {
            counter: C::NAME,
            rebuild_exact,
            counts_equal,
            state_digest_after,
            rebuild_state_digest,
            view_digest_after,
            rebuild_view_digest,
            view_matches_bundle_bloom,
            view_density_full,
            view_density_after: b.view_density(),
            lambda_full,
            lambda_after: b.lambda(),
            certificates: CertificateSummary {
                issued: certs.len(),
                verified,
                chain_consistent,
                first_state_before_matches_full_state,
                within_bound,
                max_abs_view_z,
                max_abs_count_z,
                example: certs.first().cloned(),
            },
            readouts,
            memory,
            timing,
        },
        baselines,
        verdict: Verdict {
            exact,
            chain_consistent,
            counter_indistinguishable,
            baseline_leaks,
            pass,
        },
    })
}

fn print_row(row: &Row) {
    let c = &row.counter;
    let view = &c.readouts[0];
    let count = &c.readouts[1];
    println!(
        "n={:>6} del={:>5} ({:>4.0}%) tau={:>2} | exact={} certs {}/{} chain={} | view z_del={:+.2} z_surv={:+.1} dens={:.3} | count z_del={:+.2} z_surv={:+.1} | ins {:.0} ns, del {:.0} ns, cert del {:.0} ns, verify {:.0} us | {:.1} B/item (counter {} B, per-doc {} B/item)",
        row.n,
        row.k_deleted,
        row.delete_frac * 100.0,
        row.tau,
        c.rebuild_exact,
        c.certificates.verified,
        c.certificates.issued,
        c.certificates.chain_consistent,
        view.z_deleted_vs_random,
        view.z_survivors_vs_random,
        c.view_density_after,
        count.z_deleted_vs_random,
        count.z_survivors_vs_random,
        c.timing.insert_ns_per_item.median,
        c.timing.delete_ns_per_item.median,
        c.timing.delete_with_certificate_ns_per_item.median,
        c.timing.verify_certificate_us.median,
        c.memory.bytes_per_item_total,
        c.memory.counter_bytes,
        c.memory.per_document_bytes_per_item,
    );
    for bl in &row.baselines {
        println!(
            "        baseline {:?}: z_del={:+.2} z_surv={:+.1} dens={:.3} leaks={} distinguishable={} survivor_bits_lost={:.3}",
            bl.mode,
            bl.readout.z_deleted_vs_random,
            bl.readout.z_survivors_vs_random,
            bl.view_density_after,
            bl.leaks,
            bl.distinguishable,
            bl.survivor_bits_lost_frac,
        );
    }
    let leak = match row.verdict.baseline_leaks {
        Some(true) => "baseline leaks",
        Some(false) => "baseline does NOT leak",
        None => "baseline saturated (leak check n/a)",
    };
    println!(
        "        [{}] exact={} indistinguishable={} {}",
        if row.verdict.pass { "PASS" } else { "FAIL" },
        row.verdict.exact,
        row.verdict.counter_indistinguishable,
        leak
    );
}

fn run<C: Counter>(cli: &Cli, tau_spec: TauSpec, policy: CapacityPolicy) -> Result<Report> {
    let mut rows = Vec::new();
    for &n in &cli.ns {
        for &frac in &cli.delete_fracs {
            let row = run_row::<C>(cli, tau_spec, policy, n, frac)?;
            print_row(&row);
            rows.push(row);
        }
    }
    let rows_pass = rows.iter().filter(|r| r.verdict.pass).count();
    let rows_baseline_not_applicable = rows.iter().filter(|r| r.verdict.baseline_leaks.is_none()).count();
    let overall = Overall {
        pass: rows_pass == rows.len() && !rows.is_empty(),
        rows_pass,
        rows_total: rows.len(),
        rows_baseline_not_applicable,
    };
    let generated_unix_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok(Report {
        schema: SCHEMA,
        generated_unix_s,
        config: Config {
            dim: cli.dim,
            ns: cli.ns.clone(),
            delete_fracs: cli.delete_fracs.clone(),
            tau: tau_spec.label(),
            counter: cli.counter,
            probes: cli.probes,
            survivor_sample: cli.survivor_sample,
            runs: cli.runs,
            warmup: cli.warmup,
            seed: cli.seed,
            sigma: cli.sigma,
            z_threshold: cli.z_threshold,
            policy,
            threads: 1,
        },
        shards: ShardNote {
            per_document: true,
            deletion_already_exact: true,
            note: "The engine's document-store shards keep one EntangledHVec per id (no superposition); remove() drops the map entry, removes the id's postings from the inverted index and invalidates the IVF/NSG snapshots, so deletion there is already exact. Counter bundles apply to the engine's superposition stores (bloom / counting-bloom / composite memories), which have no delete today.",
            evidence: vec![
                "src/core/engine/shard.rs:24-35 Shard { vectors: RwLock<FxHashMap<String, EntangledHVec>>, inverted, nsg, ivf, .. }",
                "src/core/engine/shard.rs:100-116 Shard::remove: vectors.remove(id); inverted.remove_doc(slot, old.indices()); ivf = None; nsg = None",
                "src/core/engine/mutation.rs:451-454,600 shards.remove(chunk, dims) / shards.insert(chunk, vector, dims)",
                "src/core/counting_bloom_memory.rs:31-61 CountingBloomMemory { counts: Vec<u32> } insert only, saturating_add, no delete",
                "src/core/entangled.rs:356-443 bundle (majority threshold, top-k truncation) and bundle_bloom (union)",
            ],
        },
        rows,
        overall,
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.dim == 0 {
        bail!("--dim must be > 0");
    }
    if cli.runs == 0 {
        bail!("--runs must be >= 1");
    }
    if cli.ns.is_empty() || cli.delete_fracs.is_empty() {
        bail!("--ns and --delete-fracs must be non-empty");
    }
    if cli.ns.iter().any(|&n| n == 0) {
        bail!("--ns entries must be >= 1");
    }
    if cli.delete_fracs.iter().any(|f| !(f.is_finite() && *f > 0.0 && *f <= 1.0)) {
        bail!("--delete-fracs must lie in (0, 1]");
    }
    let tau_spec = TauSpec::parse(&cli.tau)?;
    let policy = CapacityPolicy {
        max_items: cli.max_items.unwrap_or(usize::MAX),
        max_view_density: cli.max_view_density,
    };
    let counter_label = match cli.counter {
        CounterKind::I16 => "i16",
        CounterKind::I32 => "i32",
    };
    let out = cli.out.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "benchmarks/research/forget/out/forget_d{}_{}_tau{}.json",
            cli.dim,
            counter_label,
            tau_spec.label()
        ))
    });

    let report = match cli.counter {
        CounterKind::I16 => run::<i16>(&cli, tau_spec, policy)?,
        CounterKind::I32 => run::<i32>(&cli, tau_spec, policy)?,
    };

    println!(
        "[{}] {}/{} rows pass ({} rows with saturated baseline, leak check n/a)",
        if report.overall.pass { "PASS" } else { "FAIL" },
        report.overall.rows_pass,
        report.overall.rows_total,
        report.overall.rows_baseline_not_applicable
    );

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let json = serde_json::to_string_pretty(&report).context("serialising report")?;
    fs::write(&out, json).with_context(|| format!("writing {}", out.display()))?;
    println!("wrote {}", out.display());
    Ok(())
}
