// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use clap::Parser;
use nalgebra::Complex;
use serde::Serialize;
use std::mem::size_of;
use std::path::PathBuf;
use std::time::Instant;
use writerslogic_warm::core::algebraic_codebook::{
    PackedPhaseCodes, ValueCodeKind, ValueCodebook, MAX_DIM,
};
use writerslogic_warm::core::entangled::hash_u64;

const KINDS: [ValueCodeKind; 4] = [
    ValueCodeKind::UnpackedRandom,
    ValueCodeKind::PackedRandom,
    ValueCodeKind::RegeneratedRandom,
    ValueCodeKind::Walsh,
];
const KEY_CODES: usize = 28;

#[derive(Parser, Serialize)]
#[command(
    about = "Finite-alphabet holographic memory: implicit Walsh vs stored and regenerated random codes"
)]
struct Args {
    #[arg(long, default_value_t = 65536)]
    budget: usize,
    #[arg(long, default_value_t = 512)]
    dim: usize,
    #[arg(long, value_delimiter = ',', default_value = "16,64,256")]
    labels: Vec<usize>,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "8,16,24,32,48,64,96,128,192,256,384,512"
    )]
    facts: Vec<usize>,
    #[arg(long, default_value_t = 30)]
    seeds: usize,
    #[arg(long, default_value_t = 64)]
    queries: usize,
    #[arg(long, default_value_t = 20281008)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
}

impl Args {
    fn validate(&self) -> Result<()> {
        ensure!(
            (8192..=524288).contains(&self.budget),
            "budget must be in 8192..=524288"
        );
        ensure!(
            (1..=MAX_DIM).contains(&self.dim),
            "dimension must be in 1..=8192"
        );
        ensure!(
            !self.labels.is_empty()
                && self.labels.len() <= 5
                && self.labels.iter().all(|&n| (2..=256).contains(&n)
                    && n.is_power_of_two()
                    && self.dim.is_multiple_of(n)),
            "labels must be powers of two in 2..=256 dividing the fixed dimension"
        );
        ensure!(
            !self.facts.is_empty()
                && self.facts.len() <= 16
                && self.facts.iter().all(|&n| (1..=1024).contains(&n)),
            "fact loads must be in 1..=1024"
        );
        ensure!(
            self.facts.windows(2).all(|w| w[0] < w[1])
                && self.labels.windows(2).all(|w| w[0] < w[1]),
            "condition lists must be strictly increasing"
        );
        ensure!(
            (2..=64).contains(&self.seeds) && (1..=128).contains(&self.queries),
            "seeds must be in 2..=64 and queries in 1..=128"
        );
        ensure!(
            self.seed.checked_add(self.seeds as u64).is_some(),
            "seed range overflows"
        );
        ensure!(
            self.labels.len() * self.facts.len() * 3 * 2 * self.seeds * self.queries * KINDS.len()
                <= 2_000_000,
            "query count exceeds limit"
        );
        for &labels in &self.labels {
            for &facts in &self.facts {
                for kind in KINDS {
                    fit_dimension(kind, labels, facts, self.budget)?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Serialize)]
struct Corruption {
    name: &'static str,
    phase_replacement: f64,
    erasure: f64,
}

const CORRUPTIONS: [Corruption; 3] = [
    Corruption {
        name: "clean",
        phase_replacement: 0.0,
        erasure: 0.0,
    },
    Corruption {
        name: "phase_5_percent",
        phase_replacement: 0.05,
        erasure: 0.0,
    },
    Corruption {
        name: "erasure_10_percent",
        phase_replacement: 0.0,
        erasure: 0.1,
    },
];

#[derive(Clone, Copy)]
struct Witness {
    key: u16,
    value: u8,
}

impl Witness {
    fn subject(self) -> usize {
        usize::from(self.key & 15)
    }
    fn role0(self) -> usize {
        usize::from((self.key >> 4) & 7)
    }
    fn role1(self) -> usize {
        usize::from((self.key >> 7) & 7)
    }
    fn attribute(self) -> usize {
        usize::from(self.key >> 10)
    }
    fn relation_key(self) -> u16 {
        self.key >> 4
    }
}

struct Fixture {
    witnesses: Vec<Witness>,
    queries: Vec<usize>,
    crc32: u32,
}

fn fixture(args: &Args, facts: usize, labels: usize, seed: u64) -> Fixture {
    let stride = (hash_u64(seed ^ 0xA01, 0) as u16 & 4095) | 1;
    let offset = hash_u64(seed ^ 0xA02, 0) as u16 & 4095;
    let witnesses: Vec<_> = (0..facts)
        .map(|i| Witness {
            key: (stride.wrapping_mul(i as u16).wrapping_add(offset)) & 4095,
            value: (hash_u64(seed ^ 0xA03, i as u64) % labels as u64) as u8,
        })
        .collect();
    let count = facts.min(args.queries);
    let shift = hash_u64(seed ^ 0xA04, 0) as usize % facts;
    let queries = (0..count)
        .map(|i| (shift + i * facts / count) % facts)
        .collect();
    let mut crc = crc32fast::Hasher::new();
    for record in &witnesses {
        crc.update(&record.key.to_le_bytes());
        crc.update(&[record.value]);
    }
    Fixture {
        witnesses,
        queries,
        crc32: crc.finalize(),
    }
}

struct Model {
    dim: usize,
    codes: ValueCodebook,
    keys: PackedPhaseCodes,
    field: Vec<Complex<f64>>,
    witnesses: Vec<Witness>,
}

struct Workspace {
    cue: Vec<u16>,
    raw: Vec<Complex<f64>>,
    scores: Vec<f64>,
}

#[derive(Serialize, Debug, PartialEq)]
struct Memory {
    model_metadata: usize,
    field: usize,
    key_buffers: usize,
    value_buffers: usize,
    witnesses: usize,
    workspace_metadata: usize,
    cue: usize,
    raw: usize,
    scores: usize,
    model_bytes: usize,
    query_workspace_bytes: usize,
    total_bytes: usize,
}

fn memory(kind: ValueCodeKind, dim: usize, labels: usize, facts: usize) -> Memory {
    let model_metadata = size_of::<Model>();
    let field = dim * size_of::<Complex<f64>>();
    let key_buffers = (KEY_CODES * dim).div_ceil(2);
    let value_buffers = match kind {
        ValueCodeKind::UnpackedRandom => dim * labels * size_of::<u16>(),
        ValueCodeKind::PackedRandom => (dim * labels).div_ceil(2),
        _ => 0,
    };
    let witnesses = facts * size_of::<Witness>();
    let workspace_metadata = size_of::<Workspace>();
    let cue = dim * size_of::<u16>();
    let raw = dim * size_of::<Complex<f64>>();
    let scores = labels * size_of::<f64>();
    let model_bytes = model_metadata + field + key_buffers + value_buffers + witnesses;
    let query_workspace_bytes = workspace_metadata + cue + raw + scores;
    Memory {
        model_metadata,
        field,
        key_buffers,
        value_buffers,
        witnesses,
        workspace_metadata,
        cue,
        raw,
        scores,
        model_bytes,
        query_workspace_bytes,
        total_bytes: model_bytes + query_workspace_bytes,
    }
}

fn fit_dimension(kind: ValueCodeKind, labels: usize, facts: usize, budget: usize) -> Result<usize> {
    let mut low = 0;
    let mut high = MAX_DIM + 1;
    while high - low > 1 {
        let mid = (low + high) / 2;
        if memory(kind, mid, labels, facts).total_bytes <= budget {
            low = mid;
        } else {
            high = mid;
        }
    }
    if kind == ValueCodeKind::Walsh {
        low = low / labels * labels;
    }
    ensure!(
        low > 0,
        "budget cannot fit {:?} for {labels} labels and {facts} facts",
        kind
    );
    Ok(low)
}

impl Workspace {
    fn new(dim: usize, labels: usize) -> Self {
        Self {
            cue: vec![0; dim],
            raw: vec![Complex::new(0.0, 0.0); dim],
            scores: vec![0.0; labels],
        }
    }

    fn bytes(&self) -> usize {
        size_of::<Self>()
            + self.cue.capacity() * size_of::<u16>()
            + self.raw.capacity() * size_of::<Complex<f64>>()
            + self.scores.capacity() * size_of::<f64>()
    }
}

impl Model {
    fn new(
        kind: ValueCodeKind,
        dim: usize,
        labels: usize,
        data: &Fixture,
        seed: u64,
    ) -> Result<Self> {
        let model = Self {
            dim,
            codes: ValueCodebook::new(kind, dim, labels, seed ^ 0xB01)?,
            keys: PackedPhaseCodes::random(dim, KEY_CODES, seed ^ 0xB02)?,
            field: vec![Complex::new(0.0, 0.0); dim],
            witnesses: data.witnesses.clone(),
        };
        Ok(model)
    }

    fn key_phase(&self, subject: u16, roles: [usize; 2], attribute: usize, d: usize) -> u16 {
        (subject
            + self.keys.phase(16 + roles[0], (d + 1) % self.dim)
            + self.keys.phase(16 + roles[1], (d + 2) % self.dim)
            + self.keys.phase(24 + attribute, d))
            & 15
    }

    fn write(&mut self, record: Witness, sign: f64) {
        for d in 0..self.dim {
            let key = self.key_phase(
                self.keys.phase(record.subject(), d),
                [record.role0(), record.role1()],
                record.attribute(),
                d,
            );
            let phase = (key + self.codes.phase(usize::from(record.value), d)) & 15;
            self.field[d] += self.codes.phasor(phase) * sign;
        }
    }

    fn read(
        &self,
        roles: [usize; 2],
        attribute: usize,
        workspace: &mut Workspace,
    ) -> Result<usize> {
        for d in 0..self.dim {
            workspace.raw[d] = if workspace.cue[d] == 16 {
                Complex::new(0.0, 0.0)
            } else {
                let phase = self.key_phase(workspace.cue[d], roles, attribute, d);
                self.field[d] * self.codes.phasor(phase).conj()
            };
        }
        self.codes.scores(&workspace.raw, &mut workspace.scores)?;
        Ok(argmax(&workspace.scores))
    }

    fn bytes(&self) -> usize {
        size_of::<Self>() + self.codes.storage_bytes() - size_of::<ValueCodebook>()
            + self.keys.storage_bytes()
            - size_of::<PackedPhaseCodes>()
            + self.field.capacity() * size_of::<Complex<f64>>()
            + self.witnesses.capacity() * size_of::<Witness>()
    }
}

fn argmax(scores: &[f64]) -> usize {
    let mut best = (0, f64::NEG_INFINITY);
    for (i, &score) in scores.iter().enumerate() {
        if score > best.1 {
            best = (i, score);
        }
    }
    best.0
}

fn uniform(seed: u64, index: u64) -> f64 {
    ((hash_u64(seed, index) >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
}

fn cue(
    keys: &PackedPhaseCodes,
    subject: usize,
    index: usize,
    corruption: Corruption,
    seed: u64,
    out: &mut [u16],
) {
    for (d, p) in out.iter_mut().enumerate() {
        let counter = (index * MAX_DIM + d) as u64;
        *p = if uniform(seed ^ 0xC01, counter) < corruption.erasure {
            16
        } else if uniform(seed ^ 0xC02, counter) < corruption.phase_replacement {
            (hash_u64(seed ^ 0xC03, counter) & 15) as u16
        } else {
            keys.phase(subject, d)
        };
    }
}

fn p95(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * 95 / 100]
}

#[derive(Serialize)]
struct Run {
    seed: u64,
    fixture_crc32: u32,
    trace_crc32: u32,
    prediction_crc32: u32,
    dimensions: usize,
    memory: Memory,
    correct: usize,
    queries: usize,
    accuracy: f64,
    query_p95_us: f64,
    build_ms: f64,
    update_roundtrip_us: f64,
    update_max_abs_error: f64,
}

fn evaluate(
    data: &Fixture,
    kind: ValueCodeKind,
    dim: usize,
    labels: usize,
    corruption: Corruption,
    seed: u64,
) -> Result<Run> {
    let start = Instant::now();
    let mut model = Model::new(kind, dim, labels, data, seed)?;
    for &record in &data.witnesses {
        model.write(record, 1.0);
    }
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut workspace = Workspace::new(dim, labels);
    let memory = memory(kind, dim, labels, data.witnesses.len());
    ensure!(
        model.bytes() == memory.model_bytes && workspace.bytes() == memory.query_workspace_bytes,
        "memory formula disagrees with allocated capacities"
    );
    let reference = model.field.clone();
    let start = Instant::now();
    for _ in 0..16 {
        for &record in data.witnesses.iter().take(8) {
            model.write(record, -1.0);
        }
        for &record in data.witnesses.iter().take(8) {
            model.write(record, 1.0);
        }
    }
    let update_roundtrip_us = start.elapsed().as_secs_f64() * 1e6;
    let update_max_abs_error = model
        .field
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max);
    ensure!(
        update_max_abs_error < 1e-9,
        "write/remove roundtrip drift exceeds threshold"
    );
    let mut crc = crc32fast::Hasher::new();
    for z in &model.field {
        crc.update(&z.re.to_le_bytes());
        crc.update(&z.im.to_le_bytes());
    }
    let trace_crc32 = crc.finalize();
    let mut predictions = crc32fast::Hasher::new();
    let mut correct = 0;
    let mut times = Vec::with_capacity(data.queries.len());
    for (j, &index) in data.queries.iter().enumerate() {
        let record = data.witnesses[index];
        cue(
            &model.keys,
            record.subject(),
            index,
            corruption,
            seed,
            &mut workspace.cue,
        );
        if j == 0 {
            std::hint::black_box(model.read(
                [record.role0(), record.role1()],
                record.attribute(),
                &mut workspace,
            )?);
        }
        let start = Instant::now();
        let predicted = std::hint::black_box(model.read(
            [record.role0(), record.role1()],
            record.attribute(),
            &mut workspace,
        )?);
        times.push(start.elapsed().as_secs_f64() * 1e6);
        predictions.update(&(predicted as u16).to_le_bytes());
        correct += usize::from(predicted == usize::from(record.value));
    }
    Ok(Run {
        seed,
        fixture_crc32: data.crc32,
        trace_crc32,
        prediction_crc32: predictions.finalize(),
        dimensions: dim,
        memory,
        correct,
        queries: data.queries.len(),
        accuracy: correct as f64 / data.queries.len() as f64,
        query_p95_us: p95(&times),
        build_ms,
        update_roundtrip_us,
        update_max_abs_error,
    })
}

#[derive(Serialize)]
struct ExactRun {
    seed: u64,
    dimensions: usize,
    payload_bytes: usize,
    query_workspace_bytes: usize,
    total_bytes: usize,
    correct: usize,
    queries: usize,
    accuracy: f64,
    query_p95_us: f64,
}

fn exact_control(
    data: &Fixture,
    dim: usize,
    corruption: Corruption,
    seed: u64,
) -> Result<ExactRun> {
    let subjects = PackedPhaseCodes::random(dim, 16, seed ^ 0xB02)?;
    let book = ValueCodebook::new(ValueCodeKind::RegeneratedRandom, 1, 2, 0)?;
    let mut cue_buffer = vec![0; dim];
    let mut correct = 0;
    let mut times = Vec::with_capacity(data.queries.len());
    for &index in &data.queries {
        let query = data.witnesses[index];
        cue(
            &subjects,
            query.subject(),
            index,
            corruption,
            seed,
            &mut cue_buffer,
        );
        let start = Instant::now();
        let mut best = (0, f64::NEG_INFINITY);
        for record in &data.witnesses {
            if record.relation_key() != query.relation_key() {
                continue;
            }
            let score: f64 = cue_buffer
                .iter()
                .enumerate()
                .filter(|(_, p)| **p != 16)
                .map(|(d, &p)| {
                    book.phasor((p + 16 - subjects.phase(record.subject(), d)) & 15)
                        .re
                })
                .sum();
            if score > best.1 {
                best = (usize::from(record.value), score);
            }
        }
        std::hint::black_box(best);
        times.push(start.elapsed().as_secs_f64() * 1e6);
        correct += usize::from(best.0 == usize::from(query.value));
    }
    let payload_bytes = subjects.storage_bytes()
        + book.storage_bytes()
        + size_of::<Vec<Witness>>()
        + data.witnesses.len() * size_of::<Witness>();
    let query_workspace_bytes = size_of::<Vec<u16>>() + cue_buffer.capacity() * size_of::<u16>();
    Ok(ExactRun {
        seed,
        dimensions: dim,
        payload_bytes,
        query_workspace_bytes,
        total_bytes: payload_bytes + query_workspace_bytes,
        correct,
        queries: data.queries.len(),
        accuracy: correct as f64 / data.queries.len() as f64,
        query_p95_us: p95(&times),
    })
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn interval(values: &[f64], seed: u64) -> [f64; 2] {
    let mut draws: Vec<_> = (0..2000)
        .map(|i| {
            (0..values.len())
                .map(|j| {
                    values[(hash_u64(seed, (i * values.len() + j) as u64) % values.len() as u64)
                        as usize]
                })
                .sum::<f64>()
                / values.len() as f64
        })
        .collect();
    draws.sort_unstable_by(f64::total_cmp);
    [draws[49], draws[1949]]
}

fn main() -> Result<()> {
    let args = Args::parse();
    args.validate()?;
    let started = Instant::now();
    let mut rows = Vec::new();
    for mode in ["fixed_dimension", "fixed_total_budget"] {
        for &labels in &args.labels {
            for corruption in CORRUPTIONS {
                for &facts in &args.facts {
                    let dims: Vec<_> = KINDS
                        .into_iter()
                        .map(|kind| {
                            if mode == "fixed_dimension" {
                                Ok(args.dim)
                            } else {
                                fit_dimension(kind, labels, facts, args.budget)
                            }
                        })
                        .collect::<Result<_>>()?;
                    let mut runs: Vec<Vec<Run>> = (0..KINDS.len()).map(|_| Vec::new()).collect();
                    let mut exact = Vec::new();
                    for i in 0..args.seeds {
                        let seed = args.seed + i as u64;
                        let data = fixture(&args, facts, labels, seed);
                        for offset in 0..KINDS.len() {
                            let arm = (i + offset) % KINDS.len();
                            runs[arm].push(evaluate(
                                &data, KINDS[arm], dims[arm], labels, corruption, seed,
                            )?);
                        }
                        exact.push(exact_control(&data, dims[3], corruption, seed)?);
                    }
                    let arms: Vec<_> = KINDS.iter().enumerate().map(|(i, kind)| {
                        let scores: Vec<_> = runs[i].iter().map(|r| r.accuracy).collect();
                        serde_json::json!({"kind": kind, "accuracy": mean(&scores), "accuracy_ci95": interval(&scores, args.seed ^ 0xD01), "runs": runs[i]})
                    }).collect();
                    let contrasts: Vec<_> = (0..3).map(|i| {
                        let differences: Vec<_> = runs[3].iter().zip(&runs[i]).map(|(a, b)| a.accuracy - b.accuracy).collect();
                        let ratios: Vec<_> = runs[3].iter().zip(&runs[i]).map(|(a, b)| a.query_p95_us / b.query_p95_us).collect();
                        serde_json::json!({"control": KINDS[i], "accuracy_delta": mean(&differences), "accuracy_delta_ci95": interval(&differences, args.seed ^ 0xD02), "p95_ratio": mean(&ratios), "p95_ratio_ci95": interval(&ratios, args.seed ^ 0xD02)})
                    }).collect();
                    eprintln!("{mode} labels={labels} {} facts={facts} packed={:.3} regenerated={:.3} Walsh={:.3}", corruption.name, arms[1]["accuracy"].as_f64().unwrap(), arms[2]["accuracy"].as_f64().unwrap(), arms[3]["accuracy"].as_f64().unwrap());
                    rows.push(serde_json::json!({"mode": mode, "labels": labels, "corruption": corruption, "facts": facts, "arms": arms, "contrasts": contrasts, "exact_control": exact}));
                }
            }
        }
    }
    let report = serde_json::json!({
        "schema_version": 1, "experiment": "implicit-algebraic-codebook", "args": args,
        "protocol": {
            "selection": "No fitted parameters or arm selection. Fixed label/load/corruption grid, disjoint from earlier experiment seeds. Same fixture and prefix-coupled code/corruption seeds across arms; execution order rotates by seed.",
            "encoding": "Two ordered relation roles, 16 subjects and four attributes. Common bit-packed four-bit key banks. Random value codes are identical across their three representations at equal D; Walsh uses repeated orthogonal real characters. Only subject phase cues are corrupted; exact ordered role and attribute IDs are supplied to every arm.",
            "memory": "Budget includes allocated-capacity model payload, finite-alphabet mappings as fixed integer labels, source tuple witnesses, lookup table, key/value banks, raw accumulator and preallocated cue/readout/score workspace. No padding is charged; Walsh rounds dimension down to a multiple of L. Fixture copies, timing/report data and the diagnostic checkpoint used to measure update drift are excluded. Allocator overhead, RSS and text/provenance payloads are not modeled.",
            "timing": "Query includes positional key encoding, unbinding, scoring and argmax, including regenerated-code computation. Allocations precede timing and their workspace is charged. Cue corruption precedes timing; first query is warmed. Local p95 is indicative, not load-gated production throughput.",
            "updates": "Before queries, remove and reinsert the first min(8,M) facts for 16 cycles. Report cumulative update time and maximum complex-coordinate drift against an excluded diagnostic checkpoint.",
            "acceptance": "Primary: twice the largest contiguous tested load whose held-out recall CI lower bound is >=0.95, at fixed total budget and no worse p95 than each eligible random control. Secondary decoder result: Walsh recall CI lower >=0.95, paired accuracy delta CI lower >=-0.005 and paired p95 ratio CI upper <=0.5 against all three controls at the same load. Bootstrap intervals are exploratory without multiplicity correction. Neither result establishes novelty or superiority over exact attributed storage."
        },
        "source_crc32": {"codebook": crc32fast::hash(include_bytes!("../../core/experimental/algebraic_codebook.rs")), "harness": crc32fast::hash(include_bytes!("algebraic-codebook-bench.rs")), "transform": crc32fast::hash(include_bytes!("../../core/qgraph/kernels.rs")), "hash": crc32fast::hash(include_bytes!("../../core/entangled.rs"))},
        "layout": {"model_metadata": size_of::<Model>(), "workspace_metadata": size_of::<Workspace>(), "witness": size_of::<Witness>(), "packed_codes": size_of::<PackedPhaseCodes>(), "value_codebook": size_of::<ValueCodebook>(), "vec": size_of::<Vec<u16>>()},
        "platform": {"arch": std::env::consts::ARCH, "os": std::env::consts::OS, "usize_bytes": size_of::<usize>()},
        "elapsed_seconds": started.elapsed().as_secs_f64(), "rows": rows,
    });
    if let Some(parent) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", args.out.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args::parse_from(["algebraic-codebook-bench", "--out", "unused.json"])
    }

    #[test]
    fn total_budget_pays_for_witnesses_and_workspace_and_uses_maximal_dimension() {
        let args = args();
        let data = fixture(&args, 128, 64, 2);
        for kind in KINDS {
            let dim = fit_dimension(kind, 64, 128, args.budget).unwrap();
            let model = Model::new(kind, dim, 64, &data, 2).unwrap();
            let workspace = Workspace::new(dim, 64);
            let counted = memory(kind, dim, 64, 128);
            assert_eq!(model.bytes(), counted.model_bytes);
            assert_eq!(workspace.bytes(), counted.query_workspace_bytes);
            assert!(counted.total_bytes <= args.budget);
            let next = dim + if kind == ValueCodeKind::Walsh { 64 } else { 1 };
            assert!(memory(kind, next, 64, 128).total_bytes > args.budget);
            for budget in [
                counted.total_bytes - 1,
                counted.total_bytes,
                counted.total_bytes + 1,
            ] {
                let fitting = fit_dimension(kind, 64, 128, budget).unwrap();
                assert!(memory(kind, fitting, 64, 128).total_bytes <= budget);
                assert_eq!(fitting >= dim, budget >= counted.total_bytes);
            }
        }
    }

    #[test]
    fn one_fact_recall_and_deletion_preserve_the_additive_field() {
        let args = args();
        let data = fixture(&args, 2, 64, 3);
        for kind in KINDS {
            let mut model = Model::new(kind, 512, 64, &data, 3).unwrap();
            let mut workspace = Workspace::new(512, 64);
            let record = data.witnesses[0];
            model.write(record, 1.0);
            for corruption in CORRUPTIONS {
                cue(
                    &model.keys,
                    record.subject(),
                    0,
                    corruption,
                    3,
                    &mut workspace.cue,
                );
                assert_eq!(
                    model
                        .read(
                            [record.role0(), record.role1()],
                            record.attribute(),
                            &mut workspace
                        )
                        .unwrap(),
                    usize::from(record.value)
                );
            }
            let reference = model.field.clone();
            model.write(data.witnesses[1], 1.0);
            model.write(data.witnesses[1], -1.0);
            assert!(reference
                .iter()
                .zip(&model.field)
                .all(|(a, b)| (a - b).norm() < 1e-14));
            for _ in 0..256 {
                model.write(record, -1.0);
                model.write(record, 1.0);
            }
            assert!(reference
                .iter()
                .zip(&model.field)
                .all(|(a, b)| (a - b).norm() < 1e-12));
        }
    }

    #[test]
    fn fixtures_are_unique_and_cue_noise_is_prefix_coupled() {
        let args = args();
        let data = fixture(&args, 1024, 256, 4);
        let unique: std::collections::BTreeSet<_> = data.witnesses.iter().map(|r| r.key).collect();
        assert_eq!(unique.len(), 1024);
        let small = PackedPhaseCodes::random(64, 28, 9).unwrap();
        let large = PackedPhaseCodes::random(128, 28, 9).unwrap();
        for corruption in CORRUPTIONS {
            let mut a = vec![0; 64];
            let mut b = vec![0; 128];
            cue(&small, 3, 7, corruption, 10, &mut a);
            cue(&large, 3, 7, corruption, 10, &mut b);
            assert_eq!(a, b[..64]);
        }
    }
}
