// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//! Security-matrix probe (benchmarks/public/security_matrix.py drives it): seeds and opens
//! stores with and without encryption at rest, verifies signed audit entries, scans its own
//! memory for key material on Linux, and times the constant-time comparison the server uses.
use std::hint::black_box;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use subtle::ConstantTimeEq;
use writerslogic_warm::{
    core::{EmbeddingSpace, WarmConfig},
    DocumentInput, SearchOptions, WarmCore,
};
use zeroize::Zeroize;

const DIM: usize = 128;
const QUERIES: usize = 64;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create a store with `docs` synthetic documents (passphrase from stdin when `--encrypted`).
    Seed {
        #[arg(long)]
        store: PathBuf,
        #[arg(long, default_value_t = 2000)]
        docs: usize,
        #[arg(long)]
        encrypted: bool,
        #[arg(long)]
        audit: bool,
    },
    /// Open a store, run the fixed queries, print their results; with `--expect`, exit 0 when
    /// identical, 3 when they differ, 2 on any error.
    Open {
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        encrypted: bool,
        #[arg(long)]
        expect: Option<PathBuf>,
    },
    /// Seed then time `queries` single-thread queries; print QPS and peak RSS.
    Bench {
        #[arg(long)]
        store: PathBuf,
        #[arg(long, default_value_t = 2000)]
        docs: usize,
        #[arg(long, default_value_t = 2000)]
        queries: usize,
        #[arg(long)]
        encrypted: bool,
    },
    /// Verify every signed audit entry of a store against its stored signing key.
    AuditVerify {
        #[arg(long)]
        store: PathBuf,
    },
    /// Linux only: open an encrypted store, then scan this process's memory for the
    /// passphrase and the derived key, before and after dropping the core.
    Memscan {
        #[arg(long)]
        store: PathBuf,
    },
    /// Welch t-test of `ct_eq` (and the plain `==`) timing between equal and
    /// first-byte-different 32-byte secrets.
    Ct {
        #[arg(long, default_value_t = 200_000)]
        iters: usize,
    },
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> Vec<f64> {
        let v: Vec<f64> = (0..DIM)
            .map(|_| (self.next() % 2001) as f64 / 1000.0 - 1.0)
            .collect();
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
        v.into_iter().map(|x| x / n).collect()
    }
}

const WORDS: [&str; 24] = [
    "ledger", "harbor", "signal", "copper", "meadow", "quartz", "lantern", "timber", "orbit",
    "canyon", "velvet", "beacon", "marble", "fjord", "tundra", "saffron", "cobalt", "ember",
    "glacier", "willow", "prism", "anvil", "sonnet", "zephyr",
];

fn document(number: usize, rng: &mut SplitMix) -> DocumentInput {
    let text: Vec<&str> = (0..40).map(|_| WORDS[(rng.next() % 24) as usize]).collect();
    DocumentInput {
        id: format!("d{number:06}"),
        text: format!("document {number} {}", text.join(" ")),
        source_uri: Some(format!("probe://doc/{number}")),
        version: Some("1".into()),
        metadata: Some(json!({"number": number})),
        chunk_words: None,
        overlap_words: None,
        store_text: None,
        embeddings: Some(vec![rng.unit()]),
        title_weight: None,
    }
}

fn read_passphrase() -> Result<String> {
    let mut p = String::new();
    std::io::stdin().read_to_string(&mut p)?;
    let p = p.trim_end_matches(['\n', '\r']).to_string();
    if p.is_empty() {
        return Err(anyhow!("an encrypted store needs a passphrase on stdin"));
    }
    Ok(p)
}

fn config(passphrase: Option<String>, audit: bool) -> WarmConfig {
    let mut config = WarmConfig {
        embedding_space: Some(EmbeddingSpace {
            model: "synthetic-probe".into(),
            revision: "1".into(),
            dimensions: DIM,
            normalization: "l2".into(),
            metric: "cosine".into(),
        }),
        ..Default::default()
    };
    if let Some(p) = passphrase {
        config.security.encryption_enabled = true;
        config.security.encryption_passphrase = Some(p);
    }
    if audit {
        config.security.signing_enabled = true;
        config.security.audit_enabled = true;
    }
    config
}

fn open(store: &Path, passphrase: Option<String>, audit: bool) -> Result<WarmCore> {
    WarmCore::new(
        1024,
        Some(store.display().to_string()),
        Some(config(passphrase, audit)),
    )
}

fn seed(core: &WarmCore, docs: usize, audit: bool) -> Result<()> {
    let mut rng = SplitMix(11);
    for n in 0..docs {
        core.memorize_document(document(n, &mut rng))?;
    }
    if audit {
        // Document writes carry no audit entry; the vector API's memorize does.
        for n in 0..docs.min(64) {
            let v = core.encode_text(&format!("audited vector {n}"));
            core.memorize(format!("v{n:04}"), v)?;
        }
    }
    core.flush()?;
    Ok(())
}

fn queries(count: usize) -> Vec<Vec<f64>> {
    let mut rng = SplitMix(29);
    (0..count).map(|_| rng.unit()).collect()
}

fn run_queries(core: &WarmCore, count: usize) -> Result<Vec<Value>> {
    let mut out = Vec::with_capacity(count);
    for q in queries(count) {
        let options = SearchOptions {
            k: Some(10),
            embedding: Some(q),
            ..Default::default()
        };
        let results = core.search_documents("probe", &options)?;
        out.push(serde_json::to_value(results)?);
    }
    Ok(out)
}

fn dir_bytes(path: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for e in entries.flatten() {
            let p = e.path();
            total += if p.is_dir() {
                dir_bytes(&p)
            } else {
                p.metadata().map(|m| m.len()).unwrap_or(0)
            };
        }
    }
    total
}

fn welch_t(a: &[f64], b: &[f64]) -> f64 {
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    let var =
        |x: &[f64], m: f64| x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() - 1) as f64;
    let (ma, mb) = (mean(a), mean(b));
    let (va, vb) = (var(a, ma), var(b, mb));
    (ma - mb) / (va / a.len() as f64 + vb / b.len() as f64).sqrt()
}

fn ct_test(iters: usize) -> Value {
    let mut rng = SplitMix(7);
    let secret: Vec<u8> = (0..32).map(|_| (rng.next() & 0xff) as u8).collect();
    let equal = secret.clone();
    let mut first_differs = secret.clone();
    first_differs[0] ^= 0x80;
    let mut last_differs = secret.clone();
    last_differs[31] ^= 0x80;
    // Interleave the classes at random so drift affects all alike; each sample times a
    // batch of 256 calls of one class (the macOS timer ticks every ~42 ns).
    const BATCH: usize = 256;
    let mut classes = vec![vec![]; 3];
    let mut naive = vec![vec![]; 3];
    // One buffer receives the class's bytes before each sample, so every class is compared
    // at the same address and alignment (separate allocations would differ by cache line).
    let mut other = vec![0u8; 32];
    for _ in 0..iters {
        let which = (rng.next() % 3) as usize;
        other.copy_from_slice(match which {
            0 => &equal,
            1 => &first_differs,
            _ => &last_differs,
        });
        let other = &other;
        let t = Instant::now();
        let mut acc = 0u32;
        for _ in 0..BATCH {
            let ok: bool = black_box(&secret).ct_eq(black_box(other)).into();
            acc += u32::from(ok);
        }
        let ns = t.elapsed().as_nanos() as f64 / BATCH as f64;
        black_box(acc);
        classes[which].push(ns);
        let t = Instant::now();
        let mut acc = 0u32;
        for _ in 0..BATCH {
            acc += u32::from(black_box(&secret) == black_box(other));
        }
        let ns = t.elapsed().as_nanos() as f64 / BATCH as f64;
        black_box(acc);
        naive[which].push(ns);
    }
    let trim = |mut x: Vec<f64>| -> Vec<f64> {
        // dudect's upper-percentile cropping: drop the slowest 5% (interrupts, migrations).
        x.sort_by(f64::total_cmp);
        let keep = x.len() * 95 / 100;
        x.truncate(keep.max(2));
        x
    };
    let c: Vec<Vec<f64>> = classes.into_iter().map(trim).collect();
    let n: Vec<Vec<f64>> = naive.into_iter().map(trim).collect();
    let med = |x: &[f64]| x[x.len() / 2];
    json!({
        "samples_per_class": iters / 3,
        "calls_per_sample": BATCH,
        "secret_bytes": 32,
        "ct_eq": {
            "t_equal_vs_first_differs": welch_t(&c[0], &c[1]),
            "t_equal_vs_last_differs": welch_t(&c[0], &c[2]),
            "median_ns": [med(&c[0]), med(&c[1]), med(&c[2])],
        },
        "slice_eq": {
            "t_equal_vs_first_differs": welch_t(&n[0], &n[1]),
            "t_equal_vs_last_differs": welch_t(&n[0], &n[2]),
            "median_ns": [med(&n[0]), med(&n[1]), med(&n[2])],
        },
        "rule": "dudect threshold: |t| < 4.5 is indistinguishable at this sample size",
    })
}

#[cfg(feature = "security")]
fn audit_verify(store: &Path) -> Result<Value> {
    use ed25519_dalek::{Signature, SigningKey, Verifier};
    let core = open(store, None, true)?;
    let entries = core.audit_since(0)?;
    let mut key_bytes =
        std::fs::read(store.join("hms_signing.key")).context("reading hms_signing.key")?;
    let secret: [u8; 32] = key_bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("signing key is not 32 bytes"))?;
    let verifying = SigningKey::from_bytes(&secret).verifying_key();
    key_bytes.zeroize();
    let (mut verified, mut failed) = (0usize, 0usize);
    for e in &entries {
        let mut msg = Vec::with_capacity(41);
        msg.extend_from_slice(&e.timestamp_ms.to_le_bytes());
        msg.push(e.op as u8);
        msg.extend_from_slice(&e.id_hash);
        let sig = Signature::from_bytes(&e.signature);
        if verifying.verify(&msg, &sig).is_ok() {
            verified += 1;
        } else {
            failed += 1;
        }
    }
    Ok(json!({"entries": entries.len(), "verified": verified, "failed": failed}))
}

#[cfg(target_os = "linux")]
fn memscan(store: &Path) -> Result<Value> {
    use argon2::Argon2;
    use std::io::{Seek, SeekFrom};
    let mut pass = read_passphrase()?;
    let mask: Vec<u8> = {
        let mut rng = SplitMix(0xA5);
        (0..64).map(|_| (rng.next() & 0xff) as u8).collect()
    };
    let masked_pass: Vec<u8> = pass.bytes().zip(&mask).map(|(a, m)| a ^ m).collect();
    let salt = std::fs::read(store.join("encryption.salt")).context("encryption.salt")?;
    let mut key = [0u8; 32];
    Argon2::default()
        .hash_password_into(pass.as_bytes(), &salt, &mut key)
        .map_err(|e| anyhow!("argon2: {e}"))?;
    let masked_key: Vec<u8> = key.iter().zip(&mask).map(|(a, m)| a ^ m).collect();
    key.zeroize();
    // The probe's own derivation left the key in callee frames; scrub them so every copy
    // the scan finds belongs to the store or its cipher.
    writerslogic_warm::core::scrub_stack();
    let core = open(store, Some(pass.clone()), false)?;
    pass.zeroize();
    drop(pass);
    let _ = run_queries(&core, 4)?;
    let scan = |label: &str| -> Result<Value> {
        let maps = std::fs::read_to_string("/proc/self/maps")?;
        let mut mem = std::fs::File::open("/proc/self/mem")?;
        let (mut pass_hits, mut key_hits, mut bytes) = (0usize, 0usize, 0u64);
        for line in maps.lines() {
            let mut parts = line.split_whitespace();
            let (Some(range), Some(perms)) = (parts.next(), parts.next()) else {
                continue;
            };
            let name = parts.nth(3).unwrap_or("");
            if !perms.starts_with('r') || name.contains("vvar") || name.contains("vsyscall") {
                continue;
            }
            let (lo, hi) = range.split_once('-').context("maps range")?;
            let (lo, hi) = (u64::from_str_radix(lo, 16)?, u64::from_str_radix(hi, 16)?);
            if hi <= lo || hi - lo > 1 << 32 {
                continue;
            }
            let mut buf = vec![0u8; (hi - lo) as usize];
            if mem.seek(SeekFrom::Start(lo)).is_err() || mem.read_exact(&mut buf).is_err() {
                continue;
            }
            bytes += buf.len() as u64;
            let count = |needle: &[u8]| -> usize {
                buf.windows(needle.len())
                    .filter(|w| {
                        w.iter()
                            .zip(needle)
                            .zip(&mask)
                            .all(|((b, n), m)| b ^ m == *n)
                    })
                    .count()
            };
            pass_hits += count(&masked_pass);
            key_hits += count(&masked_key);
            // A freed scan buffer would otherwise be found by a later range or scan.
            buf.zeroize();
        }
        Ok(
            json!({"phase": label, "bytes_scanned": bytes, "passphrase_copies": pass_hits, "derived_key_copies": key_hits}),
        )
    };
    let live = scan("core_open")?;
    drop(core);
    let dropped = scan("core_dropped")?;
    Ok(json!({"host": "linux", "scans": [live, dropped],
        "note": "the needles are held XOR-masked, so every copy found belongs to the store or its cipher; a live AES-256 cipher keeps its key schedule (round keys 0 and 1 are the key) by necessity"}))
}

#[cfg(not(target_os = "linux"))]
fn memscan(_store: &Path) -> Result<Value> {
    Ok(
        json!({"excluded": "process memory scan runs on Linux (/proc/self/mem); this host is not Linux"}),
    )
}

#[cfg(not(feature = "security"))]
fn audit_verify(_store: &Path) -> Result<Value> {
    Err(anyhow!("built without the security feature"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Seed {
            store,
            docs,
            encrypted,
            audit,
        } => {
            let pass = if encrypted {
                Some(read_passphrase()?)
            } else {
                None
            };
            let t = Instant::now();
            let core = open(&store, pass, audit)?;
            seed(&core, docs, audit)?;
            let expect = run_queries(&core, QUERIES)?;
            drop(core);
            println!(
                "{}",
                json!({"docs": docs, "seed_s": t.elapsed().as_secs_f64(), "store_bytes": dir_bytes(&store),
                       "files": std::fs::read_dir(&store)?.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect::<Vec<_>>(),
                       "expected": expect})
            );
        }
        Cmd::Open {
            store,
            encrypted,
            expect,
        } => {
            let pass = if encrypted {
                Some(read_passphrase()?)
            } else {
                None
            };
            let outcome = (|| -> Result<Vec<Value>> {
                let core = open(&store, pass, false)?;
                run_queries(&core, QUERIES)
            })();
            match outcome {
                Err(e) => {
                    println!("{}", json!({"detected": e.to_string()}));
                    std::process::exit(2);
                }
                Ok(results) => {
                    if let Some(path) = expect {
                        let want: Value = serde_json::from_slice(&std::fs::read(path)?)?;
                        if want != Value::Array(results) {
                            println!("{}", json!({"served": "different results, no error"}));
                            std::process::exit(3);
                        }
                    }
                    println!("{}", json!({"served": "identical"}));
                }
            }
        }
        Cmd::Bench {
            store,
            docs,
            queries: n,
            encrypted,
        } => {
            let pass = if encrypted {
                Some(read_passphrase()?)
            } else {
                None
            };
            let t = Instant::now();
            let core = open(&store, pass, false)?;
            seed(&core, docs, false)?;
            let seed_s = t.elapsed().as_secs_f64();
            let qs = queries(n);
            let t = Instant::now();
            let mut hits = 0usize;
            for q in qs {
                let options = SearchOptions {
                    k: Some(10),
                    embedding: Some(q),
                    ..Default::default()
                };
                hits += core.search_documents("probe", &options)?.len();
            }
            let secs = t.elapsed().as_secs_f64();
            println!(
                "{}",
                json!({"docs": docs, "queries": n, "seed_s": seed_s, "query_s": secs, "qps": n as f64 / secs,
                       "hits": hits, "store_bytes": dir_bytes(&store),
                       "encrypted": encrypted})
            );
        }
        Cmd::AuditVerify { store } => {
            let v = audit_verify(&store)?;
            let failed = v["failed"].as_u64().unwrap_or(1);
            println!("{v}");
            if failed > 0 {
                std::process::exit(2);
            }
        }
        Cmd::Memscan { store } => println!("{}", memscan(&store)?),
        Cmd::Ct { iters } => println!("{}", ct_test(iters)),
    }
    Ok(())
}
