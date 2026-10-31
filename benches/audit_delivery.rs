// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::{Duration, Instant};
use writerslogic_warm::core::WarmConfig;
use writerslogic_warm::{EntangledHVec, WarmCore};

const OPERATIONS: u64 = 16;

fn benchmark_audit_delivery(c: &mut Criterion) {
    let mut group = c.benchmark_group("durable_vector_writes");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(OPERATIONS));
    let modes = [("disabled", false, false), ("unsigned", true, false)];
    #[cfg(feature = "security")]
    let modes = [modes[0], modes[1], ("signed", true, true)];
    for (name, audit, signing) in modes {
        group.bench_function(BenchmarkId::new(name, OPERATIONS), |b| {
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let directory = tempfile::tempdir().unwrap();
                    let mut config = WarmConfig::default();
                    config.security.audit_enabled = audit;
                    config.security.signing_enabled = signing;
                    let core = WarmCore::new(
                        256,
                        Some(directory.path().display().to_string()),
                        Some(config.clone()),
                    )
                    .unwrap();
                    let inputs: Vec<_> = (0..OPERATIONS)
                        .map(|i| {
                            (
                                format!("record-{i}"),
                                EntangledHVec::new_deterministic(256, i),
                            )
                        })
                        .collect();
                    let start = Instant::now();
                    for (id, vector) in inputs {
                        core.memorize(id, vector).unwrap();
                    }
                    elapsed += start.elapsed();
                    assert_eq!(core.pending_audit_effects(), 0);
                    drop(core);
                    let reopened = WarmCore::new(
                        256,
                        Some(directory.path().display().to_string()),
                        Some(config),
                    )
                    .unwrap();
                    assert_eq!(reopened.pending_audit_effects(), 0);
                    let entries = reopened.audit_since(0).unwrap();
                    assert_eq!(entries.len(), if audit { OPERATIONS as usize } else { 0 });
                    for i in 0..OPERATIONS {
                        let vector = EntangledHVec::new_deterministic(256, i);
                        assert_eq!(reopened.query(&vector, 1)[0].id, format!("record-{i}"));
                    }
                    #[cfg(feature = "security")]
                    if signing {
                        use std::io::Read;
                        let mut secret = zeroize::Zeroizing::new([0; 32]);
                        std::fs::File::open(directory.path().join("hms_signing.key"))
                            .unwrap()
                            .read_exact(&mut *secret)
                            .unwrap();
                        let verifying =
                            ed25519_dalek::SigningKey::from_bytes(&secret).verifying_key();
                        for entry in entries {
                            verifying
                                .verify_strict(
                                    &entry.signable_bytes(),
                                    &ed25519_dalek::Signature::from_bytes(&entry.signature),
                                )
                                .unwrap();
                        }
                    }
                }
                elapsed
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_audit_delivery);
criterion_main!(benches);
