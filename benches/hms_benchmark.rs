use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use holographic_memory::core::algebra::HolographicAlgebra;
use holographic_memory::core::clifford::CliffordVec;
use holographic_memory::{EntangledHVec, HmsCore};

fn benchmark_entangled(c: &mut Criterion) {
    let dim = 16384;

    c.bench_function("EntangledHVec random D=16384", |b| {
        b.iter(|| EntangledHVec::new_deterministic(black_box(dim), black_box(42)))
    });

    let e1 = EntangledHVec::new_deterministic(dim, 1);
    let e2 = EntangledHVec::new_deterministic(dim, 2);

    c.bench_function("EntangledHVec hamming", |b| {
        b.iter(|| e1.hamming(black_box(&e2)))
    });

    c.bench_function("EntangledHVec similarity", |b| {
        b.iter(|| e1.similarity(black_box(&e2)))
    });

    c.bench_function("EntangledHVec bind", |b| b.iter(|| e1.bind(black_box(&e2))));

    c.bench_function("EntangledHVec permute", |b| {
        b.iter(|| e1.permute(black_box(7)))
    });

    let evecs: Vec<EntangledHVec> = (0..10)
        .map(|i| EntangledHVec::new_deterministic(dim, i))
        .collect();
    c.bench_function("EntangledHVec bundle 10", |b| {
        b.iter(|| EntangledHVec::bundle(black_box(&evecs)))
    });
}

fn benchmark_clifford(c: &mut Criterion) {
    let dim = 16384;

    c.bench_function("CliffordVec from_seed D=16384", |b| {
        b.iter(|| CliffordVec::from_seed(black_box(dim), black_box(42)))
    });

    let c1 = CliffordVec::from_seed(dim, 1);
    let c2 = CliffordVec::from_seed(dim, 2);

    c.bench_function("CliffordVec similarity", |b| {
        b.iter(|| c1.similarity(black_box(&c2)))
    });

    c.bench_function("CliffordVec bind (geometric product)", |b| {
        b.iter(|| c1.bind(black_box(&c2)))
    });

    c.bench_function("CliffordVec reverse", |b| {
        b.iter(|| c1.reverse())
    });

    c.bench_function("CliffordVec permute", |b| {
        b.iter(|| c1.permute(black_box(7)))
    });

    let cvecs: Vec<CliffordVec> = (0..10)
        .map(|i| CliffordVec::from_seed(dim, i))
        .collect();
    c.bench_function("CliffordVec bundle 10", |b| {
        b.iter(|| CliffordVec::bundle(black_box(&cvecs)))
    });

    let e1 = EntangledHVec::new_deterministic(dim, 1);
    c.bench_function("CliffordVec from_entangled", |b| {
        b.iter(|| CliffordVec::from_entangled(black_box(&e1)))
    });

    let cf = CliffordVec::from_entangled(&e1);
    c.bench_function("CliffordVec to_entangled", |b| {
        b.iter(|| cf.to_entangled(black_box(dim)))
    });
}

fn benchmark_comparison(c: &mut Criterion) {
    let dim = 16384;

    let mut group = c.benchmark_group("bind");
    let e1 = EntangledHVec::new_deterministic(dim, 1);
    let e2 = EntangledHVec::new_deterministic(dim, 2);
    let c1 = CliffordVec::from_seed(dim, 1);
    let c2 = CliffordVec::from_seed(dim, 2);

    group.bench_function("EntangledHVec XOR", |b| b.iter(|| e1.bind(black_box(&e2))));
    group.bench_function("CliffordVec GP", |b| b.iter(|| c1.bind(black_box(&c2))));
    group.finish();

    let mut group = c.benchmark_group("similarity");
    group.bench_function("EntangledHVec Jaccard", |b| {
        b.iter(|| e1.similarity(black_box(&e2)))
    });
    group.bench_function("CliffordVec reverse-inner", |b| {
        b.iter(|| c1.similarity(black_box(&c2)))
    });
    group.finish();

    let mut group = c.benchmark_group("bundle_10");
    let evecs: Vec<EntangledHVec> = (0..10)
        .map(|i| EntangledHVec::new_deterministic(dim, i))
        .collect();
    let cvecs: Vec<CliffordVec> = (0..10)
        .map(|i| CliffordVec::from_seed(dim, i))
        .collect();
    group.bench_function("EntangledHVec", |b| {
        b.iter(|| EntangledHVec::bundle(black_box(&evecs)))
    });
    group.bench_function("CliffordVec", |b| {
        b.iter(|| CliffordVec::bundle(black_box(&cvecs)))
    });
    group.finish();
}

fn benchmark_hms(c: &mut Criterion) {
    let dim = 10_000;
    let hms = HmsCore::new(dim as u32, None, None).unwrap();

    for i in 0..100 {
        let text = format!("This is a test sentence number {}", i);
        let vec = hms.encode_text(&text);
        hms.memorize(format!("id_{}", i), vec).unwrap();
    }

    c.bench_function("HmsCore query brute-force 100 items", |b| {
        b.iter(|| {
            let q_vec = hms.encode_text(black_box("test sentence"));
            hms.query(black_box(&q_vec), black_box(5))
        })
    });

    c.bench_function("HmsCore query_hopfield 100 items", |b| {
        b.iter(|| {
            let q_vec = hms.encode_text(black_box("test sentence"));
            hms.query_hopfield(black_box(&q_vec), black_box(5))
        })
    });

    c.bench_function("HmsCore encode_text short", |b| {
        b.iter(|| hms.encode_text(black_box("hello world")))
    });

    c.bench_function("HmsCore encode_text sentence", |b| {
        b.iter(|| {
            hms.encode_text(black_box(
                "The quick brown fox jumps over the lazy dog near the river",
            ))
        })
    });
}

fn benchmark_hms_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_scaling");
    group.sample_size(20);

    for &n in &[100, 500, 1000] {
        let dim = 10_000;
        let hms = HmsCore::new(dim as u32, None, None).unwrap();
        for i in 0..n {
            let text = format!("Sentence about topic {} with variation {}", i % 20, i);
            let vec = hms.encode_text(&text);
            hms.memorize(format!("id_{}", i), vec).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("brute_force", n), &n, |b, _| {
            b.iter(|| {
                let q = hms.encode_text(black_box("topic about variation"));
                hms.query(black_box(&q), black_box(5))
            })
        });

        group.bench_with_input(BenchmarkId::new("hopfield", n), &n, |b, _| {
            b.iter(|| {
                let q = hms.encode_text(black_box("topic about variation"));
                hms.query_hopfield(black_box(&q), black_box(5))
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_entangled,
    benchmark_clifford,
    benchmark_comparison,
    benchmark_hms,
    benchmark_hms_scaling,
);
criterion_main!(benches);
