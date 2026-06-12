use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hms_native::{EntangledHVec, HmsCore};

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

    // Bundle of 10 sparse vectors
    let evecs: Vec<EntangledHVec> = (0..10).map(|i| EntangledHVec::new_deterministic(dim, i)).collect();
    c.bench_function("EntangledHVec bundle 10", |b| {
        b.iter(|| EntangledHVec::bundle(black_box(&evecs)))
    });
}

fn benchmark_hms(c: &mut Criterion) {
    let dim = 10_000;
    let hms = HmsCore::new(dim as u32, None, None).unwrap();

    // Pre-populate
    for i in 0..100 {
        let text = format!("This is a test sentence number {}", i);
        let vec = hms.encode_text(&text);
        hms.memorize(format!("id_{}", i), vec).unwrap();
    }

    c.bench_function("HmsCore query 100 items", |b| {
        b.iter(|| {
            let q_vec = hms.encode_text(black_box("test sentence"));
            hms.query(black_box(&q_vec), black_box(5))
        })
    });
}

criterion_group!(benches, benchmark_entangled, benchmark_hms);
criterion_main!(benches);
