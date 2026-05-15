use hms_native::core::entangled::EntangledHVec;
use hms_native::core::intersection::sparse_intersection_count;
use std::time::Instant;

fn main() {
    println!("HMS Micro-Benchmark: Sparse Intersection (sorted merge)");
    println!("------------------------------------------------------");

    let dim = 16384;
    let iterations = 1_000_000;

    // rho = 1/256
    let v1 = EntangledHVec::random(dim, 1);
    let v2 = EntangledHVec::random(dim, 2);

    let start = Instant::now();
    let mut total_intersection = 0;

    for _ in 0..iterations {
        // We use black_box or equivalent to prevent compiler elision
        let count = sparse_intersection_count(v1.indices(), v2.indices());
        total_intersection += count;
    }

    let duration = start.elapsed();
    let nanos_per_op = duration.as_nanos() as f64 / iterations as f64;

    println!("Dimensions: {}", dim);
    println!("Active Bits: ~{}", dim / 256);
    println!("Iterations: {}", iterations);
    println!("Total Time: {:?}", duration);
    println!("Average Time: {:.2} ns / op", nanos_per_op);
    println!("Throughput: {:.2} M ops/sec", (1000.0 / nanos_per_op));

    // Ensure result isn't optimized away
    if total_intersection == 0 {
        println!("Warning: 0 intersections found.");
    }
}
