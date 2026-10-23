// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#[test]
fn audit2_rate_limit_churn_cost_and_bound() {
    let limiter = RateLimiter::new(1.0, 1).unwrap();
    let now = Instant::now();
    let first = Instant::now();
    for i in 0..MAX_BUCKETS {
        limiter.check_at(&format!("tenant{i}"), now).unwrap();
    }
    let below = first.elapsed();
    let start = Instant::now();
    for i in MAX_BUCKETS..MAX_BUCKETS + 1000 {
        assert!(limiter.check_at(&format!("tenant{i}"), now).is_err());
    }
    let above = start.elapsed();
    let count = limiter.buckets.lock().entries.len();
    eprintln!(
        "AUDIT2 rate: first_10000_us={} next_1000_us={} buckets={count}",
        below.as_micros(),
        above.as_micros()
    );
    assert!(
        count <= MAX_BUCKETS,
        "bucket map exceeds its intended bound"
    );
    assert!(limiter
        .check_at("after-refill", now + Duration::from_secs(1))
        .is_ok());
}

#[test]
fn audit2_capacity_pressure_reexecutes_running_request() {
    let store = Idempotency::new(Duration::from_secs(60), 1);
    let fp = fingerprint("POST", "/x", "t", b"body");
    let Begin::Proceed(first) = store.begin("t", "first", fp).unwrap() else {
        panic!()
    };
    assert!(store.begin("t", "second", fp).is_err());
    let retry = store.begin("t", "first", fp);
    assert!(
        retry.is_err(),
        "a retry is executing while its original request is still running"
    );
    drop(first);
    assert!(matches!(
        store.begin("t", "second", fp).unwrap(),
        Begin::Proceed(_)
    ));
}
