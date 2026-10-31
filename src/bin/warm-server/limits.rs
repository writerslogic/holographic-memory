// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Admission control: bounded work slots with an optional bounded wait queue, per-tenant token
//! buckets, and idempotency-key replay.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use crate::error::ApiError;

pub struct WorkLimit {
    pub slots: Arc<Semaphore>,
    pub capacity: usize,
    accepting: AtomicBool,
    max_queue: usize,
    queue_timeout: Duration,
    queued: AtomicUsize,
}

struct Queued<'a>(&'a AtomicUsize);

impl Drop for Queued<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl WorkLimit {
    pub fn new(capacity: usize, max_queue: usize, queue_timeout: Duration) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(capacity)),
            capacity,
            accepting: AtomicBool::new(true),
            max_queue,
            queue_timeout,
            queued: AtomicUsize::new(0),
        }
    }

    pub fn in_use(&self) -> usize {
        self.capacity.saturating_sub(self.slots.available_permits())
    }

    pub fn queue_depth(&self) -> usize {
        self.queued.load(Ordering::Acquire)
    }

    fn closed() -> ApiError {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "shutting_down",
            "server is shutting down",
        )
    }

    fn busy() -> ApiError {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_busy",
            "server work capacity is full; retry later",
        )
    }

    /// Take a work slot, waiting in the bounded queue if one is configured.
    pub async fn acquire(&self) -> Result<OwnedSemaphorePermit, ApiError> {
        if !self.accepting.load(Ordering::Acquire) {
            return Err(Self::closed());
        }
        let permit = match self.slots.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::Closed) => return Err(Self::closed()),
            Err(TryAcquireError::NoPermits) => {
                if self.max_queue == 0 {
                    return Err(Self::busy());
                }
                let depth = self.queued.fetch_add(1, Ordering::AcqRel);
                let _queued = Queued(&self.queued);
                if depth >= self.max_queue {
                    return Err(Self::busy());
                }
                match tokio::time::timeout(self.queue_timeout, self.slots.clone().acquire_owned())
                    .await
                {
                    Ok(Ok(permit)) => permit,
                    Ok(Err(_)) => return Err(Self::closed()),
                    Err(_) => return Err(Self::busy()),
                }
            }
        };
        if !self.accepting.load(Ordering::Acquire) {
            return Err(Self::closed());
        }
        Ok(permit)
    }

    pub fn stop(&self) {
        self.accepting.store(false, Ordering::Release);
    }
}

struct Bucket {
    tokens: f64,
    last: Instant,
}

pub struct RateLimiter {
    rate: f64,
    burst: f64,
    buckets: Mutex<RateState>,
}

#[derive(Default)]
struct RateState {
    entries: HashMap<String, Bucket>,
    next_sweep: Option<Instant>,
}

const MAX_BUCKETS: usize = 10_000;

impl RateLimiter {
    /// `None` when `rps` is zero.
    pub fn new(rps: f64, burst: u32) -> Option<Self> {
        (rps > 0.0).then(|| Self {
            rate: rps,
            burst: if burst == 0 {
                rps.ceil().max(1.0)
            } else {
                f64::from(burst)
            },
            buckets: Mutex::new(RateState::default()),
        })
    }

    /// Consume one token for `tenant`, or return the seconds to wait.
    pub fn check(&self, tenant: &str) -> Result<(), u64> {
        self.check_at(tenant, Instant::now())
    }

    fn check_at(&self, tenant: &str, now: Instant) -> Result<(), u64> {
        let mut buckets = self.buckets.lock();
        let known = buckets.entries.contains_key(tenant);
        if !known && buckets.entries.len() >= MAX_BUCKETS {
            // A bucket idle long enough to be full again is indistinguishable from a new one.
            if buckets.next_sweep.is_none_or(|next| now >= next) {
                buckets.entries.retain(|_, b| {
                    b.tokens + now.saturating_duration_since(b.last).as_secs_f64() * self.rate
                        < self.burst
                });
                buckets.next_sweep = now.checked_add(Duration::from_secs(1));
            }
            if buckets.entries.len() >= MAX_BUCKETS {
                return Err(1);
            }
        }
        if !known {
            // The tenant name is copied only for a new bucket, not on every request.
            buckets.entries.insert(
                tenant.to_string(),
                Bucket {
                    tokens: self.burst,
                    last: now,
                },
            );
        }
        let bucket = buckets
            .entries
            .get_mut(tenant)
            .expect("the tenant's bucket exists or was just inserted");
        let elapsed = now.saturating_duration_since(bucket.last).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * self.rate).min(self.burst);
        bucket.last = now;
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            Ok(())
        } else {
            Err(((1.0 - bucket.tokens) / self.rate).ceil().max(1.0) as u64)
        }
    }
}

#[derive(Clone)]
pub struct StoredResponse {
    pub status: StatusCode,
    pub body: bytes::Bytes,
    allocated_bytes: usize,
    committed: bool,
}

impl StoredResponse {
    pub fn new(status: StatusCode, body: &serde_json::Value, committed: bool) -> Self {
        let serialized = body.to_string();
        Self {
            status,
            allocated_bytes: serialized.capacity(),
            body: serialized.into(),
            committed,
        }
    }
}

enum State {
    Running,
    Done(StoredResponse),
    Unreplayable,
}

struct Record {
    fingerprint: [u8; 32],
    state: State,
}

type Key = (String, String);

#[derive(Default)]
struct Inner {
    records: HashMap<Key, Record>,
    completed: VecDeque<(Instant, Key)>,
    body_bytes: usize,
}

impl Inner {
    fn remove(&mut self, key: &Key) {
        if let Some(Record {
            state: State::Done(response),
            ..
        }) = self.records.remove(key)
        {
            self.body_bytes = self
                .body_bytes
                .checked_sub(response.allocated_bytes)
                .expect("retained replay bytes are accounted");
        }
    }
}

pub struct Idempotency {
    ttl: Duration,
    capacity: usize,
    max_bytes: usize,
    inner: Arc<Mutex<Inner>>,
}

pub enum Begin {
    Proceed(IdempotencyGuard),
    Replay(StoredResponse),
}

pub struct IdempotencyGuard {
    inner: Arc<Mutex<Inner>>,
    key: Key,
    done: bool,
    max_bytes: usize,
}

impl IdempotencyGuard {
    /// Retain the outcome or a reconciliation marker; only known transient failures release the key.
    pub fn complete(mut self, response: &StoredResponse) {
        let mut inner = self.inner.lock();
        let retryable = !response.committed
            && (response.status.is_server_error() || matches!(response.status.as_u16(), 408 | 429));
        if retryable {
            inner.remove(&self.key);
        } else {
            let state =
                if response.allocated_bytes <= self.max_bytes.saturating_sub(inner.body_bytes) {
                    inner.body_bytes += response.allocated_bytes;
                    State::Done(response.clone())
                } else {
                    State::Unreplayable
                };
            let now = Instant::now();
            let record = inner
                .records
                .get_mut(&self.key)
                .expect("running request retains its key");
            record.state = state;
            inner.completed.push_back((now, self.key.clone()));
        }
        self.done = true;
    }
}

impl Drop for IdempotencyGuard {
    fn drop(&mut self) {
        if !self.done {
            let mut inner = self.inner.lock();
            let now = Instant::now();
            let record = inner
                .records
                .get_mut(&self.key)
                .expect("running request retains its key");
            record.state = State::Unreplayable;
            inner.completed.push_back((now, self.key.clone()));
        }
    }
}

pub fn fingerprint(method: &str, path: &str, tenant: &str, body: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    for part in [method.as_bytes(), path.as_bytes(), tenant.as_bytes()] {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.update(body);
    h.finalize().into()
}

impl Idempotency {
    pub fn new(ttl: Duration, capacity: usize, max_bytes: usize) -> Self {
        Self {
            ttl,
            capacity,
            max_bytes,
            inner: Arc::new(Mutex::new(Inner::default())),
        }
    }

    pub fn usage(&self) -> (usize, usize, usize) {
        let inner = self.inner.lock();
        (
            inner.records.len(),
            inner.body_bytes,
            inner
                .records
                .values()
                .filter(|r| matches!(r.state, State::Unreplayable))
                .count(),
        )
    }

    pub fn begin(&self, tenant: &str, key: &str, fingerprint: [u8; 32]) -> Result<Begin, ApiError> {
        self.begin_at(tenant, key, fingerprint, Instant::now())
    }

    fn begin_at(
        &self,
        tenant: &str,
        key: &str,
        fingerprint: [u8; 32],
        now: Instant,
    ) -> Result<Begin, ApiError> {
        let id = (tenant.to_string(), key.to_string());
        let mut inner = self.inner.lock();
        while inner
            .completed
            .front()
            .is_some_and(|(created, _)| now.saturating_duration_since(*created) >= self.ttl)
        {
            let (_, expired) = inner
                .completed
                .pop_front()
                .expect("expired completion exists");
            inner.remove(&expired);
        }
        if let Some(record) = inner.records.get(&id) {
            if record.fingerprint != fingerprint {
                return Err(ApiError::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "idempotency_key_reused",
                    "this Idempotency-Key was already used with a different request",
                ));
            }
            return match &record.state {
                    State::Done(response) => Ok(Begin::Replay(response.clone())),
                    State::Running => Err(ApiError::new(
                        StatusCode::CONFLICT,
                        "idempotency_in_flight",
                        "a request with this Idempotency-Key is still running",
                    )
                    .retry_after(1)),
                    State::Unreplayable => Err(ApiError::new(
                        StatusCode::CONFLICT,
                        "idempotency_reconciliation_required",
                        "this request may have executed but its response is unavailable; reconcile state before retrying",
                    )),
                };
        }
        if inner.records.len() >= self.capacity {
            let retry_after = inner.completed.front().map_or(1, |(completed, _)| {
                let remaining = self
                    .ttl
                    .saturating_sub(now.saturating_duration_since(*completed));
                remaining
                    .as_secs()
                    .saturating_add(u64::from(remaining.subsec_nanos() != 0))
                    .max(1)
            });
            return Err(ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_busy",
                "idempotency capacity is full; retry later",
            )
            .retry_after(retry_after));
        }
        inner.records.insert(
            id.clone(),
            Record {
                fingerprint,
                state: State::Running,
            },
        );
        Ok(Begin::Proceed(IdempotencyGuard {
            inner: self.inner.clone(),
            key: id,
            done: false,
            max_bytes: self.max_bytes,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rate_limit_bucket_cap_rejects_new_tenants_when_full() {
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
            "rate: first_10000_us={} next_1000_us={} buckets={count}",
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
    fn idempotency_keeps_running_entries_at_capacity() {
        let store = Idempotency::new(Duration::from_secs(60), 1, 1024);
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
        assert_eq!(
            store.begin("t", "first", fp).err().unwrap().code,
            "idempotency_reconciliation_required"
        );
        assert_eq!(
            store.begin("t", "second", fp).err().unwrap().code,
            "server_busy"
        );
    }

    #[tokio::test]
    async fn queue_admits_up_to_its_bound_then_sheds() {
        let work = Arc::new(WorkLimit::new(1, 1, Duration::from_millis(300)));
        let held = work.acquire().await.unwrap();
        let waiter = {
            let work = work.clone();
            tokio::spawn(async move { work.acquire().await.map(|_| ()) })
        };
        while work.queue_depth() == 0 {
            tokio::task::yield_now().await;
        }
        assert_eq!(work.acquire().await.unwrap_err().code, "server_busy");
        drop(held);
        waiter.await.unwrap().unwrap();
        assert_eq!(work.queue_depth(), 0);
    }

    #[tokio::test]
    async fn zero_queue_rejects_immediately_and_timeout_sheds() {
        let work = WorkLimit::new(1, 0, Duration::from_millis(50));
        let _held = work.acquire().await.unwrap();
        assert_eq!(work.acquire().await.unwrap_err().code, "server_busy");
        let work = WorkLimit::new(1, 4, Duration::from_millis(50));
        let _held = work.acquire().await.unwrap();
        let started = Instant::now();
        assert_eq!(work.acquire().await.unwrap_err().code, "server_busy");
        assert!(started.elapsed() >= Duration::from_millis(40));
        work.stop();
        assert_eq!(work.acquire().await.unwrap_err().code, "shutting_down");
    }

    #[test]
    fn token_bucket_allows_burst_then_refills_at_rate() {
        let limiter = RateLimiter::new(10.0, 3).unwrap();
        let t0 = Instant::now();
        for _ in 0..3 {
            limiter.check_at("a", t0).unwrap();
        }
        assert_eq!(limiter.check_at("a", t0).unwrap_err(), 1);
        limiter.check_at("b", t0).unwrap();
        // 0.1 s refills one token at 10/s.
        limiter
            .check_at("a", t0 + Duration::from_millis(110))
            .unwrap();
        assert!(limiter
            .check_at("a", t0 + Duration::from_millis(111))
            .is_err());
        assert!(RateLimiter::new(0.0, 5).is_none());
    }

    #[test]
    fn idempotency_replays_rejects_mismatch_and_retries_after_failure() {
        let store = Idempotency::new(Duration::from_secs(60), 8, 1024);
        let fp = fingerprint("POST", "/x", "t", b"body");
        let Begin::Proceed(guard) = store.begin("t", "k", fp).unwrap() else {
            panic!()
        };
        assert_eq!(
            store.begin("t", "k", fp).err().unwrap().code,
            "idempotency_in_flight"
        );
        let ok = StoredResponse::new(StatusCode::OK, &serde_json::json!({"added": 1}), false);
        guard.complete(&ok);
        let Begin::Replay(r) = store.begin("t", "k", fp).unwrap() else {
            panic!()
        };
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&r.body).unwrap(),
            serde_json::json!({"added": 1})
        );
        assert_eq!(
            r.body.as_ptr(),
            ok.body.as_ptr(),
            "replays share the serialized allocation"
        );
        let other = fingerprint("POST", "/x", "t", b"other");
        assert_eq!(
            store.begin("t", "k", other).err().unwrap().code,
            "idempotency_key_reused"
        );
        // Another tenant's identical key is independent.
        assert!(matches!(
            store.begin("u", "k", fp).unwrap(),
            Begin::Proceed(_)
        ));
        // A failed attempt is not recorded.
        let Begin::Proceed(guard) = store.begin("t", "f", fp).unwrap() else {
            panic!()
        };
        guard.complete(&StoredResponse::new(
            StatusCode::SERVICE_UNAVAILABLE,
            &serde_json::json!({}),
            false,
        ));
        assert!(matches!(
            store.begin("t", "f", fp).unwrap(),
            Begin::Proceed(_)
        ));
    }

    #[test]
    fn idempotency_capacity_is_bounded() {
        let store = Idempotency::new(Duration::from_secs(60), 4, 1024);
        let fp = fingerprint("POST", "/x", "t", b"");
        for i in 0..50 {
            let result = store.begin("t", &format!("k{i}"), fp);
            if i < 4 {
                let Begin::Proceed(g) = result.unwrap() else {
                    panic!()
                };
                g.complete(&StoredResponse::new(
                    StatusCode::OK,
                    &serde_json::json!({}),
                    false,
                ));
            } else {
                assert_eq!(result.err().unwrap().code, "server_busy");
            }
            assert!(store.usage().0 <= 4);
        }
        for i in 0..4 {
            assert!(matches!(
                store.begin("t", &format!("k{i}"), fp).unwrap(),
                Begin::Replay(_)
            ));
        }
    }

    #[test]
    fn replay_byte_budget_preserves_keys_at_allocation_boundaries() {
        let response = StoredResponse::new(
            StatusCode::OK,
            &serde_json::json!({"result": "x".repeat(1024)}),
            false,
        );
        let size = response.allocated_bytes;
        for limit in [size - 1, size, size + 1] {
            let store = Idempotency::new(Duration::from_secs(60), 2, limit);
            let Begin::Proceed(g) = store.begin("t", "k", [1; 32]).unwrap() else {
                panic!()
            };
            g.complete(&response);
            if limit < size {
                assert_eq!(store.usage(), (1, 0, 1));
                assert_eq!(
                    store.begin("t", "k", [1; 32]).err().unwrap().code,
                    "idempotency_reconciliation_required"
                );
            } else {
                assert_eq!(store.usage(), (1, size, 0));
                assert!(matches!(
                    store.begin("t", "k", [1; 32]).unwrap(),
                    Begin::Replay(_)
                ));
            }
            assert_eq!(
                store.begin("t", "k", [2; 32]).err().unwrap().code,
                "idempotency_key_reused"
            );
            let finished = store.inner.lock().completed.front().unwrap().0;
            let Begin::Proceed(next) = store
                .begin_at("t", "k", [2; 32], finished + Duration::from_secs(60))
                .unwrap()
            else {
                panic!()
            };
            assert_eq!(store.usage(), (1, 0, 0));
            drop(next);
        }
    }

    #[test]
    fn expiration_reclaims_bytes_without_expiring_running_requests() {
        let ttl = Duration::from_secs(60);
        let response = StoredResponse::new(StatusCode::OK, &serde_json::json!({"ok": true}), false);
        let store = Idempotency::new(ttl, 2, response.allocated_bytes);
        let Begin::Proceed(running) = store.begin("t", "running", [0; 32]).unwrap() else {
            panic!()
        };
        let Begin::Proceed(done) = store.begin("t", "done", [0; 32]).unwrap() else {
            panic!()
        };
        done.complete(&response);
        let finished = store.inner.lock().completed.front().unwrap().0;
        assert_eq!(
            store
                .begin_at("t", "new", [0; 32], finished + Duration::from_secs(30))
                .err()
                .unwrap()
                .retry_after,
            Some(30)
        );
        assert_eq!(
            store
                .begin_at(
                    "t",
                    "new",
                    [0; 32],
                    finished + ttl - Duration::from_nanos(1)
                )
                .err()
                .unwrap()
                .code,
            "server_busy"
        );
        assert_eq!(
            store
                .begin_at("t", "running", [0; 32], finished + ttl)
                .err()
                .unwrap()
                .code,
            "idempotency_in_flight"
        );
        assert_eq!(store.usage(), (1, 0, 0));
        let Begin::Proceed(next) = store
            .begin_at(
                "t",
                "new",
                [0; 32],
                finished + ttl + Duration::from_nanos(1),
            )
            .unwrap()
        else {
            panic!()
        };
        next.complete(&response);
        assert_eq!(store.usage(), (2, response.allocated_bytes, 0));
        drop(running);
        assert_eq!(store.usage(), (2, response.allocated_bytes, 1));
    }

    #[test]
    fn concurrent_completions_cannot_exceed_the_shared_byte_budget() {
        let response = StoredResponse::new(
            StatusCode::OK,
            &serde_json::json!({"payload": "x".repeat(512)}),
            false,
        );
        let store = Idempotency::new(Duration::from_secs(60), 8, response.allocated_bytes);
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for i in 0..8 {
                let Begin::Proceed(guard) = store.begin("t", &i.to_string(), [0; 32]).unwrap()
                else {
                    panic!()
                };
                let response = &response;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    guard.complete(response);
                });
            }
        });
        assert_eq!(store.usage(), (8, response.allocated_bytes, 7));
        for i in 0..8 {
            assert!(!matches!(
                store.begin("t", &i.to_string(), [0; 32]),
                Ok(Begin::Proceed(_))
            ));
        }
    }

    #[test]
    fn repeated_transient_failures_do_not_retain_keys_or_queue_entries() {
        let store = Idempotency::new(Duration::from_secs(60), 1, 1024);
        let response = StoredResponse::new(
            StatusCode::SERVICE_UNAVAILABLE,
            &serde_json::json!({}),
            false,
        );
        for _ in 0..100 {
            let Begin::Proceed(g) = store.begin("t", "retry", [0; 32]).unwrap() else {
                panic!()
            };
            g.complete(&response);
            assert_eq!(store.usage(), (0, 0, 0));
            assert!(store.inner.lock().completed.is_empty());
        }
    }
}
