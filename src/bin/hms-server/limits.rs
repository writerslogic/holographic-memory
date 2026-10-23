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
    buckets: Mutex<HashMap<String, Bucket>>,
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
            buckets: Mutex::new(HashMap::new()),
        })
    }

    /// Consume one token for `tenant`, or return the seconds to wait.
    pub fn check(&self, tenant: &str) -> Result<(), u64> {
        self.check_at(tenant, Instant::now())
    }

    fn check_at(&self, tenant: &str, now: Instant) -> Result<(), u64> {
        let mut buckets = self.buckets.lock();
        let known = buckets.contains_key(tenant);
        if !known && buckets.len() >= MAX_BUCKETS {
            // A bucket idle long enough to be full again is indistinguishable from a new one.
            let refill = Duration::from_secs_f64(self.burst / self.rate);
            buckets.retain(|_, b| now.duration_since(b.last) < refill);
        }
        if !known {
            // The tenant name is copied only for a new bucket, not on every request.
            buckets.insert(
                tenant.to_string(),
                Bucket {
                    tokens: self.burst,
                    last: now,
                },
            );
        }
        let bucket = buckets
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
    pub body: Arc<serde_json::Value>,
}

enum State {
    Running,
    Done(StoredResponse),
}

struct Record {
    fingerprint: [u8; 32],
    created: Instant,
    state: State,
}

type Key = (String, String);

#[derive(Default)]
struct Inner {
    records: HashMap<Key, Record>,
    order: VecDeque<Key>,
}

pub struct Idempotency {
    ttl: Duration,
    capacity: usize,
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
}

impl IdempotencyGuard {
    /// Record the outcome. Server-side failures and shed load are not recorded, so a retry runs.
    pub fn complete(mut self, response: &StoredResponse) {
        self.done = true;
        let mut inner = self.inner.lock();
        let retryable =
            response.status.is_server_error() || matches!(response.status.as_u16(), 408 | 429);
        if retryable {
            inner.records.remove(&self.key);
        } else if let Some(record) = inner.records.get_mut(&self.key) {
            record.state = State::Done(response.clone());
            record.created = Instant::now();
        }
    }
}

impl Drop for IdempotencyGuard {
    fn drop(&mut self) {
        if !self.done {
            self.inner.lock().records.remove(&self.key);
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
    pub fn new(ttl: Duration, capacity: usize) -> Self {
        Self {
            ttl,
            capacity,
            inner: Arc::new(Mutex::new(Inner::default())),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().records.len()
    }

    pub fn begin(&self, tenant: &str, key: &str, fingerprint: [u8; 32]) -> Result<Begin, ApiError> {
        let id = (tenant.to_string(), key.to_string());
        let mut inner = self.inner.lock();
        let now = Instant::now();
        if let Some(record) = inner.records.get(&id) {
            let expired = matches!(record.state, State::Done(_))
                && now.duration_since(record.created) > self.ttl;
            if !expired {
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
                };
            }
            inner.records.remove(&id);
        }
        while inner.records.len() >= self.capacity {
            let Some(oldest) = inner.order.pop_front() else {
                break;
            };
            inner.records.remove(&oldest);
        }
        inner.order.push_back(id.clone());
        // Keep the order queue proportional to the live records.
        if inner.order.len() > self.capacity * 2 {
            let Inner { records, order } = &mut *inner;
            order.retain(|k| records.contains_key(k));
        }
        inner.records.insert(
            id.clone(),
            Record {
                fingerprint,
                created: now,
                state: State::Running,
            },
        );
        Ok(Begin::Proceed(IdempotencyGuard {
            inner: self.inner.clone(),
            key: id,
            done: false,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let store = Idempotency::new(Duration::from_secs(60), 8);
        let fp = fingerprint("POST", "/x", "t", b"body");
        let Begin::Proceed(guard) = store.begin("t", "k", fp).unwrap() else {
            panic!()
        };
        assert_eq!(
            store.begin("t", "k", fp).err().unwrap().code,
            "idempotency_in_flight"
        );
        let ok = StoredResponse {
            status: StatusCode::OK,
            body: Arc::new(serde_json::json!({"added": 1})),
        };
        guard.complete(&ok);
        let Begin::Replay(r) = store.begin("t", "k", fp).unwrap() else {
            panic!()
        };
        assert_eq!(*r.body, serde_json::json!({"added": 1}));
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
        guard.complete(&StoredResponse {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: Arc::new(serde_json::json!({})),
        });
        assert!(matches!(
            store.begin("t", "f", fp).unwrap(),
            Begin::Proceed(_)
        ));
    }

    #[test]
    fn idempotency_capacity_is_bounded() {
        let store = Idempotency::new(Duration::from_secs(60), 4);
        let fp = fingerprint("POST", "/x", "t", b"");
        for i in 0..50 {
            if let Begin::Proceed(g) = store.begin("t", &format!("k{i}"), fp).unwrap() {
                g.complete(&StoredResponse {
                    status: StatusCode::OK,
                    body: Arc::new(serde_json::json!({})),
                });
            }
            assert!(store.len() <= 4);
        }
    }
}
