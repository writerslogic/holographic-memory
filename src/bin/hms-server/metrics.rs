// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Prometheus text exposition (format 0.0.4) with no dependencies: request counters,
//! latency histograms per route template, saturation gauges and process memory.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

const BOUNDS: [f64; 14] = [
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

struct Histogram {
    buckets: [AtomicU64; BOUNDS.len()],
    count: AtomicU64,
    sum_micros: AtomicU64,
}

impl Histogram {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum_micros: AtomicU64::new(0),
        }
    }

    fn observe(&self, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        if let Some(i) = BOUNDS.iter().position(|bound| seconds <= *bound) {
            self.buckets[i].fetch_add(1, Relaxed);
        }
        self.count.fetch_add(1, Relaxed);
        self.sum_micros.fetch_add(
            u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
            Relaxed,
        );
    }
}

#[derive(Default)]
pub struct TenantGauge {
    pub name: String,
    pub documents: u64,
    pub vectors: u64,
    pub used_bytes: f64,
    pub index_trained: bool,
    pub compaction_runs: u64,
    pub compaction_failures: u64,
    pub compaction_seconds: f64,
}

pub struct Saturation {
    pub work_in_use: usize,
    pub work_capacity: usize,
    pub queue_depth: usize,
    pub tenants: Vec<TenantGauge>,
    pub idempotency_records: usize,
    pub ready: bool,
}

pub struct Metrics {
    started: Instant,
    histograms: HashMap<&'static str, Histogram>,
    requests: Mutex<HashMap<(&'static str, &'static str, u16), u64>>,
    pub in_flight: AtomicI64,
    pub connections_open: AtomicI64,
    pub connections_total: AtomicU64,
    pub tls_handshake_failures: AtomicU64,
    pub auth_failures: AtomicU64,
    pub rate_limited: AtomicU64,
    pub quota_rejections: AtomicU64,
    pub busy_rejections: AtomicU64,
    pub idempotent_replays: AtomicU64,
    pub tenant_evictions: AtomicU64,
    pub ingest_documents: AtomicU64,
}

impl Metrics {
    pub fn new(routes: impl IntoIterator<Item = &'static str>) -> Self {
        let mut histograms: HashMap<_, _> =
            routes.into_iter().map(|r| (r, Histogram::new())).collect();
        histograms.insert("unmatched", Histogram::new());
        Self {
            started: Instant::now(),
            histograms,
            requests: Mutex::new(HashMap::new()),
            in_flight: AtomicI64::new(0),
            connections_open: AtomicI64::new(0),
            connections_total: AtomicU64::new(0),
            tls_handshake_failures: AtomicU64::new(0),
            auth_failures: AtomicU64::new(0),
            rate_limited: AtomicU64::new(0),
            quota_rejections: AtomicU64::new(0),
            busy_rejections: AtomicU64::new(0),
            idempotent_replays: AtomicU64::new(0),
            tenant_evictions: AtomicU64::new(0),
            ingest_documents: AtomicU64::new(0),
        }
    }

    pub fn observe_request(
        &self,
        route: &'static str,
        method: &'static str,
        status: u16,
        elapsed: Duration,
    ) {
        if let Some(h) = self.histograms.get(route) {
            h.observe(elapsed);
        }
        *self
            .requests
            .lock()
            .entry((route, method, status))
            .or_insert(0) += 1;
    }

    pub fn render(&self, sat: &Saturation) -> String {
        let mut out = String::with_capacity(8192);
        let counter = |out: &mut String, name: &str, help: &str, value: u64| {
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}"
            );
        };
        let gauge = |out: &mut String, name: &str, help: &str, value: String| {
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}"
            );
        };
        let _ = writeln!(
            out,
            "# HELP hms_build_info Build information.\n# TYPE hms_build_info gauge\nhms_build_info{{version=\"{}\"}} 1",
            env!("CARGO_PKG_VERSION")
        );
        gauge(&mut out, "hms_up", "1 while serving.", "1".into());
        gauge(
            &mut out,
            "hms_ready",
            "1 when /readyz succeeds.",
            u8::from(sat.ready).to_string(),
        );
        gauge(
            &mut out,
            "hms_uptime_seconds",
            "Seconds since start.",
            self.started.elapsed().as_secs().to_string(),
        );
        if let Some(rss) = resident_bytes() {
            gauge(
                &mut out,
                "process_resident_memory_bytes",
                "Resident set size of the process.",
                rss.to_string(),
            );
        }
        gauge(
            &mut out,
            "hms_http_in_flight_requests",
            "Requests being served.",
            self.in_flight.load(Relaxed).to_string(),
        );
        gauge(
            &mut out,
            "hms_work_slots_in_use",
            "Work slots held by admitted requests.",
            sat.work_in_use.to_string(),
        );
        gauge(
            &mut out,
            "hms_work_slots_capacity",
            "Configured work slots.",
            sat.work_capacity.to_string(),
        );
        gauge(
            &mut out,
            "hms_queue_depth",
            "Requests waiting for a work slot.",
            sat.queue_depth.to_string(),
        );
        gauge(
            &mut out,
            "hms_connections_open",
            "Open client connections.",
            self.connections_open.load(Relaxed).to_string(),
        );
        gauge(
            &mut out,
            "hms_tenants_open",
            "Tenant stores open.",
            sat.tenants.len().to_string(),
        );
        gauge(
            &mut out,
            "hms_idempotency_records",
            "Idempotency records held.",
            sat.idempotency_records.to_string(),
        );
        counter(
            &mut out,
            "hms_connections_total",
            "Connections accepted.",
            self.connections_total.load(Relaxed),
        );
        counter(
            &mut out,
            "hms_tls_handshake_failures_total",
            "Failed or timed-out TLS handshakes.",
            self.tls_handshake_failures.load(Relaxed),
        );
        counter(
            &mut out,
            "hms_auth_failures_total",
            "Requests rejected for credentials or scope.",
            self.auth_failures.load(Relaxed),
        );
        counter(
            &mut out,
            "hms_rate_limited_total",
            "Requests rejected by the per-tenant rate limit.",
            self.rate_limited.load(Relaxed),
        );
        counter(
            &mut out,
            "hms_quota_rejections_total",
            "Writes rejected by a tenant quota.",
            self.quota_rejections.load(Relaxed),
        );
        counter(
            &mut out,
            "hms_busy_rejections_total",
            "Requests rejected because work capacity was full.",
            self.busy_rejections.load(Relaxed),
        );
        counter(
            &mut out,
            "hms_idempotent_replays_total",
            "Responses replayed from an Idempotency-Key.",
            self.idempotent_replays.load(Relaxed),
        );
        counter(
            &mut out,
            "hms_tenant_evictions_total",
            "Idle tenant stores closed.",
            self.tenant_evictions.load(Relaxed),
        );
        counter(
            &mut out,
            "hms_ingest_documents_total",
            "Documents stored.",
            self.ingest_documents.load(Relaxed),
        );

        out.push_str("# HELP hms_http_requests_total Requests by route template, method and status.\n# TYPE hms_http_requests_total counter\n");
        let mut rows: Vec<_> = self.requests.lock().iter().map(|(k, v)| (*k, *v)).collect();
        rows.sort();
        for ((route, method, status), n) in rows {
            let _ = writeln!(out, "hms_http_requests_total{{route=\"{route}\",method=\"{method}\",status=\"{status}\"}} {n}");
        }
        out.push_str("# HELP hms_http_request_duration_seconds Request latency by route template.\n# TYPE hms_http_request_duration_seconds histogram\n");
        let mut routes: Vec<_> = self.histograms.iter().collect();
        routes.sort_by_key(|(route, _)| **route);
        for (route, h) in routes {
            let count = h.count.load(Relaxed);
            if count == 0 {
                continue;
            }
            let mut cumulative = 0;
            for (bound, bucket) in BOUNDS.iter().zip(&h.buckets) {
                cumulative += bucket.load(Relaxed);
                let _ = writeln!(out, "hms_http_request_duration_seconds_bucket{{route=\"{route}\",le=\"{bound}\"}} {cumulative}");
            }
            let _ = writeln!(
                out,
                "hms_http_request_duration_seconds_bucket{{route=\"{route}\",le=\"+Inf\"}} {count}"
            );
            let _ = writeln!(
                out,
                "hms_http_request_duration_seconds_sum{{route=\"{route}\"}} {}\nhms_http_request_duration_seconds_count{{route=\"{route}\"}} {count}",
                h.sum_micros.load(Relaxed) as f64 / 1e6
            );
        }
        for (name, help, kind) in [
            (
                "hms_tenant_documents",
                "Documents stored per tenant.",
                "gauge",
            ),
            ("hms_tenant_vectors", "Vectors indexed per tenant.", "gauge"),
            (
                "hms_tenant_index_bytes",
                "Bytes in use by the tenant's store arena.",
                "gauge",
            ),
            (
                "hms_tenant_document_index_trained",
                "1 when the document index is trained.",
                "gauge",
            ),
            (
                "hms_tenant_compactions_total",
                "Completed compactions per tenant.",
                "counter",
            ),
            (
                "hms_tenant_compaction_failures_total",
                "Failed compactions per tenant.",
                "counter",
            ),
            (
                "hms_tenant_compaction_seconds_total",
                "Seconds spent compacting per tenant.",
                "counter",
            ),
        ] {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
            for t in &sat.tenants {
                let value = match name {
                    "hms_tenant_documents" => t.documents.to_string(),
                    "hms_tenant_vectors" => t.vectors.to_string(),
                    "hms_tenant_index_bytes" => format!("{}", t.used_bytes),
                    "hms_tenant_document_index_trained" => u8::from(t.index_trained).to_string(),
                    "hms_tenant_compactions_total" => t.compaction_runs.to_string(),
                    "hms_tenant_compaction_failures_total" => t.compaction_failures.to_string(),
                    _ => format!("{}", t.compaction_seconds),
                };
                let _ = writeln!(out, "{name}{{tenant=\"{}\"}} {value}", t.name);
            }
        }
        out
    }
}

/// Resident set size in bytes, or `None` where the platform offers no cheap source.
pub fn resident_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
        // SAFETY: sysconf with a valid name has no memory-safety preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        (page > 0).then(|| pages * page as u64)
    }
    #[cfg(target_os = "macos")]
    {
        let mut info = std::mem::MaybeUninit::<libc::proc_taskinfo>::zeroed();
        let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        // SAFETY: `info` is a writable buffer of exactly `size` bytes for PROC_PIDTASKINFO.
        let written = unsafe {
            libc::proc_pidinfo(
                libc::getpid(),
                libc::PROC_PIDTASKINFO,
                0,
                info.as_mut_ptr().cast(),
                size,
            )
        };
        // SAFETY: proc_pidinfo filled the whole struct when it returned `size`.
        (written == size).then(|| unsafe { info.assume_init() }.pti_resident_size)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_is_cumulative_and_counts_every_observation() {
        let m = Metrics::new(["/x"]);
        m.observe_request("/x", "GET", 200, Duration::from_micros(500));
        m.observe_request("/x", "GET", 200, Duration::from_millis(30));
        m.observe_request("/x", "GET", 500, Duration::from_secs(60));
        let text = m.render(&Saturation {
            work_in_use: 0,
            work_capacity: 4,
            queue_depth: 0,
            tenants: vec![],
            idempotency_records: 0,
            ready: true,
        });
        assert!(text.contains("le=\"0.001\"} 1"));
        assert!(text.contains("le=\"0.05\"} 2"));
        assert!(text.contains("le=\"30\"} 2"));
        assert!(text.contains("le=\"+Inf\"} 3"));
        assert!(text.contains("hms_http_request_duration_seconds_count{route=\"/x\"} 3"));
        assert!(
            text.contains("hms_http_requests_total{route=\"/x\",method=\"GET\",status=\"500\"} 1")
        );
    }

    #[test]
    fn resident_memory_is_reported_on_supported_platforms() {
        if cfg!(any(target_os = "linux", target_os = "macos")) {
            assert!(resident_bytes().unwrap() > 1_000_000);
        }
    }
}
