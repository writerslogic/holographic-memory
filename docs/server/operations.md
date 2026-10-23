# Operating hms-server

Reference for running the server in production. Settings are in [`configuration.md`](configuration.md); the API is [`openapi.json`](openapi.json).

## Deployment artifacts

| Target | Path | Notes |
|---|---|---|
| Container | `deploy/docker/Dockerfile` | Static musl binary on `gcr.io/distroless/static-debian12:nonroot`, uid 65532, `/data` volume, self-probing `HEALTHCHECK`. |
| Compose | `deploy/compose/compose.yaml` | One service, read-only root, all capabilities dropped, `127.0.0.1` published, key required. |
| Kubernetes | `deploy/helm/hms-server` | Single-replica StatefulSet with a PVC, probes on `/healthz` and `/readyz`, restricted pod security context, optional Ingress, NetworkPolicy and ServiceMonitor. `helm lint` and `kubeconform -strict` clean. |
| systemd | `deploy/systemd/hms-server.service` | `DynamicUser`, `StateDirectory`, strict filesystem and syscall sandbox, `ExecReload` sends SIGHUP. |
| Benchmarks | `deploy/bench/server_bench.py` | The comparison and soak harness behind `benchmarks/results/server_v1.json` and `server_soak_v1.json`. |

One replica per data directory: a tenant store is owned by a single process through an exclusive file lock. Scale a deployment by giving the pod more CPU and memory, or by running several independent servers for disjoint tenant sets.

## TLS and sockets

`--tls-cert`/`--tls-key` serve HTTPS with rustls (TLS 1.2 and 1.3, ALPN `h2` and `http/1.1`). Replace the files and send SIGHUP or `POST /api/v1/admin/tls/reload`; a bad pair is rejected and the running certificate stays in use. `--unix-socket PATH` serves on a socket with mode 0600 instead of TCP (a stale socket file is replaced, a live one or a non-socket file is refused). Without TLS on a non-loopback address the server logs a `plaintext_on_public_address` warning at startup.

## Health, readiness and shutdown

- `GET /healthz`: the process is serving. Use as the liveness probe.
- `GET /readyz`: 200 once the listener is bound; 503 `not_ready` from the moment SIGTERM or SIGINT arrives. Use as the readiness probe.
- Shutdown order: readiness fails, `--shutdown-delay-seconds` pass with the server still answering (so a balancer can notice), the listener closes, open connections get `--shutdown-grace-seconds`, admitted work finishes, every store is flushed, the process exits 0. Set the orchestrator's termination grace to at least delay + grace + the time a flush takes (the Helm chart defaults to 60 s for 5 + 30).

## Metrics

`GET /metrics` (admin scope, or public with `--metrics-public`) is Prometheus text format 0.0.4. For a key limited to some tenants, the per-tenant series and `hms_tenants_open` cover only those tenants:

| Series | Meaning |
|---|---|
| `hms_http_request_duration_seconds` (histogram, `route`) | Latency by route template, buckets 1 ms to 30 s. |
| `hms_http_requests_total{route,method,status}` | Requests; routes are templates (`/api/v1/documents/{id}`), so cardinality is bounded. |
| `hms_http_in_flight_requests`, `hms_connections_open`, `hms_connections_total` | Load. |
| `hms_work_slots_in_use`, `hms_work_slots_capacity`, `hms_queue_depth` | Admission saturation; sustained `in_use == capacity` with a growing queue means raise `--max-concurrent-work` or add capacity. |
| `process_resident_memory_bytes` | RSS (Linux `/proc`, macOS `proc_pidinfo`). |
| `hms_tenants_open`, `hms_tenant_documents{tenant}`, `hms_tenant_vectors{tenant}`, `hms_tenant_index_bytes{tenant}`, `hms_tenant_document_index_trained{tenant}` | Per-tenant size; a tenant that is mid-snapshot is skipped for that scrape. |
| `hms_tenant_compactions_total`, `_failures_total`, `_seconds_total` | Compaction. |
| `hms_auth_failures_total`, `hms_rate_limited_total`, `hms_quota_rejections_total`, `hms_busy_rejections_total`, `hms_idempotent_replays_total`, `hms_tenant_evictions_total`, `hms_tls_handshake_failures_total`, `hms_ingest_documents_total` | Policy outcomes. |
| `hms_ready`, `hms_up`, `hms_uptime_seconds`, `hms_build_info{version}` | State. |

Per-tenant series are bounded by `--max-tenants` (only open stores report).

## Logs

One JSON object per line on stderr (`--log-format text` for key=value). Every request logs `method`, `route` template, `status`, `latency_ms`, `tenant`, `key_id`, `bytes_in` and `request_id`; `X-Request-Id` from the client is honored when it is 1 to 64 characters of `[A-Za-z0-9._-]`, otherwise a fresh id is generated, and the id appears in the response header and in error bodies. Request bodies, vectors, query text, metadata, document ids and credentials are never written. Lifecycle events (`listening`, `shutdown_started`, `shutdown_complete`, `tenant_evicted`, `snapshot_created`, `snapshot_restored`, `compaction_done`, `keys_reloaded`, `tls_reloaded`) and failures carry identifiers and counts only. A test checks that secrets and content sent through the API never appear in the log.

## Backup and restore

- `POST /api/v1/snapshots` flushes the tenant's store and copies it to `<data-dir>/snapshots/<tenant>/<id>/`. The tenant pauses (requests wait) while the copy runs; other tenants are unaffected. Expect roughly the store's size in extra disk per snapshot; an empty store is about 64 MiB because the engine preallocates its first arena segment.
- `GET /api/v1/snapshots/{id}/archive` streams the snapshot as a tar. This is the off-box backup: store it anywhere.
- `POST /api/v1/snapshots/{id}/restore` replaces the tenant's store with the snapshot; `PUT /api/v1/restore` does the same from an uploaded tar (to the same server or a different one). Both verify the candidate store (checksums, manifests and a full open that authenticates every frame) before touching the live one, refuse archives with links, device nodes or paths outside the store, and, if anything fails after the swap starts, put the previous store back. The replaced store is kept as snapshot `pre-restore-<time>`; delete it when you no longer need it.
- Snapshots on the same disk protect against mistakes, not against disk loss. Copy the archive off the machine for that.

## Compaction

`--compaction-interval-seconds N` compacts every open store that changed since its last compaction, one at a time. `GET /api/v1/compaction` shows the schedule and the last run (`runs`, `failures`, `last_trigger`, `last_started_at`, `last_duration_ms`); `POST /api/v1/compaction` runs it now. Compaction rewrites the store's logs without the superseded and deleted entries. It runs on a blocking worker outside the request slots; the engine decides how long it holds writers back, so schedule it for quiet periods on large stores.

## Capacity planning

- Memory is dominated by open stores. `--tenant-idle-seconds` closes idle ones; `hms_tenant_index_bytes` and `process_resident_memory_bytes` show the effect.
- `--max-concurrent-work` is the number of requests the engine runs at once; ingest and search are CPU-bound, so set it near the number of cores and use `--max-queue` for bursts.
- Rate limits and quotas are per tenant and off by default; enable them when tenants are mutually untrusted.

## Measured behavior

All numbers come from `benchmarks/results/server_v1.json`, `server_soak_v1.json` and the container checks below. The machine was a shared 10-core M4 at load average 25 to 40, with client and servers on one host, so absolute figures are indicative. Synthetic clustered vectors, 384 dimensions, top 10, 6 s per level, one run.

**Query throughput (QPS at 1 / 4 / 16 clients, and p50 at 16)**

| N | hms-server (exact) | Qdrant HNSW | Qdrant exact | SQLite + FastAPI |
|---|---|---|---|---|
| 1,000 | 829 / 2538 / **7781**, 1.2 ms | 599 / 1654 / 3499, 1.2 ms | 517 / 1980 / 3329, 1.1 ms | 381 / 476 / 911, 14.5 ms |
| 10,000 | 135 / 487 / 895, 13 ms | **451 / 1572 / 2442**, 3.1 ms | 452 / 1550 / 2500, 3.0 ms | 300 / 538 / 818, 17 ms |
| 50,000 | 22 / 92 / 178, 85 ms | **621 / 1561 / 1867**, 5.0 ms | 247 / 573 / 646, 20 ms | 172 / 176 / 167, 93 ms |

Recall at 10 was 1.0 for every system and size. hms-server wins at 1,000 documents and loses at 10,000 and 50,000: its vector query is an exact scan inside the engine (about 6.9 of 7.2 ms at 10,000 documents is spent in the engine search), slower than a numpy scan from 10,000 up, and the optional document index (`/index/maintain`) did not help in these runs. This is an engine cost, not server overhead, and it is the largest open performance item. Ingest was 2.1k to 2.9k documents/s for hms-server against 3.1k to 4.7k for the others (batches of 500, durable commit per batch).

**Soak** (4 minutes, 16 clients, 137,634 successful mixed requests, tenants rotating through 12 ids with at most 6 open): no errors other than 27 expected `tenant_limit` refusals, open file descriptors 26 to 35, work slots idle at the end, every tenant's contents matched the client's model both before and after a SIGTERM restart, SIGTERM exit code 0. RSS moved between 55 and 270 MB following tenant open and eviction cycles and ended at 144 MB from 139 MB at the start; the second-half linear fit reads +4.4 MB/min over that oscillation, so a longer run is needed to call it flat.

**Container** (Lima VM, Docker 29, arm64): image 15,015,731 bytes (15 MB, of which 5.84 MB is the server binary); uid 65532, read-only root, no capabilities; RSS 10.0 MB (cgroup 13.5 MiB) after loading 2,000 documents of 16 dimensions; `docker stop` took 0.19 s and exited 0; data survived a restart on the volume; 118 MiB at 10,000 documents of 384 dimensions. Compose and the systemd unit were run for real (`systemd-analyze security`: exposure 1.2 OK). Not measured: x86_64 images, musl against glibc allocator throughput, a live Kubernetes cluster (no kind available; `helm lint` and `kubeconform -strict` pass).
