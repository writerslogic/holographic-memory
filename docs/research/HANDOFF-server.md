# Handoff: server

Done and verified (this PR): module tree, hashed scoped per-tenant keys with rotation, problem+json, health/metrics/logs, OpenAPI with drift test, TLS, unix socket, limits/queue/quotas, idempotency, ETag, NDJSON ingest, snapshots/restore, compaction, tenant eviction, tenant quiesce, memory API routes, list/stats endpoints, deploy artifacts, abuse fuzzing and black-box tests.

Left:

1. Query speed at N >= 10k is an engine cost (exact scan slower than numpy; the server adds under 0.5 ms). Fix in the engine search path, then rerun `uv run deploy/bench/server_bench.py compare --hms-bin <release bin> --qdrant-bin <qdrant> --out benchmarks/results/server_v1.json`.
2. The soak was 4 minutes. For a flat-RSS claim run 30 minutes: `uv run deploy/bench/server_bench.py soak --hms-bin <release bin> --minutes 30 --out benchmarks/results/server_soak_v1.json`.
3. Not done: live Kubernetes (kind) run, x86_64 image, musl vs glibc allocator comparison.
4. Per-document GET and per-document ETag need an engine get-by-id.
