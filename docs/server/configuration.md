# hms-server configuration

Every knob is a command-line flag and an `HMS_*` environment variable with the same meaning. A flag wins over its variable. The server starts with safe defaults, and refuses to start on contradictory settings. A test keeps this page, the flags and the variables in step.

Secrets never appear as flags: `HMS_API_KEY` is read from the environment only.

## Listener and TLS

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--bind` | `HMS_BIND` | `127.0.0.1:8080` | TCP listen address. Loopback unless you change it. |
| `--unix-socket` | `HMS_UNIX_SOCKET` | unset | Listen on this unix socket (mode 0600) instead of TCP. A stale socket file is replaced; a live one is an error. |
| `--tls-cert` | `HMS_TLS_CERT` | unset | PEM certificate chain. Enables HTTPS (rustls, TLS 1.2 and 1.3, ALPN `h2` and `http/1.1`). Needs `--tls-key`. Reloaded on SIGHUP or `POST /api/v1/admin/tls/reload`. |
| `--tls-key` | `HMS_TLS_KEY` | unset | PEM private key for `--tls-cert`. |
| `--max-connections` | `HMS_MAX_CONNECTIONS` | `1024` | Simultaneous connections; more wait in the kernel backlog. |
| `--header-read-timeout-seconds` | `HMS_HEADER_READ_TIMEOUT_SECONDS` | `10` | Time a client has to send request headers (and finish a TLS handshake) before the connection is dropped. 1..=300. |
| `--shutdown-delay-seconds` | `HMS_SHUTDOWN_DELAY_SECONDS` | `0` | After SIGTERM or SIGINT, keep serving this long with `/readyz` answering 503, so a load balancer or Kubernetes endpoint controller can drain the instance before the listener closes. |
| `--shutdown-grace-seconds` | `HMS_SHUTDOWN_GRACE_SECONDS` | `30` | After SIGTERM or SIGINT, time open connections get to finish before they are cut. Admitted work still completes before the final flush. |

## Storage and engine

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--data-dir` | `HMS_DATA_DIR` | `./hms-data` | Root for stores (`tenants/`), snapshots (`snapshots/`) and restore staging (`staging/`). |
| `--memory-config` | `HMS_MEMORY_CONFIG` | unset | JSON file with the memory API configuration (policies and defaults) applied to every tenant; see `docs/memory-api`. Validated at startup. |
| `--dim` | `HMS_DIM` | `4096` | Hypervector dimensions per store (>= 256). Fixed once a store exists. |
| `--input-dim` | `HMS_INPUT_DIM` | `384` | Required length of every client embedding. |
| `--embedding-model` | `HMS_EMBEDDING_MODEL` | `client-supplied` | Embedding model name recorded in each store. |
| `--embedding-revision` | `HMS_EMBEDDING_REVISION` | `unversioned` | Embedding model revision recorded in each store. |
| `--tenant-idle-seconds` | `HMS_TENANT_IDLE_SECONDS` | `900` | A store with no request for this long is flushed and closed, freeing its `--max-tenants` slot; the next request reopens it. 0 keeps stores open. |
| `--compaction-interval-seconds` | `HMS_COMPACTION_INTERVAL_SECONDS` | `0` | Compact every open store that changed since its last compaction this often. 0 disables scheduling; `POST /api/v1/compaction` still works. |

## Authentication and browsers

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--no-auth` | `HMS_NO_AUTH` | off | Serve without authentication: every caller has every scope. Refused unless the listener is loopback or a unix socket, and refused when credentials are also configured. |
| `--keys-file` | `HMS_KEYS_FILE` | unset | JSON file of hashed per-tenant keys; see `docs/server/auth.md`. Reloaded on SIGHUP and by `POST /api/v1/admin/keys/reload`. |
| | `HMS_API_KEY` (environment only, no flag) | unset | One key valid for every tenant with every scope. Empty is an error. |
| `--metrics-public` | `HMS_METRICS_PUBLIC` | off | Serve `/metrics` without credentials. Otherwise it needs the admin scope. `/healthz`, `/readyz` and `/openapi.json` are always public. |
| `--cors-origins` | `HMS_CORS_ORIGINS` | empty | Comma-separated origins allowed by CORS, or `*`. Empty sends no CORS headers. |
| `--cors-max-age-seconds` | `HMS_CORS_MAX_AGE_SECONDS` | `600` | Preflight cache lifetime. |

With neither `HMS_API_KEY` nor `--keys-file`, the server only starts if `--no-auth` is given.

## Request limits and backpressure

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--max-body-bytes` | `HMS_MAX_BODY_BYTES` | 16 MiB | Request body limit (413 above it) for JSON endpoints. |
| `--body-timeout-seconds` | `HMS_BODY_TIMEOUT_SECONDS` | `30` | Deadline to read a body, and the longest idle gap in a streamed upload. 1..=300. |
| `--max-batch` | `HMS_MAX_BATCH` | `1000` | Documents per batch request, at most 4096. |
| `--max-top-k` | `HMS_MAX_TOP_K` | `100` | Upper bound for `top_k`. |
| `--query-index` | `HMS_QUERY_INDEX` | `auto` | Route of unfiltered vector queries: `auto` (the graph index once maintenance built one, else the exact scan), `exact`, or `graph` (409 until a graph index exists). A request's `index` field overrides it. |
| `--query-ef` | `HMS_QUERY_EF` | `128` | Beam width of graph-index queries, 1..=4096: the recall knob. A request's `ef` field overrides it. |
| `--graph-min-rows` | `HMS_GRAPH_MIN_ROWS` | `4096` | Fewest embedded chunks for which index maintenance builds the graph index; smaller stores answer from the exact scan. |
| `--ingest-encode` | `HMS_INGEST_ENCODE` | `eager` | When ingest computes chunk hypervectors (the text-vector channel): `eager` (at ingest), `deferred` (on the first vector query or maintenance; needs stored text and a text encoder) or `lexical` (never: embedding-only workloads skip the channel's time and memory). Text and `query_vector` search see every document in every mode. |
| `--max-tenants` | `HMS_MAX_TENANTS` | `64` | Stores open at once (503 `tenant_limit` above it). |
| `--max-concurrent-work` | `HMS_MAX_CONCURRENT_WORK` | `4` | Requests reading bodies or running engine work at once, 1..=256. |
| `--max-queue` | `HMS_MAX_QUEUE` | `0` | Requests allowed to wait for a work slot. 0 answers 503 `server_busy` immediately. |
| `--queue-timeout-ms` | `HMS_QUEUE_TIMEOUT_MS` | `1000` | Longest a queued request waits before 503. |
| `--rate-limit-rps` | `HMS_RATE_LIMIT_RPS` | `0` | Per-tenant sustained requests per second (token bucket); 429 `rate_limited` with `Retry-After`. 0 disables. |
| `--rate-limit-burst` | `HMS_RATE_LIMIT_BURST` | `0` | Per-tenant burst. 0 means one second of the sustained rate. |
| `--max-documents-per-tenant` | `HMS_MAX_DOCUMENTS_PER_TENANT` | `0` | Per-tenant document ingest and memory `remember` quota ([scope and retry limits](../SERVER.md#limits-and-backpressure)); exceeding it gets 429 `quota_exceeded`. Replacing an existing id never counts. 0 disables. |
| `--idempotency-ttl-seconds` | `HMS_IDEMPOTENCY_TTL_SECONDS` | `600` | How long an `Idempotency-Key` replays its response. |
| `--idempotency-capacity` | `HMS_IDEMPOTENCY_CAPACITY` | `10000` | Records kept in memory; the oldest are evicted first. |

## Bulk transfer

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--max-stream-bytes` | `HMS_MAX_STREAM_BYTES` | 1 GiB | Largest NDJSON ingest stream. |
| `--stream-batch` | `HMS_STREAM_BATCH` | `500` | Documents committed per transaction while streaming, 1..=4096. |
| `--max-stream-line-bytes` | `HMS_MAX_STREAM_LINE_BYTES` | 4 MiB | Largest single NDJSON line. |
| `--max-restore-bytes` | `HMS_MAX_RESTORE_BYTES` | 8 GiB | Largest snapshot archive accepted by `PUT /api/v1/restore`. |

## Logging

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--log-format` | `HMS_LOG_FORMAT` | `json` | `json` (one object per line) or `text`, on stderr. |
| `--log-level` | `HMS_LOG_LEVEL` | `info` | `error`, `warn`, `info` or `debug`. Health and metrics requests log at `debug`. |

Log lines carry identifiers, counts and timings only: method, route template, status, latency, tenant, key id, request id. Request bodies, query text, vectors, metadata, document ids in paths and credentials are never logged.

## On-device models (`--features local-models` builds)

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--embed-model` | `HMS_EMBED_MODEL` | unset | Qwen3-Embedding directory. |
| `--embed-revision` | `HMS_EMBED_REVISION` | unset | Revision the embedding directory must record. |
| `--rerank-model` | `HMS_RERANK_MODEL` | unset | Qwen3-Reranker directory. |
| `--rerank-revision` | `HMS_RERANK_REVISION` | unset | Revision for the reranker directory. |
| `--llm-model` | `HMS_LLM_MODEL` | unset | Qwen3 instruct LLM directory. |
| `--llm-revision` | `HMS_LLM_REVISION` | unset | Revision for the LLM directory. |
| `--extract-facts` | `HMS_EXTRACT_FACTS` | off | Extract facts at ingest with the LLM. |
| `--rewrite-queries` | `HMS_REWRITE_QUERIES` | off | Rewrite text queries with the LLM. |

## Subcommands

- `hms-server openapi` prints the OpenAPI 3.1 document for this build (the committed `docs/server/openapi.json` is its output).
- `hms-server keygen --id ID --tenants a,b|* --scopes read,write [--expires-in-days N] [--keys-file PATH]` creates a key; see `docs/server/auth.md`.
