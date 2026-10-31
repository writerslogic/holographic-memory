---
search:
  boost: 1.6
---

# warm-server

`warm-server` is the HTTP service for WARM. It implements the endpoints used by the `warm-sdk` Python client (`bindings/python/warm-sdk`) and any other HTTP client, wrapping one `WarmCore` store per tenant, and adds what a service needs: scoped per-tenant authentication, limits and backpressure, observability, TLS and unix sockets, snapshots, compaction and graceful shutdown.

| Document | Contents |
|---|---|
| this page | Run it, the API at a glance, behavior and limits |
| [`server/openapi.json`](server/openapi.json) | The OpenAPI 3.1 contract, generated from the server (also served at `/openapi.json`) |
| [`server/configuration.md`](server/configuration.md) | Every flag and `WARM_*` variable, defaults, meaning |
| [`server/auth.md`](server/auth.md) | Keys, scopes, tenant binding, rotation |
| [`server/errors.md`](server/errors.md) | The RFC 9457 error model and every error code |
| [`server/versioning.md`](server/versioning.md) | What may change within `/api/v1` and how breaking changes ship |
| [`server/operations.md`](server/operations.md) | Deploying, TLS, metrics, logs, backup and restore, shutdown, measured behavior |
| [`../deploy/`](../deploy) | Dockerfile, Compose file, Helm chart, systemd unit, benchmark harness |

## Build and run

```sh
cargo build --release --features server --bin warm-server

# One key valid for every tenant (the original setup):
WARM_API_KEY="$(openssl rand -hex 32)" ./target/release/warm-server --input-dim 384

# Per-tenant keys with scopes (docs/server/auth.md):
./target/release/warm-server keygen --id acme-rw --tenants acme --scopes read,write --keys-file keys.json
./target/release/warm-server --keys-file keys.json --input-dim 384

# Local development without credentials (loopback or a unix socket only):
./target/release/warm-server --no-auth --input-dim 384
```

The server refuses to start without credentials unless `--no-auth` is given, and refuses `--no-auth` on a non-loopback address. It prints `warm-server listening on <addr>` on stdout once bound; everything else goes to stderr as JSON log lines.

```python
from writerslogic_warm_sdk import WarmClient
client = WarmClient(url="http://127.0.0.1:8080", api_key="<token>", tenant_id="acme")
```

## API keys

Every request carries an API key. `WARM_API_KEY` sets one key valid for every tenant; `--keys-file` loads per-tenant API keys with read, write or admin scopes, created with `warm-server keygen`. Key format, scopes and rotation are in [server/auth.md](server/auth.md).

## API at a glance

The contract is [`server/openapi.json`](server/openapi.json). Every operation lists the scope it needs (`read`, `write`, `admin`) as its security requirement. Errors are `application/problem+json` ([`server/errors.md`](server/errors.md)).

| Request | Scope | Success |
|---|---|---|
| `POST /api/v1/documents` one document | write | 200 `{"added": 1}` |
| `POST /api/v1/documents/batch` 1..=`max-batch` documents, one atomic commit | write | 200 `{"added": n}` |
| `POST /api/v1/documents/stream` NDJSON, committed every `--stream-batch` lines | write | 200 `{"added","rejected","errors":[{"line","message"}],"errors_truncated"}` |
| `DELETE /api/v1/documents/{id}[?purge=true]` (percent-encoded; `purge` also rewrites the journal and scrubs the deleted data) | write | 200 `{"deleted": true}` |
| `GET /api/v1/documents?limit&after` ids and metadata, id cursor | read | 200 `{"documents":[{"id","metadata"}],"next_cursor","count"}` |
| `GET /api/v1/documents/{id}/chunks/{index}` retained chunk, zero-based index | read | 200 `{"revision","document","index","id","startByte","endByte","text"}`; 404 if absent |
| `GET /api/v1/stats` | read | 200 `{"tenant","documents","vectors","documentIndexTrained","usedBytes","capacityBytes"}` |
| `POST /api/v1/query` | read | 200 `{"matches":[{"id","text","metadata","score"}]}` best first |
| `POST /api/v1/search` | read | 200 `{"hits":[{"id","score"}]}` best first (BM25; model stages when configured) |
| `POST /api/v1/index/maintain` | write | 200 `{"documentIndexTrained": bool}` |
| `POST /api/v1/memory/{remember,recall,explain,forget,get,history,applyForgetting,consolidate}` | write / read | the memory API, `docs/memory-api` |
| `GET /api/v1/compaction`, `POST /api/v1/compaction` | read / admin | status, run now |
| `POST /api/v1/snapshots`, `GET /api/v1/snapshots`, `DELETE /api/v1/snapshots/{id}` | admin | server-side snapshots |
| `GET /api/v1/snapshots/{id}/archive`, `POST /api/v1/snapshots/{id}/restore`, `PUT /api/v1/restore` | admin | tar download, restore from a snapshot or an uploaded archive |
| `POST /api/v1/admin/keys/reload`, `POST /api/v1/admin/tls/reload`, `GET /api/v1/admin/tenants` | admin, all tenants | reload credentials, list tenants |
| `GET /healthz`, `GET /readyz`, `GET /metrics`, `GET /openapi.json` | none (`/metrics`: admin unless `--metrics-public`) | liveness, readiness, Prometheus, spec |

### Documents and queries

A document is `{"id": str, "vector": [float], "text"?: str, "metadata"?: object}`. `vector` has exactly `--input-dim` finite f64 values, not all zero; query vectors follow the same rule. Normalization handles very large and very small finite magnitudes without narrowing to f32. `id` is 1 to 256 bytes with no control characters and not starting with `hms:`. `text` must fit one engine chunk (at most 4096 words and 64 KiB), because one vector describes one stored chunk; split longer documents and send one vector per part. `metadata` is at most 64 KiB. Adding an existing id replaces it.

`GET /api/v1/documents/{id}/chunks/{index}` reads one retained chunk with document
identity, metadata, version, source URI and `chunkCount`. It requires read scope and
the selected tenant. `text: null` means text was not retained; a missing document
or out-of-range index returns 404. HTTP-ingested documents have one chunk; native
stores restored into the server may have more. A chunk contains at most 8 MiB of UTF-8
text (the native whole-document limit), plus bounded document metadata. JSON
escaping and serialization copies can make the response and peak allocation larger.

Chunks preserve their stored passages, including internal whitespace. They do not
recover omitted leading/trailing whitespace, gaps between nonoverlapping chunks,
unretained text, or original pre-normalization embedding values. Recorded byte
spans describe the ingested chunk; vector-only HTTP records use an unretained
placeholder, so their spans are not evidence of original source text. Overlapping
chunks must not be concatenated blindly.

`revision` is an opaque string captured with the chunk. When assembling multiple
reads, require the same revision throughout and retry if it changes. It is not the
user-supplied document version and is not a durable cross-restart sequence number.
Concurrent replacement/deletion or text-segment rebuilding triggers bounded internal
retries; repeated changes return 503 `concurrent_mutation` with `Retry-After`.
Storage failures remain errors, not missing text. Chunk reads support ETag and
If-None-Match for the complete returned representation.

`POST /api/v1/query` takes `{"query_vector": [...], "top_k"?: 1..=max-top-k (default 3), "filter"?: object}`. `filter` keeps documents whose metadata contains every filter key with an equal JSON value; it is applied before ranking, so it never reduces the number of matches below `top_k` when enough documents match. `score` is the exact cosine similarity between the query and the stored embedding.

`POST /api/v1/search` takes `{"query": "...", "top_k"?: 1..=max-top-k (default 10), "include"?: ["text", "metadata"]}` and ranks the tenant's documents, memory facts included, by BM25 (through the on-device model stages instead when the server has them), one hit per document. Hits carry only `id` and `score` unless `include` names stored text or metadata; it applies no memory semantics (validity time, supersession), for which use `memory/recall`.

Batch totals are limited to 8 MiB of ids and text, 8 MiB of metadata and 1,048,576 embedding values; the native prepared transaction is limited to 32 MiB. All documents are validated before one atomic commit; repeated ids keep their last occurrence while `added` stays the incoming count.

`index` routes an unfiltered query: `auto` (the default; `--query-index` sets it) uses the graph index when one is built and the exact scan otherwise, `exact` always scans, `graph` answers 409 `index_unavailable` until maintenance has built one. `ef` is the beam width of a graph query (`--query-ef`, default 128). The graph index is persisted with the tenant store: every document ingested with a vector is appended to the tenant's vector log before the engine commits it, index maintenance builds the graph over those rows and writes it next to the store, and a restart opens the written graph instead of rebuilding it. Maintenance after a restart is a no-op until something changed; scheduled compaction rebuilds after 64 mutations. A graph file that does not read back is ignored and its rows still answer, exactly, until the next maintenance; after a stop that was not clean the index checks each of its documents against the engine before serving. The graph covers documents ingested through the documents API; while a tenant also holds embeddings written through the memory API, its unfiltered queries go to the engine's own snapshot instead.

Index maintenance also builds the tenant's optional document index over stored embeddings and reports whether one is trained; a store too small to train returns `false`. Ingest does not trigger training. If the store changes during training, maintenance returns a fixed internal error and can be retried.

### Bulk ingest

`POST /api/v1/documents/stream` with `Content-Type: application/x-ndjson` reads one document object per line (blank lines ignored, the last line may lack a newline) and groups every `--stream-batch` nonblank lines into one batch, independent of transport frames. Valid documents in each batch commit in one transaction. Rejected lines are reported with their 1-based line number (the first 100) and do not stop the stream. The stream is not atomic across batches: after a failure mid-stream, the committed batches stay. Size limits are `--max-stream-bytes` and `--max-stream-line-bytes`, independent of `--max-body-bytes`; a stalled upload times out after `--body-timeout-seconds` of silence.

### Retries and conditional requests

- `Idempotency-Key` on `POST /api/v1/documents`, `/documents/batch` and the memory operations replays the recorded response of an earlier request with the same key, tenant, method, path and body (`Idempotent-Replayed: true`); the same key with a different request is 422 `idempotency_key_reused`, and a key still running is 409 `idempotency_in_flight`. Transient failures (5xx, 429, 408) release the key, except typed `mutation_committed` errors. Completed keys are protected for `--idempotency-ttl-seconds`; running keys do not expire. Panics or insufficient replay-buffer capacity retain a key that returns 409 `idempotency_reconciliation_required`, preventing blind reexecution. Repeated `Idempotency-Key` fields are rejected with 422 before reserving a key or executing the write. Keys do not survive restart. An upsert alone does not guarantee that repeating every operation after restart is safe; reconcile IDs, versions and resulting state first.
- `GET /api/v1/stats`, `GET /api/v1/documents`, and retained-chunk reads return a weak `ETag` derived from the serialized response and process epoch, and answer matching `If-None-Match` conditions with 304. Validators use weak comparison across repeated header lines and quoted lists; a lone `*` matches an existing representation. Malformed conditions are ignored rather than used to produce 304. See [RFC 9110 §13.1.2](https://www.rfc-editor.org/rfc/rfc9110.html#section-13.1.2). Validators are response-specific, not tenant revision tokens: unchanged pages may retain their tag after unrelated writes. Conditional reads render and hash the response before deciding whether to omit its body; this prevents reused validators during concurrent writes but adds work to 304 responses. Chunk validators include the captured revision; they are not whole-document snapshot handles.

## Tenants

`X-Tenant-ID` selects the store. Repeated fields, even identical ones, return 400 `invalid_tenant` before store access. Ids are 1 to 64 characters of `[A-Za-z0-9_-]`, anything else is 400; they are case-sensitive. A key bound to one tenant needs no header (see [`server/auth.md`](server/auth.md)); otherwise the default is `default`. Each tenant maps to `<data-dir>/tenants/<hex of id>/`, so ids never reach the filesystem as paths.

Stores open on first use. Simultaneous first requests share one initialization; pending initializations count toward `--max-tenants`; filesystem work runs outside the registry lock; a failed initialization releases its reservation. A store idle for `--tenant-idle-seconds` is flushed and closed, so `--max-tenants` bounds open stores, not tenants ever served.

## Limits and backpressure

- `--max-concurrent-work` bounds admitted requests reading bodies or running engine work. When full, a request is answered 503 `server_busy` with `Retry-After` immediately, or, with `--max-queue` above zero, waits in a bounded queue for up to `--queue-timeout-ms` first. `warm_queue_depth` and `warm_work_slots_in_use` expose the saturation.
- `--rate-limit-rps`/`--rate-limit-burst` is a per-tenant token bucket (429 `rate_limited`, `Retry-After`). `--max-documents-per-tenant` is a per-tenant document quota (429 `quota_exceeded`). Both are off by default.
- `--max-connections` caps simultaneous connections; `--header-read-timeout-seconds` drops clients that dribble headers; `--body-timeout-seconds` bounds body reads. A disconnected client releases body admission immediately, but a running blocking operation keeps its work slot until it finishes.
- Parsing, validation, store initialization, ingest, search and maintenance run on blocking workers; the async runtime only moves bytes.

Document ingestion and memory `remember` serialize quota checks through commit, so concurrent
requests share the same limit. Supply explicit memory ids for retries at a full quota:
requests without ids are conservatively counted as new facts. Summaries created by an opt-in
consolidation policy currently bypass quota admission.

## Shutdown

SIGINT or SIGTERM makes `/readyz` fail at once. With `--shutdown-delay-seconds` the server keeps serving for that long so a load balancer can drain it; then the listener closes, open connections get `--shutdown-grace-seconds` to finish, admitted work (including operations whose clients disconnected) runs to completion, every open store is flushed and the metadata logs are synced, and the process exits 0. An interrupted flush exits non-zero.

## On-device model stages (`local-models`)

Built with `--features server,local-models`, the server can run the model stages itself instead of requiring client vectors. Every stage is off unless its flag is given; with none given the server behaves exactly as described above.

| Flag | Environment | Stage |
|---|---|---|
| `--embed-model DIR --embed-revision SHA` | `WARM_EMBED_MODEL`, `WARM_EMBED_REVISION` | Qwen3-Embedding. Documents may omit `vector` (the text is embedded); `--input-dim` must equal the model's hidden size (1024 for 0.6B). |
| `--rerank-model DIR --rerank-revision SHA` | `WARM_RERANK_MODEL`, `WARM_RERANK_REVISION` | Qwen3-Reranker over the fused top 20 of a text query. |
| `--llm-model DIR --llm-revision SHA` | `WARM_LLM_MODEL`, `WARM_LLM_REVISION` | Qwen3 instruct LLM (greedy decoding). |
| `--extract-facts` | `WARM_EXTRACT_FACTS` | At ingest, the LLM lists the personal facts in each document; they are indexed with it (lexical and embedding key), the stored text is unchanged. |
| `--rewrite-queries` | `WARM_REWRITE_QUERIES` | The LLM rewrites each text query; the original and the rewrite are searched and merged by RRF. |

A query may then be `{"query": str, "today"?: str, "top_k"?, "filter"?, "query_vector"?}`: hybrid BM25 + cosine per query form, weighted RRF across forms, then the re-rank, with the tuned LongMemEval parameters. `score` is then the fused rank score, not a cosine. Vector-only queries keep the cosine path.

**What runs where.** Every stage runs in the server process: candle on the CPU, on the GPU through Metal on macOS, or through CUDA in builds made with `RUSTFLAGS="--cfg warm_cuda"`. Weights are read from the given local directories, which must record the given revision (a `REVISION` file or the metadata `hf download --revision <sha> --local-dir <dir>` writes); the server never downloads a model and makes no network calls, so no document or query text leaves the machine. Fact extraction adds one LLM generation per document and query rewriting one per text query; see `docs/PUBLIC-BENCHMARKS.md` for measured on-device costs.

## Storage

Each tenant is one engine store. Documents (embedding, text and metadata) go through the engine's document API. A batch writes its final distinct ids in one transaction; a validation failure leaves existing documents unchanged. Embeddings are kept at full precision, so the sparse-code compression described in the main README does not apply to this service.

The store records the embedding model, revision and input dimension (`--embedding-model`, `--embedding-revision`, `--input-dim`). Reopening a store with different values is refused, so vectors from different models are never mixed.

`<tenant>/vindex/` holds the persisted graph index: `rows.log` (one checksummed frame per inserted, replaced or deleted vector, synced per request at the engine's durability class), the snapshot maintenance last published (`CURRENT`, `graph-<epoch>.vertex`, `rows-<epoch>.bin`, `ids-<epoch>.bin`, each written through a temporary file and a rename) and `dirty`, present between the first mutation after open and a clean close. Snapshots copy the directory with the store.

Snapshots, compaction and restore are described in [`server/operations.md`](server/operations.md).

New vector snapshots record a checksum of `rows-<epoch>.bin` in `CURRENT`; a mismatch disables the snapshot so queries fall back to the engine. Older snapshots remain readable and gain the checksum on the next maintenance run.

Admission limits preserve both running and unexpired completed idempotency keys. At
`--idempotency-capacity`, new keys receive 503 before execution; existing keys can still
replay. Size this capacity for the number of distinct keys admitted during the TTL plus
in-flight work: the defaults retain 10,000 keys for 600 seconds, so a sustained stream
of new keys above roughly 16 per second requires a larger capacity or shorter TTL.
`--idempotency-max-bytes` independently bounds retained serialized response-buffer
allocations (64 MiB by default). A response that does not fit is still returned to the
original caller, but subsequent use of its key requires reconciliation. This limit
does not cover response construction, active HTTP sends, key/record overhead, or total
process RSS. The key-count limit bounds record overhead separately. Response clones
share one serialized buffer instead of retaining and repeatedly serializing a JSON tree.

The rate limiter holds at most 10,000 tenant buckets; when full, requests for a new tenant
receive 429 until idle buckets can be reclaimed. Existing tenant buckets continue operating
normally.

## What it does not provide

- Single node: no replication, clustering or HA. A tenant store is owned by one process.
- No encryption at rest (use an encrypted volume) and no audit or attestation features at this layer.
- Keys are bearer tokens: use TLS (`--tls-cert`, or a terminating proxy) when the listener is not loopback or a unix socket.
- The SDK's `zero_trust_key` vector masking is obfuscation, not encryption. The server sees masked vectors, learns every pairwise similarity, and stores text and metadata in plaintext.
- No mutual TLS, no OIDC, no per-key rate limits; limits are per tenant.
