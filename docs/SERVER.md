# hms-server

`hms-server` is a small HTTP service that implements the endpoints used by the `holographic-sdk` Python client (`bindings/python/holographic-sdk`). It wraps one `HmsCore` store per tenant.

## Build and run

```sh
cargo build --release --features server --bin hms-server
HMS_API_KEY=change-me ./target/release/hms-server --input-dim 384
```

It prints `hms-server listening on <addr>` once bound. SIGINT or SIGTERM stops accepting new work, finishes admitted requests and their blocking operations (including operations whose clients disconnected), then flushes every open store and syncs the metadata logs before exiting.

Point the SDK at it:

```python
from holographic_sdk import HolographicClient
client = HolographicClient(url="http://127.0.0.1:8080", api_key="change-me", tenant_id="acme")
```

## Configuration

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--bind` | `HMS_BIND` | `127.0.0.1:8080` | Listen address. Loopback unless you change it. |
| `--data-dir` | `HMS_DATA_DIR` | `./hms-data` | Root for per-tenant stores. |
| `--dim` | `HMS_DIM` | `4096` | Hypervector dimensions per store (>= 256). Fixed once a store exists. |
| `--input-dim` | `HMS_INPUT_DIM` | `384` | Required length of every client embedding. |
| `--embedding-model` | `HMS_EMBEDDING_MODEL` | `client-supplied` | Embedding model name recorded in each store. |
| `--embedding-revision` | `HMS_EMBEDDING_REVISION` | `unversioned` | Embedding model revision recorded in each store. |
| `--max-body-bytes` | | 16 MiB | Request body limit (413 above it). |
| `--body-timeout-seconds` | `HMS_BODY_TIMEOUT_SECONDS` | `30` | Body read deadline, 1..=300 seconds (408 on timeout). |
| `--max-batch` | | `1000` | Incoming documents per batch request, at most 4096. |
| `--max-top-k` | | `100` | Upper bound for `top_k`. |
| `--max-tenants` | | `64` | Stores open at once (503 above it). |
| `--max-concurrent-work` | `HMS_MAX_CONCURRENT_WORK` | `4` | Admitted requests reading bodies or running blocking work, 1..=256. |
| | `HMS_API_KEY` | unset | If set, every request needs `Authorization: Bearer <key>`; compared in constant time. Empty is rejected at startup. Environment only, never a flag. |

Starting on a non-loopback address without `HMS_API_KEY` prints a warning.

## Tenants

`X-Tenant-ID` selects the store; absent means `default`. Ids are 1 to 64 characters of `[A-Za-z0-9_-]`, anything else is 400. Ids are case-sensitive. Each tenant maps to `<data-dir>/tenants/<hex of id>/`, so ids never reach the filesystem as paths.

Simultaneous first requests share one initialization per tenant. Pending initializations count toward `max-tenants`; filesystem work runs outside the registry lock. A failed initialization releases its reservation so a later request can retry.

## Endpoints

Errors are JSON: `{"error": {"code": "...", "message": "..."}}`. Internal failures return a fixed 500 message; detail goes to the server's stderr.

A document is `{"id": str, "vector": [float], "text"?: str, "metadata"?: object}`. `vector` must have exactly `--input-dim` finite f64 values, not all zero. Query vectors follow the same rule. Normalization handles very large and very small finite magnitudes without narrowing to f32. `id` is 1 to 256 bytes with no control characters and not starting with `hms:`. `text` is optional; when present it must fit one engine chunk (at most 4096 words and 64 KiB), because one vector describes one stored chunk. Split longer documents and send one vector per part. `metadata` is at most 64 KiB. Adding an existing id replaces it.

| Request | Success | Failure |
|---|---|---|
| `POST /api/v1/documents` with one document | 200 `{"added": 1}` | 400 `invalid_json`, 422 `invalid_request`, 413 |
| `POST /api/v1/documents/batch` with an array of 1..=`max-batch` documents | 200 `{"added": n}` | as above; all documents are validated before one atomic commit |
| `DELETE /api/v1/documents/{id}[?purge=true]` (percent-encoded) | 200 `{"deleted": true}` | 404 `not_found` |
| `POST /api/v1/query` with `{"query_vector": [...], "top_k"?: 1..=max-top-k (default 3), "filter"?: object}` | 200 `{"matches": [{"id","text","metadata","score"}]}` best first | 400, 422 |
| `POST /api/v1/index/maintain` with an empty body or `{}` | 200 `{"documentIndexTrained": bool}` | 400, 422, 500 |

Other statuses: 401 (missing or wrong key), 400 `invalid_tenant`, 404 unknown route, 405, 408 `request_timeout`, 503 `tenant_limit`, `server_busy` or `shutting_down`. Work saturation returns immediately with `Retry-After: 1`; blocking jobs are admitted before dispatch. A disconnected client releases body admission immediately, but a running blocking operation retains its permit until it finishes. Parsing, validation, store initialization, ingest, search and index maintenance run on blocking workers.

Batch totals are limited to 8 MiB of IDs/text, 8 MiB of metadata, and 1,048,576 embedding values; the native prepared transaction is limited to 32 MiB. Every incoming record counts toward validation and budgets. Repeated IDs keep their last occurrence's text, vector and metadata, while `added` remains the original incoming count.

Index maintenance explicitly builds the tenant's optional document index over already stored embeddings. The response reports whether an index is trained; a store too small to train returns `false`. Ingest does not trigger training. Maintenance shares authentication, tenant selection and work limits with the other endpoints, and preserves query response fields and cosine scores. If the store changes during training, maintenance returns a fixed internal error and can be retried.

`filter` keeps documents whose metadata contains every filter key with an equal JSON value. It is applied before ranking, so it never reduces the number of matches below `top_k` when enough documents match.

`score` is the exact cosine similarity between the query and the stored embedding. Explicit index maintenance changes the search backend while preserving query response fields and cosine score semantics.

## On-device model stages (`local-models`)

Built with `--features server,local-models`, the server can run the model stages itself instead of requiring client vectors. Every stage is off unless its flag is given; with none given the server behaves exactly as described above.

| Flag | Environment | Stage |
|---|---|---|
| `--embed-model DIR --embed-revision SHA` | `HMS_EMBED_MODEL`, `HMS_EMBED_REVISION` | Qwen3-Embedding. Documents may omit `vector` (the text is embedded); `--input-dim` must equal the model's hidden size (1024 for 0.6B). |
| `--rerank-model DIR --rerank-revision SHA` | `HMS_RERANK_MODEL`, `HMS_RERANK_REVISION` | Qwen3-Reranker over the fused top 20 of a text query. |
| `--llm-model DIR --llm-revision SHA` | `HMS_LLM_MODEL`, `HMS_LLM_REVISION` | Qwen3 instruct LLM (greedy decoding). |
| `--extract-facts` | `HMS_EXTRACT_FACTS` | At ingest, the LLM lists the personal facts in each document; they are indexed with it (lexical and embedding key), the stored text is unchanged. |
| `--rewrite-queries` | `HMS_REWRITE_QUERIES` | The LLM rewrites each text query; the original and the rewrite are searched and merged by RRF. |

A query may then be `{"query": str, "today"?: str, "top_k"?, "filter"?, "query_vector"?}`: hybrid BM25 + cosine per query form, weighted RRF across forms, then the re-rank, with the tuned LongMemEval parameters. `score` is then the fused rank score, not a cosine. Vector-only queries keep the cosine path.

**What runs where.** Every stage runs in the server process: candle on the CPU, on the GPU through Metal on macOS, or through CUDA in builds made with `RUSTFLAGS="--cfg hms_cuda"`. Weights are read from the given local directories, which must record the given revision (a `REVISION` file or the metadata `hf download --revision <sha> --local-dir <dir>` writes); the server never downloads a model and makes no network calls, so no document or query text leaves the machine. Fact extraction adds one LLM generation per document and query rewriting one per text query; see `docs/PUBLIC-BENCHMARKS.md` for measured on-device costs.

## Storage

Each tenant is one engine store. Documents (embedding, text and metadata) go through the engine's document API. A batch writes its final distinct IDs in one transaction; a validation failure leaves existing documents unchanged. Embeddings are kept at full precision, so the sparse-code compression described in the main README does not apply to this service.

The store records the embedding model, revision and input dimension (`--embedding-model`, `--embedding-revision`, `--input-dim`). Reopening a store with different values is refused, so vectors from different models are never mixed.

## What it does not provide

- No TLS. Put it behind a TLS-terminating proxy if it leaves loopback.
- No rate limiting or per-tenant quotas beyond the limits above.
- Single node: no replication, clustering, or HA.
- One shared API key, no per-tenant authorization: any holder of the key can use any tenant id.
- No encryption at rest and no audit or attestation features.
- The SDK's `zero_trust_key` vector masking is obfuscation, not encryption. The server sees masked vectors, learns every pairwise similarity, and stores text and metadata in plaintext.
