# hms-server

`hms-server` is a small HTTP service that implements the endpoints used by the `holographic-sdk` Python client (`bindings/python/holographic-sdk`). It wraps one `HmsCore` store per tenant.

## Build and run

```sh
cargo build --release --features server --bin hms-server
HMS_API_KEY=change-me ./target/release/hms-server --input-dim 384
```

It prints `hms-server listening on <addr>` once bound. SIGINT or SIGTERM stops accepting connections, finishes in-flight requests, then flushes every open store and syncs the metadata logs before exiting.

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
| `--max-batch` | | `1000` | Documents per batch request. |
| `--max-top-k` | | `100` | Upper bound for `top_k`. |
| `--max-tenants` | | `64` | Stores open at once (503 above it). |
| | `HMS_API_KEY` | unset | If set, every request needs `Authorization: Bearer <key>`; compared in constant time. Empty is rejected at startup. Environment only, never a flag. |

Starting on a non-loopback address without `HMS_API_KEY` prints a warning.

## Tenants

`X-Tenant-ID` selects the store; absent means `default`. Ids are 1 to 64 characters of `[A-Za-z0-9_-]`, anything else is 400. Ids are case-sensitive. Each tenant maps to `<data-dir>/tenants/<hex of id>/`, so ids never reach the filesystem as paths.

## Endpoints

Errors are JSON: `{"error": {"code": "...", "message": "..."}}`. Internal failures return a fixed 500 message; detail goes to the server's stderr.

A document is `{"id": str, "vector": [float], "text"?: str, "metadata"?: object}`. `vector` must have exactly `--input-dim` finite values (representable as f32, not all zero). `id` is 1 to 256 bytes with no control characters and not starting with `hms:`. `text` is optional; when present it must fit one engine chunk (at most 4096 words and 64 KiB), because one vector describes one stored chunk. Split longer documents and send one vector per part. `metadata` is at most 64 KiB. Adding an existing id replaces it.

| Request | Success | Failure |
|---|---|---|
| `POST /api/v1/documents` with one document | 200 `{"added": 1}` | 400 `invalid_json`, 422 `invalid_request`, 413 |
| `POST /api/v1/documents/batch` with an array of 1..=`max-batch` documents | 200 `{"added": n}` | as above; all documents are validated before any is stored |
| `DELETE /api/v1/documents/{id}` (percent-encoded) | 200 `{"deleted": true}` | 404 `not_found` |
| `POST /api/v1/query` with `{"query_vector": [...], "top_k"?: 1..=max-top-k (default 3), "filter"?: object}` | 200 `{"matches": [{"id","text","metadata","score"}]}` best first | 400, 422 |

Other statuses: 401 (missing or wrong key), 400 `invalid_tenant`, 404 unknown route, 405, 503 `tenant_limit`.

`filter` keeps documents whose metadata contains every filter key with an equal JSON value. It is applied before ranking, so it never reduces the number of matches below `top_k` when enough documents match.

`score` is the exact cosine similarity between the query and the stored embedding. Every query is a linear scan over the tenant's stored embeddings, so latency grows linearly with the number of documents; there is no approximate index on this path.

## Storage

Each tenant is one engine store. Documents (embedding, text and metadata) go through the engine's document API, so each document is written in a single transaction. A batch is not atomic as a whole: a failure part-way leaves the earlier documents stored. Embeddings are kept at full precision (about 4 bytes per dimension per document), so the sparse-code compression described in the main README does not apply to this service.

The store records the embedding model, revision and input dimension (`--embedding-model`, `--embedding-revision`, `--input-dim`). Reopening a store with different values is refused, so vectors from different models are never mixed.

## What it does not provide

- No TLS. Put it behind a TLS-terminating proxy if it leaves loopback.
- No rate limiting or per-tenant quotas beyond the limits above.
- Single node: no replication, clustering, or HA.
- One shared API key, no per-tenant authorization: any holder of the key can use any tenant id.
- No encryption at rest and no audit or attestation features.
- The SDK's `zero_trust_key` vector masking is obfuscation, not encryption. The server sees masked vectors, learns every pairwise similarity, and stores text and metadata in plaintext.
