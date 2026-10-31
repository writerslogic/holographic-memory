# warm-mcp: WARM as an MCP server

`warm-mcp` exposes a WARM store to Model Context Protocol clients (see `docs/integrations/editors/`). It runs on your machine and keeps memories in the same document store as `warm-server`. Optional remote OAuth mode makes token-introspection requests to the configured authorization server.

Transports: stdio (one client per process) and streamable HTTP (one server, many clients). Protocol: the `initialize` handshake revisions 2025-03-26, 2025-06-18 and 2025-11-25, and the stateless 2026-07-28 form, all served by the official Rust SDK (`rmcp` 3.7). OAuth mode uses stateless routing for every revision.

## Install and run

```sh
cargo build --release --features mcp,security --bin warm-mcp     # add local-models for semantic recall (Rust >= 1.94)
./target/release/warm-mcp                                         # stdio
./target/release/warm-mcp --transport http                        # http://127.0.0.1:8765/mcp
```

After the first release the launchers fetch a prebuilt binary for you:

```sh
npx -y writerslogic-warm-mcp          # npm launcher
uvx writerslogic-warm-mcp             # PyPI launcher
```

Both verify the download against a SHA-256 table shipped inside the package and refuse to run a binary that does not match. `WARM_MCP_BINARY=/path/to/warm-mcp` makes them run your own build. The release assets are named `warm-mcp-<rust target triple>[.exe]` under the `mcp-v<version>` release; `integrations/mcp/scripts/build-release.sh` builds one and records its checksum.

## Tools

| Tool | Purpose | Read-only |
|---|---|---|
| `remember` | Store or replace a memory. Optional `id`, `title`, `tags`, `metadata`, `if_version` (compare-and-set), `create_only`. | no |
| `recall` | Memories most relevant to a natural-language query, best first, with full text. | yes |
| `search` | Same retrieval with explicit control: `tags`, metadata `filter`, `min_score`, `lexical_weight`, `semantic_weight`, `fusion` (`blend` or `rrf`). Returns previews and the score breakdown. | yes |
| `get` | One memory by id: text, tags, metadata, version, timestamps. | yes |
| `list` | Page through memories in id order by `prefix` or `tag`; `cursor` continues. | yes |
| `forget` | Delete by `id` or `ids`, or by `prefix` with `confirm: true`. | no |
| `stats` | Counts, recall mode, encryption status, storage size and active limits. | yes |
| `export` | Memories as `json`, `jsonl` or `markdown`, size-bounded, paged by `next_cursor`. | yes |

Every call returns `structuredContent` plus the same data as JSON text (`export` returns the payload as text and the paging metadata as structured content). Tool arguments are validated strictly: unknown fields, wrong types and out-of-range values are errors, never ignored.

A memory is `{id, text, bytes, title?, tags, metadata, created_at, updated_at, version}`. Writing an existing `id` replaces the memory. `version` changes on every write, so a read-modify-write client passes it back as `if_version` and gets `version_conflict` if someone else wrote in between (the check and the write are atomic per store). Text round-trips exactly, including leading and trailing whitespace; blank text needs `allow_empty`.

### Errors

Tool failures are results with `isError: true` and `structuredContent.error = {code, message, details?}`, so a model can read them and correct itself.

| Code | Meaning |
|---|---|
| `invalid_argument` | Failed validation (the message names the field). |
| `too_large` | Text or metadata over its limit. |
| `not_found` | No memory with that id. |
| `already_exists` | `create_only` and the id exists. |
| `version_conflict` | `if_version` did not match; `details.current_version` has the current one. |
| `store_locked` | Another process holds the store; see Sharing below. |
| `store_unavailable` | The store could not be opened with this encryption setting. |
| `too_many_stores` / `too_many` | A configured cap was reached. |
| `forbidden` | A `namespace` argument on a server that does not accept one. |
| `internal` | Unexpected; the cause is on the server's stderr, never in the reply. |

An unknown or disabled tool, a malformed request or an unknown resource or prompt is a protocol error (`-32602`), as the specification asks.

## Resources and prompts

- `hms://memory/{id}` (id percent-encoded) reads one memory as `text/plain` with its metadata in `_meta`; `resources/list` pages through the default namespace and also lists `hms://stats`. With `--namespace-arg`, `hms://namespace/{namespace}/memory/{id}` reads another store.
- Prompts: `recall-context` (retrieves memories for a topic and embeds them), `save-memory`, `review-memories`. A prompt is offered only when the tool it relies on is enabled.

## Where memories live

| Setting | Flag / environment | Default |
|---|---|---|
| Root directory | `--data-dir` / `WARM_MCP_DATA_DIR` | `~/Library/Application Support/warm-mcp` (macOS), `$XDG_DATA_HOME/warm-mcp` or `~/.local/share/warm-mcp` (Linux), `%LOCALAPPDATA%\warm-mcp` (Windows) |
| One exact directory | `--store` / `WARM_MCP_STORE` | unset |
| Namespace strategy | `--namespace-strategy global\|project` / `WARM_MCP_NAMESPACE_STRATEGY` | `global` |
| Global namespace name | `--namespace` / `WARM_MCP_NAMESPACE` | `default` |
| Project directory | `--project` / `WARM_MCP_PROJECT` | working directory; its nearest `.git` ancestor is the root |
| Per-call namespace | `--namespace-arg` / `WARM_MCP_NAMESPACE_ARG` | off |

Each namespace is its own engine store at `<data-dir>/stores/<namespace>`. Namespace names are `[a-z0-9_-]{1,64}`; the `project` strategy derives `<directory-name>-<8 hex digits of a path hash>`, so two projects with the same folder name do not collide. With `--namespace-arg` every tool takes an optional `namespace`, which is how a multi-user application keeps one store per user. At most `--max-stores` (16) active or opening stores occupy the cache. The least recently used idle store is closed before another namespace is opened; durable data remains available on reopen. Active requests retain their store lease. When every slot is active, admission returns `too_many_stores` and can be retried after an operation finishes. Initialization and flushing release the global namespace-map lock before disk I/O.

### Sharing a store

An engine store has one writer. A second stdio process on the same store fails at startup with `store_locked`. Several clients that must share memories (Claude Code and Cursor on one project, say) should use one `warm-mcp --transport http` and point each at it; the editor pages show the remote form.

## Encryption at rest

```sh
export WARM_MCP_PASSPHRASE='correct horse battery staple'
warm-mcp --encrypt            # needs a build with --features security
```

`--encrypt` (`WARM_MCP_ENCRYPT`) encrypts the store with a key derived from the passphrase in the environment variable named by `--passphrase-env` (default `WARM_MCP_PASSPHRASE`). The server refuses to start without the variable, never logs it, and a wrong passphrase fails the open. Reopening an encrypted store without `--encrypt`, or the reverse, fails rather than mixing modes. The end-to-end test stores a distinctive string and confirms it appears in no file under the data directory. What encryption does not cover: process memory while running, and the passphrase itself, which lives in whatever sets the environment.

## Recall: what "semantic" means for each build

- Default build: recall is lexical. Text is stemmed and ranked with BM25, and `tags` and `filter` narrow the candidates. Use the words a memory would contain. `stats.recall` says `lexical (BM25) with metadata filters`.
- `--features mcp,local-models` with `--embed-model DIR --embed-revision SHA` (a Qwen3-Embedding directory, see `docs/SERVER.md`): memories and queries are embedded on the machine and ranking is hybrid lexical plus semantic. `local-models` needs Rust 1.94 or newer; the `mcp` feature alone builds on the 1.89 MSRV. The semantic path compiles but has not been exercised end to end here (no model weights were available), so treat it as unverified until you run it against your weights.

`score` is a relative rank in `0..=1` (the top hit of a blend is 1.0), not a probability; `lexical_score` and `semantic_score` are the raw components.

## Exposure and limits

| Setting | Flag / environment | Default |
|---|---|---|
| Tools to expose | `--tools a,b` / `WARM_MCP_TOOLS` | all |
| Read-only (drops `remember`, `forget`) | `--read-only` / `WARM_MCP_READ_ONLY` | off |
| Resources / prompts off | `--no-resources`, `--no-prompts` | on |
| Memory text | `--max-text-bytes` | 32768 (hard cap 65536, and at most 4096 words, one storage chunk) |
| Metadata | `--max-metadata-bytes` | 16384 |
| Tags per memory | `--max-tags` | 32 |
| `k` | `--max-top-k` | 50 |
| `list` page | `--max-page` | 200 |
| `forget` batch | `--max-forget` | 1000 |
| `export` payload | `--max-export-bytes` | 4 MiB per call, paged |
| Concurrent tool calls | `--max-concurrent` | 4 |
| Waiting tool calls | `--max-pending` | 32; overload returns `server_busy` |
| Worker wait deadline | `--queue-timeout-ms` | 5000; expired work does not run |
| Dimensions per store | `--dim` | 4096, fixed once the store exists |

Every flag has an `WARM_MCP_*` environment variable (see `warm-mcp --help`). Long documents are not split for you: a memory that does not fit one chunk is rejected with `too_large` and the advice to split it, which keeps `get` exact.

Cancelled calls retain their worker permit until the blocking operation finishes. Cancellation
does not roll back a committed write; retry with the same ID and content for an idempotent outcome.

## HTTP transport

```sh
WARM_MCP_TOKEN="$(openssl rand -hex 24)" warm-mcp --transport http --bind 127.0.0.1:8765
```

- Loopback by default. A non-loopback `--bind` requires `--allow-remote` and static bearer or OAuth authorization. Terminate TLS at a trusted proxy; the listener itself uses HTTP.
- `WARM_MCP_TOKEN` (at least 16 characters, environment only) requires `Authorization: Bearer <token>` on the MCP endpoint, compared in constant time. `/health` answers `ok` without auth.
- `Host` must be a loopback name (extend with `--allow-host`) and a request carrying an `Origin` header must match `--allow-origin`; both reject DNS rebinding and cross-site calls with 403. Requests without `Origin` (non-browser clients) pass.
- Request bodies are capped (`--max-request-bytes`, 1 MiB). HTTP/1.1 connections, request admission, response streams and legacy protocol sessions have separate limits below. Requests above HTTP admission capacity return 503 immediately; excess TCP connections close before parsing.
- SIGINT or SIGTERM cancels HTTP streams and sessions, allows the configured connection shutdown grace, then flushes stores before exit.
- It prints `warm-mcp listening on http://<addr><path>` on stdout once bound (`--bind 127.0.0.1:0` picks a free port).

| HTTP resource | Flag | Default |
| --- | --- | --- |
| Accepted TCP connections, including idle keep-alive | `--max-http-connections` | 64 |
| Requests and response streams sharing admission capacity | `--max-http-inflight` | 32 |
| Header / response-header deadline, including request body reads | `--http-header-timeout-ms` / `--http-request-timeout-ms` | 5,000 / 30,000 ms |
| Response-body production deadline / absolute TCP-connection lifetime | `--http-stream-timeout-ms` / `--http-connection-timeout-ms` | 300,000 / 600,000 ms |
| Encoded response or SSE-stream bytes | `--max-http-response-bytes` | 32 MiB |
| Retained legacy sessions | `--max-http-sessions` | 16 |
| Idle legacy-session retention | `--http-session-idle-ms` | 300,000 ms |
| Legacy session operations / cumulative serialized outbound bytes | `--max-http-session-requests` / `--max-http-session-bytes` | 4,096 / 64 MiB |
| Connection shutdown grace | `--http-shutdown-grace-ms` | 10,000 ms |

The environment variable for each flag is its uppercase spelling with `WARM_MCP_` prepended
and hyphens replaced by underscores. Headers also have fixed caps of 64 fields and a 16 KiB
read buffer. Admission permits remain held until the response finishes or is dropped;
they do not become available when response headers are sent. The response deadline is
checked while polling body frames; a writer blocked by its peer remains bounded by the
absolute TCP lifetime.

Protocol revisions before `2026-07-28` retain in-memory session/replay state. Session
limits apply before that state can grow without bound: outbound messages are counted
before entering SDK replay caches, and request/notification/stream operations consume a
session quota. Idle sessions are removed from both application and SDK maps within one
reaper tick (at most one second). Interrupted initialization also retires the session.
At the session cap the SDK currently returns HTTP 500; after expiry or retirement the
client must initialize again. `2026-07-28` stateless requests use the HTTP limits instead.
Response byte or lifetime limits can end a stream after a mutation has been accepted;
reconcile the stored value/version before retrying an unacknowledged mutation. These are
memory/transport bounds, not a durable replay or exactly-once delivery guarantee.

## Security and logging

Remote OAuth mode is a resource-server profile using an external authorization server. Configure its RFC 8414 discovery, OAuth authorization/code/PKCE/client registration policies and RFC 7662 endpoint there. Set these values through the deployment's secret/configuration mechanism:

```text
WARM_MCP_TRANSPORT=http
WARM_MCP_OAUTH_ISSUER=https://identity.example/tenant
WARM_MCP_OAUTH_RESOURCE=https://memory.example/mcp
WARM_MCP_OAUTH_INTROSPECTION=https://identity.example/introspect
WARM_MCP_OAUTH_INTROSPECTION_TOKEN=<resource-server credential>
WARM_MCP_NAMESPACE=team
WARM_MCP_ALLOW_HOSTS=memory.example
```

Do not also set `WARM_MCP_TOKEN`. The public resource path must equal `--path`; OAuth mode requires the global namespace strategy and forbids `--namespace-arg`. The TLS proxy must restrict access to the internal listener and preserve the configured public host. Add allowed browser origins explicitly when needed.

Clients discover the external issuer through `/.well-known/oauth-protected-resource/mcp` and the `WWW-Authenticate` challenge. Each request requires a bearer token whose fresh introspection response has `active:true`, exact `iss`, resource in `aud`, nonempty `sub`, future integer `exp`, `token_type:Bearer`, and exact `hms_namespace:team`. Optional `nbf` and `iat` cannot be in the future. Every operation requires `hms:read`; `remember` and `forget` additionally require `hms:write`. The namespace claim is a deployment extension that the authority must issue from its own authorization policy, never from an unchecked client value.

This profile uses bearer tokens. It rejects sender-constrained `cnf` tokens because it does not verify their proofs. A stolen, still-active bearer token remains usable until expiry/revocation; there is no one-use or exactly-once mutation claim. Requests carrying session IDs are refused, and tokens are never used as session identifiers. Unacknowledged writes still require reconciliation before retrying.

Introspection uses only the administrator-pinned HTTPS issuer origin, with TLS verification, no redirects, proxy, token cache or automatic retries. Eight workers maximum retain admission until their blocking calls finish, even after client cancellation; the HTTP client deadline is five seconds. Token input is at most 8 KiB, responses 16 KiB, and request bodies at most the configured limit (hard maximum 16 MiB). Identity-service failure/overflow returns 503, invalid identity returns 401, and insufficient scope returns 403. The administrator controls the authority's DNS/network trust; clients cannot select outbound URLs. The full requirement and rejection map is in [remote-trust-profiles](../../research/technical-audit-20261029/remote-trust-profiles.md). Runtime validation is deferred under the campaign policy.

stdout carries only protocol (stdio) or the one listening line (http). Logs go to stderr at `--log` level (`warn` default) and contain operation names, durations and error codes, never memory text, queries, ids or the token or passphrase. An internal failure returns a fixed message to the client and the cause to stderr.

## Structure and the memory API

Handlers are written against the `MemoryBackend` trait (`src/bin/warm-mcp/memory.rs`): `remember`, `get`, `search`, `list`, `forget`, `stats` over a namespace. `WarmBackend` implements it on the document store. A richer memory-API module can implement the same trait and replace `WarmBackend` in `main` without touching tool parsing, validation, resources or prompts. The store gained three read-only methods for this server: `WarmCore::get_document`, `list_documents` and `document_count`.

## Testing

```sh
cargo test --features mcp --bin warm-mcp
cargo build --features mcp,security --bin warm-mcp
cd integrations/mcp/tests && npm install && WARM_MCP_BIN=$PWD/../../../target/debug/warm-mcp WARM_MCP_SECURITY=1 npm test
```

The end-to-end suite drives the binary with the official TypeScript SDK client: initialize and capabilities, tool listing and annotations, every tool, strict validation at the limit and limit plus one, resources and prompts, 60 concurrent writes with 30 concurrent reads, compare-and-set races, restart persistence over stdio and HTTP, lock contention, namespaces, tool exposure, encryption (no plaintext on disk, wrong passphrase refused, secrets absent from logs), HTTP auth, Host and Origin checks, body limits, graceful shutdown, and protocol negotiation over raw HTTP for all four protocol revisions.
