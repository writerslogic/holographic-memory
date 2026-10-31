# warm-server errors

Every failure of an API call is an RFC 9457 `application/problem+json` document:

```json
{
  "type": "https://github.com/writerslogic/warm/blob/main/docs/server/errors.md#invalid_request",
  "title": "Invalid request",
  "status": 422,
  "detail": "top_k must be 1..=100",
  "instance": "urn:hms:request:8f2c1b7a90d3e451",
  "code": "invalid_request",
  "request_id": "8f2c1b7a90d3e451",
  "error": {"code": "invalid_request", "message": "top_k must be 1..=100"}
}
```

- `type` is a stable URI that resolves to the matching section below; `code` is the same identifier as a plain string. Branch on `code`, never on `detail`, which is prose and may change.
- `request_id` echoes the `X-Request-Id` the client sent (1 to 64 characters of `[A-Za-z0-9._-]`) or a generated one. Every response carries it as a header, and the server's log line for the request carries the same id.
- `error` is the object the first SDK releases read. It is deprecated but present for the whole of `/api/v1` (see `versioning.md`).
- Internal failures return a fixed `detail`; the cause is in the server log under the same request id. Details never include document text, metadata, vectors or credentials.
- `Retry-After` (seconds) accompanies `rate_limited`, `server_busy`, `shutting_down`, `idempotency_in_flight` and `quota_exceeded`.
- Failures raised by the HTTP layer before routing (a request line or header that is not valid HTTP, headers larger than the limit) are answered by the transport with a bare status and no body.

### invalid_json
400. The body is not valid JSON, or has the wrong shape for the endpoint.

### invalid_tenant
400. `X-Tenant-ID` occurs more than once or is not 1 to 64 characters of `[A-Za-z0-9_-]`.

### invalid_request
422. The body parsed but a value is not acceptable: wrong vector length or non-finite values, `top_k` out of range, an id with control characters or the reserved `hms:` prefix, text that does not fit one chunk, a batch over the limits, an archive that is not a valid store, a malformed query string. The request was not applied.

### unauthorized
401. Missing, unknown, expired, disabled or wrong credentials. `WWW-Authenticate: Bearer` is set.

### forbidden
403. The key is valid but lacks the scope the operation needs (`read`, `write` or `admin`).

### tenant_forbidden
403. The key is not valid for the tenant addressed, or the operation needs a key valid for every tenant.

### not_found
404. No such route, document or snapshot.

### method_not_allowed
405. The route exists but not for this method.

### request_timeout
408. The body did not arrive within `--body-timeout-seconds`.

### conflict
409. The request conflicts with the current state (for example a tenant store closed while the operation was starting; retry).

### idempotency_in_flight
409. Another request with the same `Idempotency-Key` is still running. Retry after the indicated delay to receive its recorded response.

### idempotency_key_reused
422. The `Idempotency-Key` was already used with a different method, path, tenant or body.

### idempotency_reconciliation_required
409. The original request may have executed, but its response is unavailable: the worker
panicked or the response could not fit the replay byte budget. The key remains reserved
until its TTL expires. Inspect the resulting state before deliberately issuing a new
operation. This response has no `Retry-After`; blindly retrying with a new key or after
expiry can repeat a committed mutation.

### payload_too_large
413. The body, stream, line or archive exceeds its limit (`--max-body-bytes`, `--max-stream-bytes`, `--max-stream-line-bytes`, `--max-restore-bytes`).

### unsupported_media_type
415. The stream endpoint needs `Content-Type: application/x-ndjson`.

### rate_limited
429. The tenant exceeded `--rate-limit-rps` / `--rate-limit-burst`. Retry after `Retry-After`.

### quota_exceeded
429. The write would take the tenant over `--max-documents-per-tenant`. Nothing was written. Delete documents or raise the quota.

### internal
500. An unexpected failure; see the server log for the request id.

### resource_limit
429. The query exceeded its work, frontier, or retained-byte budget. No partial answer is
returned. Narrow the query or change the applicable budget before retrying; waiting alone
does not resolve this limit, so there is no `Retry-After` header. The Node memory API uses
the same `resource_limit` code; raw native Node calls carry a `[resource_limit]` prefix.

### mutation_committed
500. The journal mutation is durable, but publication, audit, provenance, or cleanup failed.
Inspect the resulting state and reopen or reconcile it before retrying. A failure with this
code does not mean that nothing was written. Use stable IDs and the idempotency key when
retrying a logical operation.
Unlike transient server errors, this response remains in the idempotency cache. The same
key replays it during its TTL without repeating the durable mutation. If its response
cannot fit the replay byte budget, the key instead returns
`idempotency_reconciliation_required`. Reconcile state before starting a deliberate new
attempt. All retained keys are process-local and lost on restart.

### not_implemented
501. Reserved for operations a build does not provide.

### not_ready
503. `/readyz` while the server is starting or shutting down.

### server_busy
503. Work admission is full or timed out, or a new idempotency key cannot fit
`--idempotency-capacity` while existing keys remain protected. Safe to retry; this
rejected attempt did not execute the operation.

### concurrent_mutation
503. Concurrent writes exhausted the memory API's bounded snapshot or planning retries.
Retry after `Retry-After`. Multi-batch forgetting can have a committed prefix; the same
scope can be retried to finish it. Reads return a coherent revision or this error.

### tenant_limit
503. `--max-tenants` stores are open and none is idle enough to close.

### tenant_busy
503. A snapshot or restore of this tenant is running and holds its store exclusively. Requests are refused at once (with `Retry-After`) rather than queued, so one tenant's backup cannot occupy the server's work slots. Other tenants are unaffected.

### shutting_down
503. The server is draining and accepts no new work.
