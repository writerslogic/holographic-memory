# Redis Stack / RediSearch

`hms-import redis --url redis://host:6379 --index idx` or `--prefix doc: --text-field content --vector-field embedding`

- Reads documents by `SCAN` over the key prefix (hashes via `HGETALL`, JSON via `JSON.GET key $`), pipelined per page. `--index` runs `FT.INFO` to discover the prefix, key type and vector attribute; without RediSearch use `--prefix`.
- Hash vector fields are raw little-endian float32 bytes (kept as bytes, never UTF-8 decoded); JSON vectors are arrays. Other binary hash values are stored base64 in metadata. Hash values are strings in Redis, so numeric metadata arrives as strings.
- Password from the URL or `$REDIS_PASSWORD`; never logged. Unreadable JSON documents are rejected with a reason, not dropped.
- SCAN has no stable order across mutation: resume re-reads and hash-verifies the committed prefix and restarts from the first record if it changed. Keys can be returned twice by SCAN; writes are idempotent by id.

Verification: hash/SCAN path REAL (redis-server 8, no modules): round trip, resume, missing-RediSearch error. `FT.INFO` parsing and JSON documents: MOCK (canned RESP2 reply and a fake client). RediSearch/RedisJSON modules themselves unverified.
