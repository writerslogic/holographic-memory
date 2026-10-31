# pgvector (PostgreSQL and Supabase)

`warm-import pgvector --dsn ... --table docs --key-column id --map text=content --map vector=embedding --map metadata=meta --map namespace=tenant --map created_at=created_at`

- Connection: `--dsn`, else `$DATABASE_URL`, else libpq `PG*` variables. A Supabase project is a Postgres DSN (use the direct or pooler connection string). The password is registered as a secret and never printed, stored or reported.
- `--table` pages by `--key-column` (unique, sortable; keyset `WHERE key > last`), so runs resume from a cursor. `--query` streams a server-side cursor and resumes by skipping a verified prefix; give it a stable `ORDER BY`.
- `vector` and `halfvec` columns arrive as `[..]` text; `sparsevec` (`{1:0.5,3:1}/5`) is densified. `--map metadata=col` spreads a jsonb column into metadata keys; other listed columns copy by name; default is every unmapped column.
- Identifiers are validated (plain, optionally schema-qualified names only).

Verification: REAL. Postgres 18.4 with pgvector 0.8.7 (vector, halfvec, sparsevec), temporary cluster started by the test. Tests: round trip with `assert_faithful`, crash/resume, text keys, query mode.
Not covered: Supabase-hosted TLS/pooler behavior (same protocol, unverified); other pgvector-compatible forks.
