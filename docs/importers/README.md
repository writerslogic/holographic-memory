# hms-import: migrate into Holographic Memory

`hms-import` copies memories from another vector store or memory system into an `hms-server`. It streams, resumes from a checkpoint, is idempotent, and ends with a fidelity report so nothing is lost silently.

```sh
uv add 'holographic-memory-sdk[import-chroma]'          # one extra per source, see the table
export HMS_API_KEY=...                                  # destination credential (env, never logged)
hms-import chroma --path ./chroma-db --hms-url http://127.0.0.1:8080 --tenant acme \
  --report migration.json                               # dry run first: add --dry-run
```

`hms-import list` prints the sources; `hms-import <source> --help` lists each source's options and the environment variable holding its credential. The same pipeline is a library: `from holographic_sdk.importers import HmsSink, ImportOptions, run_import, source_class`.

## What a run does

1. Probes the server for its input dimension and whether it embeds text (`hms-server --features local-models`).
2. Streams records from the source in deterministic order, maps each to HMS documents, sends batches (`--batch-size`, default 256, clamped to the server's batch and body limits) and advances the checkpoint after each acknowledged batch.
3. Builds the document index (`/api/v1/index/maintain`) unless `--no-maintain-index`.
4. Runs the fidelity check on a sample of queries and writes the report.

Re-running the same source is idempotent. Each mapped part carries a source ownership identity;
the server atomically rejects a different owner or an unowned destination with the same id.
The importer requires a server whose OpenAPI contract exposes `import_identity` and
`import_adopt_legacy`; older servers fail before the first write. A completed checkpoint is
not resumed, so a later run imports the source from the start.

`--id-mapping v1` remains the default and preserves existing destination ids. Ambiguous
namespace/part spellings fail as identity conflicts instead of overwriting data. For a new
destination, use `--id-mapping v2`: bounded `h2:<sha256>` ids encode the typed source,
namespace, original id and part identity. Original identities remain in metadata. Switching
an existing import to v2 is an explicit layout migration, described below.

Old imports without ownership stamps require `--adopt-legacy`. Adoption requires an exact
match of original source, namespace, source id and part metadata; sharing an id is insufficient.
Adopt the existing layout before changing its id mapping or splitting options.

Changed part counts or mapped ids are rejected by default. `--split-policy replace` writes
the complete new layout in one atomic batch, then deletes obsolete parts conditionally on
their ownership identity. A record may have at most 1000 parts and a replacement must fit the
server's batch/body budgets. Checkpoints advance only after cleanup completes. If cleanup is
interrupted, rerun with the same options to finish; already absent parts are harmless.
`stale_parts_deleted_this_run` counts deletions in that process, including in partial error reports.

## Vectors: carry, project or re-encode

HMS stores one embedding space per store, and the server refuses to mix models. The importer therefore decides the space once per run (`--vectors`):

| Mode | Behavior |
|---|---|
| `auto` (default) | The first record decides. If it has a usable vector, source vectors are carried verbatim and every record must have one of the server's dimension (others are rejected, never encoded into a different space). Otherwise every text is encoded. A dimension mismatch stops the run before anything is written and names the alternatives. Nothing is ever projected silently. |
| `carry` | Send source vectors verbatim; records without a usable one are rejected. |
| `reencode` | Ignore source vectors, encode every text with the server's on-device model, or with `--embedder module:func` (a callable `list[str] -> list[list[float]]`) when the server has none. Switching models means queries must then be encoded the same way. |
| `project` | Deterministic seeded Gaussian random projection to the server dimension (`--projection-seed`), zero padding when the source is smaller (exact for cosine). Approximate when shrinking: the dry run reports the recall it costs. |
| `none` | Text and metadata only (server must embed). |

Vectors with NaN, infinity or all zeros are unusable; they are counted (`vectors_unusable`) and handled as missing.

## What is preserved

| Source datum | HMS |
|---|---|
| id | v1 preserves `id` (or its namespace prefix); original kept in `metadata.hms_source_id`. Invalid v1 ids become `h:<sha256>` deterministically (`ids_rewritten`). v2 uses typed `h2:<sha256>` identities. Every mode checks ownership before replacement. |
| text | `text`. A text over 4000 words or 60 KiB is split at whitespace into parts `id#part-N` (`hms_parent_id`, `hms_part`, `hms_parts` in metadata); with carried vectors every part gets the record's vector. The server trims whitespace around each stored text, so whitespace at part boundaries is not retained. |
| metadata | Stored flat under the same keys (filterable with `/query` `filter`). Non-JSON values are converted reversibly (bytes as `base64:...`, non-finite floats as strings) and counted under `metadata_coercions`. Keys starting `hms_` are reserved; a colliding source key moves into `hms_orig_meta`. |
| namespace | `--namespace-mode metadata` (default): `metadata.hms_namespace` and the id becomes `<namespace>::<id>` (`--id-prefix-namespace`). `tenant`: one HMS tenant per namespace (`--tenant-prefix`; names are sanitized and hashed when needed; the server allows 64 tenants by default). |
| timestamps | `metadata.hms_created_at` / `hms_updated_at` as ISO-8601 UTC (seconds, milliseconds, microseconds, ISO strings and datetimes are recognized). An unparseable value is kept verbatim in `hms_created_at_raw` / `hms_updated_at_raw`. |
| oversize metadata (over 60 KiB) | rejected by default; `--oversize-metadata truncate` drops the largest keys and lists them in `hms_dropped_keys`. |

Every record that cannot be imported is rejected with a reason, listed in `<checkpoint>.rejects.jsonl` (ids and reasons only, never content), and the exit code is 3 unless `--allow-rejects`. Records with neither text nor vector are skipped and counted (`--keep-empty` rejects them instead).

## Dry run and the fidelity report

`--dry-run` reads and maps the whole source and writes nothing. The report (`--report file.json`, summary on stdout) holds counts read/planned/acknowledged, skipped and rejected by reason with sample ids, per-namespace counts, source vector dimensions, metadata coercions, server capabilities and timing.

The recall check samples `--verify-sample` records (default 50). Ground truth is exact cosine top-`--verify-k` over the source's own vectors, computed by streaming the source once (bounded memory), so it does not depend on the old system's ANN index. After a real import the new side is the live server queried with the same vectors (carry/project) or the text (re-encode); in a dry run it is exact cosine over the vectors that would be sent, which measures the loss from `project`. The report gives `recall_at_k_mean`, `recall_at_k_min`, `self_hit_rate` and `roundtrip_mismatches` (returned text and metadata compared with what was sent). Sources without vectors have no old-system recall; only self retrieval and the round trip are checked. With `reencode` against a source that has vectors, recall measures how much the two embedding models agree on neighbors, not an error.

## Checkpoints and resuming

`--checkpoint` defaults to a source/target/mapping-derived path under `$HMS_IMPORT_STATE_DIR`
or `~/.cache/hms-import`. Checkpoint format 2 hashes each committed record's identity, text,
vector, metadata and timestamps. Resume re-reads and verifies the committed prefix, including
for sources with cursors: a cursor alone cannot prove the source stayed unchanged. A mismatch
restarts from record zero and sets `resume_prefix_mismatch`. Prefix verification is streaming
and costs O(committed records) source reads. Format 1 checkpoints require an explicit
`--restart`; the destination's ownership checks still apply.

Imports consume an observed source stream. The importer does not create a cross-system
snapshot or a transaction spanning source pages. For a point-in-time migration, freeze the
source or supply an immutable export / source-managed snapshot. A matching committed prefix
does not establish that the uncommitted suffix or a concurrent verification scan was unchanged.
The report's `source_consistency` records this boundary and the retention of source-deleted
records. A completed rerun updates encountered records; it does not mirror deletions.

## Credentials

Only from the environment or arguments (`HMS_API_KEY`, `HMS_ZERO_TRUST_KEY`, and each source's variable). They are never written to logs, reports, checkpoints or error output: known secrets and connection-string passwords are masked, and unexpected library errors print as a scrubbed one-line message, not a traceback.

## Limits

- The importer scans the server's paginated metadata into a temporary disk-backed ownership
  inventory; it does not retain all document text. General deletions from a source are not
  propagated. Obsolete split parts require the explicit replacement policy above.
- The server embeds text only in builds with `local-models`; otherwise text-only sources need `--embedder`.
- `--zero-trust-key` masks vectors with the SDK's `VectorMask` (obfuscation, not encryption).

## Sources and how each was verified

R = verified against a real instance in this repository's tests; M = faithful mock of the documented API (not a live service). Per-source notes, options and mappings: `docs/importers/sources/<name>.md`.

| Source | Name | Install extra | Verified |
|---|---|---|---|
| JSONL, CSV, Parquet, SQLite, DuckDB, NumPy `.npy/.npz` with `--map` column mapping | `jsonl` `csv` `parquet` `sqlite` `duckdb` `numpy` | `import-files` | R |
| Markdown memory folders (CLAUDE.md, Claude memory tool dir, auto-memory) | `markdown` | none | R |
| Chroma | `chroma` | `import-chroma` | R (local persistent client) |
| sqlite-vec | `sqlite-vec` | `import-sqlite-vec` | R |
| LanceDB | `lancedb` | `import-lancedb` | R (local) |
| Qdrant | `qdrant` | `import-qdrant` | R local path; M server |
| FAISS (index + id map) | `faiss` | `import-faiss` | R |
| hnswlib, Annoy | `hnswlib` `annoy` | `import-hnswlib` `import-annoy` | R |
| Milvus / Milvus Lite | `milvus` | `import-milvus` | R (Milvus Lite) |
| pgvector (Postgres, Supabase) | `pgvector` | `import-pgvector` | R (Postgres 18 with pgvector) |
| Redis Stack / RediSearch | `redis` | `import-redis` | R hash and SCAN path; M `FT.INFO` and JSON |
| MongoDB Atlas | `mongodb` | `import-mongodb` | M |
| Elasticsearch, OpenSearch | `elasticsearch` `opensearch` | none (httpx) | M |
| Weaviate | `weaviate` | none (httpx) | M; R with embedded Weaviate when `HMS_TEST_WEAVIATE_EMBEDDED=1` |
| Vespa | `vespa` | none (httpx) | M |
| Pinecone (namespaces) | `pinecone` | `import-pinecone` | M (real client, mock data plane) |
| OpenAI vector stores | `openai-vector-store` | `import-openai` | M (real SDK, mock API) |
| LangChain (FAISS folder, InMemoryVectorStore, docstores) | `langchain` | `import-langchain` | R |
| LlamaIndex persisted storage | `llamaindex` | `import-llamaindex` | R |
| mem0 (OSS, platform), Zep, Letta | `mem0` `mem0-platform` `zep` `letta` | `import-mem0` `import-zep` `import-letta` | see each source page |

Exit codes: 0 success, 1 failure (resume with the same command), 2 usage, 3 finished with rejected records.
