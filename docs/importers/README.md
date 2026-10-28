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

Re-running is idempotent: HMS replaces a document with the same id. A crash at any point loses nothing; rerun the same command and it resumes after the last acknowledged batch. A completed checkpoint is not resumed, so a later run re-syncs the source from the start.

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
| id | `id`. Original kept in `metadata.hms_source_id`. Ids the server cannot hold (over 256 bytes, control characters, `hms:` prefix) become `h:<sha256>` deterministically and are counted (`ids_rewritten`). |
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

`--checkpoint` (default: derived from source, target and mapping under `$HMS_IMPORT_STATE_DIR` or `~/.cache/hms-import`). Seekable sources continue from a cursor; the others re-read the committed prefix while a chained hash of record ids confirms the order did not change. If it did, the import restarts from the first record (safe, writes replace by id) and the report says so (`resume_prefix_mismatch`).

## Credentials

Only from the environment or arguments (`HMS_API_KEY`, `HMS_ZERO_TRUST_KEY`, and each source's variable). They are never written to logs, reports, checkpoints or error output: known secrets and connection-string passwords are masked, and unexpected library errors print as a scrubbed one-line message, not a traceback.

## Limits

- HMS has no list or count endpoint: the report's counts come from the source and the server's acknowledgements, and the sampled check reads documents back. Documents deleted from the source, or parts of a re-split text, are not removed from HMS by a re-run.
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
