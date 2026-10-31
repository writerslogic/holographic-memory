# lancedb

`warm-import lancedb --uri DIR_OR_URI [--table NAME ...] [--vector-column COL] [--map FIELD=COLUMN ...]`

Reads every table (or the named ones) through the real `lancedb` client. Each table is checked out at its current version first, so concurrent writes cannot shift pages, then read with `limit/offset` pages (no `pylance` needed). The vector column is the first fixed-size-list column. Other columns become metadata (structs and lists stay structured). Without `--map id=...`, an `id` column is used if present, else the row ordinal.

Namespace is the table name unless `--map namespace=COLUMN`. Timestamps: `--map created_at=COL` / `updated_at=COL`.

Credentials: LanceDB Cloud (`db://`) key from `$LANCEDB_API_KEY`, never logged; `--region` for cloud.

Resume: not seekable; prefix-hash verified re-read.

Verification: real local LanceDB directory (round trip, nested metadata, all-tables default, resume). LanceDB Cloud was not run.
