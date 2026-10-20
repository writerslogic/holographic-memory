# Generic files and databases: jsonl, csv, parquet, sqlite, duckdb, numpy

Verified against real files and real SQLite and DuckDB engines (no mocks).

`--map FIELD=COLUMN` (repeatable) maps columns to `id`, `text`, `vector`, `namespace`, `created_at`, `updated_at` and `metadata`. `metadata` takes comma-separated columns, `*` (default: every unmapped column), or one JSON/dict column that is spread into the metadata. Without `--map id` the row number is the id. Vector columns may be a list, JSON text (`[0.1, 0.2]`), delimited text, or little-endian float32 bytes. If no `text` mapping is given, a column named `text`, `content` or `document` is used.

| Source | Input | Resume |
|---|---|---|
| `jsonl` | `--path` file or glob, repeatable. A line that is not a JSON object is rejected with its byte offset. | seekable by byte offset |
| `csv` | header row; `--delimiter` (`\t` for TSV) | prefix-hash skip |
| `parquet` | streamed in 1024-row batches | prefix-hash skip |
| `sqlite` | `--db` read-only, `--table` (ordered by rowid, seekable) or `--query` | table: rowid; query: skip |
| `duckdb` | `--db` optional, `--query` (can read Parquet/CSV/JSON files directly); add `ORDER BY` for a stable resume order | prefix-hash skip |
| `numpy` | `--path` `.npy` (memory-mapped) or `.npz` (each array loaded whole); `--ids`, `--texts`, `--metadata` from an `.npz` key or a `.npy`/`.txt`/`.jsonl` file streamed in lockstep; `--namespace` | seekable by row |

Pickled object arrays are refused (`allow_pickle=False`); convert them to unicode arrays or JSONL.
