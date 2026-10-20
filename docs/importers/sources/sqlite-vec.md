# sqlite-vec

`hms-import sqlite-vec --db FILE --table VEC0 [--side-table T --side-key COL] [--map FIELD=COLUMN ...]`

Opens the database read-only, loads the `sqlite-vec` extension and pages the `vec0` table by `rowid` (keyset, so resume is exact and cheap: the cursor is the last rowid). The vector column is the first `float[N]`, `int8[N]` or `bit[N]` column of the `vec0` definition (override with `--vector-column`); int8 and bit vectors are decoded to floats.

vec0 stores only vectors plus its own metadata/auxiliary columns. Text normally lives in an ordinary side table joined on `--side-key` = vec0 rowid; side-table columns and vec0 metadata/auxiliary columns are all mappable with `--map` (`id`, `text`, `namespace`, `created_at`, `updated_at`, `metadata`). Unmapped columns become metadata. Without `--map id=...` the rowid is the id. `--namespace` sets one namespace for all rows.

Verification: real SQLite with the real extension (float and int8 tables). Limit: the Python build must support `enable_load_extension` (python.org and uv builds do; some system builds do not, which raises a clear error).
