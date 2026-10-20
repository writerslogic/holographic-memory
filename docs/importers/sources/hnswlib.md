# Source: hnswlib

`hms-import hnswlib --index FILE --dim D --space {l2,ip,cosine} [--id-map F] [--texts F] [--metadata F] [--namespace NS]`

Reads an hnswlib index saved with `Index.save_index`. Verified against real `hnswlib` 0.8; no mocks.

- Ids: the integer labels, sorted ascending (stable order); `str(label)` unless an id map is given. The sorted label list is held in memory (N Python ints); vectors are fetched 1024 at a time with `get_items`.
- Sidecars (`--id-map`, `--texts`, `--metadata`, `--namespace`, `--created-at-key`, `--updated-at-key`): same format and semantics as [faiss](faiss.md), aligned to the ascending-label order. A `.json` object id map renames labels.
- `--dim` and `--space` cannot be read back from the file reliably. The file header records the element size, so a wrong `--dim` is detected and reported with the dimension the index was built with; a wrong `--space` cannot be detected and does not change the carried vectors.
- `cosine` indexes store L2-normalized vectors (hnswlib normalizes on insert); the normalized vectors are what is imported. Cosine ranking is unchanged.
- Deleted labels (`mark_deleted`) stay in `get_ids_list` but their vectors are unreadable. They are yielded without a vector, so the run reports them under `skipped: empty` with their ids rather than dropping them unseen.
- Seekable by row position.

Credentials: none.
