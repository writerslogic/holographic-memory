# Source: annoy

`warm-import annoy --index FILE --dim D --metric {angular,euclidean,manhattan,hamming,dot} [--id-map F] [--texts F] [--metadata F] [--namespace NS]`

Reads an Annoy index saved with `AnnoyIndex.save`. Verified against real `annoy` 1.17; no mocks.

- Items are `0 .. n_items-1` and the vectors are read with `get_item_vector` (Annoy keeps the original vectors, including for `angular`). Ids are `str(item)` unless an id map is given.
- Sidecars: same as [faiss](faiss.md), aligned to item number.
- Item numbers that were never added (sparse `add_item` ids) read back as all-zero vectors. They are indistinguishable from real zero vectors, which WARM cannot store either; they are yielded vector-less and reported under `skipped: empty` with their ids.
- `--dim` and `--metric` are not stored in the file. A wrong `--dim` usually fails to load (reported with that hint) but a wrong `--metric` cannot be detected and has no effect on the carried vectors.
- The file is memory-mapped by Annoy; reading is item by item. Seekable by item number.

Credentials: none.
